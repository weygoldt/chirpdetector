"""Output files of the pipeline: per-chirp table, metadata, legacy arrays.

Layout of a run::

    OUT/<run_name>/
      run.json                   manifest: config, code, models, recordings
      run.log                    log of every invocation (appended)
      config.toml                the config file the run was started with
      columns.md                 the chirp table's columns, documented
      candidates.csv             validation candidates (labeller format)
      <recording>/
        chirps_<model>.parquet   one row per detection >= store threshold
        chunks_<model>.parquet   one row per chunk (span): clicks, timings
        chirp_times_<model>.npy  legacy export (see `legacy_arrays`)
        chirp_ids_<model>.npy
        meta.json                inputs, models, code, config, counts, timings
        qc/                      QC report (qc.png, qc.md)
        windows/                 annotated sample windows (png + npz)

``<model>`` names the detector family (``rtdetr``, ``yolo``); the weights'
sha256 and architecture are in ``meta.json`` and in the parquet file's
metadata (key ``chirpdetector``).

A recording's directory appears only when everything in it is written:
outputs go to a hidden temporary sibling that is renamed at the end
(`atomic_dir`). ``meta.json`` with ``"status": "complete"`` marks a
finished recording.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from chirpdetector.bundle import Bundle
from chirpdetector.pipeline.assign import ATTRIBUTED

__all__ = [
    "COLUMN_DOCS",
    "LEGACY_DTYPE",
    "atomic_dir",
    "atomic_write_text",
    "file_record",
    "legacy_arrays",
    "model_tag",
    "read_chirps",
    "recording_start",
    "sha256_of",
    "write_json",
    "write_parquet",
]

LEGACY_DTYPE = np.float64
"""dtype of both legacy arrays (as v1's ``chirp_*_rcnn.npy``)."""


COLUMN_DOCS: dict[str, str] = {
    "recording": "recording directory name",
    "chirp_id": "row number after sorting by t_center_s (0-based, per recording and run)",
    "t_start_s": "box start, seconds from the recording's first sample",
    "t_end_s": "box end, seconds from the recording's first sample",
    "t_center_s": "box centre (t_start_s + t_end_s) / 2; the legacy chirp time",
    "f_low_hz": "box lower edge (Hz; halved for re-attributed copies)",
    "f_high_hz": "box upper edge (Hz; halved for re-attributed copies)",
    "score": "detector score (stored down to the store threshold)",
    "operating": "score >= the run's operating threshold",
    "kept": "not flagged by post-processing (pp_drop_reason == '')",
    "chirp": "operating and kept: counts as a chirp",
    "emitter_hz": "emitter baseline EODf predicted by the dual-head (NaN for box-only models; halved for re-attributed copies)",
    "track_id": "assigned wavetracker identity, ident_v value unchanged (float64; NaN = unassigned)",
    "track_hz": "assigned track's baseline before the onset (Hz; NaN if unassigned)",
    "emitter_track_df_hz": "|emitter_hz - track_hz| of the assigned track",
    "assign_status": "assigned | reattributed | beyond_tolerance | no_live_track | no_tracks | no_emitter",
    "assign_confidence": "posterior that the nearest live track is the emitter (Student-t error model, see assign.py; NaN for re-attributed copies)",
    "nearest_track_id": "nearest live track to emitter_hz, regardless of the tolerance",
    "nearest_track_hz": "its baseline (Hz)",
    "nearest_df_hz": "|emitter_hz - nearest_track_hz|",
    "second_track_id": "second-nearest live track",
    "second_track_hz": "its baseline (Hz)",
    "second_df_hz": "|emitter_hz - second_track_hz|",
    "assign_margin_hz": "second_df_hz - nearest_df_hz",
    "n_live_tracks": "tracks live at the onset",
    "track_df_other_hz": "|baseline of the nearest track - baseline of the closest other live track| (ΔF)",
    "height_hz": "decoder: chirp height above baseline (calibrated, Hz)",
    "height_sd_hz": "decoder: calibrated sd of height_hz",
    "fwhm_s": "decoder: full width at half height (s)",
    "fwhm_sd_s": "decoder: calibrated sd of fwhm_s",
    "width10_s": "decoder: width at 10 % of the height (s)",
    "width10_sd_s": "decoder: calibrated sd of width10_s",
    "undershoot_hz": "decoder: undershoot below baseline after the peak (Hz)",
    "undershoot_sd_hz": "decoder: calibrated sd of undershoot_hz",
    "params_ok": "decoder: box inside the decoder's validated domain (False for re-attributed copies)",
    "params_flag": "decoder: why not (edge, band, size, uncertain; 'harmonic' for re-attributed copies, whose decoder values are NaN)",
    "height_raw_hz": "decoder: uncalibrated height (shrunk toward the training mix; for population spreads)",
    "fwhm_raw_s": "decoder: uncalibrated FWHM",
    "width10_raw_s": "decoder: uncalibrated 10 % width",
    "undershoot_raw_hz": "decoder: uncalibrated undershoot",
    "pp_drop_reason": "post-processing: '' (kept), 'harmonic' or 'duplicate'",
    "pp_harmonic_of_track": "post-processing: track whose 2nd harmonic the box lies on",
    "pp_harmonic_pair_of": "post-processing (trackless pair rule): chirp_id of the detection at half the frequency",
    "pp_duplicate_of": "post-processing: chirp_id of the kept twin",
    "pp_reattributed": "post-processing: 'harmonic' for a copy whose fundamental was not detected, kept and given to its fish (pp_harmonic_of_track) at half the frequency; evaluated at the operating threshold",
    "pp_harmonic_emitter_hz": "re-attributed copies: the detected emitter_hz (at the harmonic)",
    "pp_harmonic_f_low_hz": "re-attributed copies: the detected box's lower edge",
    "pp_harmonic_f_high_hz": "re-attributed copies: the detected box's upper edge",
    "click_fraction_box": "fraction of the box's samples with a repaired click (any electrode; NaN: no click removal)",
    "span_click_fraction": "fraction of the span's samples with a repaired click",
    "span_n_click_runs": "repaired click runs in the span",
    "tile": "global tile index (tiles of 3.2 s stepping by 2.2 s from the recording's start)",
    "span_id": "chunk (span) index; see chunks_<model>.parquet",
    "model_sha256": "sha256 of the detector weights",
}
"""Documentation of the chirp table's columns, in table order."""


def model_tag(bundle: Bundle) -> str:
    """Short name of the detector family for file names (``rtdetr``, ``yolo``)."""
    arch = str(bundle.model.get("architecture", bundle.model_type)).lower()
    match = re.match(r"[a-z]+", arch)
    return match.group(0) if match else re.sub(r"\W+", "", bundle.model_type)


def sha256_of(path: Path, block: int = 1 << 20) -> str:
    """sha256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path, *, hash_: bool = False) -> dict[str, Any]:
    """Path, size and modification time of a file (and its sha256)."""
    st = path.stat()
    out: dict[str, Any] = {
        "path": str(path),
        "bytes": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    }
    if hash_:
        out["sha256"] = sha256_of(path)
    return out


def recording_start(
    directory: Path, entries: Mapping[str, str] | None
) -> dict[str, Any]:
    """Absolute start time of a recording, from fishgrid.cfg and its name.

    ``fishgrid``: ``StartDate``/``StartTime`` of fishgrid.cfg (None if
    absent or ``~``); ``dirname``: from a ``YYYY-MM-DD-HH_MM`` directory
    name; ``agree``: whether both exist and agree to the minute. Times are
    local lab time without a time zone.
    """
    out: dict[str, Any] = {"fishgrid": None, "dirname": None, "agree": None}
    if entries:
        date = (entries.get("StartDate") or "").strip()
        clock = (entries.get("StartTime") or "").strip()
        nums = re.findall(r"\d+", clock)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) and len(nums) >= 2:
            h, m = int(nums[0]), int(nums[1])
            s = int(nums[2]) if len(nums) > 2 else 0
            out["fishgrid"] = f"{date}T{h:02d}:{m:02d}:{s:02d}"
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})-(\d{2})_(\d{2})", directory.name)
    if match:
        out["dirname"] = f"{match.group(1)}T{match.group(2)}:{match.group(3)}:00"
    if out["fishgrid"] and out["dirname"]:
        out["agree"] = out["fishgrid"][:16] == out["dirname"][:16]
    return out


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically (temporary file + rename)."""
    atomic_write_text(path, json.dumps(data, indent=1, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def atomic_write_text(path: Path, text: str) -> None:
    """Write a text file via a temporary sibling and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_parquet(path: Path, table: pd.DataFrame, metadata: Mapping[str, Any]) -> None:
    """Write a table with JSON metadata under the key ``chirpdetector``."""
    arrow = pa.Table.from_pandas(table, preserve_index=False)
    meta = dict(arrow.schema.metadata or {})
    meta[b"chirpdetector"] = json.dumps(metadata, default=_json_default).encode()
    pq.write_table(arrow.replace_schema_metadata(meta), path, compression="zstd")


def read_chirps(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A chirp table and its ``chirpdetector`` metadata."""
    table = pq.read_table(path)
    raw = (table.schema.metadata or {}).get(b"chirpdetector", b"{}")
    return table.to_pandas(), json.loads(raw)


def legacy_arrays(chirps: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``chirp_times`` / ``chirp_ids`` with the semantics of v1's ``_rcnn`` arrays.

    v1 (tag paper-chirp-competition-v1, ``detect_chirps.py``) wrote the box
    centre in seconds from the recording's first sample and the raw
    ``ident_v`` value of the assigned track, both float64 and 1-D, for the
    detections with score >= 0.5 that had an assigned track (unassigned
    detections are not in the arrays; no NaN, no -1). Here: rows with
    ``chirp`` (score >= operating threshold and not flagged by
    post-processing) and ``assign_status`` ``assigned`` or ``reattributed``
    (a harmonic copy whose fundamental was missed counts for its fish),
    sorted by time
    (v1 sorted by box start, so its times were only nearly sorted).
    """
    if not len(chirps):
        return np.zeros(0, dtype=LEGACY_DTYPE), np.zeros(0, dtype=LEGACY_DTYPE)
    sel = chirps.loc[
        chirps["chirp"].to_numpy(dtype=bool)
        & np.isin(chirps["assign_status"].to_numpy(dtype=object), ATTRIBUTED)
    ].sort_values("t_center_s", kind="stable")
    times = sel["t_center_s"].to_numpy(dtype=LEGACY_DTYPE)
    ids = sel["track_id"].to_numpy(dtype=LEGACY_DTYPE)
    if np.isnan(ids).any() or np.isnan(times).any():  # pragma: no cover - guarded above
        msg = "legacy arrays must not contain NaN"
        raise ValueError(msg)
    return times, ids


@contextlib.contextmanager
def atomic_dir(final: Path) -> Iterator[Path]:
    """A temporary directory renamed to ``final`` on success.

    An existing ``final`` (an earlier, incomplete or outdated result) is
    moved aside and removed only after the rename. On error the temporary
    directory is removed and ``final`` is left as it was.
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    tmp.mkdir()
    try:
        yield tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    old = None
    if final.exists():
        old = final.with_name(f".{final.name}.old-{uuid.uuid4().hex[:6]}")
        os.replace(final, old)
    os.replace(tmp, final)
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)
