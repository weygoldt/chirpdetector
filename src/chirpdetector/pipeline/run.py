"""Run the pipeline over recordings: resumable, one failure never stops the run.

`run_pipeline` processes recordings in order. A recording is skipped when
its directory holds a ``meta.json`` with ``"status": "complete"`` made with
the same detection settings (`PipelineConfig.detection_hash`) and time
range; otherwise it is (re)computed into a temporary directory that
replaces the old one only when complete (`outputs.atomic_dir`). Errors are
logged with their traceback, recorded in ``run.json`` and the run moves on;
the exit status reports them. The run log (``run.log``) is appended to.

Per recording (`process_recording`):

1. open the recording (memory-mapped) and its tracks;
2. detect and decode chunk by chunk (`stream.detect_recording`);
3. post-process (`chirpdetector.postprocess`, if available: harmonic copies
   and duplicates are flagged, never removed);
4. assign to tracks (`assign.assign_detections`);
5. write the chirp table, chunk table, legacy arrays, QC report, sample
   windows and ``meta.json`` (last).
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import platform
import shutil
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gridtools.io import Tracks, load_tracks, open_recording, read_fishgrid_config

from chirpdetector.bundle import Bundle, load_bundle
from chirpdetector.pipeline.assign import (
    ASSIGN_COLUMNS,
    SIM_V4_DUALHEAD,
    ErrorModel,
    assign_detections,
)
from chirpdetector.pipeline.config import PipelineConfig
from chirpdetector.pipeline.outputs import (
    COLUMN_DOCS,
    atomic_dir,
    file_record,
    legacy_arrays,
    model_tag,
    recording_start,
    sha256_of,
    write_json,
    write_parquet,
)
from chirpdetector.pipeline.stream import detect_recording
from chirpdetector.prediction import load_detector
from chirpdetector.resources import apply_vram_limit

__all__ = [
    "Models",
    "RunSummary",
    "discover_recordings",
    "load_models",
    "process_recording",
    "run_pipeline",
    "tracks_dir",
]

log = logging.getLogger(__name__)

META_FILE = "meta.json"
TRACK_FILES = ("times", "fund_v", "ident_v", "idx_v", "sign_v")


# ------------------------------------------------------------------ inputs


def is_recording_dir(path: Path) -> bool:
    """A fishgrid directory (``traces-grid1.raw``) or one holding one wav."""
    if not path.is_dir():
        return False
    if (path / "traces-grid1.raw").exists():
        return True
    return len(list(path.glob("*.wav"))) == 1


def discover_recordings(
    paths: Sequence[Path] = (), dataset: Path | None = None
) -> list[Path]:
    """Recording directories: the given ones, plus every one under ``dataset``.

    Raises
    ------
    FileNotFoundError
        If a given path is not a recording directory.
    """
    found: list[Path] = []
    for p in paths:
        if not is_recording_dir(p):
            msg = (
                f"{p} is not a recording directory (no traces-grid1.raw or single wav)"
            )
            raise FileNotFoundError(msg)
        found.append(p.resolve())
    if dataset is not None:
        found += sorted(d.resolve() for d in dataset.iterdir() if is_recording_dir(d))
    unique: dict[Path, None] = dict.fromkeys(found)
    names = [p.name for p in unique]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        msg = f"recordings with the same directory name: {sorted(dup)}"
        raise ValueError(msg)
    return list(unique)


def tracks_dir(recording_dir: Path, source: str) -> Path | None:
    """Directory of the wavetracker arrays for a recording (None: disabled)."""
    if source.strip().lower() == "none":
        return None
    if "{recording}" in source:
        return Path(source.format(recording=recording_dir.name))
    p = Path(source)
    return p if p.is_absolute() else (recording_dir / p).resolve()


def _load_tracks(directory: Path | None) -> tuple[Tracks | None, dict[str, Any]]:
    info: dict[str, Any] = {"dir": None if directory is None else str(directory)}
    if directory is None:
        info["status"] = "disabled"
        return None, info
    files = {n: directory / f"{n}.npy" for n in TRACK_FILES}
    missing = [n for n, f in files.items() if not f.exists() and n != "sign_v"]
    if missing:
        info["status"] = f"missing {missing}"
        log.warning("%s: no tracks (missing %s)", directory, missing)
        return None, info
    tracks = load_tracks(directory)
    records = {n: file_record(f, hash_=True) for n, f in files.items() if f.exists()}
    combined = "".join(records[n]["sha256"] for n in sorted(records))
    info |= {
        "status": "loaded",
        "files": records,
        "sha256": hashlib_sha256(combined),
        "n_tracks": len(tracks.ids),
        "n_detections": len(tracks.freqs_hz),
        "unassigned_fraction": float(np.mean(~np.isfinite(tracks.idents)))
        if len(tracks.idents)
        else 0.0,
        "time_range_s": [float(tracks.times_s[0]), float(tracks.times_s[-1])]
        if len(tracks.times_s)
        else None,
    }
    return tracks, info


def hashlib_sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


# ------------------------------------------------------------------ models


@dataclass
class Models:
    """Loaded models of a run."""

    bundle: Bundle
    detector: Any
    estimator: Any | None
    tag: str
    has_emitter: bool
    error_model: ErrorModel | None
    info: dict[str, Any] = field(default_factory=dict)


def load_models(config: PipelineConfig) -> Models:
    """Load (and cross-check) the detector and the decoder."""
    device = config.compute.device
    apply_vram_limit(config.compute.max_vram_fraction, device)
    bundle = load_bundle(config.models.detector)
    detector = load_detector(bundle, device=device)
    estimator = None
    dec_info = None
    if config.models.decoder is not None:
        from chirpdetector.params.inference import ParamsEstimator

        estimator = ParamsEstimator(config.models.decoder, device=device)
        estimator.check_detector(
            bundle, allow_mismatch=config.models.allow_detector_mismatch
        )
        dec_info = {
            "path": str(config.models.decoder),
            "weights_sha256": estimator.bundle.weights_sha256,
            "detector_sha256": (estimator.bundle.detector or {}).get("weights_sha256"),
            "architecture": estimator.bundle.model.get("architecture"),
        }
    has_emitter = bundle.model_type == "dualhead"
    if not has_emitter:
        log.warning(
            "detector %s (%s) predicts no emitter frequency: track assignment is "
            "not supported; every detection gets assign_status 'no_emitter'",
            bundle.path,
            bundle.model.get("architecture"),
        )
    info = {
        "detector": {
            "path": str(bundle.path),
            "model_type": bundle.model_type,
            "architecture": bundle.model.get("architecture"),
            "weights_sha256": bundle.weights_sha256,
            "weights_short": bundle.weights_sha256[:8],
            "bundle_score_threshold": bundle.score_threshold,
            "training_run": bundle.training.get("run_id"),
            "preprocessing": dict(bundle.preprocessing),
            "spectrogram": bundle.spectrogram.to_dict(),
            "tiling": bundle.tiling.to_dict(),
            "emitter_head": has_emitter,
        },
        "decoder": dec_info,
        "tag": model_tag(bundle),
    }
    return Models(
        bundle=bundle,
        detector=detector,
        estimator=estimator,
        tag=model_tag(bundle),
        has_emitter=has_emitter,
        error_model=SIM_V4_DUALHEAD if has_emitter else None,
        info=info,
    )


# ------------------------------------------------------------- postprocess


def _postprocess(
    detections: pd.DataFrame, tracks: Tracks | None, name: str, enabled: bool
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Flag harmonic copies and duplicates, as `predict_span` does.

    Uses `chirpdetector.postprocess` if this installation has it; then
    emitters are first assigned with `assign_by_frequency` at the
    post-processing's fundamental tolerance, as `predict_span` does, and
    that provisional assignment is removed again (the pipeline's own
    assignment follows). Without the module, nothing is flagged.
    """
    n = len(detections)
    empty = {
        "pp_drop_reason": np.full(n, "", dtype=object),
        "pp_harmonic_of_track": np.full(n, np.nan),
        "pp_harmonic_pair_of": pd.array([pd.NA] * n, dtype="Int64"),
        "pp_duplicate_of": pd.array([pd.NA] * n, dtype="Int64"),
    }
    if not enabled:
        return detections.assign(**empty), {"status": "off"}
    try:
        pp = importlib.import_module("chirpdetector.postprocess")
    except ImportError:
        log.warning("chirpdetector.postprocess is not available; nothing flagged")
        return detections.assign(**empty), {"status": "unavailable"}
    config = pp.PostprocessConfig()
    table = detections.reset_index(drop=True)
    provisional = False
    if tracks is not None and "emitter_hz" in table.columns:
        from chirpdetector.evaluation.assignment import assign_by_frequency

        table = assign_by_frequency(
            table, {name: tracks}, float(config.fundamental_tolerance_hz)
        )
        provisional = True
    flagged = pp.flag_detections(
        table, None if tracks is None else {name: tracks}, config
    )
    out = detections.reset_index(drop=True).copy()
    out["pp_drop_reason"] = flagged["drop_reason"].astype(object).to_numpy()
    out["pp_harmonic_of_track"] = (
        flagged["harmonic_of_id"]
        .astype("Float64")
        .to_numpy(dtype=np.float64, na_value=np.nan)
    )
    out["pp_harmonic_pair_of"] = flagged["harmonic_pair_of"].astype("Int64").to_numpy()
    out["pp_duplicate_of"] = flagged["duplicate_of"].astype("Int64").to_numpy()
    for col in ("pp_harmonic_pair_of", "pp_duplicate_of"):
        out[col] = out[col].astype("Int64")
    info = {
        "status": "applied",
        "config": _asdict(config),
        "provisional_assignment": (
            f"assign_by_frequency at the box centre, tolerance "
            f"{config.fundamental_tolerance_hz} Hz (as predict_span)"
            if provisional
            else None
        ),
        "counts": {k: int(v) for k, v in pp.summarize(flagged).items()},
    }
    return out, info


def _asdict(obj: Any) -> dict[str, Any]:
    from dataclasses import asdict, is_dataclass

    return asdict(obj) if is_dataclass(obj) and not isinstance(obj, type) else {}


# ------------------------------------------------------------------ table


def build_table(
    name: str,
    detections: pd.DataFrame,
    assignment: pd.DataFrame,
    chunks: pd.DataFrame,
    operating_threshold: float,
) -> pd.DataFrame:
    """The per-chirp table (columns as `COLUMN_DOCS`, sorted by time)."""
    df = pd.concat(
        [detections.reset_index(drop=True), assignment.reset_index(drop=True)], axis=1
    )
    n = len(df)
    if n == 0:
        return _empty_table(detections)
    df["recording"] = name
    df["t_center_s"] = 0.5 * (df["t_start_s"] + df["t_end_s"])
    df["operating"] = df["score"].to_numpy() >= operating_threshold
    df["kept"] = df["pp_drop_reason"].to_numpy(dtype=object) == ""
    df["chirp"] = df["operating"] & df["kept"]
    if "emitter_hz" not in df.columns:
        df["emitter_hz"] = np.nan
    # sort by time; references (row positions) become chirp ids
    order = np.argsort(df["t_center_s"].to_numpy(), kind="stable")
    new_id = np.empty(n, dtype=np.int64)
    new_id[order] = np.arange(n)
    for col in ("pp_harmonic_pair_of", "pp_duplicate_of"):
        ref = df[col].astype("Int64")
        mapped = [pd.NA if pd.isna(r) else int(new_id[int(r)]) for r in ref]
        df[col] = pd.array(mapped, dtype="Int64")
    df = df.iloc[order].reset_index(drop=True)
    df["chirp_id"] = np.arange(n, dtype=np.int64)
    span = chunks.set_index("span_id") if len(chunks) else None
    if span is not None:
        df["span_click_fraction"] = (
            df["span_id"].map(span["click_fraction"]).astype(float)
        )
        df["span_n_click_runs"] = (
            df["span_id"].map(span["n_click_runs"]).astype("int64")
        )
    else:
        df["span_click_fraction"] = np.nan
        df["span_n_click_runs"] = 0
    df = df.rename(columns={"tile": "tile"})
    ordered = [c for c in COLUMN_DOCS if c in df.columns]
    raw_decoder = [c for c in df.columns if c.endswith("_raw") or c.startswith("raw_")]
    extra = [c for c in df.columns if c not in ordered and c not in raw_decoder]
    extra = [c for c in extra if c not in ("image_index",)]
    # raw decoder columns right after the calibrated ones
    pos = ordered.index("params_flag") + 1 if "params_flag" in ordered else len(ordered)
    columns = ordered[:pos] + raw_decoder + ordered[pos:] + extra
    return df.loc[:, columns]


def _empty_table(detections: pd.DataFrame) -> pd.DataFrame:
    cols = list(COLUMN_DOCS)
    data: dict[str, Any] = {}
    for c in cols:
        if c in (
            "recording",
            "assign_status",
            "params_flag",
            "pp_drop_reason",
            "model_sha256",
        ):
            data[c] = pd.Series(dtype=object)
        elif c in ("operating", "kept", "chirp", "params_ok"):
            data[c] = pd.Series(dtype=bool)
        elif c in ("chirp_id", "tile", "span_id", "n_live_tracks", "span_n_click_runs"):
            data[c] = pd.Series(dtype=np.int64)
        elif c in ("pp_harmonic_pair_of", "pp_duplicate_of"):
            data[c] = pd.Series(dtype="Int64")
        else:
            data[c] = pd.Series(dtype=np.float64)
    out = pd.DataFrame(data)
    if "params_flag" not in detections.columns:
        out = out.drop(
            columns=[
                c
                for c in out.columns
                if c.startswith(("height", "fwhm", "width10", "undershoot", "params_"))
            ]
        )
    return out


# --------------------------------------------------------------- recording


@dataclass
class RecordingResult:
    """Outcome of one recording."""

    name: str
    status: str
    duration_s: float = 0.0
    wall_s: float = 0.0
    counts: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _done(directory: Path, detection_hash: str, t_range: list[float] | None) -> bool:
    meta = directory / META_FILE
    if not meta.exists():
        return False
    try:
        data = json.loads(meta.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        data.get("status") == "complete"
        and data.get("detection_hash") == detection_hash
        and data.get("t_range_s") == t_range
    )


def process_recording(
    recording_dir: Path,
    config: PipelineConfig,
    models: Models,
    *,
    t_range_s: tuple[float, float] | None = None,
    progress: Any = None,
) -> RecordingResult:
    """Detect, assign and write all outputs of one recording (module docstring)."""
    t_wall = time.perf_counter()
    name = recording_dir.name
    final = config.run_dir / name
    recording = open_recording(recording_dir)
    fs = float(recording.samplerate_hz)
    trk_dir = tracks_dir(recording_dir, config.assignment.tracks)
    tracks, track_info = _load_tracks(trk_dir)
    entries = None
    if (recording_dir / "fishgrid.cfg").exists():
        entries = read_fishgrid_config(recording_dir).entries
    tag = models.tag
    bundle = models.bundle
    operating = (
        bundle.score_threshold
        if config.detection.operating_threshold is None
        else config.detection.operating_threshold
    )
    result = detect_recording(
        bundle,
        recording,
        name=name,
        detector=models.detector,
        estimator=models.estimator,
        store_threshold=config.detection.store_threshold,
        chunk_tiles=config.compute.chunk_tiles,
        click_workers=config.compute.click_workers,
        prefetch=config.compute.prefetch,
        clean_processes=config.compute.clean_processes,
        device=config.compute.device,
        t_range_s=t_range_s,
        progress=progress,
    )
    t = time.perf_counter()
    detections, pp_info = _postprocess(
        result.detections, tracks, name, config.detection.postprocess
    )
    pp_s = time.perf_counter() - t
    t = time.perf_counter()
    assignment = assign_detections(
        detections,
        tracks if models.has_emitter else None,
        config.assignment,
        models.error_model or SIM_V4_DUALHEAD,
    )
    if not models.has_emitter:
        assignment["assign_status"] = "no_emitter"
    assign_s = time.perf_counter() - t
    table = build_table(name, detections, assignment, result.chunks, operating)
    times, ids = legacy_arrays(table)
    counts = _counts(table, result)
    duration = recording.duration_s
    processed = (
        duration
        if t_range_s is None
        else float(
            result.chunks["t_end_s"].to_numpy().max()
            - result.chunks["t_start_s"].to_numpy().min()
        )
        if len(result.chunks)
        else 0.0
    )
    with atomic_dir(final) as tmp:
        files: dict[str, Path] = {}
        files["chirps"] = tmp / f"chirps_{tag}.parquet"
        files["chunks"] = tmp / f"chunks_{tag}.parquet"
        files["times"] = tmp / f"chirp_times_{tag}.npy"
        files["ids"] = tmp / f"chirp_ids_{tag}.npy"
        file_meta = {
            "recording": name,
            "run": config.name,
            "model": models.info["detector"]["architecture"],
            "model_tag": tag,
            "weights_sha256": bundle.weights_sha256,
            "weights_short": bundle.weights_sha256[:8],
            "decoder_sha256": (models.info.get("decoder") or {}).get("weights_sha256"),
            "store_threshold": config.detection.store_threshold,
            "operating_threshold": operating,
            "time_base": "seconds from the recording's first sample",
            "recording_start": recording_start(recording_dir, entries),
            "detection_hash": config.detection_hash(),
            "assignment_tolerance_hz": config.assignment.tolerance_hz,
        }
        write_parquet(files["chirps"], table, file_meta)
        write_parquet(files["chunks"], result.chunks, file_meta)
        np.save(files["times"], times)
        np.save(files["ids"], ids)
        t = time.perf_counter()
        qc_info: dict[str, Any] = {}
        if config.validation.qc:
            from chirpdetector.pipeline.qc import write_qc

            try:
                qc_info = write_qc(
                    tmp / "qc", table, result.chunks, file_meta, duration
                )
            except Exception as err:  # QC must never lose a recording's results
                log.exception("%s: QC report failed", name)
                qc_info = {"error": repr(err)}
        qc_s = time.perf_counter() - t
        t = time.perf_counter()
        win_info: dict[str, Any] = {}
        if config.validation.windows > 0 and len(table):
            from chirpdetector.pipeline.windows import sample_windows, write_windows

            try:
                windows = sample_windows(
                    table,
                    duration if t_range_s is None else t_range_s,
                    n=config.validation.windows,
                    window_s=config.validation.window_s,
                    seed=config.validation.seed + _name_seed(name),
                )
                win_info = write_windows(
                    tmp / "windows",
                    windows,
                    recording,
                    bundle,
                    table,
                    tracks,
                    title=f"{name} ({config.name})",
                    device="cpu",  # 10-s spectrograms; keeps the GPU for detection
                )
            except Exception as err:
                log.exception("%s: sample windows failed", name)
                win_info = {"error": repr(err)}
        win_s = time.perf_counter() - t
        wall = time.perf_counter() - t_wall
        meta = {
            "status": "complete",
            "recording": name,
            "run": config.name,
            "created": datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(),
            "detection_hash": config.detection_hash(),
            "t_range_s": None if t_range_s is None else list(t_range_s),
            "inputs": {
                "recording_dir": str(recording_dir),
                "files": _input_files(recording_dir, recording),
                "samplerate_hz": fs,
                "n_samples": recording.n_samples,
                "duration_s": duration,
                "n_electrodes": recording.n_channels,
                "start": recording_start(recording_dir, entries),
            },
            "tracks": track_info,
            "models": models.info,
            "code": _code(),
            "config": config.to_dict(),
            "click_removal": {
                "applied": result.click_removal,
                "config": bundle.preprocessing.get("click_removal"),
                "electrodes": recording.n_channels,
                "fraction_repaired": float(
                    np.mean(result.chunks["click_fraction"].to_numpy(dtype=float))
                )
                if len(result.chunks) and result.click_removal
                else None,
            },
            "postprocess": pp_info,
            "assignment": {
                "supported": models.has_emitter,
                "rule": "nearest live track to emitter_hz at the onset (assign.py)",
                "config": config.to_dict()["assignment"],
                "error_model": None
                if models.error_model is None
                else models.error_model.to_dict(),
            },
            "operating_threshold": operating,
            "counts": counts,
            "timings": {
                **result.timings,
                "postprocess_s": pp_s,
                "assign_s": assign_s,
                "qc_s": qc_s,
                "windows_s": win_s,
                "total_s": wall,
                "processed_s": processed,
                "realtime_factor": processed / wall if wall > 0 else None,
            },
            "qc": qc_info,
            "windows": win_info,
            "outputs": {
                k: {"file": v.name, "bytes": v.stat().st_size} for k, v in files.items()
            },
            "legacy": {
                "times": files["times"].name,
                "ids": files["ids"].name,
                "selection": "chirp (score >= operating threshold, not flagged) and assigned",
                "n": len(times),
                "dtype": "float64",
                "time": "box centre, seconds from the recording's first sample",
            },
        }
        write_json(tmp / META_FILE, meta)
    return RecordingResult(
        name=name,
        status="complete",
        duration_s=processed,
        wall_s=wall,
        counts=counts,
    )


def _name_seed(name: str) -> int:
    return int(hashlib_sha256(name)[:8], 16)


def _input_files(directory: Path, recording: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    path = Path(getattr(recording, "path", directory))
    if path.exists():
        out["samples"] = file_record(path)
    cfg = directory / "fishgrid.cfg"
    if cfg.exists():
        out["fishgrid.cfg"] = file_record(cfg, hash_=True)
    return out


_CODE: dict[str, Any] | None = None


def _code() -> dict[str, Any]:
    global _CODE
    if _CODE is None:
        from chirpdetector.training.provenance import code_state, uv_lock, versions

        _CODE = {"git": code_state(), "versions": versions(), "uv_lock": uv_lock()}
    return _CODE


def _counts(table: pd.DataFrame, result: Any) -> dict[str, Any]:
    if not len(table):
        return {"stored": 0, "owned_before_merge": result.n_owned}
    op = table["operating"].to_numpy(dtype=bool)
    chirp = table["chirp"].to_numpy(dtype=bool)
    status = table["assign_status"].to_numpy(dtype=object)
    out: dict[str, Any] = {
        "owned_before_merge": int(result.n_owned),
        "stored": len(table),
        "operating": int(op.sum()),
        "chirps": int(chirp.sum()),
        "dropped_by_postprocess": {
            str(k): int(v)
            for k, v in table.loc[op, "pp_drop_reason"].value_counts().items()
            if k != ""
        },
        "assign_status_chirps": {
            str(k): int(v) for k, v in pd.Series(status[chirp]).value_counts().items()
        },
        "legacy": int(np.sum(chirp & (status == "assigned"))),
    }
    if "params_ok" in table.columns:
        flags = table.loc[chirp, "params_flag"].astype(str)
        out["params_ok_chirps"] = int(table.loc[chirp, "params_ok"].sum())
        out["params_flags_chirps"] = {
            f: int(flags.str.contains(f).sum())
            for f in ("edge", "band", "size", "uncertain")
        }
    ids = table.loc[chirp & (status == "assigned"), "track_id"]
    out["per_track"] = {
        str(int(k)): int(v) for k, v in ids.value_counts().sort_index().items()
    }
    return out


# --------------------------------------------------------------------- run


@dataclass
class RunSummary:
    """What a run invocation did."""

    run_dir: Path
    done: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


class _Progress:
    """Logs progress with real-time factor and ETA."""

    def __init__(self, total_s: float, every_s: float = 30.0) -> None:
        self.total_s = total_s
        self.done_s = 0.0
        self.t0 = time.perf_counter()
        self.every_s = every_s
        self.last = 0.0
        self.name = ""
        self.rec_total = 0.0
        self.rec_start = 0.0
        self.rec_t0 = 0.0
        self.index = (0, 0)

    def start(self, name: str, duration_s: float, index: tuple[int, int]) -> None:
        self.name, self.rec_total, self.index = name, duration_s, index
        self.rec_t0 = time.perf_counter()
        self.rec_start = 0.0
        self.last = 0.0

    def __call__(self, done: int, total: int, t_end_s: float) -> None:
        now = time.perf_counter()
        if done < total and now - self.last < self.every_s:
            return
        self.last = now
        frac = done / max(total, 1)
        rec_done = frac * self.rec_total
        rec_wall = now - self.rec_t0
        rtf = rec_done / rec_wall if rec_wall > 0 else math.nan
        run_done = self.done_s + rec_done
        run_wall = now - self.t0
        run_rtf = run_done / run_wall if run_wall > 0 else math.nan
        remaining = (self.total_s - run_done) / run_rtf if run_rtf > 0 else math.nan
        rec_left = (self.rec_total - rec_done) / rtf if rtf > 0 else math.nan
        log.info(
            "[%d/%d] %s: %3.0f%% (%.2f / %.2f h) %.0fx real time, ETA recording %s, run %s",
            self.index[0],
            self.index[1],
            self.name,
            100 * frac,
            rec_done / 3600,
            self.rec_total / 3600,
            rtf,
            _hms(rec_left),
            _hms(remaining),
        )

    def finish(self, duration_s: float) -> None:
        self.done_s += duration_s


def _hms(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "?"
    s = round(seconds)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


def _attach_log(run_dir: Path) -> logging.Handler:
    run_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(run_dir / "run.log")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    return handler


def _manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def run_pipeline(
    config: PipelineConfig,
    recordings: Sequence[Path],
    *,
    config_file: Path | None = None,
    t_range_s: tuple[float, float] | None = None,
    redo: bool = False,
    models: Models | None = None,
) -> RunSummary:
    """Process recordings into ``config.run_dir`` (module docstring).

    Parameters
    ----------
    config
        Run configuration.
    recordings
        Recording directories (`discover_recordings`).
    config_file
        Copied to ``run_dir/config.toml`` (first invocation).
    t_range_s
        Only this part of every recording (dry runs).
    redo
        Recompute recordings that are complete.
    models
        Already loaded models (tests).

    Raises
    ------
    ValueError
        If the run directory was made with other detection settings.
    """
    run_dir = config.run_dir
    handler = _attach_log(run_dir)
    summary = RunSummary(run_dir=run_dir)
    try:
        manifest = _manifest(run_dir)
        dhash = config.detection_hash()
        if manifest and manifest.get("detection_hash") != dhash:
            msg = (
                f"{run_dir} was made with other models or detection settings "
                f"(detection_hash {manifest.get('detection_hash', '?')[:12]} vs "
                f"{dhash[:12]}); use another run name"
            )
            raise ValueError(msg)
        if config_file is not None and not (run_dir / "config.toml").exists():
            shutil.copyfile(config_file, run_dir / "config.toml")
        (run_dir / "columns.md").write_text(_columns_md())
        manifest.setdefault("run", config.name)
        manifest["detection_hash"] = dhash
        manifest["config"] = config.to_dict()
        manifest.setdefault("created", datetime.now().isoformat(timespec="seconds"))
        manifest.setdefault("recordings", {})
        manifest.setdefault("invocations", [])
        manifest["invocations"].append(
            {
                "started": datetime.now().isoformat(timespec="seconds"),
                "host": platform.node(),
                "recordings": [p.name for p in recordings],
                "t_range_s": None if t_range_s is None else list(t_range_s),
                "redo": redo,
            }
        )
        todo: list[Path] = []
        for path in recordings:
            rng = None if t_range_s is None else list(t_range_s)
            if not redo and _done(run_dir / path.name, dhash, rng):
                summary.skipped.append(path.name)
            else:
                todo.append(path)
        log.info(
            "run %s: %d recordings, %d complete (skipped), %d to do -> %s",
            config.name,
            len(recordings),
            len(summary.skipped),
            len(todo),
            run_dir,
        )
        durations = {}
        for p in todo:
            try:
                rec = open_recording(p)
                d = rec.duration_s
                if t_range_s is not None:
                    d = max(0.0, min(d, t_range_s[1]) - t_range_s[0])
                durations[p.name] = d
            except Exception:
                durations[p.name] = 0.0
        progress = _Progress(sum(durations.values()))
        log.info("to do: %.1f h of recordings", progress.total_s / 3600)
        if todo and models is None:
            models = load_models(config)
        if models is not None:
            manifest["models"] = models.info
        manifest["code"] = _code()
        write_json(run_dir / "run.json", manifest)
        for k, path in enumerate(todo, 1):
            name = path.name
            progress.start(name, durations.get(name, 0.0), (k, len(todo)))
            log.info("[%d/%d] %s: start", k, len(todo), name)
            assert models is not None
            try:
                res = process_recording(
                    path, config, models, t_range_s=t_range_s, progress=progress
                )
                summary.done.append(name)
                manifest["recordings"][name] = {
                    "status": "complete",
                    "path": str(path),
                    "duration_s": res.duration_s,
                    "wall_s": res.wall_s,
                    "realtime_factor": res.duration_s / res.wall_s
                    if res.wall_s
                    else None,
                    "chirps": res.counts.get("chirps"),
                    "legacy": res.counts.get("legacy"),
                    "finished": datetime.now().isoformat(timespec="seconds"),
                }
                log.info(
                    "[%d/%d] %s: done in %s (%.0fx real time): %s chirps, %s assigned",
                    k,
                    len(todo),
                    name,
                    _hms(res.wall_s),
                    res.duration_s / res.wall_s if res.wall_s else math.nan,
                    res.counts.get("chirps"),
                    res.counts.get("legacy"),
                )
            except KeyboardInterrupt:
                manifest["recordings"][name] = {
                    "status": "interrupted",
                    "path": str(path),
                }
                write_json(run_dir / "run.json", manifest)
                raise
            except Exception as err:
                tb = traceback.format_exc()
                log.error("[%d/%d] %s: FAILED: %r\n%s", k, len(todo), name, err, tb)
                summary.failed[name] = repr(err)
                manifest["recordings"][name] = {
                    "status": "failed",
                    "path": str(path),
                    "error": repr(err),
                    "traceback": tb,
                    "failed": datetime.now().isoformat(timespec="seconds"),
                }
                _free_gpu()
            progress.finish(durations.get(name, 0.0))
            manifest["updated"] = datetime.now().isoformat(timespec="seconds")
            write_json(run_dir / "run.json", manifest)
        if config.validation.candidates_per_cell > 0:
            from chirpdetector.pipeline.candidates import write_candidates

            try:
                info = write_candidates(run_dir, config)
                manifest["candidates"] = info
                write_json(run_dir / "run.json", manifest)
            except Exception:
                log.exception("validation candidates failed")
        log.info(
            "run %s: %d done, %d skipped, %d failed%s",
            config.name,
            len(summary.done),
            len(summary.skipped),
            len(summary.failed),
            f" ({', '.join(summary.failed)})" if summary.failed else "",
        )
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    return summary


def _free_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


def _columns_md() -> str:
    lines = [
        "# Columns of chirps_<model>.parquet",
        "",
        "One row per detection with score >= the store threshold, after the",
        "cross-tile merge. Rows flagged by post-processing are kept",
        "(`kept == False`, reason in `pp_drop_reason`). Times are seconds from",
        "the recording's first sample; the recording's absolute start is in",
        "meta.json (`inputs.start`).",
        "",
        "| column | meaning |",
        "|---|---|",
    ]
    lines += [f"| `{k}` | {v} |" for k, v in COLUMN_DOCS.items()]
    lines += [
        "",
        "Uncalibrated decoder columns (if the decoder provides them) follow",
        "`params_flag`.",
        "",
        "Assignment columns: " + ", ".join(f"`{c}`" for c in ASSIGN_COLUMNS) + ".",
    ]
    return "\n".join(lines) + "\n"


def sha256_file(path: Path) -> str:
    """sha256 of a file (re-export for the CLI)."""
    return sha256_of(path)
