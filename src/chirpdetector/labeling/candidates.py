"""Label any list of boxes: the candidates file (``label-emitters --candidates``).

A candidates file is a CSV or Parquet table with one row per box to label.
This format is a stable interface: the production pipeline writes its
validation samples in it, and other tools may too.

Required columns
----------------
``recording``
    The recording directory (anything :func:`gridtools.io.open_recording`
    opens). Relative paths are relative to the candidates file's directory.
``t_start_s``, ``t_end_s``, ``f_low_hz``, ``f_high_hz``
    The box, in seconds from the start of the recording and in Hz.

Optional columns
----------------
``candidate_id``
    Identifies the box within its recording (text without ``#``). The
    labelled item is ``"<recording name>#<candidate_id>"``; a box with the
    same id in another candidates file is the same item, so its label is
    reused. Default: ``b`` + 10 hex digits of a hash of the recording name
    and the box, so the same box always gets the same id.
``score``
    Detector score.
``emitter_hz``
    The model's predicted emitter EOD frequency (NaN: none).
``track_id``
    The model's predicted emitter track (nullable integer).
``source``
    Free text naming what produced the box (model, run, sample).
``tracks``
    Directory of the recording's wavetracker arrays (``fund_v.npy``,
    ``ident_v.npy``, ``idx_v.npy``, ``times.npy``), if not the recording
    directory. Relative paths as for ``recording``.

Any other column is kept as it is and stored with every label of the box
(prefixed ``candidate_``). ``score``, ``emitter_hz``, ``track_id`` and
``source`` are hidden while labelling unless the labeller shows them.

The queue keeps the file's rows (no de-duplication, no sampling), with the
:math:`|\\Delta F|` bin of each box, and is shuffled with the seed unless
asked not to. It is saved in ``<out>/queues/<queue_id>.json``, where
``queue_id`` hashes the file's content, the tracks and the options, so the
same command resumes the same queue.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from gridtools.io import load_tracks

from chirpdetector.labeling.queue import (
    QUEUE_FORMAT,
    CandidateValue,
    LabelQueue,
    QueueConfig,
    QueueItem,
    TrialInput,
    bin_labels,
    delta_f_bins,
    write_json_atomic,
)

__all__ = [
    "BOX_COLUMNS",
    "CANDIDATE_FIELDS",
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "build_candidate_queue",
    "default_candidate_id",
    "load_or_build_candidate_queue",
    "read_candidates",
    "tracks_fingerprint",
]

log = logging.getLogger(__name__)

BOX_COLUMNS = ("t_start_s", "t_end_s", "f_low_hz", "f_high_hz")
REQUIRED_COLUMNS = ("recording", *BOX_COLUMNS)
"""Columns every candidates file must have."""

OPTIONAL_COLUMNS = (
    "candidate_id",
    "score",
    "emitter_hz",
    "track_id",
    "source",
    "tracks",
)
"""Columns a candidates file may have (others are kept as extras)."""

CANDIDATE_FIELDS = (
    "candidate_id",
    "candidate_score",
    "candidate_emitter_hz",
    "candidate_track_id",
    "candidate_source",
)
"""Keys of :attr:`QueueItem.candidate` (then ``candidate_<extra column>``).

They name the columns ``candidate_id``, ``score``, ``emitter_hz``,
``track_id`` and ``source`` of the candidates file, and are the column
names in the label files.
"""

_TRACK_FILES = ("times.npy", "fund_v.npy", "ident_v.npy", "idx_v.npy")
_MAX_ERRORS = 10


def default_candidate_id(recording: str, box: tuple[float, ...]) -> str:
    """Content-derived id of a box: the same box always gets the same id."""
    text = "|".join([recording, *(f"{v:.6f}" for v in box)])
    return "b" + hashlib.sha256(text.encode()).hexdigest()[:10]


def _scalar(value: Any) -> CandidateValue:
    """A JSON-safe Python scalar (NaN and NA become None)."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if pd.isna(value):
        return None
    return str(value)


def _resolve(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path)


def read_candidates(path: str | Path) -> pd.DataFrame:
    """Read and validate a candidates file.

    Parameters
    ----------
    path
        ``.csv`` or ``.parquet`` file.

    Returns
    -------
    One row per candidate, in file order (index = row in the file), with
    the required columns, ``recording`` and ``tracks`` as absolute paths,
    ``_name`` (the recording directory's name), ``candidate_id`` (text,
    filled in where missing), ``score`` and ``emitter_hz`` (float, NaN where
    missing), ``track_id`` (Int64), ``source`` (text), and the extra columns.

    Raises
    ------
    ValueError
        If the file type is not supported, a required column is missing, or
        rows are invalid (all problems are listed, up to ten).
    FileNotFoundError
        If the file, a recording or a tracks directory does not exist.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(
            path,
            dtype={"recording": str, "tracks": str, "source": str, "candidate_id": str},
            keep_default_na=False,
            na_values=[""],
        )
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        msg = f"{path}: candidates must be .csv or .parquet, not {suffix!r}"
        raise ValueError(msg)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        msg = f"{path}: missing required columns {missing}"
        raise ValueError(msg)
    n = len(df)
    errors: list[str] = []

    def bad(mask: npt.NDArray[np.bool_], what: str) -> None:
        rows = np.flatnonzero(mask)
        if rows.size:
            shown = ", ".join(str(int(r)) for r in rows[:5])
            more = f" (+{rows.size - 5} more)" if rows.size > 5 else ""
            errors.append(f"rows {shown}{more}: {what}")

    def column(name: str) -> list[object]:
        return (
            list(df[name].to_numpy(dtype=object)) if name in df.columns else [None] * n
        )

    recording = column("recording")
    bad(np.array([_missing(v) for v in recording], dtype=bool), "recording is empty")
    box = (
        np.column_stack([_floats(column(c)) for c in BOX_COLUMNS])
        if n
        else np.zeros((0, 4))
    )
    for k, name in enumerate(BOX_COLUMNS):
        bad(~np.isfinite(box[:, k]), f"{name} is not a number")
    with np.errstate(invalid="ignore"):
        bad(box[:, 1] <= box[:, 0], "t_end_s <= t_start_s")
        bad(box[:, 3] <= box[:, 2], "f_high_hz <= f_low_hz")
        bad((box[:, 0] < 0) | (box[:, 2] < 0), "negative time or frequency")
    score = _floats(column("score"))
    emitter = _floats(column("emitter_hz"))
    with np.errstate(invalid="ignore"):
        bad(emitter <= 0, "emitter_hz must be positive (or empty)")
    track_raw = _floats(column("track_id"))
    bad(
        np.isfinite(track_raw) & (track_raw != np.round(track_raw)),
        "track_id is not an integer",
    )
    given_tracks = [_missing(v) for v in column("track_id")]
    bad(
        ~np.isfinite(track_raw) & ~np.array(given_tracks, dtype=bool),
        "track_id is not a number",
    )
    if errors:
        listed = "\n  ".join(errors[:_MAX_ERRORS])
        msg = f"{path}: invalid candidates:\n  {listed}"
        raise ValueError(msg)

    base = path.parent
    recordings = [_resolve(v, base) for v in recording]
    tracks = [
        _resolve(t, base)
        if not _missing(t)
        else (r if r.is_dir() or not r.suffix else r.parent)
        for r, t in zip(recordings, column("tracks"), strict=True)
    ]
    names = [
        r.name if r.is_dir() or not r.suffix else r.parent.name for r in recordings
    ]
    for rec in sorted(set(recordings)):
        if not rec.exists():
            raise FileNotFoundError(f"{path}: recording {rec} does not exist")
    for directory in sorted(set(tracks)):
        lacking = [f for f in _TRACK_FILES if not (directory / f).exists()]
        if lacking:
            msg = f"{path}: tracks directory {directory} lacks {lacking}"
            raise FileNotFoundError(msg)
    sources_of: dict[str, set[tuple[Path, Path]]] = {}
    for name, rec, trk in zip(names, recordings, tracks, strict=True):
        sources_of.setdefault(name, set()).add((rec, trk))
    clash = sorted(name for name, found in sources_of.items() if len(found) > 1)
    if clash:
        msg = (
            f"{path}: recording names {clash} refer to different recordings or "
            "tracks; label files are named by recording"
        )
        raise ValueError(msg)

    ids = [
        default_candidate_id(name, tuple(float(v) for v in row))
        if _missing(cid)
        else str(cid).strip()
        for cid, name, row in zip(column("candidate_id"), names, box, strict=True)
    ]
    hashes = [i for i, c in enumerate(ids) if "#" in c]
    if hashes:
        msg = f"{path}: candidate_id must not contain '#' (rows {hashes[:5]})"
        raise ValueError(msg)
    counts = Counter(zip(names, ids, strict=True))
    dup = [i for i, key in enumerate(zip(names, ids, strict=True)) if counts[key] > 1]
    if dup:
        msg = (
            f"{path}: duplicate candidate_id within a recording (rows {dup[:5]}); "
            "without candidate_id this means the same box is listed twice"
        )
        raise ValueError(msg)

    out = pd.DataFrame(
        {
            "recording": [str(r) for r in recordings],
            **{c: box[:, k] for k, c in enumerate(BOX_COLUMNS)},
            "candidate_id": ids,
            "score": score,
            "emitter_hz": emitter,
            "track_id": pd.array(
                [None if not np.isfinite(v) else int(v) for v in track_raw],
                dtype="Int64",
            ),
            "source": ["" if _missing(v) else str(v) for v in column("source")],
            "tracks": [str(t) for t in tracks],
            "_name": names,
        }
    )
    for extra in _extras(df):
        out[extra] = df[extra].to_numpy()
    return out


def _missing(value: object) -> bool:
    """True for None, NA, NaN and blank text."""
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, str):
        return not value.strip()
    return False


def _floats(values: list[object]) -> npt.NDArray[np.float64]:
    """Floats of arbitrary values (NaN where not a number)."""
    out = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        if _missing(value):
            continue
        try:
            out[i] = float(value)  # pyright: ignore[reportArgumentType]
        except (TypeError, ValueError):
            continue
    return out


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracks_fingerprint(directory: str | Path) -> str:
    """SHA-256 over a directory's wavetracker arrays."""
    directory = Path(directory)
    digest = hashlib.sha256()
    for name in _TRACK_FILES:
        digest.update(name.encode())
        digest.update((directory / name).read_bytes())
    return digest.hexdigest()


def _candidate_queue_id(
    file_sha256: str, fingerprints: dict[str, str], config: QueueConfig, shuffle: bool
) -> str:
    payload = json.dumps(
        {
            "format": QUEUE_FORMAT,
            "kind": "candidates",
            "file": file_sha256,
            "tracks": sorted(fingerprints.items()),
            "bin_edges_hz": list(config.bin_edges_hz),
            "max_gap_s": config.max_gap_s,
            "seed": config.seed,
            "shuffle": shuffle,
        },
        sort_keys=True,
    )
    return "c" + hashlib.sha256(payload.encode()).hexdigest()[:11]


def _extras(df: pd.DataFrame) -> list[str]:
    known = {*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS, "_name"}
    return [str(c) for c in df.columns if c not in known]


def _candidate_key(column: str) -> str:
    """Label-file column of a candidates-file column."""
    return column if column.startswith("candidate_") else f"candidate_{column}"


def build_candidate_queue(
    path: str | Path,
    config: QueueConfig | None = None,
    *,
    shuffle: bool = True,
    candidates: pd.DataFrame | None = None,
) -> LabelQueue:
    """Queue of every row of a candidates file (not saved).

    Parameters
    ----------
    path
        The candidates file.
    config
        Only ``seed``, ``bin_edges_hz`` and ``max_gap_s`` are used.
    shuffle
        Shuffle the rows with ``config.seed`` (default); otherwise keep the
        file order.
    candidates
        The already validated table (:func:`read_candidates`) of ``path``.

    Returns
    -------
    The queue, one item per row.
    """
    config = config or QueueConfig()
    path = Path(path).expanduser().resolve()
    df = read_candidates(path) if candidates is None else candidates
    extras = _extras(df)
    labels = bin_labels(config.bin_edges_hz)
    trials: list[TrialInput] = []
    items: list[QueueItem | None] = [None] * len(df)
    fingerprints: dict[str, str] = {}
    for name, group in df.groupby("_name", sort=True):
        name = str(name)
        rec_dir, tracks_dir = group["recording"].iloc[0], group["tracks"].iloc[0]
        fingerprints[name] = tracks_fingerprint(tracks_dir)
        boxes = group[list(BOX_COLUMNS)].to_numpy(dtype=np.float64)
        centre = 0.5 * (boxes[:, 0] + boxes[:, 1])
        n_live, delta_f, bins = delta_f_bins(
            load_tracks(tracks_dir), centre, config.bin_edges_hz, config.max_gap_s
        )
        per_bin = Counter(bins)
        trials.append(
            TrialInput(
                name=name,
                path=str(rec_dir),
                fingerprint=fingerprints[name],
                n_detections=len(group),
                n_kept=len(group),
                kept_per_bin={b: per_bin[b] for b in labels},
                tracks_path=str(tracks_dir) if tracks_dir != rec_dir else "",
            )
        )
        rows = group.index.to_numpy(dtype=np.int64)
        for k, values in enumerate(group.to_dict("records")):
            t0, t1, f0, f1 = (float(v) for v in boxes[k])
            columns = [
                "candidate_id",
                "score",
                "emitter_hz",
                "track_id",
                "source",
                *extras,
            ]
            candidate = tuple((_candidate_key(c), _scalar(values[c])) for c in columns)
            items[int(rows[k])] = QueueItem(
                item_id=f"{name}#{values['candidate_id']}",
                trial=name,
                det_index=int(rows[k]),
                t_start_s=t0,
                t_end_s=t1,
                f_low_hz=f0,
                f_high_hz=f1,
                n_candidates=int(n_live[k]),
                delta_f_hz=float(delta_f[k]) if np.isfinite(delta_f[k]) else None,
                df_bin=bins[k],
                candidate=candidate,
            )
    ordered = [i for i in items if i is not None]
    if shuffle:
        order = np.random.default_rng(config.seed).permutation(len(ordered))
        ordered = [ordered[i] for i in order]
    file_sha = _sha256_file(path)
    return LabelQueue(
        queue_id=_candidate_queue_id(file_sha, fingerprints, config, shuffle),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        config=config,
        trials=tuple(trials),
        items=tuple(ordered),
        origin={
            "kind": "candidates",
            "path": str(path),
            "sha256": file_sha,
            "shuffle": shuffle,
            "n_rows": len(df),
        },
    )


def load_or_build_candidate_queue(
    path: str | Path,
    config: QueueConfig | None,
    out_dir: str | Path,
    *,
    shuffle: bool = True,
) -> LabelQueue:
    """Load the queue of this candidates file from ``out_dir``, or build it.

    The candidates are always read and validated (so a broken file fails
    early); the queue is rebuilt only if no queue with the same file
    content, tracks and options was saved before.
    """
    config = config or QueueConfig()
    path = Path(path).expanduser().resolve()
    df = read_candidates(path)
    fingerprints = {
        str(name): tracks_fingerprint(group["tracks"].iloc[0])
        for name, group in df.groupby("_name", sort=True)
    }
    queue_id = _candidate_queue_id(_sha256_file(path), fingerprints, config, shuffle)
    queue_path = Path(out_dir) / "queues" / f"{queue_id}.json"
    if queue_path.exists():
        queue = LabelQueue.from_json(json.loads(queue_path.read_text()))
        if queue.queue_id != queue_id:
            msg = (
                f"{queue_path}: queue id {queue.queue_id} does not match its file name"
            )
            raise ValueError(msg)
        log.info("resuming queue %s (%d candidates)", queue_id, len(queue))
        return queue
    queue = build_candidate_queue(path, config, shuffle=shuffle, candidates=df)
    write_json_atomic(queue.to_json(), queue_path)
    log.info(
        "new queue %s (%d candidates from %s) written to %s",
        queue_id,
        len(queue),
        path,
        queue_path,
    )
    return queue
