"""Which v1 detections to label, and in which order.

Every v1 detection is a candidate (the labeller also decides whether it is
a chirp at all). Detections are stratified by the frequency difference
between the two closest EOD frequency tracks live at the detection centre
(:math:`|\\Delta F|`), because close-:math:`\\Delta F` chirps are both the
hardest to assign and the rarest; detections with fewer than two live tracks
form their own bin, ``single``. Up to ``per_bin`` detections are sampled per
trial and bin, and the result is shuffled across trials.

Each trial is sampled with its own generator, seeded from the queue seed and
the trial name, so adding a trial to a queue does not change which
detections of the other trials are drawn.

The queue is written to ``<out>/queues/<queue_id>.json``. ``queue_id`` is a
hash of the configuration and the content of every trial's detections and
tracks, so running the tool again with the same arguments loads the same
queue instead of sampling again.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import tempfile
import zlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from gridtools.io import Tracks, load_tracks, read_v1_detections

__all__ = [
    "DEFAULT_BIN_EDGES_HZ",
    "SINGLE_BIN",
    "LabelQueue",
    "QueueConfig",
    "QueueItem",
    "TrialInput",
    "bin_labels",
    "binned_detections",
    "build_queue",
    "load_or_build_queue",
    "queue_id_for",
    "trial_fingerprint",
]

log = logging.getLogger(__name__)

QUEUE_FORMAT = 1
DEFAULT_BIN_EDGES_HZ = (0.0, 10.0, 25.0, 50.0, 100.0)
"""Lower edges of the :math:`|\\Delta F|` bins in Hz; the last bin is open."""

SINGLE_BIN = "single"
"""Bin of detections with fewer than two live tracks."""

_DETECTIONS_FILE = "chirpdetector_bboxes.csv"
_TRACK_FILES = ("times.npy", "fund_v.npy", "ident_v.npy", "idx_v.npy")


@dataclass(frozen=True)
class QueueConfig:
    """How to sample the queue.

    Parameters
    ----------
    per_bin
        Maximum number of detections per trial and :math:`|\\Delta F|` bin.
    seed
        Seed of the sampling generators.
    bin_edges_hz
        Increasing lower bin edges, starting at 0; the last bin is open.
    max_gap_s
        A track is live at a time if its detections around it are at most
        this far apart (see :meth:`gridtools.io.Track.frequency_at`).
    dedup_iou
        Detections overlapping a higher-scoring detection of the same trial
        with an intersection over union above this are dropped as
        duplicates.
    """

    per_bin: int = 20
    seed: int = 0
    bin_edges_hz: tuple[float, ...] = DEFAULT_BIN_EDGES_HZ
    max_gap_s: float = 1.0
    dedup_iou: float = 0.3

    def __post_init__(self) -> None:
        edges = np.asarray(self.bin_edges_hz, dtype=np.float64)
        if self.per_bin < 1:
            msg = f"per_bin must be >= 1, got {self.per_bin}"
            raise ValueError(msg)
        if edges.size < 1 or edges[0] != 0 or np.any(np.diff(edges) <= 0):
            msg = f"bin_edges_hz must increase from 0, got {self.bin_edges_hz}"
            raise ValueError(msg)
        if not 0 < self.dedup_iou <= 1:
            msg = f"dedup_iou must be in (0, 1], got {self.dedup_iou}"
            raise ValueError(msg)
        object.__setattr__(self, "bin_edges_hz", tuple(float(e) for e in edges))


def bin_labels(edges_hz: Sequence[float]) -> list[str]:
    """Names of the bins, e.g. ``["0-10", "10-25", "25+", "single"]``."""
    names = [f"{lo:g}-{hi:g}" for lo, hi in itertools.pairwise(edges_hz)]
    return [*names, f"{edges_hz[-1]:g}+", SINGLE_BIN]


@dataclass(frozen=True)
class QueueItem:
    """One detection to label.

    ``item_id`` is ``"<trial>#<det_index>"``, where ``det_index`` is the row
    of the detection in :func:`gridtools.io.read_v1_detections` (sorted by
    start time). ``delta_f_hz`` is None for items with fewer than two live
    tracks. v1's assigned emitter is deliberately not stored.
    """

    item_id: str
    trial: str
    det_index: int
    t_start_s: float
    t_end_s: float
    f_low_hz: float
    f_high_hz: float
    n_candidates: int
    delta_f_hz: float | None
    df_bin: str

    @property
    def t_center_s(self) -> float:
        """Centre of the detection box in time."""
        return 0.5 * (self.t_start_s + self.t_end_s)

    @property
    def box(self) -> tuple[float, float, float, float]:
        """``(t_start_s, t_end_s, f_low_hz, f_high_hz)``."""
        return (self.t_start_s, self.t_end_s, self.f_low_hz, self.f_high_hz)


@dataclass(frozen=True)
class TrialInput:
    """A trial a queue was sampled from."""

    name: str
    path: str
    fingerprint: str
    n_detections: int
    n_kept: int
    kept_per_bin: dict[str, int]


@dataclass(frozen=True)
class LabelQueue:
    """A persisted, ordered list of detections to label."""

    queue_id: str
    created_at: str
    config: QueueConfig
    trials: tuple[TrialInput, ...]
    items: tuple[QueueItem, ...]

    def __len__(self) -> int:
        return len(self.items)

    @property
    def bins(self) -> list[str]:
        """Bin names, in order."""
        return bin_labels(self.config.bin_edges_hz)

    def counts_per_bin(self) -> dict[str, int]:
        """Number of queued items per bin."""
        counts = dict.fromkeys(self.bins, 0)
        for item in self.items:
            counts[item.df_bin] += 1
        return counts

    def to_json(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "format": QUEUE_FORMAT,
            "queue_id": self.queue_id,
            "created_at": self.created_at,
            "config": asdict(self.config),
            "trials": [asdict(t) for t in self.trials],
            "items": [asdict(i) for i in self.items],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> LabelQueue:
        """Inverse of :meth:`to_json`."""
        if data.get("format") != QUEUE_FORMAT:
            msg = f"unsupported queue format {data.get('format')!r}"
            raise ValueError(msg)
        config = dict(data["config"])
        config["bin_edges_hz"] = tuple(config["bin_edges_hz"])
        return cls(
            queue_id=data["queue_id"],
            created_at=data["created_at"],
            config=QueueConfig(**config),
            trials=tuple(TrialInput(**t) for t in data["trials"]),
            items=tuple(QueueItem(**i) for i in data["items"]),
        )


def trial_fingerprint(trial_dir: str | Path) -> str:
    """SHA-256 over a trial's v1 detections and track arrays.

    Raises
    ------
    FileNotFoundError
        If the detections or a track file is missing.
    """
    trial_dir = Path(trial_dir)
    digest = hashlib.sha256()
    for name in (_DETECTIONS_FILE, *_TRACK_FILES):
        path = trial_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _trial_names(trial_dirs: Sequence[str | Path]) -> dict[str, Path]:
    paths = {Path(d).resolve().name: Path(d).resolve() for d in trial_dirs}
    if len(paths) != len(trial_dirs):
        msg = "trial directories must have distinct names"
        raise ValueError(msg)
    return dict(sorted(paths.items()))


def queue_id_for(fingerprints: dict[str, str], config: QueueConfig) -> str:
    """Identifier of the queue sampled from these inputs with this config."""
    payload = json.dumps(
        {
            "format": QUEUE_FORMAT,
            "config": asdict(config),
            "trials": sorted(fingerprints.items()),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _duplicates(
    boxes: npt.NDArray[np.float64], scores: npt.NDArray[np.float64], iou: float
) -> npt.NDArray[np.bool_]:
    """Greedy non-maximum suppression; True where a box is a duplicate."""
    n = len(boxes)
    t0, t1, f0, f1 = boxes.T
    area = (t1 - t0) * (f1 - f0)
    order = np.argsort(t0, kind="stable")
    neighbours: dict[int, list[int]] = {}
    for a, i in enumerate(order):
        for j in order[a + 1 :]:
            if t0[j] >= t1[i]:
                break
            inter = max(0.0, min(t1[i], t1[j]) - max(t0[i], t0[j])) * max(
                0.0, min(f1[i], f1[j]) - max(f0[i], f0[j])
            )
            union = area[i] + area[j] - inter
            if union > 0 and inter / union > iou:
                neighbours.setdefault(int(i), []).append(int(j))
                neighbours.setdefault(int(j), []).append(int(i))
    drop = np.zeros(n, dtype=np.bool_)
    for i in np.argsort(-scores, kind="stable"):
        if not drop[i]:
            for j in neighbours.get(int(i), []):
                if scores[j] < scores[i] or (scores[j] == scores[i] and j > i):
                    drop[j] = True
    return drop


def binned_detections(
    tracks: Tracks, detections: pd.DataFrame, config: QueueConfig
) -> pd.DataFrame:
    """Detections without duplicates, with their :math:`|\\Delta F|` bin.

    Parameters
    ----------
    tracks
        The trial's EOD frequency tracks.
    detections
        Output of :func:`gridtools.io.read_v1_detections`.
    config
        Queue configuration.

    Returns
    -------
    The kept rows (index = detection index) with the box columns plus
    ``n_candidates`` (live tracks at the centre), ``delta_f_hz`` (NaN with
    fewer than two) and ``df_bin``.
    """
    boxes = detections[["t_start_s", "t_end_s", "f_low_hz", "f_high_hz"]].to_numpy(
        dtype=np.float64
    )
    t_center = 0.5 * (boxes[:, 0] + boxes[:, 1])
    ids = tracks.ids
    freqs = np.full((len(boxes), len(ids)), np.nan)
    for j, ident in enumerate(ids):
        freqs[:, j] = tracks.track(int(ident)).frequency_at(t_center, config.max_gap_s)
    n_live = np.isfinite(freqs).sum(axis=1)
    keep = ~_duplicates(
        boxes, detections["score"].to_numpy(dtype=np.float64), config.dedup_iou
    )
    pair = n_live >= 2
    delta_f = np.full(len(boxes), np.nan)
    if pair.any():
        ordered = np.sort(freqs[pair], axis=1)  # NaN sort last
        delta_f[pair] = np.nanmin(np.diff(ordered, axis=1), axis=1)
    names = bin_labels(config.bin_edges_hz)
    bins = np.where(
        pair, np.digitize(np.nan_to_num(delta_f), config.bin_edges_hz) - 1, -1
    )
    out = detections.loc[keep, ["t_start_s", "t_end_s", "f_low_hz", "f_high_hz"]].copy()
    out["n_candidates"] = n_live[keep]
    out["delta_f_hz"] = delta_f[keep]
    out["df_bin"] = [names[b] for b in bins[keep]]
    return out


def _sample_trial(
    name: str, kept: pd.DataFrame, config: QueueConfig
) -> list[QueueItem]:
    rng = np.random.default_rng([config.seed, zlib.crc32(name.encode())])
    det_index = kept.index.to_numpy(dtype=np.int64)
    boxes = kept[["t_start_s", "t_end_s", "f_low_hz", "f_high_hz"]].to_numpy(
        dtype=np.float64
    )
    n_candidates = kept["n_candidates"].to_numpy(dtype=np.int64)
    delta_f = kept["delta_f_hz"].to_numpy(dtype=np.float64)
    df_bin = kept["df_bin"].to_numpy()
    items: list[QueueItem] = []
    for label in bin_labels(config.bin_edges_hz):
        rows = np.flatnonzero(df_bin == label)
        if rows.size == 0:
            continue
        size = min(config.per_bin, rows.size)
        for r in np.sort(rng.choice(rows, size=size, replace=False)):
            t0, t1, f0, f1 = (float(v) for v in boxes[r])
            items.append(
                QueueItem(
                    item_id=f"{name}#{det_index[r]}",
                    trial=name,
                    det_index=int(det_index[r]),
                    t_start_s=t0,
                    t_end_s=t1,
                    f_low_hz=f0,
                    f_high_hz=f1,
                    n_candidates=int(n_candidates[r]),
                    delta_f_hz=float(delta_f[r]) if np.isfinite(delta_f[r]) else None,
                    df_bin=label,
                )
            )
    return items


def build_queue(
    trial_dirs: Sequence[str | Path],
    config: QueueConfig,
    fingerprints: dict[str, str] | None = None,
) -> LabelQueue:
    """Sample a new queue (does not write it; see :func:`load_or_build_queue`).

    Parameters
    ----------
    trial_dirs
        Trial directories, each with ``chirpdetector_bboxes.csv`` and
        wavetracker arrays. Their order does not matter.
    config
        Queue configuration.
    fingerprints
        Precomputed :func:`trial_fingerprint` per trial name.

    Returns
    -------
    The queue, shuffled across trials.
    """
    paths = _trial_names(trial_dirs)
    fingerprints = fingerprints or {n: trial_fingerprint(p) for n, p in paths.items()}
    trials: list[TrialInput] = []
    items: list[QueueItem] = []
    labels = bin_labels(config.bin_edges_hz)
    for name, path in paths.items():
        detections = read_v1_detections(path)
        kept = binned_detections(load_tracks(path), detections, config)
        per_bin = Counter(kept["df_bin"])
        trials.append(
            TrialInput(
                name=name,
                path=str(path),
                fingerprint=fingerprints[name],
                n_detections=len(detections),
                n_kept=len(kept),
                kept_per_bin={b: per_bin[b] for b in labels},
            )
        )
        sampled = _sample_trial(name, kept, config)
        log.info(
            "%s: %d detections, %d after de-duplication, %d sampled",
            name,
            len(detections),
            len(kept),
            len(sampled),
        )
        items.extend(sampled)
    order = np.random.default_rng(config.seed).permutation(len(items))
    return LabelQueue(
        queue_id=queue_id_for(fingerprints, config),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        config=config,
        trials=tuple(trials),
        items=tuple(items[i] for i in order),
    )


def _write_json_atomic(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_or_build_queue(
    trial_dirs: Sequence[str | Path], config: QueueConfig, out_dir: str | Path
) -> LabelQueue:
    """Load the queue for these inputs from ``out_dir``, or sample and save it.

    Parameters
    ----------
    trial_dirs
        Trial directories (order does not matter).
    config
        Queue configuration.
    out_dir
        Output directory; the queue lives in ``out_dir/queues/``.

    Returns
    -------
    The queue. If a queue with the same inputs and configuration was saved
    before, it is returned unchanged (no re-sampling).
    """
    paths = _trial_names(trial_dirs)
    fingerprints = {n: trial_fingerprint(p) for n, p in paths.items()}
    queue_id = queue_id_for(fingerprints, config)
    path = Path(out_dir) / "queues" / f"{queue_id}.json"
    if path.exists():
        queue = LabelQueue.from_json(json.loads(path.read_text()))
        if queue.queue_id != queue_id:
            msg = f"{path}: queue id {queue.queue_id} does not match its file name"
            raise ValueError(msg)
        log.info("resuming queue %s (%d items)", queue_id, len(queue))
        return queue
    queue = build_queue(list(paths.values()), config, fingerprints)
    _write_json_atomic(queue.to_json(), path)
    log.info("new queue %s (%d items) written to %s", queue_id, len(queue), path)
    return queue
