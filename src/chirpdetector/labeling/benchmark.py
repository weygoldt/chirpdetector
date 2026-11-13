"""Human against model on simulated scenes, where the truth is known.

``chirpdetector label-benchmark build`` samples items from simulated test
scenes and writes two files:

candidates (``--candidates``)
    A candidates file (:mod:`chirpdetector.labeling.candidates`) with only
    ``recording``, the box and an opaque ``candidate_id``, shuffled. This is
    all the labeller sees; label it with ``label-emitters --blind``.
truth (``--truth``)
    One row per item: stratum, truth, true emitter, the chirp's
    parameters, and the model's score and assigned emitter. Never read by
    the labeller (it lives elsewhere, and nothing in
    :mod:`chirpdetector.labeling` but this module and the ``score`` command
    opens it).

Strata
------
``chirp_ambiguous``
    Visible chirps in the lowest SNR quartile (of all eligible visible
    test chirps) with an excursion below 60 Hz: the regime of the real
    candidates the author found ambiguous (small excursion, weak trough).
``chirp_other``
    The other visible chirps, spread evenly over SNR quartile x excursion
    bin (< 60, 60-150, >= 150 Hz) cells.
``negative_hard``
    Detections of the model (score >= ``min_hard_score``, post-filtered,
    so harmonic copies are gone) that match no chirp and whose time span,
    widened by ``clear_hard_s``, overlaps no chirp's box or harmonic box at
    all (any fish, any visibility): the marked time holds no chirp. Sampled
    at random, so their scores follow the model's natural distribution.
``negative_easy``
    Boxes on a simulated fish's baseline at a time with no chirp box
    within ``clear_easy_s``; box shapes (duration, offsets from the
    baseline) copied from random small visible chirps.
``invisible``
    Chirps below the visibility threshold: reported, never scored.

Items are kept ``margin_s`` away from the scene edges (the labeller shows
2 s around the box centre). Every true chirp's box is its footprint box;
true chirps carry a ``weight`` (cell population / sampled) to reweight the
benchmark mix to the population of visible test chirps.

Model decisions
---------------
Detections are matched to chirps one-to-one (box midpoints within
``tolerance_s``, overlapping frequency ranges;
:func:`chirpdetector.params.samples.match_predictions`, all chirps). A
chirp's model score is that of its matched detection (0 if none); a
negative's is the highest score of the detections whose midpoint lies
within ``tolerance_s`` of its box midpoint and whose frequency range
overlaps its box (0 if none). The model's emitter is the track assigned to
that detection (``emitter_id``).

Scoring
-------
:func:`score_benchmark` compares the human's labels (``chirp`` or
``chirp_emitter_unclear`` = chirp) with the model at a threshold: recall on
true chirps, false-alarm rate on negatives, emitter accuracy, by stratum,
SNR quartile and excursion bin; paired McNemar tests (exact, two-sided) and
stratified bootstrap CIs. :func:`write_report` writes ``report.md``,
``metrics.json``, ``items.csv`` and ``report.png``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from gridtools.io import Tracks, load_tracks

__all__ = [
    "CANDIDATE_COLUMNS",
    "STRATA",
    "BenchmarkConfig",
    "BenchmarkScore",
    "build_benchmark",
    "read_detections",
    "score_benchmark",
    "write_benchmark",
    "write_report",
]

STRATA = (
    "chirp_ambiguous",
    "chirp_other",
    "negative_hard",
    "negative_easy",
    "invisible",
)
"""Item strata, in report order."""

CANDIDATE_COLUMNS = (
    "recording",
    "t_start_s",
    "t_end_s",
    "f_low_hz",
    "f_high_hz",
    "candidate_id",
)
"""The only columns of the benchmark's candidates file."""

_BOX = ("t_start_s", "t_end_s", "f_low_hz", "f_high_hz")
_CHIRP_FLOATS = (
    "t_center_s",
    "duration_s",
    "baseline_hz",
    "excursion_hz",
    "f_low_hz",
    "f_high_hz",
    "dip",
    "box_t0_s",
    "box_t1_s",
    "box_f0_hz",
    "box_f1_hz",
    "snr_db",
    "harmonic_box_t0_s",
    "harmonic_box_t1_s",
    "harmonic_box_f0_hz",
    "harmonic_box_f1_hz",
)
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class BenchmarkConfig:
    """How to sample the benchmark (see the module docstring)."""

    seed: int = 0
    n_chirps: int = 150
    n_ambiguous: int = 75
    n_hard: int = 100
    n_easy: int = 50
    n_invisible: int = 15
    ambiguous_excursion_hz: float = 60.0
    ambiguous_snr_quantile: float = 0.25
    excursion_edges_hz: tuple[float, ...] = (60.0, 150.0)
    min_hard_score: float = 0.05
    tolerance_s: float = 0.1
    margin_s: float = 1.1
    clear_hard_s: float = 0.1
    clear_easy_s: float = 0.5
    max_easy_attempts: int = 100_000

    def __post_init__(self) -> None:
        if not 0 <= self.n_ambiguous <= self.n_chirps:
            msg = "n_ambiguous must lie in [0, n_chirps]"
            raise ValueError(msg)
        if min(self.n_hard, self.n_easy, self.n_invisible) < 0:
            msg = "item counts must be >= 0"
            raise ValueError(msg)
        object.__setattr__(
            self, "excursion_edges_hz", tuple(float(e) for e in self.excursion_edges_hz)
        )


# ------------------------------------------------------------------ inputs


def _floats(df: pd.DataFrame, column: str) -> FloatArray:
    """A column as floats (NaN where missing or not a number)."""
    out = np.full(len(df), np.nan)
    if column not in df.columns:
        return out
    for i, value in enumerate(df[column].to_numpy(dtype=object)):
        try:
            out[i] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _load_chirps(scene: Path) -> pd.DataFrame:
    path = scene / "chirps.parquet"
    chirps = (
        pd.read_parquet(path) if path.exists() else pd.read_csv(scene / "chirps.csv")
    )
    chirps = chirps.reset_index(drop=True)
    for column in _CHIRP_FLOATS:
        chirps[column] = _floats(chirps, column)
    chirps["visible"] = chirps["visible"].to_numpy(dtype=bool)
    chirps["truncated"] = (
        chirps["truncated"].to_numpy(dtype=bool)
        if "truncated" in chirps.columns
        else np.zeros(len(chirps), dtype=bool)
    )
    return chirps


def _duration_s(scene: Path) -> float:
    meta = scene / "scene.json"
    if meta.exists():
        data = json.loads(meta.read_text())
        if "duration_s" in data:
            return float(data["duration_s"])
    from gridtools.io import open_recording

    return float(open_recording(scene).duration_s)


def read_detections(directory: str | Path) -> dict[str, pd.DataFrame]:
    """Per-scene detections (``<scene>.parquet``) of a model run."""
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(Path(directory).glob("*.parquet")):
        out[path.stem] = pd.read_parquet(path)
    return out


def _kept(detections: pd.DataFrame) -> pd.DataFrame:
    """Detections the post-filter kept."""
    det = detections.reset_index(drop=True)
    if "drop_reason" in det.columns:
        reason = det["drop_reason"].to_numpy(dtype=object)
        keep = np.array([r is None or r == "" or r != r for r in reason], dtype=bool)
        det = det.loc[keep].reset_index(drop=True)
    return det


def _chirp_boxes(chirps: pd.DataFrame) -> FloatArray:
    """``(t0, t1, f0, f1)`` per chirp: footprint box, else onset/offset band."""
    t0, t1 = _floats(chirps, "box_t0_s"), _floats(chirps, "box_t1_s")
    f0, f1 = _floats(chirps, "box_f0_hz"), _floats(chirps, "box_f1_hz")
    on, off = _floats(chirps, "t_onset_s"), _floats(chirps, "t_offset_s")
    lo, hi = _floats(chirps, "f_low_hz"), _floats(chirps, "f_high_hz")
    bad = ~np.isfinite(t0 + t1 + f0 + f1)
    t0[bad], t1[bad], f0[bad], f1[bad] = on[bad], off[bad], lo[bad], hi[bad]
    return np.column_stack([t0, t1, f0, f1])


def _time_overlap(spans: FloatArray, t0: float, t1: float) -> npt.NDArray[np.bool_]:
    ok = np.isfinite(spans[:, 0]) & np.isfinite(spans[:, 1])
    return ok & (spans[:, 0] < t1) & (spans[:, 1] > t0)


def _best_overlapping(
    det: pd.DataFrame, box: Sequence[float] | FloatArray, tolerance_s: float
) -> int:
    """Row of the highest-scoring detection matching a box (-1: none)."""
    if not len(det):
        return -1
    mid = 0.5 * (det["t_start_s"].to_numpy(float) + det["t_end_s"].to_numpy(float))
    centre = 0.5 * (box[0] + box[1])
    ok = (
        (np.abs(mid - centre) <= tolerance_s)
        & (det["f_low_hz"].to_numpy(float) < box[3])
        & (det["f_high_hz"].to_numpy(float) > box[2])
    )
    if not ok.any():
        return -1
    scores = np.where(ok, det["score"].to_numpy(float), -np.inf)
    return int(np.argmax(scores))


def _model_columns(det: pd.DataFrame, row: int) -> dict[str, Any]:
    if row < 0:
        return {"model_score": 0.0, "model_emitter_id": None, "model_emitter_hz": None}
    r = det.iloc[row]
    emitter = r.get("emitter_id")
    emitter_hz = r.get("emitter_hz")
    return {
        "model_score": float(r["score"]),
        "model_emitter_id": None
        if emitter is None or pd.isna(emitter)
        else int(emitter),
        "model_emitter_hz": None
        if emitter_hz is None or pd.isna(emitter_hz)
        else float(emitter_hz),
    }


# ------------------------------------------------------------------ build


@dataclass
class _Scene:
    name: str
    path: Path
    duration_s: float
    chirps: pd.DataFrame
    boxes: FloatArray
    harmonics: FloatArray
    detections: pd.DataFrame
    chirp_det: npt.NDArray[np.int64]
    tracks: Tracks


def _prepare(
    scene_dirs: Sequence[str | Path],
    detections: Mapping[str, pd.DataFrame],
    config: BenchmarkConfig,
) -> list[_Scene]:
    from chirpdetector.params.samples import match_predictions

    scenes = []
    # absolute: the candidates file is read from another directory
    for directory in sorted(Path(d).resolve() for d in scene_dirs):
        name = directory.name
        if name not in detections:
            msg = f"no detections for scene {name}"
            raise ValueError(msg)
        chirps = _load_chirps(directory)
        det = _kept(detections[name])
        index, _ = match_predictions(det, chirps, config.tolerance_s)
        chirp_det = np.full(len(chirps), -1, dtype=np.int64)
        for d, c in enumerate(index):
            if c >= 0:
                chirp_det[c] = d
        harmonics = np.column_stack(
            [
                _floats(chirps, c)
                for c in (
                    "harmonic_box_t0_s",
                    "harmonic_box_t1_s",
                    "harmonic_box_f0_hz",
                    "harmonic_box_f1_hz",
                )
            ]
        )
        det = det.assign(_chirp=index)
        scenes.append(
            _Scene(
                name=name,
                path=directory,
                duration_s=_duration_s(directory),
                chirps=chirps,
                boxes=_chirp_boxes(chirps),
                harmonics=harmonics,
                detections=det,
                chirp_det=chirp_det,
                tracks=load_tracks(directory),
            )
        )
    return scenes


def _even(n: int, sizes: Sequence[int]) -> list[int]:
    """Split ``n`` over cells as evenly as their sizes allow."""
    take = [0] * len(sizes)
    left = n
    open_cells = [i for i, s in enumerate(sizes) if s > 0]
    while left > 0 and open_cells:
        share = max(1, left // len(open_cells))
        for i in list(open_cells):
            add = min(share, sizes[i] - take[i], left)
            take[i] += add
            left -= add
            if take[i] >= sizes[i]:
                open_cells.remove(i)
            if left == 0:
                break
    return take


def _excursion_bin(excursion: float, edges: Sequence[float]) -> str:
    names = [f"<{edges[0]:g}"]
    names += [f"{a:g}-{b:g}" for a, b in itertools.pairwise(edges)]
    names.append(f">={edges[-1]:g}")
    if not math.isfinite(excursion):
        return "unknown"
    return names[int(np.searchsorted(np.asarray(edges), excursion, side="right"))]


def _item_id(seed: int, scene: str, kind: str, key: object) -> str:
    text = f"{seed}|{scene}|{kind}|{key}"
    return "h" + hashlib.sha256(text.encode()).hexdigest()[:10]


def _chirp_row(
    scene: _Scene, c: int, stratum: str, config: BenchmarkConfig, quartiles: FloatArray
) -> dict[str, Any]:
    chirp = scene.chirps.iloc[c]
    box = scene.boxes[c]
    snr = float(chirp["snr_db"])
    det_row = int(scene.chirp_det[c])
    row: dict[str, Any] = {
        "scene": scene.name,
        "recording": str(scene.path),
        "stratum": stratum,
        "truth": "chirp" if stratum.startswith("chirp") else "invisible_chirp",
        "scored": stratum.startswith("chirp"),
        "chirp_id": int(chirp["chirp_id"]) if "chirp_id" in chirp else c,
        "emitter_id": int(chirp["emitter_id"]),
        "baseline_hz": float(chirp["baseline_hz"]),
        "snr_db": snr,
        "snr_quartile": int(np.searchsorted(quartiles, snr, side="right")) + 1,
        "excursion_hz": float(chirp["excursion_hz"]),
        "excursion_bin": _excursion_bin(
            float(chirp["excursion_hz"]), config.excursion_edges_hz
        ),
        "duration_s": float(chirp["duration_s"]),
        "dip": float(chirp["dip"]),
        **dict(zip(_BOX, (float(v) for v in box), strict=True)),
        "det_row": det_row,
        **_model_columns(scene.detections, det_row),
        "candidate_id": _item_id(config.seed, scene.name, "chirp", int(c)),
    }
    return row


def build_benchmark(
    scene_dirs: Sequence[str | Path],
    detections: Mapping[str, pd.DataFrame],
    config: BenchmarkConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Sample the benchmark items.

    Parameters
    ----------
    scene_dirs
        Simulated scenes (``chirps.parquet`` or ``chirps.csv``, wavetracker
        tracks, ``scene.json`` with ``duration_s``).
    detections
        The model's post-filtered detections per scene name (columns of
        `chirpdetector.inference.predict_span`; ``emitter_id`` optional).
    config
        Sampling.

    Returns
    -------
    candidates
        The labeller's candidates (:data:`CANDIDATE_COLUMNS`), shuffled.
    truth
        One row per item (module docstring), in candidates order.
    summary
        Counts, pool sizes, quartile edges and the configuration.
    """
    config = config or BenchmarkConfig()
    scenes = _prepare(scene_dirs, detections, config)
    rng = np.random.default_rng(config.seed)
    margin = config.margin_s

    def inside(scene: _Scene, box: FloatArray) -> bool:
        centre = 0.5 * (box[0] + box[1])
        return margin <= centre <= scene.duration_s - margin

    # visible and invisible chirps
    visible: list[tuple[int, int]] = []
    invisible: list[tuple[int, int]] = []
    for s, scene in enumerate(scenes):
        for c in range(len(scene.chirps)):
            chirp = scene.chirps.iloc[c]
            box = scene.boxes[c]
            if bool(chirp["truncated"]) or not np.isfinite(box).all():
                continue
            if not inside(scene, box):
                continue
            (visible if bool(chirp["visible"]) else invisible).append((s, c))
    snr = np.array([scenes[s].chirps["snr_db"].iloc[c] for s, c in visible], float)
    excursion = np.array(
        [scenes[s].chirps["excursion_hz"].iloc[c] for s, c in visible], float
    )
    quartiles = (
        np.quantile(snr, [0.25, 0.5, 0.75]) if len(snr) else np.array([0.0, 0.0, 0.0])
    )
    snr_q = np.searchsorted(quartiles, snr, side="right")
    exc_bin = np.searchsorted(
        np.asarray(config.excursion_edges_hz), excursion, side="right"
    )
    ambiguous = (
        snr < np.quantile(snr, config.ambiguous_snr_quantile) if len(snr) else snr > 0
    ) & (excursion < config.ambiguous_excursion_hz)
    rows: list[dict[str, Any]] = []
    weights: list[float] = []

    pool = np.flatnonzero(ambiguous)
    n_amb = min(config.n_ambiguous, pool.size)
    for k in np.sort(rng.choice(pool, size=n_amb, replace=False)) if n_amb else []:
        s, c = visible[int(k)]
        rows.append(_chirp_row(scenes[s], c, "chirp_ambiguous", config, quartiles))
        weights.append(pool.size / n_amb)
    n_bins = len(config.excursion_edges_hz) + 1
    cells = [
        np.flatnonzero(~ambiguous & (snr_q == q) & (exc_bin == b))
        for q in range(4)
        for b in range(n_bins)
    ]
    takes = _even(config.n_chirps - n_amb, [len(cell) for cell in cells])
    for cell, take in zip(cells, takes, strict=True):
        if take == 0:
            continue
        for k in np.sort(rng.choice(cell, size=take, replace=False)):
            s, c = visible[int(k)]
            rows.append(_chirp_row(scenes[s], c, "chirp_other", config, quartiles))
            weights.append(len(cell) / take)

    # hard negatives: model detections far from every chirp
    hard_pool: list[tuple[int, int]] = []
    for s, scene in enumerate(scenes):
        det = scene.detections
        if not len(det):
            continue
        boxes = det[list(_BOX)].to_numpy(dtype=np.float64)
        for d in range(len(det)):
            if (
                det["_chirp"].iloc[d] >= 0
                or det["score"].iloc[d] < config.min_hard_score
            ):
                continue
            t0, t1 = (
                boxes[d, 0] - config.clear_hard_s,
                boxes[d, 1] + config.clear_hard_s,
            )
            if _time_overlap(scene.boxes, t0, t1).any():
                continue
            if _time_overlap(scene.harmonics, t0, t1).any():
                continue
            if inside(scene, boxes[d]):
                hard_pool.append((s, d))
    n_hard = min(config.n_hard, len(hard_pool))
    picked = (
        np.sort(rng.choice(len(hard_pool), size=n_hard, replace=False))
        if n_hard
        else []
    )
    for k in picked:
        s, d = hard_pool[int(k)]
        scene = scenes[s]
        box = scene.detections[list(_BOX)].iloc[d].to_numpy(dtype=np.float64)
        rows.append(
            {
                "scene": scene.name,
                "recording": str(scene.path),
                "stratum": "negative_hard",
                "truth": "no_chirp",
                "scored": True,
                **dict(zip(_BOX, (float(v) for v in box), strict=True)),
                "det_row": d,
                **_model_columns(
                    scene.detections,
                    _best_overlapping(scene.detections, box, config.tolerance_s),
                ),
                "candidate_id": _item_id(config.seed, scene.name, "hard", d),
            }
        )
        weights.append(len(hard_pool) / n_hard)

    # easy negatives: a fish's baseline with no chirp nearby
    shapes: list[tuple[float, float, float]] = []
    for k in np.flatnonzero(excursion < config.ambiguous_excursion_hz):
        s, c = visible[int(k)]
        box, base = scenes[s].boxes[c], float(scenes[s].chirps["baseline_hz"].iloc[c])
        shapes.append((box[1] - box[0], box[2] - base, box[3] - base))
    easy: list[dict[str, Any]] = []
    attempts = 0
    while len(easy) < config.n_easy and shapes and attempts < config.max_easy_attempts:
        attempts += 1
        scene = scenes[int(rng.integers(len(scenes)))]
        fish = [int(i) for i in scene.tracks.ids if int(i) < 1000]
        if not fish:
            continue
        ident = fish[int(rng.integers(len(fish)))]
        t = float(rng.uniform(margin, scene.duration_s - margin))
        duration, lo, hi = shapes[int(rng.integers(len(shapes)))]
        clear = (
            t - duration / 2 - config.clear_easy_s,
            t + duration / 2 + config.clear_easy_s,
        )
        if (
            _time_overlap(scene.boxes, *clear).any()
            or _time_overlap(scene.harmonics, *clear).any()
        ):
            continue
        f = float(scene.tracks.track(ident).frequency_at(t, 1.0)[0])
        if not math.isfinite(f):
            continue
        box = np.array([t - duration / 2, t + duration / 2, f + lo, f + hi])
        if any(
            e["scene"] == scene.name and abs(e["t_start_s"] - box[0]) < 1.0
            for e in easy
        ):
            continue
        easy.append(
            {
                "scene": scene.name,
                "recording": str(scene.path),
                "stratum": "negative_easy",
                "truth": "no_chirp",
                "scored": True,
                "baseline_hz": f,
                "fish_id": ident,
                **dict(zip(_BOX, (float(v) for v in box), strict=True)),
                **_model_columns(
                    scene.detections,
                    _best_overlapping(scene.detections, box, config.tolerance_s),
                ),
                "candidate_id": _item_id(config.seed, scene.name, "easy", attempts),
            }
        )
    rows.extend(easy)
    weights.extend([1.0] * len(easy))

    # invisible chirps (reported only)
    n_inv = min(config.n_invisible, len(invisible))
    for k in (
        np.sort(rng.choice(len(invisible), size=n_inv, replace=False)) if n_inv else []
    ):
        s, c = invisible[int(k)]
        rows.append(_chirp_row(scenes[s], c, "invisible", config, quartiles))
        weights.append(len(invisible) / n_inv)

    truth = pd.DataFrame(rows)
    truth["weight"] = weights
    truth["item_id"] = truth["scene"] + "#" + truth["candidate_id"]
    if truth["item_id"].duplicated().any():
        msg = "duplicate benchmark items"
        raise RuntimeError(msg)
    order = rng.permutation(len(truth))
    truth = truth.iloc[order].reset_index(drop=True)
    for column in (
        "emitter_id",
        "model_emitter_id",
        "chirp_id",
        "det_row",
        "fish_id",
        "snr_quartile",
    ):
        if column in truth.columns:
            truth[column] = pd.array(
                [None if v is None or v != v else int(v) for v in truth[column]],
                dtype="Int64",
            )
    candidates = truth[list(CANDIDATE_COLUMNS)].copy()
    first = ["item_id", "candidate_id", "scene", "stratum", "truth", "scored"]
    truth = truth[[*first, *(c for c in truth.columns if c not in first)]]
    counts = truth["stratum"].value_counts()
    summary: dict[str, Any] = {
        "config": asdict(config),
        "n_scenes": len(scenes),
        "n_items": len(truth),
        "per_stratum": {s: int(counts.get(s, 0)) for s in STRATA},
        "pools": {
            "visible": len(visible),
            "ambiguous": int(pool.size),
            "hard_negatives": len(hard_pool),
            "invisible": len(invisible),
            "easy_attempts": attempts,
        },
        "snr_quartile_edges_db": [float(q) for q in quartiles],
        "ambiguous_snr_below_db": float(
            np.quantile(snr, config.ambiguous_snr_quantile) if len(snr) else math.nan
        ),
    }
    return candidates, truth, summary


def write_benchmark(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    summary: dict[str, Any],
    candidates_path: str | Path,
    truth_path: str | Path,
) -> None:
    """Write the candidates file, the truth file and ``<truth>.json``.

    Raises
    ------
    ValueError
        If both would share a directory (keep the truth away from the
        labeller's inputs) or the candidates carry other columns.
    """
    candidates_path, truth_path = Path(candidates_path), Path(truth_path)
    if candidates_path.resolve().parent == truth_path.resolve().parent:
        msg = "write the truth to another directory than the candidates"
        raise ValueError(msg)
    if tuple(candidates.columns) != CANDIDATE_COLUMNS:
        msg = f"candidates must have exactly {CANDIDATE_COLUMNS}"
        raise ValueError(msg)
    for path in (candidates_path, truth_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(candidates_path, index=False)
    truth.to_parquet(truth_path, index=False)
    sidecar = dict(summary)
    sidecar["candidates"] = str(candidates_path.resolve())
    truth_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=1))


# ------------------------------------------------------------------ scoring


def _mcnemar(a: npt.NDArray[np.bool_], b: npt.NDArray[np.bool_]) -> dict[str, Any]:
    """Exact two-sided McNemar test of paired correctness ``a`` vs ``b``."""
    from scipy.stats import binomtest

    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(min(only_a, only_b), n, 0.5).pvalue)
    return {"n": len(a), "only_human": only_a, "only_model": only_b, "p": p}


@dataclass
class BenchmarkScore:
    """Scores of one benchmark (tables keyed as in ``metrics``)."""

    items: pd.DataFrame
    metrics: dict[str, Any]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    curve: pd.DataFrame = field(default_factory=pd.DataFrame)


def _objects(df: pd.DataFrame, column: str) -> npt.NDArray[np.object_]:
    if column not in df.columns:
        return np.full(len(df), None, dtype=object)
    return np.asarray(df[column].to_numpy(dtype=object), dtype=object)


def _missing(value: object) -> bool:
    return (
        value is None or value is pd.NA or (isinstance(value, float) and value != value)
    )


def _ints(df: pd.DataFrame, column: str, fill: int) -> npt.NDArray[np.int64]:
    """Integer column with NA replaced by ``fill``."""
    return np.array(
        [fill if _missing(v) else int(float(v)) for v in _objects(df, column)],  # pyright: ignore[reportArgumentType]
        dtype=np.int64,
    )


def _text(value: object) -> str:
    if _missing(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))  # an Int64 column read back as float
    return str(value)


def _strings(df: pd.DataFrame, column: str) -> npt.NDArray[np.object_]:
    return np.array([_text(v) for v in _objects(df, column)], dtype=object)


def _order(value: str) -> tuple[float, str]:
    """Sort key of group names: numbers and bin names by their lower edge."""
    text = value.lstrip("Q")
    for prefix, low in (("<", -math.inf), (">=", None)):
        if text.startswith(prefix):
            return (low if low is not None else float(text[2:]), value)
    try:
        return (float(text.split("-")[0]), value)
    except ValueError:
        return (math.inf, value)


def _select(df: pd.DataFrame, mask: npt.NDArray[np.bool_]) -> pd.DataFrame:
    out: pd.DataFrame = df.loc[np.asarray(mask, dtype=bool)]
    return out.reset_index(drop=True)


def _human(labels: pd.DataFrame, labeler: str | None) -> pd.DataFrame:
    """One label per item: ``human_label``, ``human_emitter_id`` (-1: none),
    ``elapsed_s``."""
    if labeler is not None and len(labels):
        labels = _select(labels, _strings(labels, "labeler") == labeler)
    out = pd.DataFrame(
        {
            "item_id": _strings(labels, "item_id"),
            "human_label": _strings(labels, "label"),
            "human_emitter_id": _ints(labels, "emitter_id", -1),
            "elapsed_s": _floats(labels, "elapsed_s"),
            "labeled_at": _strings(labels, "labeled_at"),
        }
    )
    if len(set(out["item_id"])) < len(out):
        if labeler is None:
            msg = "several labellers labelled the same items; pass labeler"
            raise ValueError(msg)
        out = out.sort_values("labeled_at").drop_duplicates("item_id", keep="last")
    return out


def _rate(values: npt.NDArray[np.bool_], weights: FloatArray | None = None) -> float:
    if not len(values):
        return math.nan
    if weights is None:
        return float(values.mean())
    return float((values * weights).sum() / weights.sum())


def _bootstrap(
    groups: npt.NDArray[np.object_],
    stats: dict[str, Any],
    n_boot: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Percentile 95% CIs of statistics, resampling items within groups."""
    rng = np.random.default_rng(seed)
    index = {g: np.flatnonzero(groups == g) for g in sorted(set(groups.tolist()))}
    draws: dict[str, list[float]] = {k: [] for k in stats}
    for _ in range(n_boot):
        sample = np.concatenate(
            [rng.choice(rows, size=rows.size, replace=True) for rows in index.values()]
        )
        for key, fn in stats.items():
            draws[key].append(fn(sample))
    out = {}
    for key, values in draws.items():
        finite = np.asarray([v for v in values if math.isfinite(v)])
        out[key] = (
            (float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975)))
            if finite.size
            else (math.nan, math.nan)
        )
    return out


def score_benchmark(
    truth: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    threshold: float = 0.5,
    labeler: str | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> BenchmarkScore:
    """Compare human labels and the model on the benchmark items.

    Parameters
    ----------
    truth
        The truth file of :func:`build_benchmark`.
    labels
        The labeller's label rows (``item_id``, ``label``, ``emitter_id``,
        optionally ``elapsed_s``, ``labeler``), e.g.
        :func:`chirpdetector.labeling.summary.read_label_files`.
    threshold
        The model says chirp at ``model_score >= threshold``.
    labeler
        Only this labeller's labels.
    n_boot, seed
        Stratified bootstrap.

    Returns
    -------
    The per-item table and the metrics (module docstring). Items without a
    human label are counted (``unlabelled``) and left out.
    """
    human = _human(labels, labeler)
    items = truth.merge(human, on="item_id", how="left")
    items["labelled"] = np.array(
        [not _missing(v) for v in _objects(items, "human_label")]
    )
    label = _strings(items, "human_label")
    true_em = _ints(items, "emitter_id", -2)
    items["human_chirp"] = np.isin(label, ["chirp", "chirp_emitter_unclear"])
    items["model_chirp"] = _floats(items, "model_score") >= threshold
    items["human_emitter_ok"] = items["human_chirp"].to_numpy(dtype=bool) & (
        _ints(items, "human_emitter_id", -1) == true_em
    )
    items["model_emitter_ok"] = items["model_chirp"].to_numpy(dtype=bool) & (
        _ints(items, "model_emitter_id", -1) == true_em
    )
    labelled = items["labelled"].to_numpy(dtype=bool)
    scored = _select(items, labelled & items["scored"].to_numpy(dtype=bool))
    chirp = _strings(scored, "truth") == "chirp"
    hc = scored["human_chirp"].to_numpy(dtype=bool)
    mc = scored["model_chirp"].to_numpy(dtype=bool)
    heo = scored["human_emitter_ok"].to_numpy(dtype=bool)
    meo = scored["model_emitter_ok"].to_numpy(dtype=bool)
    weight = _floats(scored, "weight")
    stratum = _strings(scored, "stratum")
    hard = stratum == "negative_hard"
    easy = stratum == "negative_easy"
    neg = hard | easy

    def on(dec: npt.NDArray[np.bool_], mask: npt.NDArray[np.bool_]) -> Any:
        return lambda rows: _rate(dec[rows][mask[rows]])

    def weighted(dec: npt.NDArray[np.bool_]) -> Any:
        return lambda rows: _rate(dec[rows][chirp[rows]], weight[rows][chirp[rows]])

    def diff(a: Any, b: Any) -> Any:
        return lambda rows: a(rows) - b(rows)

    stats = {
        "recall_human": on(hc, chirp),
        "recall_model": on(mc, chirp),
        "recall_diff": diff(on(mc, chirp), on(hc, chirp)),
        "recall_weighted_human": weighted(hc),
        "recall_weighted_model": weighted(mc),
        "false_alarm_human": on(hc, neg),
        "false_alarm_model": on(mc, neg),
        "false_alarm_diff": diff(on(mc, neg), on(hc, neg)),
        "false_alarm_hard_human": on(hc, hard),
        "false_alarm_hard_model": on(mc, hard),
        "false_alarm_easy_human": on(hc, easy),
        "false_alarm_easy_model": on(mc, easy),
        "emitter_recall_human": on(heo, chirp),
        "emitter_recall_model": on(meo, chirp),
        "emitter_recall_diff": diff(on(meo, chirp), on(heo, chirp)),
    }
    rows = np.arange(len(scored))
    point = {k: fn(rows) for k, fn in stats.items()}
    ci = _bootstrap(stratum, stats, n_boot, seed) if len(scored) else {}

    given_h = chirp & hc & (_ints(scored, "human_emitter_id", -1) >= 0)
    given_m = chirp & mc & (_ints(scored, "model_emitter_id", -1) >= 0)
    unclear = chirp & (_strings(scored, "human_label") == "chirp_emitter_unclear")
    right_h = np.where(chirp, hc, ~hc)
    right_m = np.where(chirp, mc, ~mc)
    metrics: dict[str, Any] = {
        "threshold": threshold,
        "labeler": labeler,
        "n_items": len(truth),
        "n_labelled": int(labelled.sum()),
        "unlabelled": int((~labelled).sum()),
        "n_scored": len(scored),
        "n_true_chirps": int(chirp.sum()),
        "n_negatives": int(neg.sum()),
        "point": point,
        "ci95": {k: list(v) for k, v in ci.items()},
        "emitter_accuracy": {
            "human": _rate(heo[given_h]),
            "human_n": int(given_h.sum()),
            "model": _rate(meo[given_m]),
            "model_n": int(given_m.sum()),
            "human_unclear": int(unclear.sum()),
        },
        "mcnemar": {
            "chirps": _mcnemar(hc[chirp], mc[chirp]),
            "negatives": _mcnemar(~hc[neg], ~mc[neg]),
            "all": _mcnemar(right_h, right_m),
            "emitter_on_chirps": _mcnemar(heo[chirp], meo[chirp]),
        },
    }
    invisible = labelled & (_strings(items, "stratum") == "invisible")
    metrics["invisible"] = {
        "n": int(invisible.sum()),
        "human_chirp": int(items["human_chirp"].to_numpy(dtype=bool)[invisible].sum()),
        "model_chirp": int(items["model_chirp"].to_numpy(dtype=bool)[invisible].sum()),
    }
    elapsed = _floats(items, "elapsed_s")[labelled]
    has_time = np.isfinite(elapsed)
    metrics["time_s"] = {
        "median": float(np.median(elapsed[has_time])) if has_time.any() else math.nan,
        "total": float(elapsed[has_time].sum()),
    }

    tables: dict[str, pd.DataFrame] = {}
    for name, key in (
        ("by_stratum", "stratum"),
        ("by_snr_quartile", "snr_quartile"),
        ("by_excursion", "excursion_bin"),
    ):
        keys = _strings(scored, key)
        table = []
        for value in sorted(set(keys.tolist()) - {""}, key=_order):
            g = keys == value
            c = g & chirp
            n = g & ~chirp
            times = _floats(scored, "elapsed_s")[g]
            table.append(
                {
                    key: value,
                    "n": int(g.sum()),
                    "n_chirps": int(c.sum()),
                    "recall_human": _rate(hc[c]),
                    "recall_model": _rate(mc[c]),
                    "false_alarm_human": _rate(hc[n]),
                    "false_alarm_model": _rate(mc[n]),
                    "emitter_recall_human": _rate(heo[c]),
                    "emitter_recall_model": _rate(meo[c]),
                    "median_time_s": float(np.median(times[np.isfinite(times)]))
                    if np.isfinite(times).any()
                    else math.nan,
                }
            )
        tables[name] = pd.DataFrame(table)

    scores = _floats(scored, "model_score")
    thresholds = np.unique(np.concatenate([[0.0], scores, [np.inf]]))
    curve = pd.DataFrame(
        {
            "threshold": thresholds,
            "recall": [_rate((scores >= t)[chirp]) for t in thresholds],
            "false_alarm": [_rate((scores >= t)[neg]) for t in thresholds],
        }
    )
    return BenchmarkScore(items=items, metrics=metrics, tables=tables, curve=curve)


# ------------------------------------------------------------------ report


def _fmt(value: float, ci: Sequence[float] | None = None) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    text = f"{value:.3f}"
    if ci is not None and all(math.isfinite(c) for c in ci):
        text += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
    return text


def _markdown(table: pd.DataFrame) -> str:
    """A DataFrame as a Markdown table (floats with 3 decimals)."""

    def cell(value: object) -> str:
        if isinstance(value, float):
            return "-" if not math.isfinite(value) else f"{value:.3f}"
        return str(value)

    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    rule = "|" + "---|" * len(table.columns)
    rows = [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in table.itertuples(index=False)
    ]
    return "\n".join([header, rule, *rows])


def _plot(score: BenchmarkScore, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    items = score.items
    use = (
        items["labelled"].to_numpy(dtype=bool)
        & items["scored"].to_numpy(dtype=bool)
        & (_strings(items, "truth") == "chirp")
    )
    chirps = _select(items, use)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    rng = np.random.default_rng(0)
    edges = score.metrics.get("snr_quartile_edges_db")
    for ax, key, title in (
        (axes[0], "snr_quartile", "recall by SNR quartile"),
        (axes[1], "excursion_bin", "recall by excursion (Hz)"),
    ):
        keys = _strings(chirps, key)
        values_of = sorted(set(keys.tolist()) - {""}, key=_order)
        x = np.arange(len(values_of))
        for offset, column, colour, label in (
            (-0.08, "human_chirp", "#1f77b4", "human"),
            (
                0.08,
                "model_chirp",
                "#d62728",
                f"model (score >= {score.metrics['threshold']:g})",
            ),
        ):
            decisions = chirps[column].to_numpy(dtype=bool)
            mean, low, high = [], [], []
            for value in values_of:
                dec = decisions[keys == value]
                boots = [rng.choice(dec, dec.size).mean() for _ in range(1000)]
                mean.append(dec.mean())
                low.append(dec.mean() - np.quantile(boots, 0.025))
                high.append(np.quantile(boots, 0.975) - dec.mean())
            ax.errorbar(
                x + offset,
                mean,
                yerr=[low, high],
                fmt="o-",
                color=colour,
                label=label,
                capsize=3,
            )
        names = list(values_of)
        if key == "snr_quartile" and edges:
            bounds = [-math.inf, *edges, math.inf]
            names = [
                f"Q{v}: {bounds[int(v) - 1]:.0f} to {bounds[int(v)]:.0f} dB"
                for v in values_of
            ]
        counts = [int((keys == v).sum()) for v in values_of]
        ax.set_xticks(x, [f"{n}\n(n={c})" for n, c in zip(names, counts, strict=True)])
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("recall (true chirps called chirp)")
        ax.set_title(title)
        ax.legend(loc="lower right")
    ax = axes[2]
    curve = score.curve
    ax.plot(
        curve["false_alarm"],
        curve["recall"],
        color="#d62728",
        label="model, all thresholds",
    )
    m = score.metrics["point"]
    ax.plot(
        m["false_alarm_model"],
        m["recall_model"],
        "s",
        color="#d62728",
        ms=8,
        label=f"model at {score.metrics['threshold']:g}",
    )
    ax.plot(
        m["false_alarm_human"],
        m["recall_human"],
        "o",
        color="#1f77b4",
        ms=9,
        label="human",
    )
    ax.set_xlabel("false-alarm rate on negatives")
    ax.set_ylabel("recall on true chirps")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("human vs model (same items)")
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_report(
    score: BenchmarkScore, out_dir: str | Path, *, summary: dict[str, Any] | None = None
) -> Path:
    """Write ``report.md``, ``metrics.json``, ``items.csv``, tables and ``report.png``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if summary is not None:
        score.metrics["snr_quartile_edges_db"] = summary.get("snr_quartile_edges_db")
        score.metrics["build"] = {
            k: summary[k] for k in ("per_stratum", "pools") if k in summary
        }
    score.items.to_csv(out / "items.csv", index=False)
    for name, table in score.tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    score.curve.to_csv(out / "model_curve.csv", index=False)

    def clean(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    (out / "metrics.json").write_text(json.dumps(clean(score.metrics), indent=1))
    _plot(score, out / "report.png")
    m, ci = score.metrics["point"], score.metrics["ci95"]
    mc = score.metrics["mcnemar"]
    em = score.metrics["emitter_accuracy"]
    thr = score.metrics["threshold"]
    lines = [
        "# Human vs model on simulated chirps",
        "",
        f"{score.metrics['n_labelled']} of {score.metrics['n_items']} items labelled "
        f"({score.metrics['unlabelled']} open); {score.metrics['n_true_chirps']} true "
        f"chirps and {score.metrics['n_negatives']} negatives scored. Model: dual-head "
        f"score >= {thr:g}. 95% CIs: stratified bootstrap over items. Human 'chirp' "
        "includes 'chirp, emitter unclear'.",
        "",
        "| | human | model | model - human | McNemar p (only human / only model correct) |",
        "|---|---|---|---|---|",
        f"| recall on true chirps | {_fmt(m['recall_human'], ci.get('recall_human'))} | "
        f"{_fmt(m['recall_model'], ci.get('recall_model'))} | {_fmt(m['recall_diff'], ci.get('recall_diff'))} | "
        f"{mc['chirps']['p']:.3g} ({mc['chirps']['only_human']} / {mc['chirps']['only_model']}) |",
        f"| false alarms on negatives | {_fmt(m['false_alarm_human'], ci.get('false_alarm_human'))} | "
        f"{_fmt(m['false_alarm_model'], ci.get('false_alarm_model'))} | {_fmt(m['false_alarm_diff'], ci.get('false_alarm_diff'))} | "
        f"{mc['negatives']['p']:.3g} ({mc['negatives']['only_human']} / {mc['negatives']['only_model']}) |",
        f"| ... hard negatives (model detections) | {_fmt(m['false_alarm_hard_human'], ci.get('false_alarm_hard_human'))} | "
        f"{_fmt(m['false_alarm_hard_model'], ci.get('false_alarm_hard_model'))} | | |",
        f"| ... easy negatives (baselines) | {_fmt(m['false_alarm_easy_human'], ci.get('false_alarm_easy_human'))} | "
        f"{_fmt(m['false_alarm_easy_model'], ci.get('false_alarm_easy_model'))} | | |",
        f"| chirp found with the right emitter | {_fmt(m['emitter_recall_human'], ci.get('emitter_recall_human'))} | "
        f"{_fmt(m['emitter_recall_model'], ci.get('emitter_recall_model'))} | {_fmt(m['emitter_recall_diff'], ci.get('emitter_recall_diff'))} | "
        f"{mc['emitter_on_chirps']['p']:.3g} ({mc['emitter_on_chirps']['only_human']} / {mc['emitter_on_chirps']['only_model']}) |",
        f"| recall, reweighted to all visible test chirps | {_fmt(m['recall_weighted_human'], ci.get('recall_weighted_human'))} | "
        f"{_fmt(m['recall_weighted_model'], ci.get('recall_weighted_model'))} | | |",
        "",
        f"All decisions (chirps and negatives): McNemar p = {mc['all']['p']:.3g} "
        f"({mc['all']['only_human']} items only the human got right, "
        f"{mc['all']['only_model']} only the model).",
        "",
        f"Emitter accuracy when an emitter was given: human {_fmt(em['human'])} "
        f"(n = {em['human_n']}; {em['human_unclear']} true chirps marked 'emitter "
        f"unclear'), model {_fmt(em['model'])} (n = {em['model_n']}).",
        "",
        f"Invisible chirps (not scored): {score.metrics['invisible']['n']} labelled, "
        f"human said chirp for {score.metrics['invisible']['human_chirp']}, model for "
        f"{score.metrics['invisible']['model_chirp']}.",
        "",
        f"Human time per item: median {_fmt(score.metrics['time_s']['median'])} s, "
        f"total {score.metrics['time_s']['total'] / 60:.1f} min.",
        "",
        "Recall and false alarms depend on the benchmark mix only through the "
        "strata; precision would also depend on the chirp/negative ratio chosen "
        "here, so it is not reported. The ambiguous stratum (lowest SNR quartile, "
        "excursion < 60 Hz) is oversampled to about half the true chirps; the "
        "reweighted recall corrects for that.",
        "",
        "![recall vs SNR](report.png)",
        "",
    ]
    for name, table in score.tables.items():
        lines += [f"## {name.replace('_', ' ')}", "", _markdown(table), ""]
    path = out / "report.md"
    path.write_text("\n".join(lines))
    return path
