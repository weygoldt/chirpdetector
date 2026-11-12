"""Samples of the chirp-parameter decoder from simulated scenes.

A *sample* is one box on a simulated scene with its decoder input (crop and
scalars, `chirpdetector.params.crops.decoder_inputs`), the chirp it belongs
to (if any), that chirp's ground truth (`chirpdetector.params.truth`) and
covariates for the evaluation. `extract_samples` makes them for a set of
scene directories and writes a *sample set*::

    SAMPLES/
      manifest.json     format, configs, detector, scenes and splits, hashes
      samples.parquet   one row per sample (`SAMPLE_COLUMNS` and covariates)
      crops.npy         (n, n_windows, H, W) uint8, row ``row`` of the table

Per scene, the recording is read and cleaned of clicks exactly as the
detector's inference cleans it (the detector bundle's preprocessing), its
spectrogram is computed once with the detector's front end, and the boxes
come from one of two sources (`BoxSource`):

``detector`` (the production path)
    The detector bundle is run over the whole scene
    (`chirpdetector.inference.detect_in_spectrogram`, i.e. `predict_span`
    on the scene), with a score threshold (default the bundle's). Its
    predictions are matched one-to-one to the scene's chirps by the rule of
    `chirpdetector.evaluation`: centre times within ``match_tolerance_s``
    (100 ms; a chirp's centre is ``t_center_s``) and overlapping frequency
    ranges (the chirp's ``f_low_hz``-``f_high_hz``), the matching with the
    most pairs and the smallest total |Δt| (`chirpdetector.evaluation.
    matching.assign`). Every chirp takes part in the matching, also
    truncated and invisible ones, so that a prediction is never credited
    to the wrong chirp. Matched predictions are samples with a truth;
    unmatched ones (false positives) are kept in the splits listed in
    ``keep_unmatched`` (to evaluate what the decoder makes of them).
``footprint`` (fallback without a detector)
    The chirps' footprint boxes (``box_*`` of ``chirps.csv``), of visible
    chirps only by default, with every edge jittered by N(0, jitter_t_s)
    and N(0, jitter_f_hz) in the splits listed in ``jitter_splits``.

A sample is ``usable`` (has a valid target) if it is matched to a chirp
that is not truncated by the scene's edges and whose per-period truth is
finite and not edge-clipped.

Splits follow the detector datasets: by scene seed (`SplitRule`, default
0-1599 train, 1600-1799 val, 1800-1999 test); a scene inside a directory
named ``train``, ``val`` or ``test`` must agree with its seed's split.

Extraction is resumable: each scene is written to ``shards/`` first, and
the sample set is assembled once every scene is done.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import zlib
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from chirpdetector.bundle import Bundle
from chirpdetector.data.contract import SPLITS, sha256_file
from chirpdetector.data.scenes import DEFAULT_SPLIT, SplitRule, scene_id
from chirpdetector.params.crops import (
    SCALAR_NAMES,
    CropConfig,
    DecoderInputs,
    decoder_inputs,
)
from chirpdetector.params.truth import PARAMETERS, TRUTH_PAD_S, scene_truth
from chirpdetector.spectrogram import SpectrogramConfig

__all__ = [
    "BACKGROUNDS",
    "SAMPLES_FORMAT",
    "SAMPLE_COLUMNS",
    "BoxSource",
    "SampleSet",
    "SceneRef",
    "discover_scenes",
    "extract_samples",
    "load_samples",
    "match_predictions",
]

log = logging.getLogger(__name__)

SAMPLES_FORMAT = "chirpdetector-params-samples"
SAMPLES_FORMAT_VERSION = 1
MANIFEST_FILE = "manifest.json"
TABLE_FILE = "samples.parquet"
CROPS_FILE = "crops.npy"
SHARDS_DIR = "shards"
FISH_IN_BAND_MARGIN_HZ = 30.0
"""Another fish counts as in the chirp's band within this margin of it."""

SAMPLE_COLUMNS: tuple[str, ...] = (
    "row",
    "source_id",
    "scene_dir",
    "scene_name",
    "seed",
    "split",
    "kind",
    "n_electrodes",
    "box_source",
    "t_start_s",
    "t_end_s",
    "f_low_hz",
    "f_high_hz",
    "score",
    "matched",
    "usable",
    "chirp_id",
    "dt_match_s",
    *SCALAR_NAMES,
    "outside_time",
    "outside_freq",
    "canvas_shifted",
    *(f"true_{p}" for p in PARAMETERS),
    "truth_edge_clipped",
)
"""Columns of every ``samples.parquet`` (chirp covariates come in addition)."""

_CHIRP_COLUMNS = (
    "emitter_id",
    "t_center_s",
    "baseline_hz",
    "family",
    "template_id",
    "snr_db",
    "visible",
    "visible_fraction",
    "overlaps_other_chirp",
    "is_response",
    "truncated",
    "box_t0_s",
    "box_t1_s",
    "box_f0_hz",
    "box_f1_hz",
)
"""Columns of ``chirps.csv`` copied to each matched sample."""

_TEMPLATE_COLUMNS = {
    "template_excursion_hz": "excursion_hz",
    "template_undershoot_hz": "undershoot_hz",
    "template_duration_s": "duration_s",
}
"""Template-level values of ``chirps.csv`` (not the ground truth), renamed."""


# ------------------------------------------------------------------ config


@dataclass(frozen=True)
class BoxSource:
    """Where the boxes of the samples come from (module docstring).

    Parameters
    ----------
    mode
        ``"detector"`` (production) or ``"footprint"``.
    score_threshold
        Detector mode: keep predictions with at least this score (None:
        the detector bundle's threshold).
    match_tolerance_s
        Detector mode: largest difference of centre times of a match.
    keep_unmatched
        Detector mode: splits in which unmatched predictions are kept.
    jitter_t_s, jitter_f_hz
        Footprint mode: SD of the Gaussian jitter of each time and each
        frequency edge.
    jitter_splits
        Footprint mode: splits whose boxes are jittered.
    visible_only
        Footprint mode: only visible chirps.
    seed
        Seed of the jitter.
    """

    mode: Literal["detector", "footprint"] = "detector"
    score_threshold: float | None = None
    match_tolerance_s: float = 0.1
    keep_unmatched: tuple[str, ...] = ("val", "test")
    jitter_t_s: float = 0.010
    jitter_f_hz: float = 5.0
    jitter_splits: tuple[str, ...] = ("train",)
    visible_only: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("detector", "footprint"):
            msg = f"boxes.mode must be 'detector' or 'footprint', got {self.mode!r}"
            raise ValueError(msg)
        for name in ("keep_unmatched", "jitter_splits"):
            value = tuple(str(v) for v in getattr(self, name))
            object.__setattr__(self, name, value)
            unknown = sorted(set(value) - set(SPLITS))
            if unknown:
                msg = f"boxes.{name}: unknown splits {unknown}"
                raise ValueError(msg)
        if self.score_threshold is not None and not 0 <= self.score_threshold <= 1:
            msg = (
                f"boxes.score_threshold must lie in [0, 1], got {self.score_threshold}"
            )
            raise ValueError(msg)
        if min(self.match_tolerance_s, self.jitter_t_s, self.jitter_f_hz) < 0:
            msg = "boxes: tolerances and jitters must be >= 0"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """JSON form."""
        out = asdict(self)
        out["keep_unmatched"] = list(self.keep_unmatched)
        out["jitter_splits"] = list(self.jitter_splits)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BoxSource:
        """Inverse of `to_dict`; unknown keys raise."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            msg = f"unknown boxes keys {unknown}; known: {sorted(known)}"
            raise ValueError(msg)
        values = dict(data)
        for name in ("keep_unmatched", "jitter_splits"):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class SceneRef:
    """A scene directory and its place in the splits."""

    directory: Path
    source_id: str
    name: str
    seed: int
    split: str
    kind: str


def _scene_dirs(root: Path) -> list[Path]:
    if (root / "scene.json").is_file():
        return [root]
    found: list[Path] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if (child / "scene.json").is_file():
            found.append(child)
        else:
            found.extend(
                g for g in sorted(child.iterdir()) if (g / "scene.json").is_file()
            )
    return found


BACKGROUNDS = ("any", "hybrid", "synthetic")
"""Scene filters of `discover_scenes`: all scenes, only scenes on a real
background segment (``scene.json`` ``background.used``), only synthetic-noise
scenes."""


def discover_scenes(
    roots: Iterable[str | Path],
    split: SplitRule = DEFAULT_SPLIT,
    *,
    background: str = "any",
) -> list[SceneRef]:
    """Scenes under ``roots`` (a scene, a directory of scenes, or of split dirs).

    Parameters
    ----------
    roots
        Scene directories, directories of scenes, or directories of
        ``train``/``val``/``test`` scene directories.
    split
        Seed -> split rule (default: 0-1599 train, 1600-1799 val,
        1800-1999 test).
    background
        ``"hybrid"`` keeps only scenes on a real background segment
        (``scene.json`` ``background.used`` true; the author's rule for the
        decoder's data), ``"synthetic"`` only the others, ``"any"`` all.

    Raises
    ------
    FileNotFoundError
        If a root holds no scene.
    ValueError
        If a scene's directory names a split other than its seed's, or two
        scenes have the same source id.
    """
    infos: dict[Path, dict[str, Any]] = {}
    for root in roots:
        root = Path(root).resolve()
        dirs = _scene_dirs(root)
        if not dirs:
            msg = f"no scene directories (with scene.json) under {root}"
            raise FileNotFoundError(msg)
        for d in dirs:
            infos[d] = json.loads((d / "scene.json").read_text())
    splits = split.assign_scenes({d: int(info["seed"]) for d, info in infos.items()})
    if background not in BACKGROUNDS:
        msg = f"background must be one of {BACKGROUNDS}, got {background!r}"
        raise ValueError(msg)
    refs: list[SceneRef] = []
    seen: set[str] = set()
    for d, info in infos.items():
        used = info.get("background")
        hybrid = isinstance(used, Mapping) and bool(used.get("used"))
        if (background == "hybrid" and not hybrid) or (
            background == "synthetic" and hybrid
        ):
            continue
        name = d.parent.name
        if name in SPLITS and name != splits[d]:
            msg = f"{d} lies in a '{name}' directory but its seed is in '{splits[d]}'"
            raise ValueError(msg)
        source = scene_id(info)
        if source in seen:
            msg = f"two scenes have the source id {source}"
            raise ValueError(msg)
        seen.add(source)
        refs.append(
            SceneRef(
                directory=d,
                source_id=source,
                name=str(info["name"]),
                seed=int(info["seed"]),
                split=str(splits[d]),
                kind="hybrid" if hybrid else "simulation",
            )
        )
    return sorted(refs, key=lambda r: r.source_id)


# ------------------------------------------------------------------ matching


def match_predictions(
    predictions: pd.DataFrame, chirps: pd.DataFrame, tolerance_s: float = 0.1
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Match predictions one-to-one to simulated chirps (module docstring).

    Parameters
    ----------
    predictions
        ``t_start_s, t_end_s, f_low_hz, f_high_hz, score``.
    chirps
        ``t_center_s, f_low_hz, f_high_hz`` (the scene's ``chirps.csv``).
    tolerance_s
        Largest difference of centre times.

    Returns
    -------
    chirp_index, dt_s
        Row of ``chirps`` matched to each prediction (-1: none) and the
        signed centre-time difference (prediction minus chirp; NaN if
        unmatched).
    """
    from chirpdetector.evaluation.matching import admissible_pairs, assign

    n_pred = len(predictions)
    index = np.full(n_pred, -1, dtype=np.int64)
    dt = np.full(n_pred, np.nan)
    if n_pred == 0 or len(chirps) == 0:
        return index, dt
    pred_t = 0.5 * (
        predictions["t_start_s"].to_numpy(dtype=np.float64)
        + predictions["t_end_s"].to_numpy(dtype=np.float64)
    )
    label_t = chirps["t_center_s"].to_numpy(dtype=np.float64)
    li, pj, adt = admissible_pairs(
        label_t,
        (
            chirps["f_low_hz"].to_numpy(dtype=np.float64),
            chirps["f_high_hz"].to_numpy(dtype=np.float64),
        ),
        pred_t,
        (
            predictions["f_low_hz"].to_numpy(dtype=np.float64),
            predictions["f_high_hz"].to_numpy(dtype=np.float64),
        ),
        time_tolerance_s=tolerance_s,
    )
    label_rank = np.empty(len(chirps), dtype=np.int64)
    label_rank[np.argsort(label_t, kind="stable")] = np.arange(len(chirps))
    pred_rank = np.empty(n_pred, dtype=np.int64)
    pred_rank[np.argsort(-predictions["score"].to_numpy(), kind="stable")] = np.arange(
        n_pred
    )
    chosen = assign(li, pj, adt, label_rank, pred_rank)
    index[pj[chosen]] = li[chosen]
    dt[pj[chosen]] = pred_t[pj[chosen]] - label_t[li[chosen]]
    return index, dt


# ------------------------------------------------------------------ per scene


@dataclass
class _Prepared:
    """What a worker reads and computes for one scene (no spectrogram)."""

    ref: SceneRef
    samples: npt.NDArray[np.float32]
    samplerate_hz: float
    n_electrodes: int
    chirps: pd.DataFrame


def _covariates(scene_dir: Path, chirps: pd.DataFrame) -> pd.DataFrame:
    """ΔF to the nearest other live fish and whether one is in the chirp's band."""
    from gridtools.io import load_tracks

    out = pd.DataFrame(index=chirps.index)
    if len(chirps) == 0:
        out["min_df_live_hz"] = pd.Series(dtype=float)
        out["fish_in_band"] = pd.Series(dtype=bool)
        return out
    tracks = load_tracks(scene_dir)
    centres = chirps["t_center_s"].to_numpy(dtype=np.float64)
    track_list = list(tracks)
    if track_list:
        live = np.stack([t.frequency_at(centres) for t in track_list], axis=1)
        idents = np.array([int(t.ident) for t in track_list])
    else:
        live = np.zeros((len(centres), 0))
        idents = np.zeros(0, dtype=np.int64)
    emitters = chirps["emitter_id"].to_numpy(dtype=np.int64)
    baselines = chirps["baseline_hz"].to_numpy(dtype=np.float64)
    others = np.where(idents[None, :] != emitters[:, None], live, np.nan)
    rel = others - baselines[:, None]
    has = np.isfinite(rel).any(axis=1)
    min_df = np.full(len(centres), np.nan)
    min_df[has] = np.nanmin(np.abs(rel[has]), axis=1)
    height = chirps["true_height_hz"].to_numpy(dtype=np.float64)
    top = np.where(np.isfinite(height), height, chirps["excursion_hz"].to_numpy(float))
    with np.errstate(invalid="ignore"):
        in_band = (rel > -FISH_IN_BAND_MARGIN_HZ) & (
            rel < top[:, None] + FISH_IN_BAND_MARGIN_HZ
        )
    out["min_df_live_hz"] = min_df
    out["fish_in_band"] = in_band.any(axis=1)
    return out


def _prepare_scene(ref: SceneRef, click_removal: Mapping[str, Any] | None) -> _Prepared:
    """Worker: read and clean the recording; truth and covariates of its chirps."""
    from gridtools.io import open_recording

    from chirpdetector.preprocessing import (
        ClickRemoval,
        can_remove_clicks,
        remove_clicks,
    )

    recording = open_recording(ref.directory / "recording.wav")
    fs = float(recording.samplerate_hz)
    raw = recording.read(0, recording.n_samples)
    n_electrodes = int(raw.shape[1])
    samples = raw
    if click_removal is not None:
        config = ClickRemoval.from_dict(dict(click_removal))
        if can_remove_clicks(n_electrodes, config, samplerate_hz=fs):
            samples, _ = remove_clicks(raw, fs, config)
    chirps = pd.read_csv(ref.directory / "chirps.csv")
    truth = scene_truth(ref.directory, chirps, pad_s=TRUTH_PAD_S).rename(
        columns={
            **{p: f"true_{p}" for p in PARAMETERS},
            "edge_clipped": "truth_edge_clipped",
            "t_peak_s": "true_t_peak_s",
        }
    )
    chirps = pd.concat(
        [chirps.reset_index(drop=True), truth.drop(columns="chirp_id")], axis=1
    )
    chirps = pd.concat([chirps, _covariates(ref.directory, chirps)], axis=1)
    return _Prepared(
        ref=ref,
        samples=np.ascontiguousarray(samples, dtype=np.float32),
        samplerate_hz=fs,
        n_electrodes=n_electrodes,
        chirps=chirps,
    )


_BOOL_COLUMNS = (
    "visible",
    "overlaps_other_chirp",
    "is_response",
    "truncated",
    "fish_in_band",
    "truth_edge_clipped",
)
_INT_COLUMNS = ("chirp_id", "emitter_id", "template_id")
_STR_COLUMNS = ("family",)


def _matched_rows(chirps: pd.DataFrame, index: npt.NDArray[np.int64]) -> pd.DataFrame:
    """The chirp of each sample: row ``index[i]`` of ``chirps``, blank if -1.

    Blank rows have NaN floats, False flags, -1 ids and empty strings.
    """
    needed = [
        *_CHIRP_COLUMNS,
        *_TEMPLATE_COLUMNS.values(),
        "chirp_id",
        "min_df_live_hz",
        "fish_in_band",
        "true_t_peak_s",
        "truth_edge_clipped",
        *(f"true_{p}" for p in PARAMETERS),
    ]
    matched = index >= 0
    source = chirps.reset_index(drop=True)
    out = {}
    for name in dict.fromkeys(needed):
        values = source[name].to_numpy() if name in source.columns else None
        if name in _BOOL_COLUMNS:
            column = np.zeros(len(index), dtype=bool)
            if values is not None and matched.any():
                column[matched] = values[index[matched]].astype(bool)
        elif name in _INT_COLUMNS:
            column = np.full(len(index), -1, dtype=np.int64)
            if values is not None and matched.any():
                column[matched] = values[index[matched]].astype(np.int64)
        elif name in _STR_COLUMNS:
            column = np.full(len(index), "", dtype=object)
            if values is not None and matched.any():
                column[matched] = values[index[matched]].astype(str)
        else:
            column = np.full(len(index), np.nan)
            if values is not None and matched.any():
                column[matched] = values[index[matched]].astype(np.float64)
        out[name] = column
    return pd.DataFrame(out)


def _footprint_boxes(
    chirps: pd.DataFrame, ref: SceneRef, boxes: BoxSource
) -> tuple[pd.DataFrame, npt.NDArray[np.int64]]:
    edges = chirps[["box_t0_s", "box_t1_s", "box_f0_hz", "box_f1_hz"]].to_numpy(float)
    keep = np.isfinite(edges).all(axis=1) & ~chirps["truncated"].to_numpy(dtype=bool)
    if boxes.visible_only:
        keep &= chirps["visible"].to_numpy(dtype=bool)
    rows = np.flatnonzero(keep)
    edges = edges[rows].copy()
    if ref.split in boxes.jitter_splits and len(rows):
        rng = np.random.default_rng(
            [boxes.seed, ref.seed, zlib.crc32(ref.name.encode())]
        )
        jitter = rng.normal(size=edges.shape)
        jitter[:, :2] *= boxes.jitter_t_s
        jitter[:, 2:] *= boxes.jitter_f_hz
        jittered = edges + jitter
        ok = (jittered[:, 1] - jittered[:, 0] > 0.02) & (
            jittered[:, 3] - jittered[:, 2] > 5.0
        )
        edges[ok] = jittered[ok]
    table = pd.DataFrame(
        edges, columns=["t_start_s", "t_end_s", "f_low_hz", "f_high_hz"]
    ).assign(score=np.nan)
    return table, rows.astype(np.int64)


def _scene_samples(
    prepared: _Prepared,
    *,
    spectrogram: SpectrogramConfig,
    crop: CropConfig,
    boxes: BoxSource,
    detector: Bundle | None,
    model: Any,
    device: str,
) -> tuple[pd.DataFrame, npt.NDArray[np.uint8]]:
    """Main process: spectrogram, boxes, matching and crops of one scene."""
    from chirpdetector.spectrogram import compute_spectrogram

    ref, chirps = prepared.ref, prepared.chirps
    spec = compute_spectrogram(
        prepared.samples, prepared.samplerate_hz, spectrogram, device=device
    )
    if boxes.mode == "detector":
        from chirpdetector.inference import detect_in_spectrogram

        assert detector is not None
        threshold = (
            detector.score_threshold
            if boxes.score_threshold is None
            else boxes.score_threshold
        )
        preds = detect_in_spectrogram(
            detector,
            spec,
            name=ref.source_id,
            score_threshold=threshold,
            detector=model,
        )
        index, dt = match_predictions(preds, chirps, boxes.match_tolerance_s)
        keep = index >= 0
        if ref.split in boxes.keep_unmatched:
            keep[:] = True
        table = preds.loc[
            keep, ["t_start_s", "t_end_s", "f_low_hz", "f_high_hz", "score"]
        ]
        table = table.reset_index(drop=True)
        index, dt = index[keep], dt[keep]
    else:
        table, index = _footprint_boxes(chirps, ref, boxes)
        dt = np.zeros(len(index))
    inputs: DecoderInputs = decoder_inputs(
        spec, table[["t_start_s", "t_end_s", "f_low_hz", "f_high_hz"]].to_numpy(), crop
    )
    matched = index >= 0
    rows = _matched_rows(chirps, index)
    out = pd.DataFrame(
        {
            "source_id": ref.source_id,
            "scene_dir": str(ref.directory),
            "scene_name": ref.name,
            "seed": ref.seed,
            "split": ref.split,
            "kind": ref.kind,
            "n_electrodes": prepared.n_electrodes,
            "box_source": boxes.mode,
            "t_start_s": table["t_start_s"].to_numpy(float),
            "t_end_s": table["t_end_s"].to_numpy(float),
            "f_low_hz": table["f_low_hz"].to_numpy(float),
            "f_high_hz": table["f_high_hz"].to_numpy(float),
            "score": table["score"].to_numpy(float),
            "matched": matched,
            "chirp_id": rows["chirp_id"].to_numpy(np.int64),
            "dt_match_s": dt,
        }
    )
    for k, name in enumerate(SCALAR_NAMES):
        out[name] = inputs.scalars[:, k]
    out["outside_time"] = inputs.outside_time
    out["outside_freq"] = inputs.outside_freq
    out["canvas_shifted"] = inputs.canvas_shifted
    for p in PARAMETERS:
        out[f"true_{p}"] = rows[f"true_{p}"].to_numpy(float)
    out["truth_edge_clipped"] = rows["truth_edge_clipped"].to_numpy(bool)
    finite = np.isfinite(out[[f"true_{p}" for p in PARAMETERS]].to_numpy()).all(axis=1)
    out["usable"] = (
        matched
        & finite
        & ~rows["truncated"].to_numpy(bool)
        & ~out["truth_edge_clipped"].to_numpy()
    )
    for name in (*_CHIRP_COLUMNS, "min_df_live_hz", "fish_in_band", "true_t_peak_s"):
        out[name] = rows[name].to_numpy()
    for name, column in _TEMPLATE_COLUMNS.items():
        out[name] = rows[column].to_numpy(float)
    return out, inputs.crops


# ------------------------------------------------------------------ sample sets


@dataclass(frozen=True)
class SampleSet:
    """A loaded sample set (module docstring).

    Attributes
    ----------
    directory
        Where it lies.
    manifest
        ``manifest.json``.
    table
        ``samples.parquet``; ``row`` indexes `crops`.
    crops
        Memory-mapped ``(n, n_windows, H, W)`` uint8.
    """

    directory: Path
    manifest: Mapping[str, Any]
    table: pd.DataFrame
    crops: npt.NDArray[np.uint8]

    def select(
        self,
        split: str | None = None,
        *,
        usable: bool | None = None,
        min_score: float | None = None,
    ) -> pd.DataFrame:
        """Rows of a split (all if None), optionally only usable ones and scores >= ``min_score``."""
        mask = np.ones(len(self.table), dtype=bool)
        if split is not None:
            mask &= self.table["split"].to_numpy() == split
        if usable is not None:
            mask &= self.table["usable"].to_numpy(dtype=bool) == usable
        if min_score is not None:
            score = self.table["score"].to_numpy(dtype=np.float64)
            mask &= ~(score < min_score)  # NaN (footprint boxes) passes
        return self.table.loc[mask]

    def inputs(self, rows: pd.DataFrame) -> DecoderInputs:
        """Decoder inputs of table rows (crops read into memory)."""
        index = rows["row"].to_numpy(dtype=np.int64)
        order = np.argsort(index, kind="stable")
        crops = np.empty((len(index), *self.crops.shape[1:]), dtype=np.uint8)
        if len(index):
            crops[order] = self.crops[index[order]]
        return DecoderInputs(
            crops=crops,
            scalars=rows[list(SCALAR_NAMES)].to_numpy(dtype=np.float32),
            outside_time=rows["outside_time"].to_numpy(dtype=np.float64),
            outside_freq=rows["outside_freq"].to_numpy(dtype=np.float64),
            canvas_shifted=rows["canvas_shifted"].to_numpy(dtype=bool),
        )


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


def load_samples(directory: str | Path) -> SampleSet:
    """Load a complete sample set.

    Raises
    ------
    FileNotFoundError
        If it is missing or incomplete (no ``manifest.json``).
    ValueError
        If the manifest is of another format, or the files do not match
        its hashes.
    """
    directory = Path(directory)
    path = directory / MANIFEST_FILE
    if not path.is_file():
        msg = f"{path} does not exist (no sample set, or extraction not finished)"
        raise FileNotFoundError(msg)
    manifest = json.loads(path.read_text())
    if manifest.get("format") != SAMPLES_FORMAT:
        msg = f"{path}: not a sample set (format {manifest.get('format')!r})"
        raise ValueError(msg)
    if sha256_file(directory / TABLE_FILE) != manifest["files"][TABLE_FILE]:
        msg = f"{directory / TABLE_FILE} differs from its manifest hash"
        raise ValueError(msg)
    table = pd.read_parquet(directory / TABLE_FILE)
    crops = np.load(directory / CROPS_FILE, mmap_mode="r")
    if crops.shape[0] != len(table):
        msg = f"{directory}: {crops.shape[0]} crops but {len(table)} table rows"
        raise ValueError(msg)
    return SampleSet(directory.resolve(), manifest, table, crops)


def _identity(
    *,
    crop: CropConfig,
    boxes: BoxSource,
    spectrogram: SpectrogramConfig,
    preprocessing: Mapping[str, Any],
    detector: Bundle | None,
    split: SplitRule,
    scenes: Sequence[SceneRef],
) -> dict[str, Any]:
    """What determines a sample set's content (compared when resuming)."""
    return {
        "crop": crop.to_dict(),
        "boxes": boxes.to_dict(),
        "spectrogram": spectrogram.to_dict(),
        "preprocessing": dict(preprocessing),
        "detector": None
        if detector is None
        else {
            "weights_sha256": detector.weights_sha256,
            "path": str(detector.path),
            "run_id": detector.training.get("run_id"),
            "score_threshold": detector.score_threshold,
        },
        "split": split.to_dict(),
        "truth": {"definition": "chirpdetector.params.truth", "pad_s": TRUTH_PAD_S},
        "scenes": [
            {
                "source_id": r.source_id,
                "directory": str(r.directory),
                "split": r.split,
                "seed": r.seed,
                "kind": r.kind,
                "chirps_sha256": sha256_file(r.directory / "chirps.csv"),
            }
            for r in scenes
        ],
    }


def _resolve_device(device: str) -> str:
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _assemble(
    directory: Path, identity: Mapping[str, Any], refs: Sequence[SceneRef]
) -> SampleSet:
    """Concatenate the shards into ``samples.parquet`` and ``crops.npy``."""
    shards = directory / SHARDS_DIR
    tables = [pd.read_parquet(shards / f"{r.source_id}.parquet") for r in refs]
    n = sum(len(t) for t in tables)
    first = next((t for t in tables if len(t)), None)
    shape: tuple[int, ...]
    if first is None:
        shape = (0, 1, *identity["crop"]["size"])
    else:
        sample = np.load(shards / f"{first['source_id'].iloc[0]}.npy", mmap_mode="r")
        shape = (n, *sample.shape[1:])
    crops = np.lib.format.open_memmap(
        directory / CROPS_FILE, mode="w+", dtype=np.uint8, shape=shape
    )
    offset = 0
    for ref, table in zip(refs, tables, strict=True):
        if len(table):
            crops[offset : offset + len(table)] = np.load(
                shards / f"{ref.source_id}.npy"
            )
        table.insert(0, "row", np.arange(offset, offset + len(table), dtype=np.int64))
        offset += len(table)
    crops.flush()
    del crops
    table = pd.concat([t for t in tables if len(t)] or tables[:1], ignore_index=True)
    table.to_parquet(directory / TABLE_FILE, index=False)
    from chirpdetector.training import provenance

    split_of = table["split"].to_numpy()
    usable = table["usable"].to_numpy(dtype=bool)
    counts = {
        s: {
            "samples": int(np.count_nonzero(split_of == s)),
            "usable": int(np.count_nonzero((split_of == s) & usable)),
        }
        for s in SPLITS
    }
    manifest = {
        "format": SAMPLES_FORMAT,
        "format_version": SAMPLES_FORMAT_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **identity,
        "counts": counts,
        "files": {
            TABLE_FILE: sha256_file(directory / TABLE_FILE),
            CROPS_FILE: sha256_file(directory / CROPS_FILE),
        },
        "code": provenance.code_state(),
    }
    _write_json(directory / MANIFEST_FILE, manifest)
    shutil.rmtree(shards)
    return load_samples(directory)


def extract_samples(
    scenes: Sequence[SceneRef],
    out: str | Path,
    *,
    crop: CropConfig | None = None,
    boxes: BoxSource | None = None,
    detector: Bundle | None = None,
    spectrogram: SpectrogramConfig | None = None,
    preprocessing: Mapping[str, Any] | None = None,
    split: SplitRule = DEFAULT_SPLIT,
    workers: int = 4,
    device: str = "auto",
    progress: Callable[[str], None] | None = None,
) -> SampleSet:
    """Extract (or resume, or reuse) a sample set (module docstring).

    Parameters
    ----------
    scenes
        Scenes (`discover_scenes`).
    out
        Sample-set directory.
    crop
        Crop geometry.
    boxes
        Box source; ``mode="detector"`` needs ``detector``.
    detector
        Detector bundle; its front end and preprocessing are used.
    spectrogram, preprocessing
        Front end and preprocessing without a detector (default: the
        dataset builders' defaults, `SpectrogramConfig()` and click
        removal); must equal the detector's if both are given.
    split
        Recorded in the manifest (the scenes carry their split).
    workers
        Processes that read, clean and compute ground truth (the main
        process computes spectrograms, detections and crops).
    device
        Torch device of the main process (``auto``: CUDA if available).
    progress
        Called with a status line every few scenes (default: logging).

    Returns
    -------
    SampleSet

    Raises
    ------
    ValueError
        If ``out`` holds a sample set (or an interrupted extraction) made
        with other settings, or the detector mode lacks a detector.
    """
    from chirpdetector.preprocessing import ClickRemoval

    crop = crop or CropConfig()
    boxes = boxes or BoxSource()
    if boxes.mode == "detector" and detector is None:
        msg = "boxes.mode = 'detector' needs a detector bundle"
        raise ValueError(msg)
    if detector is not None:
        if spectrogram is not None and spectrogram != detector.spectrogram:
            msg = "the spectrogram config differs from the detector's front end"
            raise ValueError(msg)
        spectrogram = detector.spectrogram
        preprocessing = dict(detector.preprocessing)
    spectrogram = spectrogram or SpectrogramConfig()
    if preprocessing is None:
        preprocessing = {"click_removal": ClickRemoval().to_dict()}
    identity = _identity(
        crop=crop,
        boxes=boxes,
        spectrogram=spectrogram,
        preprocessing=preprocessing,
        detector=detector,
        split=split,
        scenes=scenes,
    )
    directory = Path(out)
    say = progress or log.info
    if (directory / MANIFEST_FILE).is_file():
        existing = load_samples(directory)
        stored = {k: existing.manifest.get(k) for k in identity}
        if stored != identity:
            differ = sorted(k for k in identity if stored[k] != identity[k])
            msg = f"{directory} holds a sample set made with other {differ}; use another directory"
            raise ValueError(msg)
        say(f"reusing the sample set in {directory}")
        return existing
    shards = directory / SHARDS_DIR
    shards.mkdir(parents=True, exist_ok=True)
    pending_path = directory / "extraction.json"
    if pending_path.is_file():
        if json.loads(pending_path.read_text()) != identity:
            msg = f"{directory} holds an interrupted extraction with other settings"
            raise ValueError(msg)
    else:
        _write_json(pending_path, identity)
    todo = [r for r in scenes if not (shards / f"{r.source_id}.parquet").is_file()]
    say(f"{len(scenes) - len(todo)} of {len(scenes)} scenes already extracted")
    device = _resolve_device(device)
    model = None
    if detector is not None:
        from chirpdetector.prediction import load_detector
        from chirpdetector.resources import apply_vram_limit

        if device != "cpu":
            apply_vram_limit(None, device)
        model = load_detector(detector, device=device)
    click = preprocessing.get("click_removal")
    started = time.perf_counter()
    n_done = 0

    def handle(prepared: _Prepared) -> None:
        nonlocal n_done
        table, crops = _scene_samples(
            prepared,
            spectrogram=spectrogram,
            crop=crop,
            boxes=boxes,
            detector=detector,
            model=model,
            device=device,
        )
        stem = prepared.ref.source_id
        tmp = shards / f".{stem}.npy.tmp"
        with tmp.open("wb") as f:
            np.save(f, crops, allow_pickle=False)
        tmp.replace(shards / f"{stem}.npy")
        tmp = shards / f".{stem}.parquet.tmp"
        table.to_parquet(tmp, index=False)
        tmp.replace(shards / f"{stem}.parquet")  # written last: marks the scene done
        n_done += 1
        if n_done % 25 == 0 or n_done == len(todo):
            rate = (time.perf_counter() - started) / n_done
            say(
                f"{n_done}/{len(todo)} scenes, {rate:.2f} s per scene, "
                f"{rate * (len(todo) - n_done) / 60:.0f} min left"
            )

    if workers <= 1:
        for ref in todo:
            handle(_prepare_scene(ref, click))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            queue: deque[Future[_Prepared]] = deque()
            it = iter(todo)
            for ref in it:
                queue.append(pool.submit(_prepare_scene, ref, click))
                if len(queue) >= workers + 1:
                    break
            while queue:
                prepared = queue.popleft().result()
                nxt = next(it, None)
                if nxt is not None:
                    queue.append(pool.submit(_prepare_scene, nxt, click))
                handle(prepared)
    result = _assemble(directory, identity, scenes)
    pending_path.unlink(missing_ok=True)
    return result


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
