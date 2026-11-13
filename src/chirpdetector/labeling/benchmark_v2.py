"""Human benchmark v2: edge-case strata, several models, repeats.

Extends :mod:`chirpdetector.labeling.benchmark` (whose items, truth and
labels stay valid) with items from new simulated test scenes, sampled so
that the report can say where a model works and where it does not.

Chirp features
--------------
Computed per chirp from the scene (:func:`scene_features`), for sampling and
for every report table:

``df_nearest_hz``
    Distance of the emitter's baseline to the nearest other live fish at the
    chirp, simulated or real background fish (the scene's ``min_df_hz``
    ignores the real ones).
``near_harmonic``
    The emitter's baseline lies within ``harmonic_near_hz`` of twice another
    live fish's frequency (its chirp may be taken for a harmonic copy).
``overlap``
    A chirp of another fish overlaps its box in time.
``bout``
    At least ``bout_min`` other chirps of the same fish have centres within
    ``bout_window_s``.
``drift``
    The emitter's track changes by at least ``drift_hz`` from 1 s before to
    1 s after the chirp (drift or rise).
``click``
    A click (:func:`click_events`) lies within ``click_window_s`` of the box.
``few_electrodes``
    The scene has at most ``few_electrodes`` electrodes.
``rippled_plateau``
    A big chirp whose template has a ripple flag, a ripple score in the top
    decile of big templates, or a flat top (``duration_50_s >= plateau_ratio
    x duration_10_s``).
``long``, ``short``
    ``duration_s >= 0.3`` and ``< 0.01``.
``gap``
    Set for items on *gap scenes*: copies of a scene whose emitter track has
    no detections within ``gap_half_s`` of the chirp (simulated tracks are
    gap-free; wavetracker's often break at chirps).

Strata
------
Every eligible visible chirp falls into exactly one stratum, the first of
:data:`CHIRP_STRATA` whose feature it has (rare features first); the rest
form the core cells (SNR quartile x excursion < 60 / 60-150 / >= 150 Hz).
Items are drawn uniformly within a stratum, so ``weight`` (stratum size /
items drawn) reweights any subgroup to the population of visible chirps.

Negatives (:data:`NEGATIVE_STRATA`): false alarms of each model (kept
detections with no chirp or harmonic box within ``clear_s``), stratified by
model and score bin over the whole range; harmonic copies of visible
chirps, split by whether the production model detected the fundamental;
clicks; beats (two fish within ``beat_df_hz``); rises (a track changing by
``drift_hz`` in 2 s); plain baselines. Negative boxes off detections have a
small chirp's shape on a fish's baseline. ``repeat`` items copy already
labelled items under a new id (``retest_of``).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from gridtools.io import Tracks, load_tracks, open_recording, save_tracks

from chirpdetector.labeling.benchmark import (
    _BOX,
    _chirp_boxes,
    _floats,
    _load_chirps,
    _select,
    _strings,
    _time_overlap,
)

__all__ = [
    "CHIRP_STRATA",
    "FEATURES",
    "NEGATIVE_STRATA",
    "BuildV2Config",
    "build_v2",
    "click_events",
    "make_gap_scene",
    "model_outcomes",
    "plan_gaps",
    "read_model_dir",
    "scene_features",
]

FEATURES = (
    "gap",
    "click",
    "near_harmonic",
    "rippled_plateau",
    "long",
    "short",
    "df_0_5",
    "df_5_10",
    "overlap",
    "bout",
    "drift",
    "few_electrodes",
    "df_10_25",
    "df_25_50",
)
"""Chirp features, in stratum priority order."""

CHIRP_STRATA = tuple(f"chirp_{f}" for f in FEATURES)
NEGATIVE_STRATA = (
    "negative_model_fa",
    "negative_harmonic_detected",
    "negative_harmonic_missed",
    "negative_click",
    "negative_beat",
    "negative_rise",
    "negative_baseline",
)
FloatArray = npt.NDArray[np.float64]
_TRACK_FILES = ("times.npy", "fund_v.npy", "ident_v.npy", "idx_v.npy", "sign_v.npy")


@dataclass(frozen=True)
class BuildV2Config:
    """Feature thresholds and item counts (module docstring)."""

    seed: int = 2
    n_per_stratum: tuple[tuple[str, int], ...] = (
        ("chirp_gap", 20),
        ("chirp_click", 15),
        ("chirp_near_harmonic", 15),
        ("chirp_rippled_plateau", 15),
        ("chirp_long", 25),
        ("chirp_short", 25),
        ("chirp_df_0_5", 25),
        ("chirp_df_5_10", 25),
        ("chirp_overlap", 25),
        ("chirp_bout", 20),
        ("chirp_drift", 15),
        ("chirp_few_electrodes", 25),
        ("chirp_df_10_25", 15),
        ("chirp_df_25_50", 10),
        ("invisible", 20),
        ("negative_harmonic_detected", 20),
        ("negative_harmonic_missed", 20),
        ("negative_click", 20),
        ("negative_beat", 25),
        ("negative_rise", 20),
        ("negative_baseline", 35),
    )
    n_core_per_cell: int = 10
    n_model_fa: tuple[tuple[str, int], ...] = (
        ("dualhead", 50),
        ("yolo26s", 35),
        ("v1", 35),
    )
    fa_score_edges: tuple[float, ...] = (0.05, 0.2, 0.5, 0.8)
    n_repeats: int = 30
    excursion_edges_hz: tuple[float, ...] = (60.0, 150.0)
    harmonic_near_hz: float = 20.0
    bout_window_s: float = 1.0
    bout_min: int = 2
    drift_hz: float = 2.0
    click_window_s: float = 0.05
    few_electrodes: int = 4
    plateau_ratio: float = 0.88
    beat_df_hz: float = 15.0
    clear_s: float = 0.1
    clear_negative_s: float = 0.5
    gap_half_s: float = 1.5
    margin_s: float = 2.6
    tolerance_s: float = 0.1
    fundamental_score: float = 0.5
    max_attempts: int = 200_000

    def count(self, stratum: str) -> int:
        """Items to draw from a stratum."""
        return dict(self.n_per_stratum).get(stratum, 0)


# ------------------------------------------------------------------ models


def read_model_dir(path: str | Path, kind: str = "v2") -> dict[str, pd.DataFrame]:
    """Detections of one model per scene, in one format.

    ``kind="v2"``: ``<scene>.parquet`` of `predict_span` (``keep_dropped``);
    ``kind="v1"``: the v1 runner's ``<dataset>/<scene>.csv``. Columns: the
    box, ``score``, ``status`` (``kept``, ``reattributed``, ``harmonic``,
    ``duplicate`` or ``unassignable``), ``emitter_id`` (float, NaN: none)
    and ``harmonic_f_low_hz``, ``harmonic_f_high_hz`` (the box as detected,
    for re-attributed copies; else the box).
    """
    out: dict[str, pd.DataFrame] = {}
    root = Path(path)
    files = (
        sorted(root.glob("*/*.csv")) if kind == "v1" else sorted(root.glob("*.parquet"))
    )
    for file in files:
        if file.name.endswith(".prenms.csv"):
            continue
        df = pd.read_csv(file) if kind == "v1" else pd.read_parquet(file)
        n = len(df)
        status = np.full(n, "kept", dtype=object)
        if kind == "v1":
            ok = (
                df["assignable"].to_numpy(dtype=bool)
                if "assignable" in df.columns
                else np.ones(n, bool)
            )
            status[~ok] = "unassignable"
            emitter = np.full(n, np.nan)
        else:
            reason = _strings(df, "drop_reason")
            status = np.where(reason != "", reason, status).astype(object)
            reattr = _strings(df, "reattributed") == "harmonic"
            status[reattr] = "reattributed"
            emitter = _floats(df, "emitter_id")
        lo, hi = _floats(df, "f_low_hz"), _floats(df, "f_high_hz")
        h_lo, h_hi = _floats(df, "harmonic_f_low_hz"), _floats(df, "harmonic_f_high_hz")
        out[file.stem] = pd.DataFrame(
            {
                **{c: _floats(df, c) for c in ("t_start_s", "t_end_s")},
                "f_low_hz": lo,
                "f_high_hz": hi,
                "score": _floats(df, "score"),
                "status": status,
                "emitter_id": emitter,
                "harmonic_f_low_hz": np.where(np.isfinite(h_lo), h_lo, lo),
                "harmonic_f_high_hz": np.where(np.isfinite(h_hi), h_hi, hi),
            }
        )
    return out


def _counted(det: pd.DataFrame) -> pd.DataFrame:
    """Detections a model reports (kept or re-attributed)."""
    status = _strings(det, "status")
    return _select(det, (status == "kept") | (status == "reattributed"))


def _overlapping(
    det: pd.DataFrame, box: Sequence[float], tolerance_s: float, harmonic: bool = False
) -> npt.NDArray[np.bool_]:
    """Detections whose midpoint is within ``tolerance_s`` of the box's and
    whose frequency range (as detected, with ``harmonic``) overlaps it."""
    if not len(det):
        return np.zeros(0, dtype=bool)
    mid = 0.5 * (_floats(det, "t_start_s") + _floats(det, "t_end_s"))
    lo_col, hi_col = (
        ("harmonic_f_low_hz", "harmonic_f_high_hz")
        if harmonic
        else ("f_low_hz", "f_high_hz")
    )
    centre = 0.5 * (box[0] + box[1])
    return (
        (np.abs(mid - centre) <= tolerance_s)
        & (_floats(det, lo_col) < box[3])
        & (_floats(det, hi_col) > box[2])
    )


def model_outcomes(
    items: pd.DataFrame,
    models: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    tolerance_s: float = 0.1,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Each model's answer on each item.

    True chirps (``truth`` ``chirp`` or ``invisible_chirp``): the counted
    detections (kept or re-attributed) at score >= ``threshold`` are matched
    one-to-one to all the scene's chirps, as the evaluation does
    (`chirpdetector.params.samples.match_predictions`); a chirp matched there
    gets its detection's score, any other the highest score of the counted
    detections below the threshold that could match it (box midpoint within
    ``tolerance_s`` of the chirp's, overlapping frequencies), so that the
    score curve below the threshold is meaningful. Harmonic copies: the
    highest-scoring detection at the copy's place as detected (dropped,
    kept or re-attributed). Other items: the highest-scoring counted
    detection overlapping the box.

    Returns
    -------
    One row per item and model: ``item_id``, ``model``, ``score`` (0 if
    none), ``status`` (``none`` if no detection, ``missing`` if the model
    has no detections for the scene), ``emitter_id`` (NaN: none).
    """
    from chirpdetector.params.samples import chirp_centres, match_predictions

    rows = []
    cache: dict[tuple[str, str], tuple[pd.DataFrame, dict[int, int], pd.DataFrame]] = {}
    chirps_of: dict[str, pd.DataFrame] = {}

    def answer(
        record: Mapping[str, Any],
        name: str,
        hit: pd.Series | None,
        missing: bool = False,
    ) -> dict[str, Any]:
        if missing:
            return {
                "item_id": record["item_id"],
                "model": name,
                "score": math.nan,
                "status": "missing",
                "emitter_id": math.nan,
            }
        if hit is None:
            return {
                "item_id": record["item_id"],
                "model": name,
                "score": 0.0,
                "status": "none",
                "emitter_id": math.nan,
            }
        return {
            "item_id": record["item_id"],
            "model": name,
            "score": float(np.asarray(hit["score"], dtype=np.float64)),
            "status": str(hit["status"]),
            "emitter_id": float(np.asarray(hit["emitter_id"], dtype=np.float64)),
        }

    for record in items.to_dict("records"):
        recording = str(record["recording"])
        scene = Path(recording).name
        truth = str(record["truth"])
        box = [float(record[c]) for c in _BOX]
        for name, by_scene in models.items():
            det = by_scene.get(scene)
            if det is None:
                rows.append(answer(record, name, None, missing=True))
                continue
            if truth in ("chirp", "invisible_chirp") and not _missing_int(
                record.get("chirp_id")
            ):
                key = (name, scene)
                if key not in cache:
                    if recording not in chirps_of:
                        chirps_of[recording] = _load_chirps(Path(recording))
                    counted = _counted(det)
                    high = _select(counted, _floats(counted, "score") >= threshold)
                    index, _ = match_predictions(
                        high, chirps_of[recording], tolerance_s
                    )
                    cache[key] = (
                        high,
                        {int(c): d for d, c in enumerate(index) if c >= 0},
                        _select(counted, _floats(counted, "score") < threshold),
                    )
                high, matched, low = cache[key]
                c = int(record["chirp_id"])
                if c in matched:
                    rows.append(answer(record, name, high.iloc[matched[c]]))
                    continue
                chirps = chirps_of[recording]
                centre = float(chirp_centres(chirps.iloc[[c]])[0])
                lo, hi = (
                    float(chirps["f_low_hz"].iloc[c]),
                    float(chirps["f_high_hz"].iloc[c]),
                )
                ok = _overlapping(low, [centre, centre, lo, hi], tolerance_s)
                rows.append(
                    answer(
                        record,
                        name,
                        low.iloc[
                            int(np.argmax(np.where(ok, _floats(low, "score"), -np.inf)))
                        ]
                        if ok.any()
                        else None,
                    )
                )
            elif str(record["stratum"]).startswith("negative_harmonic"):
                ok = _overlapping(det, box, tolerance_s, harmonic=True)
                rows.append(
                    answer(
                        record,
                        name,
                        det.iloc[
                            int(np.argmax(np.where(ok, _floats(det, "score"), -np.inf)))
                        ]
                        if ok.any()
                        else None,
                    )
                )
            else:
                counted = _counted(det)
                ok = _overlapping(counted, box, tolerance_s)
                rows.append(
                    answer(
                        record,
                        name,
                        counted.iloc[
                            int(
                                np.argmax(
                                    np.where(ok, _floats(counted, "score"), -np.inf)
                                )
                            )
                        ]
                        if ok.any()
                        else None,
                    )
                )
    return pd.DataFrame(rows)


def _missing_int(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return not math.isfinite(float(value))  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        return True


# ------------------------------------------------------------------ features


def click_events(scene: str | Path, *, min_gap_s: float = 0.05) -> FloatArray:
    """``(start_s, stop_s)`` of the clicks in a scene's recording.

    The samples the click removal repairs
    (`chirpdetector.preprocessing.clicks.remove_clicks`, electrode guard
    on), merged across electrodes and gaps below ``min_gap_s``. Empty for
    recordings with too few electrodes for the guard.
    """
    from chirpdetector.preprocessing.clicks import (
        ClickRemoval,
        can_remove_clicks,
        remove_clicks,
    )

    rec = open_recording(scene)
    config = ClickRemoval()
    if not can_remove_clicks(rec.n_channels, config):
        return np.zeros((0, 2))
    samples = np.asarray(rec.read(0, rec.n_samples), dtype=np.float32)
    _, mask = remove_clicks(samples, rec.samplerate_hz, config)
    hot = mask.any(axis=1)
    if not hot.any():
        return np.zeros((0, 2))
    edges = np.flatnonzero(np.diff(np.concatenate([[0], hot.astype(np.int8), [0]])))
    starts, stops = edges[0::2] / rec.samplerate_hz, edges[1::2] / rec.samplerate_hz
    merged: list[list[float]] = []
    for a, b in zip(starts, stops, strict=True):
        if merged and a - merged[-1][1] < min_gap_s:
            merged[-1][1] = float(b)
        else:
            merged.append([float(a), float(b)])
    return np.asarray(merged, dtype=np.float64)


def _template_flags(
    templates: pd.DataFrame | None, config: BuildV2Config
) -> dict[int, bool]:
    if templates is None or not len(templates):
        return {}
    family = _strings(templates, "family")
    big = family == "big"
    ripple_score = _floats(templates, "ripple_score")
    cut = np.nanquantile(ripple_score[big], 0.9) if big.any() else np.inf
    flag = _strings(templates, "ripple_flag") == "True"
    d10, d50 = _floats(templates, "duration_10_s"), _floats(templates, "duration_50_s")
    with np.errstate(invalid="ignore", divide="ignore"):
        plateau = d50 >= config.plateau_ratio * d10
    rippled = big & (flag | (ripple_score >= cut) | plateau)
    ids = _floats(templates, "template_id")
    return {
        int(i): bool(r) for i, r in zip(ids, rippled, strict=True) if math.isfinite(i)
    }


def scene_features(
    scene: str | Path,
    *,
    config: BuildV2Config | None = None,
    templates: pd.DataFrame | None = None,
    clicks: FloatArray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Chirp features (module docstring) and scene facts.

    Returns
    -------
    chirps
        The scene's chirps (``_load_chirps``) plus ``box_*`` edges and the
        feature columns (``df_nearest_hz``, ``near_harmonic``, ``overlap``,
        ``bout``, ``drift_hz``, ``drift``, ``click``, ``few_electrodes``,
        ``rippled_plateau``, ``long``, ``short``, ``gap``).
    info
        ``n_electrodes``, ``duration_s``, ``clicks`` (list of spans).
    """
    config = config or BuildV2Config()
    scene = Path(scene)
    chirps = _load_chirps(scene)
    meta = json.loads((scene / "scene.json").read_text())
    duration = float(meta["duration_s"])
    n_electrodes = (
        int(meta.get("grid", {}).get("n_electrodes", 0))
        or open_recording(scene).n_channels
    )
    tracks = load_tracks(scene)
    if clicks is None:
        clicks = click_events(scene)
    boxes = _chirp_boxes(chirps)
    n = len(chirps)
    t = _floats(chirps, "t_center_s")
    base = _floats(chirps, "baseline_hz")
    emitter = chirps["emitter_id"].to_numpy(dtype=np.int64)
    ids = [int(i) for i in tracks.ids]
    freqs = np.full((n, len(ids)), np.nan)
    before = np.full((n, len(ids)), np.nan)
    after = np.full((n, len(ids)), np.nan)
    for j, ident in enumerate(ids):
        track = tracks.track(ident)
        freqs[:, j] = track.frequency_at(t, 10.0)
        before[:, j] = track.frequency_at(t - 1.0, 10.0)
        after[:, j] = track.frequency_at(t + 1.0, 10.0)
    df_near = np.full(n, np.nan)
    near_harmonic = np.zeros(n, dtype=bool)
    drift = np.full(n, np.nan)
    for i in range(n):
        others = np.array(
            [k for k, ident in enumerate(ids) if ident != emitter[i]], dtype=np.int64
        )
        f_other = freqs[i, others] if others.size else np.zeros(0)
        f_other = f_other[np.isfinite(f_other)]
        if f_other.size:
            df_near[i] = float(np.min(np.abs(f_other - base[i])))
            near_harmonic[i] = bool(
                np.any(np.abs(2 * f_other - base[i]) <= config.harmonic_near_hz)
            )
        own = [k for k, ident in enumerate(ids) if ident == emitter[i]]
        if own:
            drift[i] = float(abs(after[i, own[0]] - before[i, own[0]]))
    overlap = np.zeros(n, dtype=bool)
    bout = np.zeros(n, dtype=bool)
    for i in range(n):
        other_fish = emitter != emitter[i]
        overlap[i] = bool(
            np.any(
                other_fish & (boxes[:, 0] < boxes[i, 1]) & (boxes[:, 1] > boxes[i, 0])
            )
        )
        same = (emitter == emitter[i]) & (np.arange(n) != i)
        bout[i] = (
            int(np.sum(same & (np.abs(t - t[i]) <= config.bout_window_s)))
            >= config.bout_min
        )
    click = np.zeros(n, dtype=bool)
    if len(clicks):
        for i in range(n):
            click[i] = bool(
                _time_overlap(
                    clicks,
                    boxes[i, 0] - config.click_window_s,
                    boxes[i, 1] + config.click_window_s,
                ).any()
            )
    flags = _template_flags(templates, config)
    template = _floats(chirps, "template_id")
    rippled = np.array(
        [
            bool(flags.get(int(x), False)) if math.isfinite(x) else False
            for x in template
        ],
        dtype=bool,
    )
    duration_s = _floats(chirps, "duration_s")
    out = chirps.copy()
    out["box_t0_s"], out["box_t1_s"], out["box_f0_hz"], out["box_f1_hz"] = boxes.T
    out["df_nearest_hz"] = df_near
    out["near_harmonic"] = near_harmonic
    out["overlap"] = overlap
    out["bout"] = bout
    out["drift_hz"] = drift
    out["drift"] = np.nan_to_num(drift) >= config.drift_hz
    out["click"] = click
    out["few_electrodes"] = n_electrodes <= config.few_electrodes
    out["rippled_plateau"] = rippled & (_strings(chirps, "family") == "big")
    out["long"] = duration_s >= 0.3
    out["short"] = duration_s < 0.01
    out["gap"] = False
    out["n_electrodes"] = n_electrodes
    info = {
        "n_electrodes": n_electrodes,
        "duration_s": duration,
        "clicks": [list(map(float, c)) for c in clicks],
        "track_ids": ids,
    }
    return out, info


def _df_features(df: FloatArray) -> dict[str, npt.NDArray[np.bool_]]:
    d = np.nan_to_num(df, nan=np.inf)
    return {
        "df_0_5": d < 5,
        "df_5_10": (d >= 5) & (d < 10),
        "df_10_25": (d >= 10) & (d < 25),
        "df_25_50": (d >= 25) & (d < 50),
    }


def make_gap_scene(
    source: str | Path,
    target: str | Path,
    ident: int,
    centres_s: Sequence[float],
    half_s: float,
) -> Path:
    """A copy of a scene whose track ``ident`` has no detections within
    ``half_s`` of the given times (other files symlinked)."""
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        if entry.name in _TRACK_FILES:
            continue
        link = target / entry.name
        if not link.exists():
            link.symlink_to(entry.resolve())
    tracks = load_tracks(source)
    times = tracks.times_s[tracks.time_indices]
    drop = np.zeros(len(tracks.freqs_hz), dtype=bool)
    for c in centres_s:
        drop |= (tracks.idents == ident) & (np.abs(times - c) <= half_s)
    keep = ~drop
    save_tracks(
        Tracks(
            times_s=tracks.times_s,
            freqs_hz=tracks.freqs_hz[keep],
            idents=tracks.idents[keep],
            time_indices=tracks.time_indices[keep],
            powers_db=None if tracks.powers_db is None else tracks.powers_db[keep],
        ),
        target,
    )
    return target


def plan_gaps(
    scenes: Mapping[str, tuple[Path, pd.DataFrame, dict[str, Any]]],
    n: int,
    *,
    seed: int,
    margin_s: float,
    target_root: str | Path,
    half_s: float = 1.5,
) -> dict[str, tuple[Path, pd.DataFrame, dict[str, Any]]]:
    """Make ``n`` gap copies, each for one visible chirp of a distinct scene.

    The copy is named ``<scene number + 10000>`` (6 digits) under
    ``target_root``; its features are the source's with ``gap`` set for
    the chosen chirp only (a gap of ``half_s`` either side does not change
    the 10-s liveness the features use).
    """
    rng = np.random.default_rng(seed)
    names = sorted(scenes)
    order = rng.permutation(len(names))
    out: dict[str, tuple[Path, pd.DataFrame, dict[str, Any]]] = {}
    for k in order:
        if len(out) == n:
            break
        name = names[int(k)]
        path, feats, info = scenes[name]
        duration = float(info["duration_s"])
        ok = [
            c
            for c in range(len(feats))
            if bool(feats["visible"].iloc[c])
            and not bool(feats["truncated"].iloc[c])
            and int(feats["emitter_id"].iloc[c]) < 1000
            and margin_s <= float(feats["t_center_s"].iloc[c]) <= duration - margin_s
        ]
        if not ok:
            continue
        c = ok[int(rng.integers(len(ok)))]
        alias = f"{int(name) + 10000:06d}" if name.isdigit() else f"{name}_gap"
        target = make_gap_scene(
            path,
            Path(target_root) / alias,
            int(feats["emitter_id"].iloc[c]),
            [float(feats["t_center_s"].iloc[c])],
            half_s,
        )
        gfeats = feats.copy()
        gfeats["gap"] = False
        gfeats.loc[gfeats.index[c], "gap"] = True
        out[alias] = (
            target,
            gfeats,
            {**info, "gap_of": name, "gap_chirp_id": int(feats["chirp_id"].iloc[c])},
        )
    return out


# ------------------------------------------------------------------ build


def _opaque(seed: int, *parts: object) -> str:
    text = "|".join(str(p) for p in (seed, *parts))
    return "h" + hashlib.sha256(text.encode()).hexdigest()[:10]


def _pick(rng: np.random.Generator, n_pool: int, n: int) -> npt.NDArray[np.int64]:
    n = min(n, n_pool)
    return (
        np.sort(rng.choice(n_pool, size=n, replace=False))
        if n
        else np.zeros(0, np.int64)
    )


@dataclass
class _Scene:
    name: str
    path: Path
    chirps: pd.DataFrame
    info: dict[str, Any]
    tracks: Tracks


def _feature_row(chirp: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "df_nearest_hz",
        "near_harmonic",
        "overlap",
        "bout",
        "drift_hz",
        "drift",
        "click",
        "few_electrodes",
        "rippled_plateau",
        "long",
        "short",
        "gap",
        "n_electrodes",
    )
    return {k: chirp.get(k) for k in keys}


def build_v2(
    scenes: Mapping[str, tuple[Path, pd.DataFrame, dict[str, Any]]],
    models: Mapping[str, Mapping[str, pd.DataFrame]],
    config: BuildV2Config | None = None,
    *,
    repeat_pool: pd.DataFrame | None = None,
    gap_scenes: Mapping[str, tuple[Path, pd.DataFrame, dict[str, Any]]] | None = None,
    production: str = "dualhead",
    exclude: set[tuple[str, int]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Sample the v2 items.

    Parameters
    ----------
    scenes
        ``{name: (scene dir, chirp features, info)}`` (:func:`scene_features`).
    models
        Per model, its detections per scene (:func:`read_model_dir`).
    config
        Counts and thresholds.
    repeat_pool
        Already labelled items (an earlier truth table) to repeat.
    gap_scenes
        Gap copies (:func:`make_gap_scene`) with their features; their
        ``gap_items`` chirps form the gap stratum.
    production
        The model whose detected fundamentals split the harmonic copies.
    exclude
        ``(scene, chirp_id)`` never drawn as chirps (e.g. those shown on gap
        copies).

    Returns
    -------
    candidates, truth, summary
        As :func:`chirpdetector.labeling.benchmark.build_benchmark`; the
        truth has the feature columns and ``weight``, ``retest_of``.
    """
    config = config or BuildV2Config()
    rng = np.random.default_rng(config.seed)
    margin = config.margin_s
    loaded: dict[str, _Scene] = {
        name: _Scene(name, path, feats, info, load_tracks(path))
        for name, (path, feats, info) in sorted(scenes.items())
    }
    rows: list[dict[str, Any]] = []
    pools: dict[str, int] = {}

    def inside(scene: _Scene, t0: float, t1: float) -> bool:
        centre = 0.5 * (t0 + t1)
        return margin <= centre <= float(scene.info["duration_s"]) - margin

    def chirp_item(
        scene: _Scene, c: int, stratum: str, weight: float, truth: str = "chirp"
    ) -> dict[str, Any]:
        chirp = scene.chirps.iloc[c].to_dict()
        return {
            "scene": scene.name,
            "recording": str(scene.path),
            "stratum": stratum,
            "truth": truth,
            "scored": truth == "chirp",
            "chirp_id": int(chirp["chirp_id"]) if "chirp_id" in chirp else c,
            "emitter_id": int(chirp["emitter_id"]),
            "baseline_hz": float(chirp["baseline_hz"]),
            "snr_db": float(chirp["snr_db"]),
            "excursion_hz": float(chirp["excursion_hz"]),
            "duration_s": float(chirp["duration_s"]),
            "dip": float(chirp["dip"]),
            "t_start_s": float(chirp["box_t0_s"]),
            "t_end_s": float(chirp["box_t1_s"]),
            "f_low_hz": float(chirp["box_f0_hz"]),
            "f_high_hz": float(chirp["box_f1_hz"]),
            **_feature_row(chirp),
            "weight": weight,
            "candidate_id": _opaque(config.seed, scene.name, "chirp", c),
        }

    # ---- visible chirps: one stratum each, by priority
    eligible: list[tuple[str, int]] = []
    invisible: list[tuple[str, int]] = []
    for scene in loaded.values():
        ch = scene.chirps
        for c in range(len(ch)):
            row = ch.iloc[c]
            box = (
                float(row["box_t0_s"]),
                float(row["box_t1_s"]),
                float(row["box_f0_hz"]),
                float(row["box_f1_hz"]),
            )
            if (
                bool(row["truncated"])
                or not all(math.isfinite(v) for v in box)
                or not inside(scene, box[0], box[1])
            ):
                continue
            if exclude and (scene.name, int(row["chirp_id"])) in exclude:
                continue
            (eligible if bool(row["visible"]) else invisible).append((scene.name, c))
    feats = pd.DataFrame([loaded[s].chirps.iloc[c].to_dict() for s, c in eligible])
    snr = _floats(feats, "snr_db")
    quartiles = np.quantile(snr, [0.25, 0.5, 0.75]) if len(snr) else np.zeros(3)
    df_bins = _df_features(_floats(feats, "df_nearest_hz"))
    assigned = np.full(len(eligible), "", dtype=object)
    for feature in FEATURES:
        if feature == "gap":
            continue
        has = (
            df_bins[feature]
            if feature in df_bins
            else feats[feature].to_numpy(dtype=bool)
        )
        assigned[(assigned == "") & has] = f"chirp_{feature}"
    q = np.searchsorted(quartiles, snr, side="right")
    e = np.searchsorted(
        np.asarray(config.excursion_edges_hz),
        _floats(feats, "excursion_hz"),
        side="right",
    )
    for k in np.flatnonzero(assigned == ""):
        assigned[k] = f"chirp_core_q{q[k] + 1}_e{e[k]}"
    for stratum in sorted(set(assigned.tolist())):
        members = np.flatnonzero(assigned == stratum)
        n = (
            config.n_core_per_cell
            if stratum.startswith("chirp_core")
            else config.count(stratum)
        )
        pools[stratum] = int(members.size)
        chosen = members[_pick(rng, members.size, n)]
        for k in chosen:
            s, c = eligible[int(k)]
            rows.append(chirp_item(loaded[s], c, stratum, members.size / len(chosen)))

    # ---- gap stratum on gap copies
    if gap_scenes:
        for name, (path, feats_g, info) in sorted(gap_scenes.items()):
            scene = _Scene(name, path, feats_g, info, load_tracks(path))
            for c in np.flatnonzero(feats_g["gap"].to_numpy(dtype=bool)):
                rows.append(chirp_item(scene, int(c), "chirp_gap", 1.0))
        pools["chirp_gap"] = sum(
            int(f["gap"].to_numpy(dtype=bool).sum()) for _, f, _ in gap_scenes.values()
        )

    # ---- invisible
    pools["invisible"] = len(invisible)
    for k in _pick(rng, len(invisible), config.count("invisible")):
        s, c = invisible[int(k)]
        rows.append(
            chirp_item(
                loaded[s],
                c,
                "invisible",
                len(invisible) / max(1, config.count("invisible")),
                truth="invisible_chirp",
            )
        )

    # ---- negatives
    def clear(scene: _Scene, t0: float, t1: float, pad: float) -> bool:
        ch = scene.chirps
        boxes = ch[["box_t0_s", "box_t1_s"]].to_numpy(dtype=np.float64)
        harm = np.column_stack(
            [_floats(ch, "harmonic_box_t0_s"), _floats(ch, "harmonic_box_t1_s")]
        )
        return not (
            _time_overlap(boxes, t0 - pad, t1 + pad).any()
            or _time_overlap(harm, t0 - pad, t1 + pad).any()
        )

    def negative(
        scene: _Scene,
        box: Sequence[float] | FloatArray,
        stratum: str,
        weight: float,
        key: object,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "scene": scene.name,
            "recording": str(scene.path),
            "stratum": stratum,
            "truth": "no_chirp",
            "scored": True,
            **dict(zip(_BOX, (float(v) for v in box), strict=True)),
            "n_electrodes": scene.info["n_electrodes"],
            "few_electrodes": scene.info["n_electrodes"] <= config.few_electrodes,
            "weight": weight,
            "candidate_id": _opaque(config.seed, scene.name, stratum, key),
            **extra,
        }

    # model false alarms, per model and score bin
    edges = (*config.fa_score_edges, math.inf)
    taken: list[tuple[str, FloatArray]] = []
    for model, n_model in config.n_model_fa:
        cells: dict[int, list[tuple[str, int]]] = {b: [] for b in range(len(edges) - 1)}
        for scene in loaded.values():
            det = models.get(model, {}).get(scene.name)
            if det is None or not len(det):
                continue
            counted = _counted(det)
            for d in range(len(counted)):
                score = float(counted["score"].iloc[d])
                t0, t1 = (
                    float(counted["t_start_s"].iloc[d]),
                    float(counted["t_end_s"].iloc[d]),
                )
                if (
                    score < edges[0]
                    or not inside(scene, t0, t1)
                    or not clear(scene, t0, t1, config.clear_s)
                ):
                    continue
                cells[int(np.searchsorted(edges, score, side="right")) - 1].append(
                    (scene.name, d)
                )
        takes = _even_split(n_model, [len(v) for v in cells.values()])
        for b, take in zip(cells, takes, strict=True):
            pool = cells[b]
            pools[f"negative_model_fa_{model}_{edges[b]:g}"] = len(pool)
            for k in _pick(rng, len(pool), take):
                s, d = pool[int(k)]
                counted = _counted(models[model][s])
                box = counted[list(_BOX)].iloc[d].to_numpy(dtype=np.float64)
                if any(n_ == s and _iou(box, other) > 0.3 for n_, other in taken):
                    continue
                taken.append((s, box))
                rows.append(
                    negative(
                        loaded[s],
                        box,
                        "negative_model_fa",
                        len(pool) / max(1, take),
                        (model, d),
                        fa_model=model,
                        fa_score=float(counted["score"].iloc[d]),
                    )
                )

    # harmonic copies, split by the production model's fundamental
    harmonic: dict[str, list[tuple[str, int]]] = {
        "negative_harmonic_detected": [],
        "negative_harmonic_missed": [],
    }
    for scene in loaded.values():
        ch = scene.chirps
        det = models.get(production, {}).get(scene.name)
        hit: dict[int, float] = {}
        if det is not None and len(det):
            from chirpdetector.params.samples import match_predictions

            counted = _counted(det)
            index, _ = match_predictions(counted, ch, config.tolerance_s)
            for d, c in enumerate(index):
                if c >= 0:
                    hit[int(c)] = max(
                        hit.get(int(c), 0.0), float(counted["score"].iloc[d])
                    )
        for c in range(len(ch)):
            row = ch.iloc[c]
            hb = [
                float(row[k])
                for k in (
                    "harmonic_box_t0_s",
                    "harmonic_box_t1_s",
                    "harmonic_box_f0_hz",
                    "harmonic_box_f1_hz",
                )
            ]
            if (
                not bool(row["visible"])
                or not all(math.isfinite(v) for v in hb)
                or hb[3] > 1500.0
            ):
                continue
            if not inside(scene, hb[0], hb[1]):
                continue
            key = (
                "negative_harmonic_detected"
                if hit.get(c, 0.0) >= config.fundamental_score
                else "negative_harmonic_missed"
            )
            harmonic[key].append((scene.name, c))
    for stratum, pool in harmonic.items():
        pools[stratum] = len(pool)
        n = config.count(stratum)
        for k in _pick(rng, len(pool), n):
            s, c = pool[int(k)]
            row = loaded[s].chirps.iloc[c]
            hb = [
                float(row[x])
                for x in (
                    "harmonic_box_t0_s",
                    "harmonic_box_t1_s",
                    "harmonic_box_f0_hz",
                    "harmonic_box_f1_hz",
                )
            ]
            rows.append(
                negative(
                    loaded[s],
                    hb,
                    stratum,
                    len(pool) / max(1, n),
                    c,
                    chirp_id=int(row["chirp_id"]),
                    emitter_id=int(row["emitter_id"]),
                    baseline_hz=float(row["baseline_hz"]),
                    snr_db=float(row["snr_db"]),
                    excursion_hz=float(row["excursion_hz"]),
                )
            )

    # boxes on baselines: clicks, beats, rises, plain
    shapes = [
        (b[1] - b[0], b[2] - base, b[3] - base)
        for scene in loaded.values()
        for b, base, small, vis in zip(
            scene.chirps[["box_t0_s", "box_t1_s", "box_f0_hz", "box_f1_hz"]].to_numpy(
                dtype=np.float64
            ),
            _floats(scene.chirps, "baseline_hz"),
            _floats(scene.chirps, "excursion_hz") < config.excursion_edges_hz[0],
            scene.chirps["visible"].to_numpy(dtype=bool),
            strict=True,
        )
        if small and vis and np.isfinite(b).all()
    ]
    scenes_list = list(loaded.values())

    def baseline_box(scene: _Scene, ident: int, t: float) -> FloatArray | None:
        f = float(scene.tracks.track(ident).frequency_at(t, 10.0)[0])
        if not math.isfinite(f) or not shapes:
            return None
        duration, lo, hi = shapes[int(rng.integers(len(shapes)))]
        return np.array([t - duration / 2, t + duration / 2, f + lo, f + hi])

    def fish_at(scene: _Scene, t: float) -> dict[int, float]:
        return scene.tracks.frequencies_at(t, 10.0)

    def slope(scene: _Scene, ident: int, t: float) -> float:
        track = scene.tracks.track(ident)
        a, b = (
            track.frequency_at(t - 1.0, 10.0)[0],
            track.frequency_at(t + 1.0, 10.0)[0],
        )
        return float(abs(b - a)) if math.isfinite(a) and math.isfinite(b) else math.nan

    found: dict[str, list[dict[str, Any]]] = {
        s: []
        for s in (
            "negative_click",
            "negative_beat",
            "negative_rise",
            "negative_baseline",
        )
    }
    attempts = {s: 0 for s in found}
    click_pool = [
        (scene.name, float(c[0]), float(c[1]))
        for scene in scenes_list
        for c in scene.info["clicks"]
        if inside(scene, float(c[0]), float(c[1]))
        and clear(scene, float(c[0]), float(c[1]), config.clear_negative_s)
    ]
    pools["negative_click"] = len(click_pool)
    for k in _pick(rng, len(click_pool), config.count("negative_click")):
        s, c0, c1 = click_pool[int(k)]
        scene = loaded[s]
        live = fish_at(scene, 0.5 * (c0 + c1))
        if not live:
            continue
        ident = sorted(live)[int(rng.integers(len(live)))]
        box = baseline_box(scene, ident, 0.5 * (c0 + c1))
        if box is not None:
            found["negative_click"].append(
                negative(
                    scene,
                    box,
                    "negative_click",
                    len(click_pool) / max(1, config.count("negative_click")),
                    (c0, ident),
                    fish_id=ident,
                )
            )
    for stratum in ("negative_beat", "negative_rise", "negative_baseline"):
        target = config.count(stratum)
        while len(found[stratum]) < target and attempts[stratum] < config.max_attempts:
            attempts[stratum] += 1
            scene = scenes_list[int(rng.integers(len(scenes_list)))]
            t = float(rng.uniform(margin, float(scene.info["duration_s"]) - margin))
            live = fish_at(scene, t)
            if not live or not clear(
                scene, t - 0.15, t + 0.15, config.clear_negative_s
            ):
                continue
            if any(abs(t - 0.5 * (c[0] + c[1])) < 1.0 for c in scene.info["clicks"]):
                continue
            idents = sorted(live)
            ident = idents[int(rng.integers(len(idents)))]
            f = live[ident]
            others = np.array([v for i, v in live.items() if i != ident])
            df = float(np.min(np.abs(others - f))) if others.size else math.inf
            rise = slope(scene, ident, t)
            if stratum == "negative_beat":
                ok = df < config.beat_df_hz
            elif stratum == "negative_rise":
                ok = math.isfinite(rise) and rise >= config.drift_hz
            else:
                ok = (
                    ident < 1000
                    and df >= 30.0
                    and (not math.isfinite(rise) or rise < 1.0)
                )
            if not ok:
                continue
            box = baseline_box(scene, ident, t)
            if box is None or any(
                x["scene"] == scene.name and abs(x["t_start_s"] - box[0]) < 1.0
                for x in found[stratum]
            ):
                continue
            found[stratum].append(
                negative(
                    scene,
                    box,
                    stratum,
                    1.0,
                    (t, ident),
                    fish_id=ident,
                    df_nearest_hz=df,
                    drift_hz=rise,
                )
            )
        pools[f"{stratum}_attempts"] = attempts[stratum]
    for items in found.values():
        rows.extend(items)

    # ---- repeats of already labelled items
    if repeat_pool is not None and len(repeat_pool) and config.n_repeats:
        scored = _select(repeat_pool, repeat_pool["scored"].to_numpy(dtype=bool))
        for k in _pick(rng, len(scored), config.n_repeats):
            old = scored.iloc[int(k)].to_dict()
            item = {c: old[c] for c in old if c not in ("item_id", "candidate_id")}
            item["retest_of"] = old["item_id"]
            item["stratum"] = f"repeat:{old['stratum']}"
            item["scored"] = False  # scored with its original
            item["candidate_id"] = _opaque(config.seed, old["item_id"], "repeat")
            rows.append(item)

    truth = pd.DataFrame(rows)
    truth["item_id"] = (
        truth["scene"].astype(str) + "#" + truth["candidate_id"].astype(str)
    )
    if truth["item_id"].duplicated().any():
        truth = truth.drop_duplicates("item_id").reset_index(drop=True)
    truth = truth.iloc[rng.permutation(len(truth))].reset_index(drop=True)
    from chirpdetector.labeling.benchmark import CANDIDATE_COLUMNS

    candidates = truth[list(CANDIDATE_COLUMNS)].copy()
    first = ["item_id", "candidate_id", "scene", "stratum", "truth", "scored"]
    truth = truth[[*first, *(c for c in truth.columns if c not in first)]]
    counts = truth["stratum"].value_counts()
    summary = {
        "config": asdict(config),
        "n_scenes": len(loaded),
        "n_items": len(truth),
        "per_stratum": {str(k): int(v) for k, v in counts.items()},
        "pools": pools,
        "snr_quartile_edges_db": [float(x) for x in quartiles],
    }
    return candidates, truth, summary


def _iou(a: FloatArray, b: FloatArray) -> float:
    dt = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    df = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = dt * df
    union = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / union if union > 0 else 0.0


def _even_split(n: int, sizes: Sequence[int]) -> list[int]:
    from chirpdetector.labeling.benchmark import _even

    return _even(n, sizes)
