"""Score human and models on merged benchmark sets (``label-benchmark score``).

Inputs: one or more *sets* (a truth table of
:mod:`chirpdetector.labeling.benchmark` or
:mod:`chirpdetector.labeling.benchmark_v2` and the labeller's label rows),
the chirp features of :func:`chirpdetector.labeling.benchmark_v2.
scene_features` (attached to items that lack them) and each model's
answers (:func:`chirpdetector.labeling.benchmark_v2.model_outcomes`).

Decisions
---------
- human: ``chirp`` or ``chirp_emitter_unclear`` = chirp; the emitter is the
  clicked or picked track;
- model: a detection that the model reports (kept or re-attributed) with
  score >= threshold; the emitter is its assigned track (models without an
  emitter head have none).

Measures
--------
Recall on true chirps (and weighted to the population of visible test
chirps), false alarms per negative type, emitter accuracy (by ΔF to the
nearest fish), chirps found with the right emitter, all with stratified
bootstrap 95% CIs and exact two-sided McNemar tests of human against each
model on the paired items. Harmonic copies are reported by outcome
(rejected; called a chirp of the fish whose harmonic it is, i.e.
re-attributed; called a chirp of another or no fish), and count as false
alarms only in the last case. Invisible chirps are reported, not scored.
Repeats (``retest_of``) give the human's test-retest agreement.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from chirpdetector.labeling.benchmark import (
    _floats,
    _ints,
    _markdown,
    _mcnemar,
    _select,
    _strings,
)

__all__ = [
    "FEATURE_COLUMNS",
    "NEGATIVE_TYPES",
    "ReportV2",
    "attach_features",
    "merge_sets",
    "retest_pairs",
    "score_v2",
    "write_report_v2",
]

FEATURE_COLUMNS = (
    "near_harmonic",
    "overlap",
    "bout",
    "drift",
    "click",
    "few_electrodes",
    "rippled_plateau",
    "long",
    "short",
    "gap",
)
"""Boolean chirp features reported as groups."""

NEGATIVE_TYPES = (
    "model_fa",
    "harmonic_detected",
    "harmonic_missed",
    "click",
    "beat",
    "rise",
    "baseline",
)
_V1_NEGATIVES = {"negative_hard": "model_fa", "negative_easy": "baseline"}
DF_EDGES_HZ = (5.0, 10.0, 25.0, 50.0)
SNR_EDGES_DB = (-20.0, -10.0, 0.0, 10.0)
EXCURSION_EDGES_HZ = (60.0, 150.0)
BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]


def _bins(
    values: FloatArray, edges: Sequence[float], unit: str
) -> npt.NDArray[np.object_]:
    names = [f"<{edges[0]:g}"]
    names += [
        f"{a:g} to {b:g}" if a < 0 else f"{a:g}-{b:g}"
        for a, b in itertools.pairwise(edges)
    ]
    names.append(f">={edges[-1]:g}")
    out = np.array(["unknown"] * len(values), dtype=object)
    ok = np.isfinite(values)
    out[ok] = np.array(names, dtype=object)[
        np.searchsorted(np.asarray(edges), values[ok], side="right")
    ]
    return np.array([v if v == "unknown" else f"{v} {unit}" for v in out], dtype=object)


def attach_features(
    truth: pd.DataFrame, features: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """Fill missing chirp features from per-scene feature tables.

    Items with a ``chirp_id`` get the features of that chirp of their scene
    (by the recording's directory name); columns already present and finite
    are kept.
    """
    out = truth.copy()
    columns = ("df_nearest_hz", *FEATURE_COLUMNS, "n_electrodes")
    for column in columns:
        if column not in out.columns:
            out[column] = None
    names = [Path(str(r)).name for r in _strings(out, "recording")]
    chirp_ids = _ints(out, "chirp_id", -1)
    for i, (scene, cid) in enumerate(zip(names, chirp_ids, strict=True)):
        table = features.get(scene)
        if table is None:
            continue
        if cid >= 0:
            match = table[table["chirp_id"].to_numpy(dtype=np.int64) == cid]
            if len(match):
                row = match.iloc[0]
                for column in columns:
                    current = out.at[i, column]
                    if current is None or (
                        isinstance(current, float) and not math.isfinite(current)
                    ):
                        out.at[i, column] = row.get(column)
        if len(table) and (
            out.at[i, "n_electrodes"] is None
            or out.at[i, "n_electrodes"] != out.at[i, "n_electrodes"]
        ):
            out.at[i, "n_electrodes"] = table["n_electrodes"].iloc[0]
    return out


def merge_sets(
    sets: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]], labeler: str | None = None
) -> pd.DataFrame:
    """One table of items with the human's answer.

    Parameters
    ----------
    sets
        ``(name, truth, labels)`` per set.
    labeler
        Only this labeller's labels.

    Returns
    -------
    Truth rows of all sets with ``set``, ``negative_type``,
    ``human_label`` (empty: unlabelled), ``human_emitter_id`` (-1: none) and
    ``elapsed_s``.
    """
    from chirpdetector.labeling.benchmark import _human

    frames = []
    for name, truth, labels in sets:
        t = truth.copy()
        t["set"] = name
        human = _human(labels, labeler)
        frames.append(t.merge(human, on="item_id", how="left"))
    items = pd.concat(frames, ignore_index=True)
    stratum = _strings(items, "stratum")
    neg = []
    for s in stratum:
        if s in _V1_NEGATIVES:
            neg.append(_V1_NEGATIVES[s])
        elif s.startswith("negative_"):
            neg.append(s.removeprefix("negative_"))
        else:
            neg.append("")
    items["negative_type"] = neg
    # a repeat is scored once, with its original; its answer feeds test-retest
    if "retest_of" in items.columns:
        repeat = _strings(items, "retest_of") != ""
        items["scored"] = items["scored"].to_numpy(dtype=bool) & ~repeat
    if items["item_id"].duplicated().any():
        msg = "item ids repeat across sets"
        raise ValueError(msg)
    return items


@dataclass
class ReportV2:
    """Per-item decisions, metrics and tables."""

    items: pd.DataFrame
    models: list[str]
    metrics: dict[str, Any]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    curves: dict[str, pd.DataFrame] = field(default_factory=dict)


def _rate(values: BoolArray, weights: FloatArray | None = None) -> float:
    if not len(values):
        return math.nan
    if weights is None:
        return float(values.mean())
    total = float(weights.sum())
    return float((values * weights).sum() / total) if total > 0 else math.nan


def _boot(
    groups: npt.NDArray[np.object_],
    stats: Mapping[str, Callable[[npt.NDArray[np.int64]], float]],
    n_boot: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    index = [np.flatnonzero(groups == g) for g in sorted(set(groups.tolist()))]
    draws: dict[str, list[float]] = {k: [] for k in stats}
    for _ in range(n_boot):
        rows = (
            np.concatenate([rng.choice(r, size=r.size, replace=True) for r in index])
            if index
            else np.zeros(0, np.int64)
        )
        for key, fn in stats.items():
            draws[key].append(fn(rows))
    out = {}
    for key, values in draws.items():
        finite = np.asarray([v for v in values if math.isfinite(v)])
        out[key] = (
            (float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975)))
            if finite.size
            else (math.nan, math.nan)
        )
    return out


def score_v2(
    items: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    threshold: float = 0.5,
    n_boot: int = 2000,
    seed: int = 0,
    snr_edges_db: Sequence[float] | None = None,
) -> ReportV2:
    """Human and models on the merged items (module docstring).

    Parameters
    ----------
    items
        :func:`merge_sets` output (features attached).
    outcomes
        ``item_id, model, score, status, emitter_id`` per item and model.
    threshold
        Models' operating score.
    n_boot, seed
        Bootstrap (resampling items within ``set`` x ``stratum``).
    snr_edges_db
        SNR bin edges (default :data:`SNR_EDGES_DB`).
    """
    items = items.copy()
    models = sorted(set(_strings(outcomes, "model").tolist()))
    for model in models:
        out = _select(outcomes, _strings(outcomes, "model") == model).set_index(
            "item_id"
        )
        score = out["score"].reindex(items["item_id"]).to_numpy(dtype=np.float64)
        status = (
            out["status"]
            .reindex(items["item_id"])
            .fillna("none")
            .astype(str)
            .to_numpy(dtype=object)
        )
        emitter = out["emitter_id"].reindex(items["item_id"]).to_numpy(dtype=np.float64)
        reports = np.isin(status, ["kept", "reattributed"]) & (
            np.nan_to_num(score) >= threshold
        )
        items[f"{model}_score"] = score
        items[f"{model}_status"] = status
        items[f"{model}_chirp"] = reports
        items[f"{model}_emitter_id"] = np.where(reports, emitter, np.nan)
    label = _strings(items, "human_label")
    labelled = label != ""
    items["labelled"] = labelled
    items["human_chirp"] = np.isin(label, ["chirp", "chirp_emitter_unclear"])
    true_em = _ints(items, "emitter_id", -2).astype(np.float64)
    h_em = _ints(items, "human_emitter_id", -1).astype(np.float64)
    items["human_emitter_ok"] = items["human_chirp"].to_numpy(dtype=bool) & (
        h_em == true_em
    )
    for model in models:
        m_em = np.nan_to_num(
            items[f"{model}_emitter_id"].to_numpy(dtype=np.float64), nan=-1.0
        )
        items[f"{model}_emitter_ok"] = items[f"{model}_chirp"].to_numpy(dtype=bool) & (
            m_em == true_em
        )
    truth = _strings(items, "truth")
    is_harmonic = np.char.startswith(
        _strings(items, "negative_type").astype(str), "harmonic"
    )
    # a harmonic copy called a chirp of the fish it belongs to is re-attributed, not a false alarm
    items["human_false_alarm"] = (
        (truth == "no_chirp")
        & items["human_chirp"].to_numpy(dtype=bool)
        & ~(is_harmonic & (h_em == true_em))
    )
    for model in models:
        m_em = np.nan_to_num(
            items[f"{model}_emitter_id"].to_numpy(dtype=np.float64), nan=-1.0
        )
        items[f"{model}_false_alarm"] = (
            (truth == "no_chirp")
            & items[f"{model}_chirp"].to_numpy(dtype=bool)
            & ~(is_harmonic & (m_em == true_em))
        )
    snr = _floats(items, "snr_db")
    if snr_edges_db is None:
        snr_edges_db = SNR_EDGES_DB
    items["snr_bin"] = _bins(snr, snr_edges_db, "dB")
    items["excursion_bin"] = _bins(
        _floats(items, "excursion_hz"), EXCURSION_EDGES_HZ, "Hz"
    )
    items["df_bin"] = _bins(_floats(items, "df_nearest_hz"), DF_EDGES_HZ, "Hz")

    use = labelled & items["scored"].to_numpy(dtype=bool)
    sc = _select(items, use)
    chirp = _strings(sc, "truth") == "chirp"
    neg = _strings(sc, "truth") == "no_chirp"
    ntype = _strings(sc, "negative_type")
    weight = np.nan_to_num(_floats(sc, "weight"), nan=1.0)
    groups = np.array(
        [
            f"{a}|{b}"
            for a, b in zip(_strings(sc, "set"), _strings(sc, "stratum"), strict=True)
        ],
        dtype=object,
    )
    deciders = ["human", *models]
    # models without an emitter head assign none: their emitter measures are undefined
    has_emitter = {"human": True} | {
        m: bool(np.isfinite(_floats(items, f"{m}_emitter_id")).any()) for m in models
    }
    said = {d: sc[f"{d}_chirp"].to_numpy(dtype=bool) for d in deciders}
    fa = {d: sc[f"{d}_false_alarm"].to_numpy(dtype=bool) for d in deciders}
    em_ok = {d: sc[f"{d}_emitter_ok"].to_numpy(dtype=bool) for d in deciders}

    def on(
        values: BoolArray, mask: BoolArray, weighted: bool = False
    ) -> Callable[[npt.NDArray[np.int64]], float]:
        return lambda rows: _rate(
            values[rows][mask[rows]], weight[rows][mask[rows]] if weighted else None
        )

    stats: dict[str, Callable[[npt.NDArray[np.int64]], float]] = {}
    for d in deciders:
        stats[f"recall_{d}"] = on(said[d], chirp)
        stats[f"recall_weighted_{d}"] = on(said[d], chirp, True)
        stats[f"false_alarm_{d}"] = on(fa[d], neg)
        stats[f"emitter_recall_{d}"] = (
            on(em_ok[d], chirp) if has_emitter[d] else (lambda r: math.nan)
        )
        for t in NEGATIVE_TYPES:
            stats[f"false_alarm_{t}_{d}"] = on(fa[d], neg & (ntype == t))
    for m in models:
        stats[f"recall_diff_{m}"] = (lambda a, b: lambda r: a(r) - b(r))(
            stats[f"recall_{m}"], stats["recall_human"]
        )
        stats[f"false_alarm_diff_{m}"] = (lambda a, b: lambda r: a(r) - b(r))(
            stats[f"false_alarm_{m}"], stats["false_alarm_human"]
        )
        stats[f"emitter_recall_diff_{m}"] = (lambda a, b: lambda r: a(r) - b(r))(
            stats[f"emitter_recall_{m}"], stats["emitter_recall_human"]
        )
    rows_all = np.arange(len(sc))
    point = {k: fn(rows_all) for k, fn in stats.items()}
    ci = _boot(groups, stats, n_boot, seed) if len(sc) else {}
    mcnemar = {}
    for m in models:
        right_h = np.where(chirp, said["human"], ~fa["human"])
        right_m = np.where(chirp, said[m], ~fa[m])
        mcnemar[m] = {
            "chirps": _mcnemar(said["human"][chirp], said[m][chirp]),
            "negatives": _mcnemar(~fa["human"][neg], ~fa[m][neg]),
            "all": _mcnemar(right_h[chirp | neg], right_m[chirp | neg]),
            "emitter_on_chirps": _mcnemar(em_ok["human"][chirp], em_ok[m][chirp])
            if has_emitter[m]
            else {"n": 0, "only_human": 0, "only_model": 0, "p": math.nan},
        }
    metrics: dict[str, Any] = {
        "threshold": threshold,
        "models": models,
        "n_items": len(items),
        "n_labelled": int(labelled.sum()),
        "unlabelled": int((~labelled).sum()),
        "n_true_chirps": int(chirp.sum()),
        "n_negatives": int(neg.sum()),
        "per_set": {
            str(k): int(v) for k, v in items.loc[labelled, "set"].value_counts().items()
        },
        "snr_edges_db": list(snr_edges_db),
        "point": point,
        "ci95": {k: list(v) for k, v in ci.items()},
        "mcnemar": mcnemar,
        "models_with_emitter": [m for m in models if has_emitter[m]],
    }

    # ---- tables by group
    tables: dict[str, pd.DataFrame] = {}

    def group_table(
        name: str, keys: npt.NDArray[np.object_], mask: BoolArray, kind: str
    ) -> pd.DataFrame:
        records = []
        for value in sorted(set(keys[mask].tolist()), key=_order):
            g = mask & (keys == value)
            record: dict[str, Any] = {name: value, "n": int(g.sum())}
            for d in deciders:
                values = said[d] if kind == "recall" else fa[d]
                rows = np.flatnonzero(g)
                record[f"{kind}_{d}"] = _rate(
                    values[rows], weight[rows] if kind == "recall" else None
                )
                boots = _boot(
                    groups[rows],
                    {
                        "x": (
                            lambda v, w: (
                                lambda r: _rate(
                                    v[r], w[r] if kind == "recall" else None
                                )
                            )
                        )(values[rows], weight[rows])
                    },
                    max(200, n_boot // 5),
                    seed,
                )
                record[f"{kind}_{d}_ci"] = boots["x"]
            for m in models:
                p = _mcnemar(
                    said["human"][g] if kind == "recall" else ~fa["human"][g],
                    said[m][g] if kind == "recall" else ~fa[m][g],
                )
                record[f"p_{m}"] = p["p"]
            if kind == "recall":
                record["median_time_s"] = _median(_floats(sc, "elapsed_s")[g])
            records.append(record)
        return pd.DataFrame(records)

    for name in ("snr_bin", "excursion_bin", "df_bin", "stratum"):
        tables[name] = group_table(name, _strings(sc, name), chirp, "recall")
    flags = {
        f: np.array(
            [v is True or v == 1 or v == "True" for v in sc[f].to_numpy(dtype=object)],
            dtype=bool,
        )
        if f in sc.columns
        else np.zeros(len(sc), dtype=bool)
        for f in FEATURE_COLUMNS
    }
    keys = np.full(len(sc), "", dtype=object)
    parts = []
    for f, has in flags.items():
        if (chirp & has).any():
            keys[:] = ""
            keys[has] = f
            parts.append(group_table("feature", keys.copy(), chirp & has, "recall"))
    tables["features"] = (
        pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    )
    tables["negative_type"] = group_table("negative_type", ntype, neg, "false_alarm")

    # emitter accuracy by ΔF (when an emitter was given)
    records = []
    dfb = _strings(sc, "df_bin")
    for value in sorted(set(dfb[chirp].tolist()), key=_order):
        g = chirp & (dfb == value)
        record: dict[str, Any] = {"df_bin": value, "n_chirps": int(g.sum())}
        for d in (d for d in deciders if has_emitter[d]):
            em = (
                _floats(sc, f"{d}_emitter_id")
                if d != "human"
                else _ints(sc, "human_emitter_id", -1).astype(np.float64)
            )
            given = g & said[d] & np.isfinite(em) & (em >= 0)
            record[f"accuracy_{d}"] = _rate(em_ok[d][given])
            record[f"n_given_{d}"] = int(given.sum())
            record[f"right_emitter_recall_{d}"] = _rate(em_ok[d][g])
        records.append(record)
    tables["emitter_by_df"] = pd.DataFrame(records)

    # harmonic copies by outcome
    records = []
    for t in ("harmonic_detected", "harmonic_missed"):
        g = neg & (ntype == t)
        if not g.any():
            continue
        h_em_sc = _ints(sc, "human_emitter_id", -1)
        t_em_sc = _ints(sc, "emitter_id", -2)
        label_sc = _strings(sc, "human_label")
        record = {
            "type": t,
            "n": int(g.sum()),
            "human_rejected": int((g & ~said["human"]).sum()),
            "human_to_its_fish": int((g & said["human"] & (h_em_sc == t_em_sc)).sum()),
            "human_other_or_unclear": int(
                (g & said["human"] & (h_em_sc != t_em_sc)).sum()
            ),
            "human_unclear": int((g & (label_sc == "chirp_emitter_unclear")).sum()),
        }
        for m in models:
            status = _strings(sc, f"{m}_status")
            m_em = np.nan_to_num(_floats(sc, f"{m}_emitter_id"), nan=-1.0)
            reattr = g & said[m] & (status == "reattributed")
            record[f"{m}_rejected"] = int((g & ~said[m]).sum())
            record[f"{m}_dropped_as_harmonic"] = int((g & (status == "harmonic")).sum())
            record[f"{m}_reattributed_right"] = int((reattr & (m_em == t_em_sc)).sum())
            record[f"{m}_false_alarm"] = int((g & fa[m]).sum())
        records.append(record)
    tables["harmonic"] = pd.DataFrame(records)

    # invisible chirps
    inv = labelled & (_strings(items, "truth") == "invisible_chirp")
    tables["invisible"] = pd.DataFrame(
        [
            {
                "decider": d,
                "n": int(inv.sum()),
                "said_chirp": int(items.loc[inv, f"{d}_chirp"].sum()),
            }
            for d in deciders
        ]
    )

    # models on every item, labelled or not
    every = items["scored"].to_numpy(dtype=bool)
    kinds = np.array(
        [
            t
            if t
            else (
                "chirp:core"
                if s.startswith("chirp_core")
                else "chirp:" + s.removeprefix("chirp_")
            )
            for t, s in zip(
                _strings(items, "negative_type"),
                _strings(items, "stratum"),
                strict=True,
            )
        ],
        dtype=object,
    )
    records = []
    truth_all = _strings(items, "truth")
    for kind in sorted(set(kinds[every].tolist())):
        g = every & (kinds == kind)
        record = {"group": kind, "n": int(g.sum())}
        for m in models:
            values = (
                items[f"{m}_chirp"].to_numpy(dtype=bool)
                if kind.startswith("chirp")
                else items[f"{m}_false_alarm"].to_numpy(dtype=bool)
            )
            record[m] = _rate(
                values[
                    g
                    & (
                        (truth_all == "chirp")
                        if kind.startswith("chirp")
                        else (truth_all == "no_chirp")
                    )
                ]
            )
        records.append(record)
    tables["models_all_items"] = pd.DataFrame(records)

    # human time
    times = _floats(items, "elapsed_s")
    tables["time"] = pd.DataFrame(
        [
            {
                "stratum": s,
                "n": int(g.sum()),
                "median_s": _median(times[g]),
                "total_min": float(np.nansum(times[g]) / 60),
            }
            for s in sorted(set(_strings(items, "stratum")[labelled].tolist()))
            for g in [labelled & (_strings(items, "stratum") == s)]
        ]
    )
    metrics["time_s"] = {
        "median": _median(times[labelled]),
        "total_min": float(np.nansum(times[labelled]) / 60),
    }

    # test-retest
    metrics["retest"] = _retest(items)

    # curves (model score threshold), pooled negatives
    curves = {}
    for m in models:
        s = np.nan_to_num(_floats(sc, f"{m}_score"))
        reports = np.isin(_strings(sc, f"{m}_status"), ["kept", "reattributed"])
        thr = np.unique(np.concatenate([[0.0], s[reports], [np.inf]]))
        curves[m] = pd.DataFrame(
            {
                "threshold": thr,
                "recall": [_rate((reports & (s >= x))[chirp]) for x in thr],
                "false_alarm": [_rate((reports & (s >= x))[neg]) for x in thr],
            }
        )
    return ReportV2(
        items=items, models=models, metrics=metrics, tables=tables, curves=curves
    )


def _median(values: FloatArray) -> float:
    ok = np.isfinite(values)
    return float(np.median(values[ok])) if ok.any() else math.nan


def _order(value: str) -> tuple[float, str]:
    text = value.split(" ")[0].lstrip("Q")
    if text.startswith("<"):
        return (-math.inf, value)
    if text.startswith(">="):
        try:
            return (float(text[2:]), value)
        except ValueError:
            return (math.inf, value)
    try:
        return (
            float(text.split("-")[0])
            if not text.startswith("-")
            else -float(text.split("-")[1]),
            value,
        )
    except (ValueError, IndexError):
        return (math.inf, value)


def _retest(items: pd.DataFrame) -> dict[str, Any]:
    """Agreement of the human's answers on repeated items."""
    repeat = (
        _strings(items, "retest_of")
        if "retest_of" in items.columns
        else np.full(len(items), "", dtype=object)
    )
    by_id = {iid: i for i, iid in enumerate(_strings(items, "item_id"))}
    # ``retest_of`` names the original by item id or by candidate id
    for i, cid in enumerate(_strings(items, "candidate_id")):
        by_id.setdefault(cid, i)
    label = _strings(items, "human_label")
    h_em = _ints(items, "human_emitter_id", -1)
    chirp = items["human_chirp"].to_numpy(dtype=bool)
    truth = _strings(items, "truth")
    pairs = [
        (i, by_id[r])
        for i, r in enumerate(repeat)
        if r and r in by_id and label[i] and label[by_id[r]]
    ]
    if not pairs:
        return {"n_pairs": 0}
    a = np.array([chirp[j] for _, j in pairs])
    b = np.array([chirp[i] for i, _ in pairs])
    agree = float((a == b).mean())
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    kappa = float((agree - pe) / (1 - pe)) if pe < 1 else math.nan
    both = [(i, j) for i, j in pairs if chirp[i] and chirp[j]]
    correct_first = float(
        np.mean([chirp[j] == (truth[j] == "chirp") for _, j in pairs])
    )
    correct_second = float(
        np.mean([chirp[i] == (truth[j] == "chirp") for i, j in pairs])
    )
    return {
        "n_pairs": len(pairs),
        "decision_agreement": agree,
        "cohen_kappa": kappa,
        "emitter_agreement": float(np.mean([h_em[i] == h_em[j] for i, j in both]))
        if both
        else math.nan,
        "n_both_chirp": len(both),
        "correct_first": correct_first,
        "correct_repeat": correct_second,
    }


def retest_pairs(
    repeat_truth: pd.DataFrame,
    repeat_labels: pd.DataFrame,
    original_truth: pd.DataFrame,
    original_labels: pd.DataFrame,
    labeler: str | None = None,
) -> dict[str, Any]:
    """Test-retest agreement of a repeat session against an earlier one.

    ``repeat_truth`` rows name their original in ``retest_of`` (item or
    candidate id of ``original_truth``); each side's labels are its own
    session's (not a merged final view, which may already hold the repeat).
    """
    from chirpdetector.labeling.benchmark import _human

    original = original_truth.merge(
        _human(original_labels, labeler), on="item_id", how="left"
    )
    repeat = repeat_truth.merge(
        _human(repeat_labels, labeler), on="item_id", how="left"
    )
    items = pd.concat([original.assign(retest_of=""), repeat], ignore_index=True)
    items["human_chirp"] = np.isin(
        _strings(items, "human_label"), ["chirp", "chirp_emitter_unclear"]
    )
    return _retest(items)


# ------------------------------------------------------------------ report

_COLOURS = {
    "human": "#1f77b4",
    "dualhead": "#d62728",
    "yolo26s": "#ff7f0e",
    "v1": "#7f7f7f",
}
_NAMES = {
    "human": "human (blind)",
    "dualhead": "v2 dual-head",
    "yolo26s": "v2 YOLO26s",
    "v1": "v1 (paper)",
}


def _fmt(value: Any, ci: Sequence[float] | None = None) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    text = f"{number:.3f}"
    if ci is not None and len(ci) == 2 and all(math.isfinite(float(c)) for c in ci):
        text += f" [{float(ci[0]):.3f}, {float(ci[1]):.3f}]"
    return text


def _plot_groups(
    ax: Any,
    table: pd.DataFrame,
    key: str,
    deciders: Sequence[str],
    kind: str,
    title: str,
) -> None:
    if not len(table):
        ax.set_visible(False)
        return
    x = np.arange(len(table))
    width = 0.8 / len(deciders)
    for k, d in enumerate(deciders):
        values = table[f"{kind}_{d}"].to_numpy(dtype=np.float64)
        cis = table[f"{kind}_{d}_ci"].tolist()
        low = [
            v - c[0] if math.isfinite(c[0]) else 0
            for v, c in zip(values, cis, strict=True)
        ]
        high = [
            c[1] - v if math.isfinite(c[1]) else 0
            for v, c in zip(values, cis, strict=True)
        ]
        ax.errorbar(
            x + (k - (len(deciders) - 1) / 2) * width,
            values,
            yerr=[low, high],
            fmt="o",
            color=_COLOURS.get(d, "black"),
            label=_NAMES.get(d, d),
            capsize=2,
            ms=5,
        )
    ax.set_xticks(
        x,
        [f"{v}\n(n={n})" for v, n in zip(table[key], table["n"], strict=True)],
        fontsize=8,
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("recall" if kind == "recall" else "false-alarm rate")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def _figures(report: ReportV2, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    deciders = ["human", *report.models]
    t = report.tables
    files = []
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    _plot_groups(
        axes[0], t["snr_bin"], "snr_bin", deciders, "recall", "recall by chirp SNR"
    )
    _plot_groups(
        axes[1],
        t["excursion_bin"],
        "excursion_bin",
        deciders,
        "recall",
        "recall by excursion",
    )
    _plot_groups(
        axes[2],
        t["df_bin"],
        "df_bin",
        deciders,
        "recall",
        "recall by ΔF to the nearest fish",
    )
    axes[0].legend(fontsize=8, loc="lower right")
    fig.savefig(out / "fig_recall.png", dpi=150)
    fig.savefig(out / "fig_recall.pdf")
    plt.close(fig)
    files.append("fig_recall.png")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(17, 4.6),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.6, 1]},
    )
    _plot_groups(
        axes[0], t["features"], "feature", deciders, "recall", "recall on edge cases"
    )
    _plot_groups(
        axes[1],
        t["negative_type"],
        "negative_type",
        deciders,
        "false_alarm",
        "false alarms by negative type",
    )
    axes[1].set_ylim(-0.03, 1.03)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.savefig(out / "fig_edge_cases.png", dpi=150)
    fig.savefig(out / "fig_edge_cases.pdf")
    plt.close(fig)
    files.append("fig_edge_cases.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    p = report.metrics["point"]
    for m in report.models:
        c = report.curves[m]
        axes[0].plot(
            c["false_alarm"],
            c["recall"],
            color=_COLOURS.get(m),
            label=f"{_NAMES.get(m, m)}, all thresholds",
        )
        axes[0].plot(
            p[f"false_alarm_{m}"], p[f"recall_{m}"], "s", color=_COLOURS.get(m), ms=8
        )
    axes[0].plot(
        p["false_alarm_human"],
        p["recall_human"],
        "o",
        color=_COLOURS["human"],
        ms=10,
        label=_NAMES["human"],
    )
    axes[0].set_xlabel("false-alarm rate (all negatives)")
    axes[0].set_ylabel("recall (true chirps)")
    axes[0].set_title(
        f"operating points (squares: score >= {report.metrics['threshold']:g})",
        fontsize=10,
    )
    axes[0].legend(fontsize=8, loc="lower right")
    e = t["emitter_by_df"]
    if len(e):
        x = np.arange(len(e))
        for k, d in enumerate(
            d
            for d in deciders
            if f"accuracy_{d}" in e.columns and _floats(e, f"n_given_{d}").sum() > 0
        ):
            axes[1].plot(
                x + 0.05 * k,
                e[f"accuracy_{d}"],
                "o-",
                color=_COLOURS.get(d),
                label=_NAMES.get(d, d),
            )
        axes[1].set_xticks(
            x,
            [f"{v}\n(n={n})" for v, n in zip(e["df_bin"], e["n_chirps"], strict=True)],
            fontsize=8,
        )
        axes[1].set_ylim(-0.03, 1.03)
        axes[1].set_ylabel("emitter accuracy (when given)")
        axes[1].set_title("emitter by ΔF to the nearest fish", fontsize=10)
        axes[1].legend(fontsize=8)
    fig.savefig(out / "fig_operating_emitter.png", dpi=150)
    fig.savefig(out / "fig_operating_emitter.pdf")
    plt.close(fig)
    files.append("fig_operating_emitter.png")
    return files


def _verdicts(report: ReportV2, model: str) -> list[str]:
    """Where a model is better, equal or worse than the human (95% CIs)."""
    lines = []
    for name, table, kind in (
        ("SNR", report.tables["snr_bin"], "recall"),
        ("excursion", report.tables["excursion_bin"], "recall"),
        ("ΔF", report.tables["df_bin"], "recall"),
        ("edge case", report.tables["features"], "recall"),
        ("negative type", report.tables["negative_type"], "false_alarm"),
    ):
        if not len(table):
            continue
        key = table.columns[0]
        for record in table.to_dict("records"):
            h, m = record.get(f"{kind}_human"), record.get(f"{kind}_{model}")
            p = record.get(f"p_{model}")
            if h is None or m is None or not (math.isfinite(h) and math.isfinite(m)):
                continue
            better = (m > h) if kind == "recall" else (m < h)
            if p is not None and math.isfinite(p) and p < 0.05:
                verdict = "better" if better else "worse"
            else:
                verdict = "no detectable difference"
            lines.append(
                f"| {name}: {record[key]} | {record['n']} | {_fmt(h)} | {_fmt(m)} | {verdict} | {_fmt(p)} |"
            )
    return lines


def write_report_v2(
    report: ReportV2,
    out_dir: str | Path,
    *,
    notes: Sequence[str] = (),
    retests: Mapping[str, dict[str, Any]] | None = None,
) -> Path:
    """``report.md``, ``metrics.json``, ``items.csv``, tables and figures."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report.items.to_csv(out / "items.csv", index=False)
    for name, table in report.tables.items():
        table.to_csv(out / f"table_{name}.csv", index=False)
    for name, curve in report.curves.items():
        curve.to_csv(out / f"curve_{name}.csv", index=False)

    def clean(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    figures = _figures(report, out)
    m, ci, mc = (
        report.metrics["point"],
        report.metrics["ci95"],
        report.metrics["mcnemar"],
    )
    deciders = ["human", *report.models]
    lines = [
        "# Human against models on simulated chirps",
        "",
        f"{report.metrics['n_labelled']} of {report.metrics['n_items']} items labelled "
        f"({', '.join(f'{k}: {v}' for k, v in report.metrics['per_set'].items())}); "
        f"{report.metrics['n_true_chirps']} true chirps and {report.metrics['n_negatives']} "
        f"negatives scored. Models at score >= {report.metrics['threshold']:g}; 95% CIs from a "
        "bootstrap over items within set x stratum; p: exact two-sided McNemar test of the "
        "paired decisions (human vs model). Human 'chirp' includes 'chirp, emitter unclear'.",
        "",
        "## Overall",
        "",
        "| measure | " + " | ".join(_NAMES.get(d, d) for d in deciders) + " |",
        "|---|" + "---|" * len(deciders),
    ]
    for key, label in (
        ("recall", "recall on true chirps"),
        ("recall_weighted", "recall, reweighted to all visible test chirps"),
        ("false_alarm", "false-alarm rate, all negatives"),
        ("emitter_recall", "chirp found with the right emitter"),
    ):
        lines.append(
            f"| {label} | "
            + " | ".join(
                _fmt(m.get(f"{key}_{d}"), ci.get(f"{key}_{d}")) for d in deciders
            )
            + " |"
        )
    for t in NEGATIVE_TYPES:
        if any(
            math.isfinite(m.get(f"false_alarm_{t}_{d}", math.nan)) for d in deciders
        ):
            lines.append(
                f"| ... false alarms on {t.replace('_', ' ')} | "
                + " | ".join(
                    _fmt(m.get(f"false_alarm_{t}_{d}"), ci.get(f"false_alarm_{t}_{d}"))
                    for d in deciders
                )
                + " |"
            )
    lines += [
        "",
        "| model - human | recall | false alarms | right emitter | McNemar p: chirps / negatives / all / emitter |",
        "|---|---|---|---|---|",
    ]
    for model in report.models:
        lines.append(
            f"| {_NAMES.get(model, model)} | {_fmt(m[f'recall_diff_{model}'], ci.get(f'recall_diff_{model}'))} | "
            f"{_fmt(m[f'false_alarm_diff_{model}'], ci.get(f'false_alarm_diff_{model}'))} | "
            f"{_fmt(m[f'emitter_recall_diff_{model}'], ci.get(f'emitter_recall_diff_{model}'))} | "
            + " / ".join(
                f"{mc[model][k]['p']:.3g} ({mc[model][k]['only_human']}:{mc[model][k]['only_model']})"
                if math.isfinite(mc[model][k]["p"])
                else "-"
                for k in ("chirps", "negatives", "all", "emitter_on_chirps")
            )
            + " |"
        )
    lines += ["", "(x:y = items only the human / only the model got right.)", ""]
    lines += [
        f"![recall]({figures[0]})",
        "",
        f"![edge cases]({figures[1]})",
        "",
        f"![operating points and emitter]({figures[2]})",
        "",
    ]
    if "dualhead" in report.models:
        lines += [
            "## Where the production model (v2 dual-head) is better, equal or worse than the human",
            "",
        ]
        lines += [
            "| group | n | human | dual-head | verdict | McNemar p |",
            "|---|---|---|---|---|---|",
            *_verdicts(report, "dualhead"),
            "",
        ]
        lines += [
            "'No detectable difference' means p >= 0.05 on these items, not equality; small groups "
            "cannot show moderate differences. Recall groups are weighted to the population of "
            "visible test chirps; false alarms are per negative type (their mix is a design choice).",
            "",
        ]
    every = {"repeats within the sets": report.metrics["retest"], **(retests or {})}
    report.metrics["retests"] = every
    shown = [(k, v) for k, v in every.items() if v.get("n_pairs")]
    if shown:
        lines += ["## Test-retest (repeated items)", ""]
        for name, rt in shown:
            lines += [
                f"- {name}: {rt['n_pairs']} pairs, decision agreement "
                f"{_fmt(rt['decision_agreement'])}, Cohen's kappa {_fmt(rt['cohen_kappa'])}; "
                f"emitter agreement {_fmt(rt['emitter_agreement'])} (n = {rt['n_both_chirp']} "
                f"pairs called chirp twice); correct the first time {_fmt(rt['correct_first'])}, "
                f"on repeat {_fmt(rt['correct_repeat'])}."
            ]
        lines.append("")
    lines += [
        f"Human time: median {_fmt(report.metrics['time_s']['median'])} s per item, "
        f"{report.metrics['time_s']['total_min']:.1f} min in total.",
        "",
    ]
    (out / "metrics.json").write_text(json.dumps(clean(report.metrics), indent=1))
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    for name, table in report.tables.items():
        if len(table):
            shown = table.drop(
                columns=[c for c in table.columns if c.endswith("_ci")], errors="ignore"
            )
            lines += [f"## Table: {name.replace('_', ' ')}", "", _markdown(shown), ""]
    path = out / "report.md"
    path.write_text("\n".join(lines))
    return path
