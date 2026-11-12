"""Detection metrics: precision, recall, F1, confidence intervals, PR curves.

All metrics are computed from pooled counts, never by averaging ratios:
precision = TP / detections, recall = TP / labels and
F1 = 2 TP / (labels + detections), each NaN when its denominator is 0
(as in ``detector_benchmark``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import beta

from chirpdetector.evaluation.matching import COUNT_COLUMNS, counts_from_events
from chirpdetector.evaluation.schema import (
    floats,
    grouped,
    ints,
    pick,
    select,
    strings,
)

__all__ = [
    "LEVELS",
    "add_rates",
    "average_precision",
    "bootstrap_regions",
    "clopper_pearson",
    "confidence_intervals",
    "pr_curve",
    "summarize",
]

LEVELS = ("recording", "dataset", "all")
"""Pooling levels of the summaries."""

_LEVEL_KEYS = {
    "recording": ["dataset", "recording"],
    "dataset": ["dataset"],
    "all": [],
}


def _ratio(num: npt.ArrayLike, den: npt.ArrayLike) -> npt.NDArray[np.float64]:
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def add_rates(counts: pd.DataFrame) -> pd.DataFrame:
    """Add ``fn``, ``fp``, ``precision``, ``recall`` and ``f1`` to counts.

    Parameters
    ----------
    counts
        Table with integer columns ``labels``, ``detections`` and ``tp``.

    Returns
    -------
    A copy with the new columns.
    """
    out = counts.copy()
    tp = out["tp"].to_numpy(dtype=np.int64)
    labels = out["labels"].to_numpy(dtype=np.int64)
    detections = out["detections"].to_numpy(dtype=np.int64)
    out["fn"] = labels - tp
    out["fp"] = detections - tp
    out["precision"] = _ratio(tp, detections)
    out["recall"] = _ratio(tp, labels)
    out["f1"] = _ratio(2 * tp, labels + detections)
    return out


def summarize(
    per_scope: pd.DataFrame, *, by: Sequence[str] = ("threshold",)
) -> pd.DataFrame:
    """Pool scope counts per recording, per dataset and over everything.

    Parameters
    ----------
    per_scope
        One row per scope (and threshold) with ``dataset``, ``recording``,
        the grouping columns ``by`` and the counts in
        :data:`~chirpdetector.evaluation.matching.COUNT_COLUMNS`.
    by
        Extra grouping columns kept at every level.

    Returns
    -------
    Columns ``dataset, recording, *by``, the counts, ``fn, fp, precision,
    recall, f1`` and ``level`` (:data:`LEVELS`); ``dataset`` and
    ``recording`` are NaN where pooled over. Rows are ordered by level, then
    by the grouping keys (the layout of ``detector_benchmark``'s
    ``summary.csv``).
    """
    counts = [c for c in COUNT_COLUMNS if c in per_scope.columns]
    rates = ["fn", "fp", "precision", "recall", "f1"]
    columns = ["dataset", "recording", *by, *counts, *rates, "level"]
    if len(per_scope) == 0:
        return pd.DataFrame(columns=columns)
    parts = []
    for level in LEVELS:
        keys = [*_LEVEL_KEYS[level], *by]
        if keys:
            sums = cast(
                "pd.DataFrame", per_scope.groupby(keys, sort=True)[counts].sum()
            )
            pooled = sums.reset_index()
        else:
            pooled = pd.DataFrame([{c: int(ints(per_scope[c]).sum()) for c in counts}])
        parts.append(add_rates(pooled).assign(level=level))
    out = pd.concat(parts, ignore_index=True)
    for column in ("dataset", "recording"):
        if column not in out.columns:
            out[column] = np.nan
    return pick(out, columns)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial confidence interval of ``k / n``.

    Returns
    -------
    ``(lower, upper)``; ``(nan, nan)`` if ``n == 0``.
    """
    if n == 0:
        return float("nan"), float("nan")
    if not 0 <= k <= n:
        msg = f"need 0 <= k <= n, got k={k}, n={n}"
        raise ValueError(msg)
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def bootstrap_regions(
    per_region: pd.DataFrame,
    *,
    rng: np.random.Generator,
    n_boot: int = 5000,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile bootstrap CIs of precision, recall and F1 over regions.

    Regions are resampled with replacement and their counts pooled. The
    draws are made one resample at a time (``rng.integers(0, n, n)``), as in
    ``detector_benchmark``, so the same seed gives the same intervals.

    Parameters
    ----------
    per_region
        One row per region with ``labels``, ``detections`` and ``tp``.
    rng
        Random generator (consumed).
    n_boot
        Number of resamples.
    alpha
        1 - confidence level.

    Returns
    -------
    ``precision_lo``, ``precision_hi``, ``recall_lo``, ``recall_hi``,
    ``f1_lo``, ``f1_hi``; resamples where a ratio is undefined are skipped.
    NaN if there are no regions.
    """
    keys = ("precision", "recall", "f1")
    n = len(per_region)
    if n == 0 or n_boot < 1:
        return {f"{k}_{b}": float("nan") for k in keys for b in ("lo", "hi")}
    idx = np.stack([rng.integers(0, n, n) for _ in range(n_boot)])
    tp = per_region["tp"].to_numpy(dtype=np.int64)[idx].sum(axis=1)
    lab = per_region["labels"].to_numpy(dtype=np.int64)[idx].sum(axis=1)
    det = per_region["detections"].to_numpy(dtype=np.int64)[idx].sum(axis=1)
    stats = np.column_stack(
        [_ratio(tp, det), _ratio(tp, lab), _ratio(2 * tp, lab + det)]
    )
    q = [50.0 * alpha, 100.0 - 50.0 * alpha]
    out: dict[str, float] = {}
    for j, key in enumerate(keys):
        column = stats[:, j]
        if np.all(np.isnan(column)):
            lo = hi = float("nan")
        else:
            lo, hi = np.nanpercentile(column, q)
        out[f"{key}_lo"], out[f"{key}_hi"] = float(lo), float(hi)
    return out


_CI_COLUMNS = (
    "precision",
    "precision_lo",
    "precision_hi",
    "recall",
    "recall_lo",
    "recall_hi",
    "boot_precision_lo",
    "boot_precision_hi",
    "boot_recall_lo",
    "boot_recall_hi",
    "f1",
    "boot_f1_lo",
    "boot_f1_hi",
    "labels",
    "detections",
    "tp",
    "n_regions",
)


def confidence_intervals(
    per_region: pd.DataFrame,
    *,
    alpha: float = 0.05,
    n_boot: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    """CIs of pooled precision and recall: exact binomial and bootstrap.

    Parameters
    ----------
    per_region
        One row per region at a single threshold, with ``dataset``,
        ``labels``, ``detections`` and ``tp``.
    alpha
        1 - confidence level.
    n_boot
        Bootstrap resamples.
    seed
        Every group is resampled with a fresh generator seeded with
        ``seed`` (as in ``detector_benchmark``), so a group's interval does
        not depend on which other groups are evaluated.

    Returns
    -------
    One row per group (``all``, then each dataset), indexed by ``group``,
    with ``precision, precision_lo, precision_hi, recall, recall_lo,
    recall_hi`` (Clopper-Pearson on the pooled counts), ``boot_precision_lo,
    boot_precision_hi, boot_recall_lo, boot_recall_hi`` (bootstrap over
    regions; these ten columns match the benchmark's table), then ``f1``,
    ``boot_f1_lo``, ``boot_f1_hi``, the counts and ``n_regions``.
    """
    if len(per_region) == 0:
        return pd.DataFrame(columns=_CI_COLUMNS).rename_axis("group")
    groups: list[tuple[str, pd.DataFrame]] = [("all", per_region)]
    groups += [(str(k), g) for k, g in per_region.groupby("dataset", sort=True)]
    rows = {}
    for name, df in groups:
        tp, lab, det = (int(ints(df[c]).sum()) for c in ("tp", "labels", "detections"))
        p_lo, p_hi = clopper_pearson(tp, det, alpha)
        r_lo, r_hi = clopper_pearson(tp, lab, alpha)
        boot = bootstrap_regions(
            df, rng=np.random.default_rng(seed), n_boot=n_boot, alpha=alpha
        )
        rows[name] = {
            "precision": _ratio(tp, det).item(),
            "precision_lo": p_lo,
            "precision_hi": p_hi,
            "recall": _ratio(tp, lab).item(),
            "recall_lo": r_lo,
            "recall_hi": r_hi,
            "boot_precision_lo": boot["precision_lo"],
            "boot_precision_hi": boot["precision_hi"],
            "boot_recall_lo": boot["recall_lo"],
            "boot_recall_hi": boot["recall_hi"],
            "f1": _ratio(2 * tp, lab + det).item(),
            "boot_f1_lo": boot["f1_lo"],
            "boot_f1_hi": boot["f1_hi"],
            "labels": lab,
            "detections": det,
            "tp": tp,
            "n_regions": len(df),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "group"
    return out


def pr_curve(events: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Precision and recall at every distinct score, per pooling level.

    Parameters
    ----------
    events
        Count events of region scopes
        (:meth:`~chirpdetector.evaluation.matching.Scope.count_events`) with
        ``dataset`` and ``recording`` columns.
    labels
        One row per region scope with ``dataset``, ``recording``,
        ``labels`` and ``negatives``.

    Returns
    -------
    Columns ``level, dataset, recording, threshold``, the counts and
    ``precision, recall, f1``; thresholds are the distinct scores, in
    decreasing order, so the curve runs from high precision to high recall.
    """
    parts = []
    for level in LEVELS:
        keys = _LEVEL_KEYS[level]
        for named, lab in grouped(labels, keys):
            mask = np.ones(len(events), dtype=np.bool_)
            for column, value in named.items():
                mask &= np.asarray(strings(events[column]), dtype=object) == value
            ev = select(events, mask)
            thresholds = np.unique(floats(ev["score"]))[::-1]
            counts = counts_from_events(
                ev,
                thresholds,
                labels=int(ints(lab["labels"]).sum()),
                negatives=int(ints(lab["negatives"]).sum()),
            )
            counts = add_rates(counts)
            counts.insert(0, "level", level)
            counts.insert(1, "dataset", named.get("dataset", np.nan))
            counts.insert(2, "recording", named.get("recording", np.nan))
            parts.append(counts)
    columns = ["level", "dataset", "recording", "threshold", *COUNT_COLUMNS]
    columns += ["fn", "fp", "precision", "recall", "f1"]
    if not parts:
        return pd.DataFrame(columns=columns)
    return pick(pd.concat(parts, ignore_index=True), columns)


def average_precision(curve: pd.DataFrame) -> pd.DataFrame:
    """Average precision of each curve in :func:`pr_curve`'s output.

    AP is the step-wise area under the precision-recall curve without
    interpolation, ``sum_k (R_k - R_{k-1}) P_k`` over thresholds in
    decreasing order with ``R_0 = 0`` (the definition of scikit-learn's
    ``average_precision_score``). Thresholds without detections have no
    precision and do not contribute.

    Returns
    -------
    Columns ``level, dataset, recording, average_precision, labels``.
    """
    rows = []
    keys = ["level", "dataset", "recording"]
    for key, g in curve.groupby(keys, sort=False, dropna=False):
        recall = floats(g["recall"])
        precision = floats(g["precision"])
        labels = int(ints(g["labels"])[0]) if len(g) else 0
        if labels == 0:
            ap = float("nan")
        else:
            d_recall = np.diff(np.concatenate([[0.0], recall]))
            ap = float(np.nansum(d_recall * precision))
        rows.append(
            {
                **dict(zip(keys, cast("tuple[object, ...]", key), strict=True)),
                "average_precision": ap,
                "labels": labels,
            }
        )
    return pd.DataFrame(rows, columns=[*keys, "average_precision", "labels"])
