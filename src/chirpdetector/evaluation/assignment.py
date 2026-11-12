"""Emitter assignment and end-to-end metrics, stratified by ΔF.

Assignment is scored on true positives only: a matched (label, prediction)
pair whose label is ``chirp`` with a known ``emitter_id``.
``chirp_emitter_unclear`` labels, and ``chirp`` labels without an emitter,
count for detection but not here. A prediction matched in both a region
and a spot scope is counted once (its region match).

- *coverage*: fraction of these pairs with an assigned emitter
  (``emitter_id`` not NA);
- *accuracy*: correct / assigned; *accuracy_all*: correct / all pairs (an
  abstention counts as an error);
- *coverage-accuracy curve*: rejecting assignments with ``emitter_prob``
  below a threshold, at every distinct probability;
- *calibration*: reliability bins of ``emitter_prob`` (equal width on
  [0, 1]) and the expected calibration error
  ``ECE = sum_b n_b / N |accuracy_b - confidence_b|`` over assigned pairs
  with a probability;
- *confusion*: counts per recording, true and predicted emitter.

Everything is reported pooled (``df_bin == "all"``) and per ΔF bin. ΔF is
the frequency difference of the two closest tracks alive at the chirp
(:func:`delta_f_at`; the same definition as the labelling queue), at the
label's centre for labelled items and at the prediction's centre for
unmatched predictions. Bins are ``0-10``, ``10-25``, ``25-50``, ``50-100``,
``100+`` Hz, ``single`` (fewer than two live tracks) and ``unknown`` (no
tracks for the recording).

End to end, a chirp is found only if it is detected *and* assigned to the
right emitter: TP = pairs with the correct emitter, precision = TP /
detections, recall = TP / labels with a known emitter. Labels whose emitter
is unknown (unclear or NA), and the predictions matched to them, are left
out. Regions take part only if every ``chirp`` label in them has an
emitter (the region was emitter-labelled); spot scopes always take part.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from gridtools.io import Tracks

from chirpdetector.evaluation.detection import add_rates
from chirpdetector.evaluation.schema import (
    POSITIVE_LABELS,
    Column,
    as_series,
    floats,
    grouped,
    ints,
    pick,
    select,
    strings,
)
from chirpdetector.labeling.queue import DEFAULT_BIN_EDGES_HZ, SINGLE_BIN, bin_labels

__all__ = [
    "ALL",
    "DEFAULT_BIN_EDGES_HZ",
    "SINGLE_BIN",
    "UNKNOWN_BIN",
    "assignment_summary",
    "assignment_table",
    "bin_names",
    "confusion",
    "coverage_accuracy",
    "delta_f_at",
    "df_bins",
    "end_to_end",
    "reliability",
]

ALL = "all"
"""``df_bin`` value of pooled rows."""

UNKNOWN_BIN = "unknown"
"""ΔF bin of items whose recording has no tracks."""


def bin_names(edges_hz: Sequence[float] = DEFAULT_BIN_EDGES_HZ) -> list[str]:
    """All ΔF bin names in order, including ``single`` and ``unknown``."""
    return [*bin_labels(edges_hz), UNKNOWN_BIN]


def delta_f_at(
    tracks: Tracks, t_s: npt.ArrayLike, max_gap_s: float = 1.0
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """ΔF of the two closest live tracks, and the number of live tracks.

    Parameters
    ----------
    tracks
        The recording's tracks.
    t_s
        Times in seconds.
    max_gap_s
        A track is live at ``t`` if its detections around ``t`` are at most
        this far apart (:meth:`gridtools.io.Track.frequency_at`).

    Returns
    -------
    ``delta_f_hz`` (NaN with fewer than two live tracks) and ``n_live``.
    """
    t = np.atleast_1d(np.asarray(t_s, dtype=np.float64))
    ids = tracks.ids
    freqs = np.full((len(t), len(ids)), np.nan)
    for j, ident in enumerate(ids):
        freqs[:, j] = tracks.track(int(ident)).frequency_at(t, max_gap_s)
    n_live = np.isfinite(freqs).sum(axis=1).astype(np.int64)
    delta = np.full(len(t), np.nan)
    pair = n_live >= 2
    if pair.any():
        ordered = np.sort(freqs[pair], axis=1)  # NaN sort last
        delta[pair] = np.nanmin(np.diff(ordered, axis=1), axis=1)
    return delta, n_live


def df_bins(
    delta_f_hz: npt.ArrayLike,
    n_live: npt.ArrayLike,
    edges_hz: Sequence[float] = DEFAULT_BIN_EDGES_HZ,
) -> npt.NDArray[np.object_]:
    """ΔF bin name of every item (``single`` with fewer than two live tracks)."""
    delta = np.asarray(delta_f_hz, dtype=np.float64)
    live = np.asarray(n_live, dtype=np.int64)
    names = np.asarray(bin_labels(edges_hz), dtype=object)
    idx = np.digitize(np.nan_to_num(delta), np.asarray(edges_hz, dtype=np.float64)) - 1
    idx = np.clip(idx, 0, len(edges_hz) - 1)
    return np.where(live >= 2, names[idx], SINGLE_BIN).astype(object)


def _nullable_ints(values: Column, fill: int) -> npt.NDArray[np.int64]:
    return as_series(values).astype("Int64").fillna(fill).to_numpy(dtype=np.int64)


def assignment_table(items: pd.DataFrame) -> pd.DataFrame:
    """True positives with a known emitter, one row per prediction.

    Parameters
    ----------
    items
        Item table of an evaluation (``EvaluationResult.items``).

    Returns
    -------
    The selected item rows plus boolean ``assigned`` and ``correct``.
    """
    label = np.asarray(strings(items["label"]), dtype=object)
    known = as_series(items["true_emitter_id"]).notna().to_numpy(dtype=np.bool_)
    outcome = np.asarray(strings(items["outcome"]), dtype=object)
    sel = select(items, (outcome == "tp") & (label == "chirp") & known)
    # A prediction matched in a region and a spot scope counts once (region).
    spot = np.asarray(strings(sel["scope"]), dtype=object) != "region"
    order = np.argsort(spot, kind="stable")
    keys = list(
        zip(
            np.asarray(strings(sel["recording"]), dtype=object)[order].tolist(),
            _nullable_ints(sel["pred_row"], -1)[order].tolist(),
            strict=True,
        )
    )
    first: dict[tuple[str, int], int] = {}
    for position, key in zip(order.tolist(), keys, strict=True):
        first.setdefault(key, position)
    sel = sel.iloc[sorted(first.values())]
    pred = _nullable_ints(sel["pred_emitter_id"], -1)
    true = _nullable_ints(sel["true_emitter_id"], -2)
    assigned = as_series(sel["pred_emitter_id"]).notna().to_numpy(dtype=np.bool_)
    return sel.assign(assigned=assigned, correct=assigned & (pred == true))


@dataclass(frozen=True)
class _Pairs:
    """Columns of an assignment table as arrays."""

    df_bin: npt.NDArray[np.object_]
    assigned: npt.NDArray[np.bool_]
    correct: npt.NDArray[np.bool_]
    prob: npt.NDArray[np.float64]

    @classmethod
    def of(cls, table: pd.DataFrame) -> _Pairs:
        return cls(
            df_bin=np.asarray(strings(table["df_bin"]), dtype=object),
            assigned=as_series(table["assigned"]).to_numpy(dtype=np.bool_),
            correct=as_series(table["correct"]).to_numpy(dtype=np.bool_),
            prob=floats(table["emitter_prob"]),
        )

    def groups(self, names: Sequence[str]) -> list[tuple[str, npt.NDArray[np.bool_]]]:
        """Pooled group, then every ΔF bin with pairs, as row masks."""
        out = [(ALL, np.ones(len(self.df_bin), dtype=np.bool_))]
        out += [(b, self.df_bin == b) for b in names if np.any(self.df_bin == b)]
        return out


def _bin_index(prob: npt.NDArray[np.float64], n_bins: int) -> npt.NDArray[np.int64]:
    return np.clip(np.floor(prob * n_bins), 0, n_bins - 1).astype(np.int64)


def _ece(
    prob: npt.NDArray[np.float64], correct: npt.NDArray[np.bool_], n_bins: int
) -> float:
    if len(prob) == 0:
        return float("nan")
    b = _bin_index(prob, n_bins)
    n = np.bincount(b, minlength=n_bins)
    conf = np.bincount(b, weights=prob, minlength=n_bins)
    acc = np.bincount(b, weights=correct.astype(np.float64), minlength=n_bins)
    used = n > 0
    gap = np.abs(acc[used] / n[used] - conf[used] / n[used])
    return float(np.sum(n[used] / len(prob) * gap))


_SUMMARY_COLUMNS = (
    "df_bin",
    "n",
    "n_assigned",
    "n_correct",
    "coverage",
    "accuracy",
    "accuracy_all",
    "n_with_prob",
    "mean_confidence",
    "ece",
)


def _rate(num: int, den: int) -> float:
    return num / den if den else float("nan")


def assignment_summary(
    table: pd.DataFrame, names: Sequence[str], n_bins: int = 10
) -> pd.DataFrame:
    """Coverage, accuracy and ECE, pooled and per ΔF bin.

    Parameters
    ----------
    table
        Output of :func:`assignment_table`.
    names
        ΔF bin names in reporting order (:func:`bin_names`).
    n_bins
        Number of reliability bins for the ECE.

    Returns
    -------
    One row per group with ``df_bin, n, n_assigned, n_correct, coverage,
    accuracy, accuracy_all, n_with_prob, mean_confidence, ece``.
    """
    if len(table) == 0:
        return pd.DataFrame(columns=list(_SUMMARY_COLUMNS))
    pairs = _Pairs.of(table)
    rows = []
    for name, rows_in in pairs.groups(names):
        n = int(rows_in.sum())
        n_assigned = int(np.sum(rows_in & pairs.assigned))
        n_correct = int(np.sum(rows_in & pairs.correct))
        with_prob = rows_in & pairs.assigned & ~np.isnan(pairs.prob)
        prob = pairs.prob[with_prob]
        rows.append(
            {
                "df_bin": name,
                "n": n,
                "n_assigned": n_assigned,
                "n_correct": n_correct,
                "coverage": _rate(n_assigned, n),
                "accuracy": _rate(n_correct, n_assigned),
                "accuracy_all": _rate(n_correct, n),
                "n_with_prob": len(prob),
                "mean_confidence": float(prob.mean()) if len(prob) else float("nan"),
                "ece": _ece(prob, pairs.correct[with_prob], n_bins),
            }
        )
    return pd.DataFrame(rows, columns=list(_SUMMARY_COLUMNS))


def reliability(
    table: pd.DataFrame, names: Sequence[str], n_bins: int = 10
) -> pd.DataFrame:
    """Reliability bins of ``emitter_prob``, pooled and per ΔF bin.

    Bins are ``[k / n_bins, (k + 1) / n_bins)``, the last one closed.

    Returns
    -------
    One row per group and non-empty bin: ``df_bin, bin, prob_lo, prob_hi,
    n, mean_confidence, accuracy``.
    """
    columns = [
        "df_bin",
        "bin",
        "prob_lo",
        "prob_hi",
        "n",
        "mean_confidence",
        "accuracy",
    ]
    if len(table) == 0:
        return pd.DataFrame(columns=columns)
    pairs = _Pairs.of(table)
    rows = []
    for name, rows_in in pairs.groups(names):
        with_prob = rows_in & pairs.assigned & ~np.isnan(pairs.prob)
        prob = pairs.prob[with_prob]
        correct = pairs.correct[with_prob].astype(np.float64)
        b = _bin_index(prob, n_bins)
        for k in np.unique(b).tolist():
            in_bin = b == k
            rows.append(
                {
                    "df_bin": name,
                    "bin": k,
                    "prob_lo": k / n_bins,
                    "prob_hi": (k + 1) / n_bins,
                    "n": int(in_bin.sum()),
                    "mean_confidence": float(prob[in_bin].mean()),
                    "accuracy": float(correct[in_bin].mean()),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def coverage_accuracy(table: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    """Accuracy against coverage when rejecting by ``emitter_prob``.

    At each distinct probability ``p`` (decreasing), assignments with
    ``emitter_prob >= p`` are kept. Coverage is relative to all pairs of the
    group (abstentions included). Groups with an assigned pair that has no
    probability are skipped.

    Returns
    -------
    ``df_bin, prob_threshold, n, n_kept, n_correct, coverage, accuracy``.
    """
    columns = [
        "df_bin",
        "prob_threshold",
        "n",
        "n_kept",
        "n_correct",
        "coverage",
        "accuracy",
    ]
    if len(table) == 0:
        return pd.DataFrame(columns=columns)
    pairs = _Pairs.of(table)
    rows = []
    for name, rows_in in pairs.groups(names):
        kept = rows_in & pairs.assigned
        n = int(rows_in.sum())
        if not kept.any() or np.isnan(pairs.prob[kept]).any():
            continue
        prob = pairs.prob[kept]
        order = np.argsort(-prob, kind="stable")
        prob = prob[order]
        cum_correct = np.cumsum(pairs.correct[kept][order].astype(np.int64))
        thresholds = np.unique(prob)[::-1]
        n_kept = len(prob) - np.searchsorted(prob[::-1], thresholds, side="left")
        for p, k in zip(thresholds.tolist(), n_kept.tolist(), strict=True):
            c = int(cum_correct[k - 1])
            rows.append(
                {
                    "df_bin": name,
                    "prob_threshold": p,
                    "n": n,
                    "n_kept": k,
                    "n_correct": c,
                    "coverage": k / n,
                    "accuracy": c / k,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def confusion(table: pd.DataFrame) -> pd.DataFrame:
    """Counts per recording, ΔF bin, true and predicted emitter (NA = none)."""
    keys = ["recording", "df_bin", "true_emitter_id", "pred_emitter_id"]
    if len(table) == 0:
        return pd.DataFrame(columns=[*keys, "n"])
    counts = as_series(table.groupby(keys, dropna=False, sort=True).size())
    return counts.reset_index(name="n")


def end_to_end(items: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    """Detection with the correct emitter (see the module docstring).

    Returns
    -------
    One row per scope (``region``, ``spot``) and group: ``scope, level,
    dataset, recording, df_bin, labels, detections, tp, fn, fp, precision,
    recall, f1, n_regions``. ``level`` is ``recording``, ``dataset`` or
    ``all``; pooled rows (``level == "all"``) are also given per ΔF bin.
    """
    columns = [
        "scope",
        "level",
        "dataset",
        "recording",
        "df_bin",
        "labels",
        "detections",
        "tp",
        "fn",
        "fp",
        "precision",
        "recall",
        "f1",
        "n_regions",
    ]
    if len(items) == 0:
        return pd.DataFrame(columns=columns)
    label = np.asarray(strings(items["label"]), dtype=object)
    scope = np.asarray(strings(items["scope"]), dtype=object)
    recording = np.asarray(strings(items["recording"]), dtype=object)
    outcome = np.asarray(strings(items["outcome"]), dtype=object)
    true = _nullable_ints(items["true_emitter_id"], -2)
    pred = _nullable_ints(items["pred_emitter_id"], -1)
    has_true = as_series(items["true_emitter_id"]).notna().to_numpy(dtype=np.bool_)
    positive = np.isin(label, POSITIVE_LABELS)
    known = (label == "chirp") & has_true
    in_region = scope == "region"
    region_key = np.array(
        [f"{r}#{g}" for r, g in zip(recording, strings(items["region"]), strict=True)],
        dtype=object,
    )
    bad = set(region_key[in_region & (label == "chirp") & ~has_true].tolist())
    eligible = ~in_region | ~np.isin(region_key, list(bad))
    keep = eligible & ~(positive & ~known)
    is_tp = outcome == "tp"
    counts = pd.DataFrame(
        {
            "scope": scope[keep],
            "dataset": np.asarray(strings(items["dataset"]), dtype=object)[keep],
            "recording": recording[keep],
            "df_bin": np.asarray(strings(items["df_bin"]), dtype=object)[keep],
            "region_key": np.where(in_region, region_key, "")[keep],
            "labels": (is_tp | (outcome == "fn"))[keep].astype(np.int64),
            "detections": (is_tp | (outcome == "fp"))[keep].astype(np.int64),
            "tp": (is_tp & (pred == true))[keep].astype(np.int64),
        }
    )
    levels = {"recording": ["dataset", "recording"], "dataset": ["dataset"], "all": []}
    parts = []
    for scope_name in sorted(set(counts["scope"])):
        sc = select(counts, counts["scope"] == scope_name)
        for level, keys in levels.items():
            strata: list[tuple[str, pd.DataFrame]] = [(ALL, sc)]
            if level == "all":
                bins = np.asarray(strings(sc["df_bin"]), dtype=object)
                strata += [
                    (b, select(sc, bins == b)) for b in names if np.any(bins == b)
                ]
            for name, df in strata:
                for key, g in grouped(df, keys):
                    regions = {k for k in strings(g["region_key"]) if k}
                    parts.append(
                        {
                            "scope": scope_name,
                            "level": level,
                            "dataset": key.get("dataset", np.nan),
                            "recording": key.get("recording", np.nan),
                            "df_bin": name,
                            "labels": int(ints(g["labels"]).sum()),
                            "detections": int(ints(g["detections"]).sum()),
                            "tp": int(ints(g["tp"]).sum()),
                            "n_regions": len(regions),
                        }
                    )
    if not parts:
        return pd.DataFrame(columns=columns)
    return pick(add_rates(pd.DataFrame(parts)), columns)
