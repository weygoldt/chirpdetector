"""Match predictions to labels: what counts as found, missed or false.

Admissible pairs
----------------
A label and a prediction may be matched if

- their centre times differ by at most ``time_tolerance_s`` (default
  100 ms). A prediction's centre is its box midpoint; a label's centre is
  its ``t_center_s`` (the box midpoint for audian labels);
- their frequency ranges overlap (``require_frequency_overlap``; open
  intervals, so boxes that only touch do not overlap).

One-to-one matching
-------------------
``method="hungarian"`` (default; the rule of ``detector_benchmark``, which
calls :func:`scipy.optimize.linear_sum_assignment` on |Δt| with 1e6 for
inadmissible pairs): the matching with the most pairs and, among those, the
smallest total |Δt|. It is solved per connected component of the graph of
admissible pairs, which gives the same matching as one big assignment
problem. Ties are broken deterministically: each pair's cost is |Δt| plus
less than :data:`TIE_COST_S` (1 ns) that grows with the prediction's rank
(score, descending) and then the label's rank (centre time, ascending). So
of two predictions equally far from one label the higher-scoring one is
matched, and a prediction equally far from two labels goes to the earlier
one; totals that differ by less than about 1 ns per pair count as tied.
The few ties this leaves (e.g. two predictions and two labels all equally
far apart) follow the solver, which is deterministic for the given order.

``method="greedy"``: pairs in order of increasing |Δt| (ties: higher
score, then earlier label), each taken if both ends are still free. It can
find fewer pairs than ``"hungarian"``.

Scopes
------
Labels are scored per *scope*:

region (labels with a ``region``; exhaustive)
    Predictions take part if their centre lies in the region's span
    (inclusive) and, with ``band_filter``, their frequency range overlaps the
    region's band; this drops boxes on EOD harmonics. With
    ``edge_tolerance``, predictions centred up to the time tolerance outside
    the span take part too, but only count if they match a chirp.
spot (labels without a region; not exhaustive), per recording
    All predictions of the recording take part.

Matching runs in three passes, each on the predictions left by the one
before: positive labels (``chirp``, ``chirp_emitter_unclear``) give true
positives; then ``ignore`` labels, whose predictions are not counted; then
``not_chirp`` labels, whose predictions are false positives.

In a region, every counted prediction that is not a true positive is a
false positive (``not_chirp`` labels only annotate some of them). In a spot
scope, predictions matching no label are not scored, so detections = true
positives + predictions matching a ``not_chirp`` label; an unmatched
``not_chirp`` label is a true negative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

from chirpdetector.evaluation.schema import POSITIVE_LABELS, pick

__all__ = [
    "COUNT_COLUMNS",
    "TIE_COST_S",
    "MatchConfig",
    "MatchMethod",
    "Scope",
    "ScopeMatch",
    "admissible_pairs",
    "assign",
]

MatchMethod = Literal["hungarian", "greedy"]

TIE_COST_S = 1e-9
"""Upper bound of the tie-breaking cost added to a pair's |Δt|."""

COUNT_COLUMNS = ("labels", "detections", "tp", "ignored", "negatives", "fp_negative")
"""Counts per scope: positive labels, counted predictions, true positives,
predictions matched to ``ignore`` labels, ``not_chirp`` labels and
predictions matched to them."""

_POSITIVE, _IGNORE, _NEGATIVE = 0, 1, 2
_PASSES = (_POSITIVE, _IGNORE, _NEGATIVE)

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class MatchConfig:
    """How predictions are matched to labels (see the module docstring).

    Parameters
    ----------
    time_tolerance_s
        Maximum difference of centre times.
    require_frequency_overlap
        Require overlapping frequency ranges.
    band_filter
        Drop predictions outside a region's frequency band (harmonics).
    edge_tolerance
        Let predictions centred just outside a region match its chirps.
    method
        ``"hungarian"`` (maximum matching, then minimum total |Δt|) or
        ``"greedy"`` (closest pairs first).
    """

    time_tolerance_s: float = 0.1
    require_frequency_overlap: bool = True
    band_filter: bool = True
    edge_tolerance: bool = False
    method: MatchMethod = "hungarian"

    def __post_init__(self) -> None:
        if not (np.isfinite(self.time_tolerance_s) and self.time_tolerance_s >= 0):
            msg = (
                f"time_tolerance_s must be finite and >= 0, got {self.time_tolerance_s}"
            )
            raise ValueError(msg)
        if self.method not in ("hungarian", "greedy"):
            msg = f"method must be 'hungarian' or 'greedy', got {self.method!r}"
            raise ValueError(msg)


def admissible_pairs(
    label_t_s: FloatArray,
    label_band_hz: tuple[FloatArray, FloatArray],
    pred_t_s: FloatArray,
    pred_band_hz: tuple[FloatArray, FloatArray],
    *,
    time_tolerance_s: float,
    require_frequency_overlap: bool = True,
) -> tuple[IntArray, IntArray, FloatArray]:
    """All admissible (label, prediction) pairs.

    Parameters
    ----------
    label_t_s, pred_t_s
        Centre times.
    label_band_hz, pred_band_hz
        ``(f_low_hz, f_high_hz)`` arrays.
    time_tolerance_s
        Maximum ``|label_t_s - pred_t_s|`` (inclusive).
    require_frequency_overlap
        Also require ``min(f_high) > max(f_low)``.

    Returns
    -------
    Label indices, prediction indices and ``|Δt|`` of every admissible
    pair, sorted by label, then prediction.
    """
    label_t = np.asarray(label_t_s, dtype=np.float64)
    pred_t = np.asarray(pred_t_s, dtype=np.float64)
    order = np.argsort(pred_t, kind="stable")
    sorted_t = pred_t[order]
    # Candidates with a margin, then the exact test |a - b| <= tol.
    margin = time_tolerance_s + 1e-6
    lo = np.searchsorted(sorted_t, label_t - margin, side="left")
    hi = np.searchsorted(sorted_t, label_t + margin, side="right")
    counts = hi - lo
    li = np.repeat(np.arange(len(label_t)), counts)
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    pj = order[np.repeat(lo, counts) + offsets]
    dt = np.abs(label_t[li] - pred_t[pj])
    ok = dt <= time_tolerance_s
    if require_frequency_overlap:
        l_lo, l_hi = (np.asarray(a, dtype=np.float64) for a in label_band_hz)
        p_lo, p_hi = (np.asarray(a, dtype=np.float64) for a in pred_band_hz)
        ok &= np.minimum(l_hi[li], p_hi[pj]) > np.maximum(l_lo[li], p_lo[pj])
    li, pj, dt = li[ok], pj[ok], dt[ok]
    resort = np.lexsort((pj, li))
    return li[resort].astype(np.int64), pj[resort].astype(np.int64), dt[resort]


def _components(li: IntArray, pj: IntArray) -> IntArray:
    """Connected component of every edge of a bipartite graph."""
    lu, l_loc = np.unique(li, return_inverse=True)
    _, p_loc = np.unique(pj, return_inverse=True)
    n = len(lu) + int(p_loc.max()) + 1
    graph = coo_array(
        (np.ones(len(li)), (l_loc, len(lu) + p_loc)), shape=(n, n)
    ).tocsr()
    _, comp = connected_components(graph, directed=False)
    return comp[l_loc].astype(np.int64)


def _local_rank(index: IntArray, rank: IntArray) -> IntArray:
    """Position of each entry's node among the distinct nodes, ordered by rank."""
    nodes = np.unique(index)
    position = np.empty(len(nodes), dtype=np.int64)
    position[np.argsort(rank[nodes], kind="stable")] = np.arange(len(nodes))
    return position[np.searchsorted(nodes, index)]


def _hungarian_component(
    li: IntArray,
    pj: IntArray,
    dt: FloatArray,
    label_rank: IntArray,
    pred_rank: IntArray,
) -> BoolArray:
    r = _local_rank(li, label_rank)
    q = _local_rank(pj, pred_rank)
    nl, npred = int(r.max()) + 1, int(q.max()) + 1
    tie = TIE_COST_S * (q * nl + r) / (nl * npred)
    big = 1e6 + 2.0 * min(nl, npred) * (float(dt.max()) + 1.0)
    cost = np.full((nl, npred), big)
    cost[r, q] = dt + tie
    allowed = np.zeros((nl, npred), dtype=np.bool_)
    allowed[r, q] = True
    rows, cols = linear_sum_assignment(cost)
    good = allowed[rows, cols]
    chosen = set(zip(rows[good].tolist(), cols[good].tolist(), strict=True))
    return np.array(
        [(a, b) in chosen for a, b in zip(r.tolist(), q.tolist(), strict=True)],
        dtype=np.bool_,
    )


def assign(
    li: IntArray,
    pj: IntArray,
    dt: FloatArray,
    label_rank: IntArray,
    pred_rank: IntArray,
    method: MatchMethod = "hungarian",
) -> BoolArray:
    """Choose a one-to-one subset of admissible pairs.

    Parameters
    ----------
    li, pj, dt
        Admissible pairs (:func:`admissible_pairs`).
    label_rank, pred_rank
        Tie-breaking rank of every label and prediction (lower wins):
        labels by centre time, predictions by decreasing score.
    method
        See the module docstring.

    Returns
    -------
    Boolean mask over the pairs: True for matched pairs.
    """
    li = np.asarray(li, dtype=np.int64)
    pj = np.asarray(pj, dtype=np.int64)
    dt = np.asarray(dt, dtype=np.float64)
    selected = np.zeros(len(li), dtype=np.bool_)
    if len(li) == 0:
        return selected
    if len(np.unique(li)) == len(li) and len(np.unique(pj)) == len(pj):
        selected[:] = True  # no label or prediction in two pairs: no conflict
        return selected
    if method == "greedy":
        order = np.lexsort((label_rank[li], pred_rank[pj], dt))
        used_l: set[int] = set()
        used_p: set[int] = set()
        for e in order.tolist():
            a, b = int(li[e]), int(pj[e])
            if a not in used_l and b not in used_p:
                used_l.add(a)
                used_p.add(b)
                selected[e] = True
        return selected
    comp = _components(li, pj)
    size = np.bincount(comp)
    single = size[comp] == 1
    selected[single] = True
    multi = np.flatnonzero(~single)
    if multi.size:
        multi = multi[np.argsort(comp[multi], kind="stable")]
        bounds = np.flatnonzero(np.diff(comp[multi])) + 1
        for edges in np.split(multi, bounds):
            selected[edges] = _hungarian_component(
                li[edges], pj[edges], dt[edges], label_rank, pred_rank
            )
    return selected


@dataclass(frozen=True)
class ScopeMatch:
    """Outcome of one scope at one threshold.

    Pairs index the scope's labels and candidate predictions
    (:attr:`Scope.label_index`, :attr:`Scope.pred_index`).
    """

    tp: tuple[IntArray, IntArray]
    ignored: tuple[IntArray, IntArray]
    negative: tuple[IntArray, IntArray]
    counted: BoolArray
    """Candidate predictions counted as detections."""

    def counts(self, scope: Scope) -> dict[str, int]:
        """Counts in :data:`COUNT_COLUMNS`."""
        return {
            "labels": int(np.sum(scope.kind == _POSITIVE)),
            "detections": int(self.counted.sum()),
            "tp": len(self.tp[0]),
            "ignored": len(self.ignored[0]),
            "negatives": int(np.sum(scope.kind == _NEGATIVE)),
            "fp_negative": len(self.negative[0]),
        }


class Scope:
    """One region, or the spot labels of one recording, with its predictions.

    Parameters
    ----------
    labels
        The scope's labels (rows of
        :attr:`~chirpdetector.evaluation.schema.GroundTruth.labels`).
    predictions
        All predictions of the recording.
    config
        Matching configuration.
    region
        The region as ``(t_start_s, t_end_s, f_low_hz, f_high_hz)``, or None
        for a spot scope.
    """

    def __init__(
        self,
        labels: pd.DataFrame,
        predictions: pd.DataFrame,
        config: MatchConfig,
        region: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.config = config
        self.is_region = region is not None
        self.label_index = labels.index.to_numpy()
        kind = np.full(len(labels), _NEGATIVE, dtype=np.int64)
        kind[labels["label"].isin(POSITIVE_LABELS).to_numpy()] = _POSITIVE
        kind[(labels["label"] == "ignore").to_numpy()] = _IGNORE
        self.kind = kind
        self.label_t_s = labels["t_center_s"].to_numpy(dtype=np.float64)
        p_t = 0.5 * (
            predictions["t_start_s"].to_numpy(dtype=np.float64)
            + predictions["t_end_s"].to_numpy(dtype=np.float64)
        )
        p_lo = predictions["f_low_hz"].to_numpy(dtype=np.float64)
        p_hi = predictions["f_high_hz"].to_numpy(dtype=np.float64)
        if region is not None:
            t0, t1, f0, f1 = (float(v) for v in region)
            in_span = (p_t >= t0) & (p_t <= t1)
            take = in_span.copy()
            if config.edge_tolerance:
                tol = config.time_tolerance_s
                take |= (p_t >= t0 - tol) & (p_t <= t1 + tol)
            if config.band_filter:
                take &= (p_lo < f1) & (p_hi > f0)
        else:
            in_span = np.zeros(len(p_t), dtype=np.bool_)
            take = np.ones(len(p_t), dtype=np.bool_)
        self.pred_index = predictions.index.to_numpy()[take]
        self.pred_t_s = p_t[take]
        self.score = predictions["score"].to_numpy(dtype=np.float64)[take]
        self.in_span = in_span[take]
        self.label_rank = np.empty(len(labels), dtype=np.int64)
        self.label_rank[np.lexsort((np.arange(len(labels)), self.label_t_s))] = (
            np.arange(len(labels))
        )
        n = len(self.score)
        self.pred_rank = np.empty(n, dtype=np.int64)
        self.pred_rank[np.lexsort((np.arange(n), self.pred_t_s, -self.score))] = (
            np.arange(n)
        )
        li, pj, dt = admissible_pairs(
            self.label_t_s,
            (
                labels["f_low_hz"].to_numpy(dtype=np.float64),
                labels["f_high_hz"].to_numpy(dtype=np.float64),
            ),
            self.pred_t_s,
            (p_lo[take], p_hi[take]),
            time_tolerance_s=config.time_tolerance_s,
            require_frequency_overlap=config.require_frequency_overlap,
        )
        self.li, self.pj, self.dt = li, pj, dt

    @property
    def positive(self) -> BoolArray:
        """Mask of the scope's positive labels."""
        return self.kind == _POSITIVE

    @property
    def negative(self) -> BoolArray:
        """Mask of the scope's ``not_chirp`` labels."""
        return self.kind == _NEGATIVE

    def _match(self, active: BoolArray, edges: IntArray | None = None) -> ScopeMatch:
        if edges is None:
            edges = np.arange(len(self.li))
        available = active.copy()
        result: dict[int, tuple[IntArray, IntArray]] = {}
        for kind in _PASSES:
            e = edges[(self.kind[self.li[edges]] == kind) & available[self.pj[edges]]]
            chosen = e[
                assign(
                    self.li[e],
                    self.pj[e],
                    self.dt[e],
                    self.label_rank,
                    self.pred_rank,
                    self.config.method,
                )
            ]
            result[kind] = (self.li[chosen], self.pj[chosen])
            available[self.pj[chosen]] = False
        tp_preds = result[_POSITIVE][1]
        if self.is_region:
            counted = active & self.in_span
            counted[result[_IGNORE][1]] = False
            counted[tp_preds] = True
        else:
            counted = np.zeros(len(active), dtype=np.bool_)
            counted[tp_preds] = True
            counted[result[_NEGATIVE][1]] = True
        return ScopeMatch(
            tp=result[_POSITIVE],
            ignored=result[_IGNORE],
            negative=result[_NEGATIVE],
            counted=counted,
        )

    def match(self, threshold: float) -> ScopeMatch:
        """Match the predictions with ``score >= threshold``."""
        return self._match(self.score >= threshold)

    def count_events(self) -> pd.DataFrame:
        """Changes of the counts as the threshold is lowered.

        Returns
        -------
        One row per (component, score) with ``score`` and the change of
        ``detections``, ``tp``, ``ignored`` and ``fp_negative`` when the
        threshold drops to ``score``. Summing the rows with
        ``score >= threshold`` gives the counts of :meth:`match` at that
        threshold, for every threshold at once. Each connected component of
        admissible pairs is matched separately at each of its own scores, so
        this is exact for both matching methods.
        """
        cols = ("detections", "tp", "ignored", "fp_negative")
        scores: list[float] = []
        deltas: list[tuple[int, int, int, int]] = []
        connected = np.zeros(len(self.score), dtype=np.bool_)
        if len(self.li):
            comp = _components(self.li, self.pj)
            order = np.argsort(comp, kind="stable")
            bounds = np.flatnonzero(np.diff(comp[order])) + 1
            for edges in np.split(order, bounds):
                preds = np.unique(self.pj[edges])
                connected[preds] = True
                previous = (0, 0, 0, 0)
                for s in np.unique(self.score[preds])[::-1]:
                    active = np.zeros(len(self.score), dtype=np.bool_)
                    active[preds[self.score[preds] >= s]] = True
                    m = self._match(active, edges)
                    now = (
                        int(m.counted.sum()),
                        len(m.tp[0]),
                        len(m.ignored[0]),
                        len(m.negative[0]),
                    )
                    scores.append(float(s))
                    deltas.append(
                        (
                            now[0] - previous[0],
                            now[1] - previous[1],
                            now[2] - previous[2],
                            now[3] - previous[3],
                        )
                    )
                    previous = now
        if self.is_region:
            lonely = ~connected & self.in_span
            scores.extend(self.score[lonely].tolist())
            deltas.extend([(1, 0, 0, 0)] * int(lonely.sum()))
        events = pd.DataFrame(deltas, columns=list(cols), dtype=np.int64)
        events.insert(0, "score", np.asarray(scores, dtype=np.float64))
        return events

    def counts_at(
        self, thresholds: npt.ArrayLike, events: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Counts (:data:`COUNT_COLUMNS`) at each threshold.

        Parameters
        ----------
        thresholds
            Score thresholds.
        events
            Output of :meth:`count_events`, if already computed.
        """
        return counts_from_events(
            self.count_events() if events is None else events,
            np.asarray(thresholds, dtype=np.float64),
            labels=int(np.sum(self.kind == _POSITIVE)),
            negatives=int(np.sum(self.kind == _NEGATIVE)),
        )


def counts_from_events(
    events: pd.DataFrame, thresholds: FloatArray, *, labels: int, negatives: int
) -> pd.DataFrame:
    """Sum count events with ``score >= threshold`` for every threshold."""
    cols = ["detections", "tp", "ignored", "fp_negative"]
    order = np.argsort(events["score"].to_numpy(), kind="stable")
    s = events["score"].to_numpy()[order]
    values = events[cols].to_numpy(dtype=np.int64)[order]
    above = np.vstack(
        [np.zeros((1, len(cols)), dtype=np.int64), np.cumsum(values[::-1], axis=0)]
    )
    n_ge = len(s) - np.searchsorted(s, thresholds, side="left")
    out = pd.DataFrame(above[n_ge], columns=cols)
    out.insert(0, "threshold", thresholds)
    out.insert(1, "labels", labels)
    out["negatives"] = negatives
    return pick(out, ["threshold", *COUNT_COLUMNS])
