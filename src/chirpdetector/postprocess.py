"""Post-processing of detections: harmonic copies and duplicate boxes.

Two kinds of false positives survive the detector and the tiling
(`chirpdetector.inference.merge_tiles`). This module flags them, with a
reason, and leaves the decision to drop them to the caller
(`drop_flagged`); `chirpdetector.inference.predict_span` drops them by
default. All functions are pure: they take prediction tables (as
`chirpdetector.prediction.predict_images` and `predict_span` make them)
and, optionally, the recordings' tracks, and return new tables.

Emitter frequency
-----------------
Both rules need the frequency of a detection's emitter. A model with an
emitter head (the dual-head RT-DETR) predicts it: a finite ``emitter_hz``
is used as is (*head* rows). A box-only model (YOLO26) does not; its
emitter's baseline lies a little above the box's lower edge, because the
box includes the power drop at the baseline and the analysis window's
smear (*edge* rows). On the sim_v4 test tiles (YOLO26s, true positives,
n = 9,630) the true baseline was 41 Hz above ``f_low_hz`` (median; 0.5 %
and 99.5 % quantiles 18 and 142 Hz). Edge rows therefore use windows
relative to the lower edge instead of a single frequency.

Harmonic copies
---------------
The EOD of a wave-type fish has harmonics. The second harmonic of a fish
below 750 Hz lies inside the default band (500-1500 Hz), and every chirp
of that fish appears there as well: at the same time, at twice the
baseline and with twice the excursion. Training treated these copies as
ignore regions, so the detector is free to box them (the dual-head model
does, often with high scores). A detection is a *harmonic copy* if

1. twice the frequency of a track alive at its centre time lies in its
   *harmonic interval*, and
2. no track alive at that time lies in its *fundamental interval*, and it
   has no assigned emitter (``emitter_id``).

Rule 2 makes sure that a detection whose emitter matches a live
fundamental is never removed, including the chirp of a fish that happens
to sit at twice another fish's frequency.

=========== ================================== ==================================
interval    head rows (``emitter_hz`` = e)      edge rows (``f_low_hz`` = l)
=========== ================================== ==================================
harmonic    e ± ``harmonic_tolerance_hz``       l + ``edge_harmonic_window_hz``
fundamental e ± ``fundamental_tolerance_hz``    l + ``edge_fundamental_window_hz``
=========== ================================== ==================================

Defaults, from the sim_v4 test tiles (``detections`` = boxes whose best
ground-truth box is a harmonic ignore region, any score >= 0.05) and the
four hand-labelled real recordings:

- ``harmonic_tolerance_hz = 15``: the dual-head's ``emitter_hz`` on
  harmonic copies was within 14.9 Hz of twice a live track for 99 % of
  them (9.7 Hz at score >= 0.5; real copies at 2022-05-12 and 2022-06-01:
  <= 6 Hz). Fundamentals were never closer than 183 Hz (1 % quantile) to
  twice another track.
- ``fundamental_tolerance_hz = 50``: the dual-head's true positives had
  ``emitter_hz`` within 9 Hz (sim, 99 % at score >= 0.5; 27 Hz at any
  score) and 6.4 Hz (real labels, all) of a live track; harmonic copies
  had no live track within 44 Hz (sim, 1 % quantile; real: > 500 Hz).
  50 Hz is also the assignment tolerance of the dual-head evaluation, so
  "assigned" and "protected" coincide.
- ``edge_harmonic_window_hz = (15, 170)``: twice the emitter's baseline
  lay 20-168 Hz above the lower edge of YOLO26s's boxes on harmonic
  copies (min-max, n = 527; real copies: 22-61 Hz).
- ``edge_fundamental_window_hz = (-20, 170)``: the baseline lay -7 to 164 Hz
  above the lower edge for 99.8 % of YOLO26s's true positives
  (0.1-99.9 % quantiles; real labels: -14 to 117 Hz).

The windows are wide, and rule 2 then also protects copies that have
any live track within 170 Hz above their lower edge (11 % of the
simulated copies at score >= 0.5, whose scenes hold fish up to 1,100 Hz;
in the competition recordings, harmonic copies lie > 400 Hz above every
fish). Wide windows trade precision for recall, on purpose.

Re-attribution
--------------
A copy whose fundamental was not detected is the only trace of a real
chirp. With ``reattribute_harmonics`` (default), such a copy is kept and
reported at the fundamental instead of dropped: ``emitter_hz``,
``f_low_hz`` and ``f_high_hz`` are halved (the detected values move to
``harmonic_emitter_hz``, ``harmonic_f_low_hz``, ``harmonic_f_high_hz``),
``emitter_id`` (if the table has one) and ``harmonic_of_id`` are the
track, ``emitter_df_hz`` is recomputed, and ``reattributed`` is
``"harmonic"``. The chirp-parameter decoder never saw harmonic crops, so
its values on these rows are NaN (calibrated and raw), with ``params_ok``
False and ``params_flag`` ``"harmonic"``. Box-only rows are kept and
halved the same way, without an assignment.

A copy of track ``k`` counts as detected at the fundamental if a kept
detection overlaps it in time (temporal IoU >= ``reattribute_min_tiou``,
0.3) and either spans ``k``'s frequency (``f_low <= f_k <= f_high``) or is
assigned to ``k``; copies are visited by decreasing score, so the second
copy of one chirp finds the first re-attributed one. On the sim_v4 test
tiles (dual-head, score >= 0.5), kept fundamentals overlapped their
copies with temporal IoU > 0.5 or not at all (< 0.1), so the threshold
is not critical. The assignment alone is not enough: 128 of the 1,399
copies had no overlapping kept detection *assigned* to their track, but
44 of these chirps had been detected and assigned to another, close fish
(re-attribution would count them twice) and 64 lay at tracks with no
labelled chirp nearby. With the frequency criterion, 1 copy remained, on
a chirp whose fundamental was missed.

Measured at score 0.5 (detections above the threshold post-processed, as
in `predict_span`): nothing re-attributed on the real labels; on sim_v4,
the dual-head re-attributed 1 copy, which matched its missed chirp, and
YOLO26s 12, all false positives at tracks without a labelled chirp (9 at
a real background fish). For box-only models the gain is not shown.

Re-attribution depends on which fundamentals are detected, so unlike the
rest it does not commute with the score threshold: apply it to the
detections that are reported (`predict_span` does, at its threshold).

Live tracks
-----------
A track is alive at a time if its detections around it are at most
``max_gap_s`` apart (10 s, the liveness rule of the production assignment;
pass the pipeline's value). Wavetracker tracks have gaps: in the 33 trials
a track lies in gaps longer than 1 s for 5 % of its time on average (up to
27 %). With a 1-s gap, 45 of the dual-head's 335 true positives on the
real labels had no live track within 15 Hz of their ``emitter_hz`` (the
other fish's track was in a gap: up to 45 Hz away, or no live track at
all); with 10 s, all lay within 6.4 Hz of one. A short gap would keep copies whose reference track
is in a gap and leave fundamentals unprotected while their own track is.

Without tracks
--------------
For a recording without tracks, rule 2 cannot be checked. The *pair
rule* then marks a detection that overlaps another, at least as confident
detection in time (temporal IoU >= ``pair_min_tiou``) at half its emitter
frequency (head: ``|e - 2 e'| <= pair_tolerance_hz``; edge: the same with
``e = f_low_hz + edge_offset_hz``, tolerance ``edge_pair_tolerance_hz``).
On the sim_v4 test tiles with the tracks withheld, it marked 1,342
dual-head detections with score >= 0.5 (1,302 on harmonic ignore
regions, 39 unmatched false positives, 1 matched to a true chirp) and 173
YOLO26s detections (172 on harmonic regions, none on true chirps); the
tracked rule dropped 1,356 of the 1,402 and 178 of the 218 detections on
harmonic regions. It still cannot tell a copy
from the chirp of a second fish at twice the frequency, which mixed
recordings (e.g. *Eigenmannia* at 300-600 Hz next to *A. leptorhynchus*
at 600-1000 Hz) contain. So by default it only flags (column
``harmonic_pair_of``); ``trackless="drop"`` drops as well, ``"off"``
skips it. The 33 competition trials all have tracks.

Duplicates
----------
Two boxes on one chirp. Detections are visited in order of decreasing
score; a detection is a *duplicate* of an already kept one if the two
overlap in frequency (open intervals) and

- head rows (both): ``|e - e'| <= duplicate_emitter_tolerance_hz`` (5 Hz,
  about the 90 % quantile of the dual-head's emitter error, 4.7 Hz), the
  same ``emitter_id`` if the table has one (equal, or both unassigned),
  and temporal IoU >= ``duplicate_head_min_tiou`` (0.7);
- otherwise: ``|f_low - f_low'| <= duplicate_edge_tolerance_hz`` (2 Hz,
  under half a frequency bin; the real duplicates of YOLO26s differ by
  0.2-0.6 Hz), temporal IoU >= ``duplicate_edge_min_tiou`` (0.5) and
  frequency IoU >= ``duplicate_edge_min_fiou`` (0.5).

The higher-scoring one is kept. As in greedy non-maximum suppression, a
detection is only suppressed by a higher-scoring kept one, so the result
at any score threshold equals the result of post-processing only the
detections above it.

Why these overlaps (sim_v4 test tiles; pairs of detections with score >=
0.5 whose best ground-truth boxes are one chirp, or two different
chirps): "centre inside the other box" is not a criterion, because 7.8 %
(dual-head) and 3.2 % (YOLO26s) of the pairs on two chirps of the same
fish in quick succession (boxes of small chirps are about 0.2 s wide, the
analysis window's smear) had one centre inside the other box, while only
0.5 % and 0 % reached a temporal IoU of 0.5. With temporal IoU >= 0.5
alone, the emitter criteria merged 2 of 1,051 (dual-head) and 3 of 1,002
(YOLO26s) pairs on different chirps, each losing a true positive: two
fish 18-23 Hz apart chirping at the same time, whose box-only boxes
shared the lower edge but not the height, and dual-head boxes of one
track overlapping at temporal IoU 0.55-0.63. The settings above merge
none of those pairs and still catch 43 (dual-head, 46 before) and 45
(YOLO26s, unchanged) pairs of boxes on one chirp.

Columns
-------
`flag_detections` adds `COLUMNS`:

``drop_reason``
    ``""`` (kept), ``"harmonic"`` or ``"duplicate"``.
``harmonic_of_id``
    Harmonic copies (tracked rule): the track whose second harmonic the
    detection lies on (Int64, NA otherwise).
``harmonic_pair_of``
    Pair rule: row of the co-occurring detection at half the frequency
    (Int64, NA otherwise); set whether the row is dropped or only flagged.
``duplicate_of``
    Duplicates: row of the kept twin (Int64, NA otherwise).
``reattributed``
    ``"harmonic"`` for a re-attributed copy (kept), ``""`` otherwise.
``harmonic_emitter_hz``, ``harmonic_f_low_hz``, ``harmonic_f_high_hz``
    The detected values of a re-attributed copy (NaN otherwise).

Rows refer to positions in the same table. `drop_flagged` removes the
dropped rows and renumbers these references (NA where the referenced row
was dropped).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, get_args

import numpy as np
import numpy.typing as npt
import pandas as pd
from gridtools.io import Tracks

__all__ = [
    "COLUMNS",
    "DEFAULT_FUNDAMENTAL_TOLERANCE_HZ",
    "DEFAULT_MAX_GAP_S",
    "REASONS",
    "PostprocessConfig",
    "TracklessMode",
    "drop_flagged",
    "duplicates",
    "flag_detections",
    "harmonic_copies",
    "harmonic_pairs",
    "postprocess",
    "reattributable_copies",
    "summarize",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

TracklessMode = Literal["off", "flag", "drop"]
"""What the pair rule does for recordings without tracks."""

KEPT = ""
REASONS: tuple[str, ...] = ("harmonic", "duplicate")
"""Possible non-empty values of ``drop_reason``."""

COLUMNS: tuple[str, ...] = (
    "drop_reason",
    "harmonic_of_id",
    "harmonic_pair_of",
    "duplicate_of",
    "reattributed",
    "harmonic_emitter_hz",
    "harmonic_f_low_hz",
    "harmonic_f_high_hz",
)
"""Columns added by `flag_detections` (module docstring)."""

_REFERENCES = ("harmonic_pair_of", "duplicate_of")

DEFAULT_FUNDAMENTAL_TOLERANCE_HZ = 50.0
"""Default ``fundamental_tolerance_hz`` (and `predict_span`'s assignment tolerance)."""

DEFAULT_MAX_GAP_S = 10.0
"""Default ``max_gap_s``: the production pipeline's liveness rule (module docstring)."""


def _window(value: Any, name: str) -> tuple[float, float]:
    lo, hi = (float(v) for v in value)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        msg = f"{name} must be a finite (lo, hi) with lo < hi, got {value!r}"
        raise ValueError(msg)
    return lo, hi


@dataclass(frozen=True)
class PostprocessConfig:
    """Settings of `flag_detections` (module docstring for each rule).

    Parameters
    ----------
    harmonic, duplicate
        Run the harmonic and the duplicate rule.
    trackless
        Pair rule for recordings without tracks: ``"flag"`` (default) sets
        ``harmonic_pair_of`` only, ``"drop"`` also drops, ``"off"`` skips.
    pair_min_tiou
        Pair rule: temporal IoU at which two detections count as
        simultaneous.
    max_gap_s
        A track is alive where its detections are at most this far apart
        (`gridtools.io.Track.frequency_at`); pass the value the emitter
        assignment uses.
    harmonic_tolerance_hz, fundamental_tolerance_hz
        Half-widths of the harmonic and fundamental intervals around
        ``emitter_hz`` (head rows).
    edge_harmonic_window_hz, edge_fundamental_window_hz
        The same intervals for edge rows, relative to ``f_low_hz``.
    pair_tolerance_hz, edge_pair_tolerance_hz, edge_offset_hz
        Pair rule: tolerance of ``|e - 2 e'|`` for head rows; for edge rows,
        with ``e = f_low_hz + edge_offset_hz``.
    duplicate_emitter_tolerance_hz, duplicate_edge_tolerance_hz
        Same emitter of two head rows (``emitter_hz``) or of two edge rows
        (``f_low_hz``).
    duplicate_head_min_tiou
        Duplicates of head rows: smallest temporal IoU.
    duplicate_edge_min_tiou, duplicate_edge_min_fiou
        Duplicates of edge rows: smallest temporal and frequency IoU.
    reattribute_harmonics
        Keep a harmonic copy without a detected fundamental, re-attributed
        to its fish at half the frequency (module docstring), instead of
        dropping it.
    reattribute_min_tiou
        Temporal IoU from which a kept detection counts as the copy's
        fundamental.
    """

    harmonic: bool = True
    duplicate: bool = True
    trackless: TracklessMode = "flag"
    max_gap_s: float = DEFAULT_MAX_GAP_S
    harmonic_tolerance_hz: float = 15.0
    fundamental_tolerance_hz: float = DEFAULT_FUNDAMENTAL_TOLERANCE_HZ
    edge_harmonic_window_hz: tuple[float, float] = (15.0, 170.0)
    edge_fundamental_window_hz: tuple[float, float] = (-20.0, 170.0)
    pair_tolerance_hz: float = 20.0
    edge_offset_hz: float = 41.0
    edge_pair_tolerance_hz: float = 40.0
    pair_min_tiou: float = 0.5
    duplicate_emitter_tolerance_hz: float = 5.0
    duplicate_head_min_tiou: float = 0.7
    duplicate_edge_tolerance_hz: float = 2.0
    duplicate_edge_min_tiou: float = 0.5
    duplicate_edge_min_fiou: float = 0.5
    reattribute_harmonics: bool = True
    reattribute_min_tiou: float = 0.3

    def __post_init__(self) -> None:
        if self.trackless not in get_args(TracklessMode):
            msg = f"trackless must be one of {get_args(TracklessMode)}, got {self.trackless!r}"
            raise ValueError(msg)
        for name in (
            "pair_min_tiou",
            "duplicate_head_min_tiou",
            "duplicate_edge_min_tiou",
            "duplicate_edge_min_fiou",
            "reattribute_min_tiou",
        ):
            value = float(getattr(self, name))
            if not 0 < value <= 1:
                msg = f"{name} must lie in (0, 1], got {value}"
                raise ValueError(msg)
        for name in (
            "max_gap_s",
            "harmonic_tolerance_hz",
            "fundamental_tolerance_hz",
            "pair_tolerance_hz",
            "edge_pair_tolerance_hz",
            "duplicate_emitter_tolerance_hz",
            "duplicate_edge_tolerance_hz",
        ):
            value = float(getattr(self, name))
            if not (np.isfinite(value) and value >= 0):
                msg = f"{name} must be finite and >= 0, got {value}"
                raise ValueError(msg)
        if not np.isfinite(self.edge_offset_hz):
            msg = f"edge_offset_hz must be finite, got {self.edge_offset_hz}"
            raise ValueError(msg)
        for name in ("edge_harmonic_window_hz", "edge_fundamental_window_hz"):
            object.__setattr__(self, name, _window(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable settings (windows as lists)."""
        return {
            k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PostprocessConfig:
        """Inverse of `to_dict`; missing keys take their defaults.

        Raises
        ------
        ValueError
            On unknown keys or invalid values.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            msg = f"unknown post-processing keys {sorted(unknown)}"
            raise ValueError(msg)
        values = dict(data)
        for name in ("edge_harmonic_window_hz", "edge_fundamental_window_hz"):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)


# ------------------------------------------------------------------ helpers


def _floats(df: pd.DataFrame, column: str) -> FloatArray:
    """A column as a fresh float64 array (never a view of ``df``)."""
    values = pd.Series(df[column]).astype("Float64")
    return np.array(values.to_numpy(dtype=np.float64, na_value=np.nan), copy=True)


def _centres(df: pd.DataFrame) -> FloatArray:
    if "t_center_s" in df.columns:
        return _floats(df, "t_center_s")
    return 0.5 * (_floats(df, "t_start_s") + _floats(df, "t_end_s"))


def _head(df: pd.DataFrame) -> BoolArray:
    """Rows with a predicted emitter frequency."""
    if "emitter_hz" not in df.columns:
        return np.zeros(len(df), dtype=np.bool_)
    return np.isfinite(_floats(df, "emitter_hz"))


def _emitter_ids(df: pd.DataFrame) -> IntArray | None:
    """``emitter_id`` with -1 for NA, or None without the column."""
    if "emitter_id" not in df.columns:
        return None
    ids = pd.Series(df["emitter_id"]).astype("Int64")
    return ids.fillna(-1).to_numpy(dtype=np.int64)


def _intervals(
    df: pd.DataFrame,
    tolerance_hz: float,
    window_hz: tuple[float, float],
) -> tuple[FloatArray, FloatArray]:
    """``[lo, hi]`` per row: ``emitter_hz ± tolerance`` or ``f_low + window``."""
    head = _head(df)
    f_low = _floats(df, "f_low_hz")
    lo = f_low + window_hz[0]
    hi = f_low + window_hz[1]
    if head.any():
        e = _floats(df, "emitter_hz")
        lo = np.where(head, e - tolerance_hz, lo)
        hi = np.where(head, e + tolerance_hz, hi)
    return lo, hi


def _live_frequencies(
    tracks: Tracks, t_s: FloatArray, max_gap_s: float
) -> tuple[FloatArray, IntArray]:
    """Frequency of every track at every time (NaN where not alive), and ids."""
    ids = tracks.ids
    freqs = np.full((len(t_s), len(ids)), np.nan)
    for j, ident in enumerate(ids):
        freqs[:, j] = tracks.track(int(ident)).frequency_at(t_s, max_gap_s)
    return freqs, ids


def _iou_1d(lo: FloatArray, hi: FloatArray, i: int, others: IntArray) -> FloatArray:
    """IoU of interval ``i`` with each of ``others`` (time or frequency)."""
    inter = np.minimum(hi[i], hi[others]) - np.maximum(lo[i], lo[others])
    union = np.maximum(hi[i], hi[others]) - np.minimum(lo[i], lo[others])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, np.clip(inter, 0, None) / union, 0.0)


def _overlap_groups(t0: FloatArray, t1: FloatArray) -> list[IntArray]:
    """Rows in groups of transitively time-overlapping boxes (sweep by start)."""
    if len(t0) == 0:
        return []
    order = np.argsort(t0, kind="stable")
    groups, current, reach = [], [int(order[0])], t1[order[0]]
    for i in order[1:].tolist():
        if t0[i] < reach:
            current.append(i)
            reach = max(reach, t1[i])
        else:
            groups.append(np.asarray(current, dtype=np.int64))
            current, reach = [i], t1[i]
    groups.append(np.asarray(current, dtype=np.int64))
    return groups


def _score_order(scores: FloatArray, rows: IntArray) -> IntArray:
    """``rows`` by decreasing score (ties: row order)."""
    return rows[np.argsort(-scores[rows], kind="stable")]


# -------------------------------------------------------------------- rules


def harmonic_copies(
    predictions: pd.DataFrame, tracks: Tracks, config: PostprocessConfig
) -> tuple[BoolArray, IntArray]:
    """Harmonic copies among one recording's detections (tracked rule).

    Parameters
    ----------
    predictions
        Detections of one recording (box columns, ``score``; optionally
        ``emitter_hz``, ``emitter_id``, ``t_center_s``).
    tracks
        The recording's tracks, on the predictions' clock.
    config
        Tolerances.

    Returns
    -------
    ``is_copy`` per row, and the id of the track whose second harmonic
    the row lies on (-1 where not a copy).
    """
    n = len(predictions)
    if n == 0 or len(tracks.ids) == 0:
        return np.zeros(n, dtype=np.bool_), np.full(n, -1, dtype=np.int64)
    freqs, ids = _live_frequencies(tracks, _centres(predictions), config.max_gap_s)
    f_lo, f_hi = _intervals(
        predictions, config.fundamental_tolerance_hz, config.edge_fundamental_window_hz
    )
    h_lo, h_hi = _intervals(
        predictions, config.harmonic_tolerance_hz, config.edge_harmonic_window_hz
    )
    with np.errstate(invalid="ignore"):
        fundamental = ((freqs >= f_lo[:, None]) & (freqs <= f_hi[:, None])).any(axis=1)
        second = 2.0 * freqs
        inside = (second >= h_lo[:, None]) & (second <= h_hi[:, None])
    assigned = _emitter_ids(predictions)
    if assigned is not None:
        fundamental |= assigned >= 0
    copy = inside.any(axis=1) & ~fundamental
    centre = 0.5 * (h_lo + h_hi)
    distance = np.where(inside, np.abs(second - centre[:, None]), np.inf)
    best = np.argmin(distance, axis=1)
    return copy, np.where(copy, ids[best], -1).astype(np.int64)


def harmonic_pairs(
    predictions: pd.DataFrame,
    config: PostprocessConfig,
    active: npt.ArrayLike | None = None,
) -> IntArray:
    """Pair rule: the co-occurring detection at half the frequency.

    Parameters
    ----------
    predictions
        Detections of one recording.
    config
        Tolerances and ``pair_min_tiou``.
    active
        Rows that take part (default: all).

    Returns
    -------
    Per row, the row of its partner (temporal IoU >= ``pair_min_tiou``, score at
    least as high, emitter frequency within the pair tolerance of half the
    row's; the closest such), or -1.
    """
    n = len(predictions)
    partner = np.full(n, -1, dtype=np.int64)
    if n < 2:
        return partner
    use = np.ones(n, dtype=np.bool_) if active is None else np.asarray(active, bool)
    head = _head(predictions)
    e = _floats(predictions, "f_low_hz") + config.edge_offset_hz
    if head.any():
        e = np.where(head, _floats(predictions, "emitter_hz"), e)
    tol = np.where(head, config.pair_tolerance_hz, config.edge_pair_tolerance_hz)
    t0, t1 = _floats(predictions, "t_start_s"), _floats(predictions, "t_end_s")
    score = _floats(predictions, "score")
    for group in _overlap_groups(t0, t1):
        group = group[use[group]]
        for i in group.tolist():
            others = group[group != i]
            if not others.size:
                continue
            gap = np.abs(e[i] - 2.0 * e[others])
            ok = (
                (gap <= tol[i])
                & (score[others] >= score[i])
                & (_iou_1d(t0, t1, i, others) >= config.pair_min_tiou)
            )
            if ok.any():
                partner[i] = int(others[ok][np.argmin(gap[ok])])
    return partner


def duplicates(
    predictions: pd.DataFrame,
    config: PostprocessConfig,
    active: npt.ArrayLike | None = None,
) -> IntArray:
    """Duplicates among one recording's detections (greedy by score).

    Parameters
    ----------
    predictions
        Detections of one recording.
    config
        Tolerances and overlaps.
    active
        Rows that take part (default: all); the others are neither kept
        nor compared.

    Returns
    -------
    Per row, the row of the kept twin it duplicates, or -1.
    """
    n = len(predictions)
    twin = np.full(n, -1, dtype=np.int64)
    if n < 2:
        return twin
    use = np.ones(n, dtype=np.bool_) if active is None else np.asarray(active, bool)
    t0, t1 = _floats(predictions, "t_start_s"), _floats(predictions, "t_end_s")
    f0, f1 = _floats(predictions, "f_low_hz"), _floats(predictions, "f_high_hz")
    score = _floats(predictions, "score")
    head = _head(predictions)
    e = _floats(predictions, "emitter_hz") if head.any() else np.full(n, np.nan)
    ids = _emitter_ids(predictions)
    for group in _overlap_groups(t0, t1):
        kept: list[int] = []
        for i in _score_order(score, group[use[group]]).tolist():
            if kept:
                k = np.asarray(kept, dtype=np.int64)
                both_head = head[i] & head[k]
                same = np.where(
                    both_head,
                    np.abs(e[i] - e[k]) <= config.duplicate_emitter_tolerance_hz,
                    np.abs(f0[i] - f0[k]) <= config.duplicate_edge_tolerance_hz,
                )
                if ids is not None:
                    same &= ~both_head | (ids[k] == ids[i])
                min_tiou = np.where(
                    both_head,
                    config.duplicate_head_min_tiou,
                    config.duplicate_edge_min_tiou,
                )
                f_iou = _iou_1d(f0, f1, i, k)
                hit = (
                    same
                    & (f_iou > 0)
                    & (both_head | (f_iou >= config.duplicate_edge_min_fiou))
                    & (_iou_1d(t0, t1, i, k) >= min_tiou)
                )
                if hit.any():
                    twin[i] = int(k[np.argmax(hit)])  # the highest-scoring twin
                    continue
            kept.append(i)
    return twin


def reattributable_copies(
    predictions: pd.DataFrame,
    tracks: Tracks,
    config: PostprocessConfig,
    reason: npt.ArrayLike,
    harmonic_of: npt.ArrayLike,
) -> BoolArray:
    """Harmonic copies of one recording without a detected fundamental.

    Copies (``reason == "harmonic"`` with a track in ``harmonic_of``) are
    visited by decreasing score. A copy of track ``k`` is re-attributable
    if no kept detection (``reason == ""``, or a copy re-attributed before
    it) overlaps it in time with temporal IoU >= ``reattribute_min_tiou``
    while spanning ``k``'s frequency at the copy's centre (``f_low <= f_k
    <= f_high``, the halved box for a re-attributed copy) or being
    assigned to ``k`` (``emitter_id``; edge rows: ``f_k`` in the
    fundamental window). Of two copies of one chirp, the second finds the
    first.

    Returns
    -------
    Per row, whether to re-attribute it.
    """
    n = len(predictions)
    out = np.zeros(n, dtype=np.bool_)
    reason = np.asarray(reason, dtype=object)
    harmonic_of = np.asarray(harmonic_of, dtype=np.int64)
    copies = np.flatnonzero((reason == "harmonic") & (harmonic_of >= 0))
    if not copies.size:
        return out
    t0, t1 = _floats(predictions, "t_start_s"), _floats(predictions, "t_end_s")
    f0, f1 = _floats(predictions, "f_low_hz"), _floats(predictions, "f_high_hz")
    head = _head(predictions)
    ids = _emitter_ids(predictions)
    centre = _centres(predictions)
    score = _floats(predictions, "score")
    w_lo, w_hi = config.edge_fundamental_window_hz
    kept = reason == KEPT
    for i in _score_order(score, copies).tolist():
        k = int(harmonic_of[i])
        f_k = float(tracks.track(k).frequency_at(centre[i], config.max_gap_s)[0])
        others = np.flatnonzero(kept | out)
        if others.size and np.isfinite(f_k):
            lo = np.where(out[others], 0.5 * f0[others], f0[others])
            hi = np.where(out[others], 0.5 * f1[others], f1[others])
            belongs = ((lo <= f_k) & (hi >= f_k)) | (
                out[others] & (harmonic_of[others] == k)
            )
            if ids is not None:
                belongs |= head[others] & (ids[others] == k)
            belongs |= (
                ~head[others]
                & (f0[others] + w_lo <= f_k)
                & (f0[others] + w_hi >= f_k)
                & ~out[others]
            )
            overlap = _iou_1d(t0, t1, i, others) >= config.reattribute_min_tiou
            if np.any(belongs & overlap):
                continue
        out[i] = np.isfinite(f_k)
    return out


# ---------------------------------------------------------------- the table


def _int64(values: IntArray) -> pd.arrays.IntegerArray:
    out = pd.array(values, dtype="Int64")
    out[values < 0] = pd.NA
    return out  # type: ignore[return-value]


def _recording_rows(df: pd.DataFrame) -> list[tuple[str | None, IntArray]]:
    if "recording" not in df.columns:
        return [(None, np.arange(len(df), dtype=np.int64))]
    names = df["recording"].astype(str).to_numpy(dtype=object)
    return [
        (str(name), np.flatnonzero(names == name).astype(np.int64))
        for name in sorted(set(names.tolist()))
    ]


def flag_detections(
    predictions: pd.DataFrame,
    tracks: Mapping[str, Tracks] | None = None,
    config: PostprocessConfig | None = None,
) -> pd.DataFrame:
    """Flag harmonic copies and duplicates; drop nothing.

    Parameters
    ----------
    predictions
        Prediction table: ``t_start_s, t_end_s, f_low_hz, f_high_hz,
        score``; optionally ``recording`` (rules run per recording; all
        rows are one recording without it), ``emitter_hz``,
        ``emitter_id``, ``t_center_s``.
    tracks
        Tracks per recording, on the predictions' clock. Recordings
        without tracks (or ``tracks=None``) get the pair rule
        (``config.trackless``) instead of the tracked harmonic rule.
    config
        Settings (default: `PostprocessConfig()`).

    Returns
    -------
    DataFrame
        A copy with a fresh index and `COLUMNS` appended (replacing
        existing ones). Harmonic copies are found first; duplicates only
        among the rows not dropped as copies.
    """
    config = config or PostprocessConfig()
    out = predictions.drop(columns=[c for c in COLUMNS if c in predictions.columns])
    out = out.reset_index(drop=True)
    n = len(out)
    reason = np.full(n, KEPT, dtype=object)
    reattributed = np.full(n, KEPT, dtype=object)
    harmonic_of = np.full(n, -1, dtype=np.int64)
    pair_of = np.full(n, -1, dtype=np.int64)
    duplicate_of = np.full(n, -1, dtype=np.int64)
    groups = _recording_rows(out) if n else []
    for name, rows in groups:
        part = out.iloc[rows]
        rec_tracks = None if tracks is None or name is None else tracks.get(name)
        if not config.harmonic:
            pass
        elif rec_tracks is not None:
            copy, of = harmonic_copies(part, rec_tracks, config)
            reason[rows[copy]] = "harmonic"
            harmonic_of[rows[copy]] = of[copy]
        elif config.trackless != "off":
            partner = harmonic_pairs(part, config)
            has = partner >= 0
            pair_of[rows[has]] = rows[partner[has]]
            if config.trackless == "drop":
                reason[rows[has]] = "harmonic"
        if config.duplicate:
            twin = duplicates(part, config, active=reason[rows] == KEPT)
            has = twin >= 0
            reason[rows[has]] = "duplicate"
            duplicate_of[rows[has]] = rows[twin[has]]
        if config.harmonic and config.reattribute_harmonics and rec_tracks is not None:
            again = reattributable_copies(
                part, rec_tracks, config, reason[rows], harmonic_of[rows]
            )
            reattributed[rows[again]] = "harmonic"
            reason[rows[again]] = KEPT
    out["drop_reason"] = pd.Series(reason, dtype=object)
    out["harmonic_of_id"] = _int64(harmonic_of)
    out["harmonic_pair_of"] = _int64(pair_of)
    out["duplicate_of"] = _int64(duplicate_of)
    _reattribute(out, reattributed, harmonic_of, tracks, config)
    out.attrs = {**predictions.attrs}
    return out


_HARMONIC_COLUMNS = {
    "emitter_hz": "harmonic_emitter_hz",
    "f_low_hz": "harmonic_f_low_hz",
    "f_high_hz": "harmonic_f_high_hz",
}


def _reattribute(
    out: pd.DataFrame,
    reattributed: npt.NDArray[np.object_],
    harmonic_of: IntArray,
    tracks: Mapping[str, Tracks] | None,
    config: PostprocessConfig,
) -> None:
    """Report re-attributed copies at the fundamental (in place)."""
    rows = np.flatnonzero(reattributed == "harmonic")
    out["reattributed"] = pd.Series(reattributed, dtype=object)
    for column, saved in _HARMONIC_COLUMNS.items():
        values = np.full(len(out), np.nan)
        if column in out.columns and rows.size:
            values[rows] = _floats(out, column)[rows]
            halved = _floats(out, column)
            halved[rows] = 0.5 * values[rows]
            out[column] = halved
        out[saved] = values
    if not rows.size:
        return
    ids = harmonic_of[rows]
    if "emitter_id" in out.columns:
        emitter = _emitter_ids(out)
        assert emitter is not None
        emitter = np.array(emitter, copy=True)
        emitter[rows] = ids
        out["emitter_id"] = _int64(emitter)
    if "emitter_df_hz" in out.columns and tracks is not None:
        df_hz = _floats(out, "emitter_df_hz")
        centre = _centres(out)
        names = out["recording"].astype(str).to_numpy() if "recording" in out else None
        e = _floats(out, "emitter_hz") if "emitter_hz" in out.columns else None
        for r, k in zip(rows.tolist(), ids.tolist(), strict=True):
            rec = tracks.get(str(names[r])) if names is not None else None
            if rec is None or e is None:
                continue
            f_k = rec.track(int(k)).frequency_at(centre[r], config.max_gap_s)[0]
            df_hz[r] = abs(e[r] - f_k)
        out["emitter_df_hz"] = df_hz
    if "params_ok" in out.columns:
        _invalidate_params(out, rows)


def _invalidate_params(out: pd.DataFrame, rows: IntArray) -> None:
    """Decoder values of re-attributed copies: NaN, flagged ``harmonic``."""
    from chirpdetector.params.inference import ESTIMATE_COLUMNS

    for column in ESTIMATE_COLUMNS:
        if column in ("params_ok", "params_flag") or column not in out.columns:
            continue
        values = _floats(out, column)
        values[rows] = np.nan
        out[column] = values
    ok = np.array(pd.Series(out["params_ok"]).astype(bool).to_numpy(), copy=True)
    ok[rows] = False
    out["params_ok"] = ok
    if "params_flag" in out.columns:
        flag = np.array(
            pd.Series(out["params_flag"]).astype(object).to_numpy(), copy=True
        )
        flag[rows] = "harmonic"
        out["params_flag"] = pd.Series(flag, dtype=object)


def drop_flagged(flagged: pd.DataFrame) -> pd.DataFrame:
    """The kept rows of a `flag_detections` table, references renumbered.

    Returns
    -------
    DataFrame
        Rows with ``drop_reason == ""``, fresh index; ``harmonic_pair_of``
        and ``duplicate_of`` point to rows of the returned table (NA where
        the referenced row was dropped).
    """
    keep = flagged["drop_reason"].astype(str).to_numpy() == KEPT
    position = np.cumsum(keep) - 1
    out = flagged.loc[keep].reset_index(drop=True)
    for column in _REFERENCES:
        old = pd.Series(flagged[column]).astype("Int64").fillna(-1)
        old = old.to_numpy(dtype=np.int64)[keep]
        valid = old >= 0
        new = np.full(len(old), -1, dtype=np.int64)
        new[valid] = np.where(keep[old[valid]], position[old[valid]], -1)
        out[column] = _int64(new)
    out.attrs = {**flagged.attrs}
    return out


def summarize(flagged: pd.DataFrame) -> dict[str, int]:
    """Counts: ``n``, ``kept``, one per reason, ``harmonic_pair_flagged`` and
    ``reattributed``."""
    reason = flagged["drop_reason"].astype(str).to_numpy()
    counts = {"n": len(flagged), "kept": int(np.sum(reason == KEPT))}
    counts.update({r: int(np.sum(reason == r)) for r in REASONS})
    pairs = pd.Series(flagged["harmonic_pair_of"]).notna().to_numpy(dtype=np.bool_)
    counts["harmonic_pair_flagged"] = int(pairs.sum())
    if "reattributed" in flagged.columns:
        again = flagged["reattributed"].astype(str).to_numpy() == "harmonic"
        counts["reattributed"] = int(again.sum())
    return counts


def postprocess(
    predictions: pd.DataFrame,
    tracks: Mapping[str, Tracks] | None = None,
    config: PostprocessConfig | None = None,
    *,
    keep_dropped: bool = False,
) -> pd.DataFrame:
    """`flag_detections`, then `drop_flagged` unless ``keep_dropped``."""
    flagged = flag_detections(predictions, tracks, config)
    return flagged if keep_dropped else drop_flagged(flagged)
