"""Assign detections to wavetracker tracks by the predicted emitter frequency.

Rule
----
The dual-head detector predicts, with every box, the emitter's baseline
EOD frequency (``emitter_hz``). A detection is assigned to the *nearest
live track* to that frequency, compared at the chirp's onset with the
track's baseline before the chirp:

- *onset*: the box's ``t_start_s`` (not its centre, where a tracker that
  follows the excursion would be pulled away from the baseline);
- *live*: the track exists at the onset (`gridtools.io.Track.frequency_at`
  with ``max_gap_s``: between its first and last detection, and not inside
  a gap longer than ``max_gap_s``). Unassigned detections (NaN ``ident_v``)
  belong to no track;
- *baseline*: the mean of the track's detections in ``[onset - window,
  onset - guard)`` (1 s before the onset by default); if it has none there (a gap, or it starts at the
  onset), its frequency interpolated at the onset;
- *tolerance*: no assignment if the nearest live track is further than
  ``tolerance_hz`` from ``emitter_hz`` (status ``beyond_tolerance``), if no
  track is live (``no_live_track``), if the recording has no tracks
  (``no_tracks``) or the detector predicts no emitter (``no_emitter``; the
  YOLO bundles: assignment is not supported for them).

The track id is the wavetracker identity (``ident_v`` value) unchanged
(float64, NaN = unassigned), so ids match the analysis code and
``trial_summary.csv``.

Confidence
----------
``assign_confidence`` is the posterior probability that the chosen
(nearest) track is the emitter, under a defined error model:

- ``emitter_hz - f_emitter_baseline`` follows a Student-t distribution
  (location 0, scale ``s``, ``nu`` degrees of freedom), fitted to the
  deployed dual-head's 8,799 true positives on the held-out sim_v4 test
  scenes (``nu = 1.73``, ``s = 1.52 Hz``; median absolute error 1.29 Hz,
  p90 4.7 Hz, p99 24 Hz);
- every live track is a priori equally likely to be the emitter, and with
  probability ``p_untracked`` (0.05) the emitter has no live track, its
  ``emitter_hz`` then uniform over the front end's band (1000 Hz).

So ``p_j = (1 - p_u) / n t(e_j) / (sum_k (1 - p_u) / n t(e_k) + p_u / B)``.
On the sim_v4 test true positives this is well calibrated (expected
calibration error 0.013, mean confidence 0.925 vs accuracy 0.914; 10 equal
bins). The simulated emitter tracks are noise-free, so on real tracks the
error is larger by the tracker's error, and the confidence is an upper
bound; ``p_untracked`` is an assumption, not a fit. Fit:
``/mnt/data1/chirp_v2/detections/calibration/`` (README.md).

The tolerance (default 20 Hz, as v1's) keeps 98.9% of the sim true
positives (97.4% within 10 Hz, 98.7% within 15 Hz). On the dense dry-run
recording (2019-11-25-09_59, 3,675 chirps at score >= 0.5) the distance to
the nearest live track was < 10 Hz for 3,363 chirps, 10-20 Hz for 36,
20-50 Hz for 7, and > 50 Hz for 269 (mostly second-harmonic copies, whose
predicted emitter lies near twice a fish's frequency): the gap between
the error distribution and the copies lies between 20 and 50 Hz.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from gridtools.io import Tracks
from scipy import stats

from chirpdetector.pipeline.config import AssignmentConfig

__all__ = [
    "ASSIGN_COLUMNS",
    "ASSIGN_STATUS",
    "SIM_V4_DUALHEAD",
    "ErrorModel",
    "assign_detections",
    "track_baselines",
]

ASSIGN_STATUS = (
    "assigned",
    "beyond_tolerance",
    "no_live_track",
    "no_tracks",
    "no_emitter",
)
"""Values of ``assign_status``."""

ASSIGN_COLUMNS = (
    "track_id",
    "track_hz",
    "emitter_track_df_hz",
    "assign_status",
    "assign_confidence",
    "nearest_track_id",
    "nearest_track_hz",
    "nearest_df_hz",
    "second_track_id",
    "second_track_hz",
    "second_df_hz",
    "assign_margin_hz",
    "n_live_tracks",
    "track_df_other_hz",
)
"""Columns added by `assign_detections`."""


@dataclass(frozen=True)
class ErrorModel:
    """Student-t model of ``emitter_hz`` minus the emitter's baseline.

    Attributes
    ----------
    nu, scale_hz, loc_hz
        Degrees of freedom, scale and location of the t distribution.
    p_untracked
        Prior probability that the emitter has no live track.
    band_hz
        Width of the band over which an untracked emitter's predicted
        frequency is uniform.
    source
        Where the parameters come from.
    """

    nu: float
    scale_hz: float
    loc_hz: float = 0.0
    p_untracked: float = 0.05
    band_hz: float = 1000.0
    source: str = ""

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)

    def posterior(self, errors_hz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Posterior of every live track (rows: detections, NaN = not live)."""
        err = np.asarray(errors_hz, dtype=np.float64)
        live = np.isfinite(err)
        n = live.sum(axis=1, keepdims=True)
        lik = np.where(
            live,
            stats.t.pdf(np.nan_to_num(err), self.nu, self.loc_hz, self.scale_hz),
            0.0,
        )
        prior = np.where(n > 0, (1 - self.p_untracked) / np.maximum(n, 1), 0.0)
        den = (prior * lik).sum(axis=1, keepdims=True) + self.p_untracked / self.band_hz
        return np.where(live, prior * lik / den, np.nan)


SIM_V4_DUALHEAD = ErrorModel(
    nu=1.724,
    scale_hz=1.516,
    loc_hz=0.0,
    p_untracked=0.05,
    band_hz=1000.0,
    source=(
        "Student-t (loc fixed at 0) fitted to emitter_hz - track baseline of the "
        "8,799 true positives of dualhead_r18_sim_v4_r1 (weights 9fcdef3c) on "
        "sim_v4 test; /mnt/data1/chirp_v2/detections/calibration/"
    ),
)
"""Error model of the deployed dual-head detector."""


def track_baselines(
    tracks: Tracks,
    onsets_s: npt.ArrayLike,
    config: AssignmentConfig,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Every track's baseline at every onset (module docstring).

    Returns
    -------
    ids
        Track identities (float64, as in ``ident_v``), shape ``(m,)``.
    baselines
        Shape ``(n, m)``; NaN where the track is not live at the onset.
    """
    t = np.atleast_1d(np.asarray(onsets_s, dtype=np.float64))
    ids = tracks.ids
    out = np.full((len(t), len(ids)), np.nan)
    lo_off, hi_off = config.baseline_window_s, config.baseline_guard_s
    for j, ident in enumerate(ids):
        track = tracks.track(int(ident))
        # only onsets the track can be live at
        near = np.flatnonzero(
            (t >= track.start_s - config.max_gap_s)
            & (t <= track.stop_s + config.max_gap_s)
        )
        if not near.size:
            continue
        at = track.frequency_at(t[near], config.max_gap_s)
        live = np.isfinite(at)
        if not live.any():
            continue
        rows = near[live]
        times, freqs = track.times_s, track.freqs_hz
        cum = np.concatenate([[0.0], np.cumsum(freqs)])
        a = np.searchsorted(times, t[rows] - lo_off, side="left")
        b = np.searchsorted(times, t[rows] - hi_off, side="left")
        count = b - a
        mean = np.where(count > 0, (cum[b] - cum[a]) / np.maximum(count, 1), at[live])
        out[rows, j] = mean
    return ids.astype(np.float64), out


def _nan_ids(n: int) -> npt.NDArray[np.float64]:
    return np.full(n, np.nan)


def assign_detections(
    detections: pd.DataFrame,
    tracks: Tracks | None,
    config: AssignmentConfig,
    model: ErrorModel = SIM_V4_DUALHEAD,
    block: int = 10_000,
) -> pd.DataFrame:
    """Assignment columns (`ASSIGN_COLUMNS`) for a detection table.

    Parameters
    ----------
    detections
        With ``t_start_s`` and, for an emitter-predicting model,
        ``emitter_hz``.
    tracks
        The recording's tracks (None: no tracks).
    config
        Tolerance, liveness and baseline window.
    model
        Error model of the confidence.
    block
        Detections per block (bounds the memory of the detection x track
        matrices).

    Returns
    -------
    DataFrame
        One row per detection (index of ``detections``).
    """
    if len(detections) > block:
        parts = [
            _assign_block(detections.iloc[a : a + block], tracks, config, model)
            for a in range(0, len(detections), block)
        ]
        return pd.concat(parts)
    return _assign_block(detections, tracks, config, model)


def _assign_block(
    detections: pd.DataFrame,
    tracks: Tracks | None,
    config: AssignmentConfig,
    model: ErrorModel,
) -> pd.DataFrame:
    n = len(detections)
    out = pd.DataFrame(
        {
            "track_id": _nan_ids(n),
            "track_hz": np.full(n, np.nan),
            "emitter_track_df_hz": np.full(n, np.nan),
            "assign_status": np.full(n, "no_emitter", dtype=object),
            "assign_confidence": np.full(n, np.nan),
            "nearest_track_id": _nan_ids(n),
            "nearest_track_hz": np.full(n, np.nan),
            "nearest_df_hz": np.full(n, np.nan),
            "second_track_id": _nan_ids(n),
            "second_track_hz": np.full(n, np.nan),
            "second_df_hz": np.full(n, np.nan),
            "assign_margin_hz": np.full(n, np.nan),
            "n_live_tracks": np.zeros(n, dtype=np.int64),
            "track_df_other_hz": np.full(n, np.nan),
        },
        index=detections.index,
    )
    if n == 0:
        return out
    has_emitter = "emitter_hz" in detections.columns
    emitter = (
        detections["emitter_hz"].to_numpy(dtype=np.float64)
        if has_emitter
        else np.full(n, np.nan)
    )
    onsets = detections["t_start_s"].to_numpy(dtype=np.float64)
    if tracks is None or len(tracks.ids) == 0:
        out.loc[np.isfinite(emitter), "assign_status"] = "no_tracks"
        return out
    ids, base = track_baselines(tracks, onsets, config)
    live = np.isfinite(base)
    n_live = live.sum(axis=1)
    out["n_live_tracks"] = n_live.astype(np.int64)
    gap = np.abs(base - emitter[:, None])
    gap_inf = np.where(np.isfinite(gap), gap, np.inf)
    order = np.argsort(gap_inf, axis=1, kind="stable")
    rows = np.arange(n)
    first, second = order[:, 0], order[:, 1] if base.shape[1] > 1 else order[:, 0]
    d1 = gap_inf[rows, first]
    d2 = gap_inf[rows, second] if base.shape[1] > 1 else np.full(n, np.inf)
    ok_emitter = np.isfinite(emitter)
    has1 = ok_emitter & np.isfinite(d1)
    has2 = ok_emitter & np.isfinite(d2)
    f1 = base[rows, first]
    f2 = base[rows, second] if base.shape[1] > 1 else np.full(n, np.nan)
    out["nearest_track_id"] = np.where(has1, ids[first], np.nan)
    out["nearest_track_hz"] = np.where(has1, f1, np.nan)
    out["nearest_df_hz"] = np.where(has1, d1, np.nan)
    out["second_track_id"] = np.where(has2, ids[second], np.nan)
    out["second_track_hz"] = np.where(has2, f2, np.nan)
    out["second_df_hz"] = np.where(has2, d2, np.nan)
    with np.errstate(invalid="ignore"):
        out["assign_margin_hz"] = np.where(has2, d2 - d1, np.nan)
    # ΔF between the nearest track and the closest other live track
    other = np.abs(base - np.where(has1, f1, np.nan)[:, None])
    other[rows, first] = np.nan
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df_other = np.where(
            n_live >= 2, np.nanmin(np.where(live, other, np.inf), axis=1), np.nan
        )
    out["track_df_other_hz"] = np.where(np.isfinite(df_other) & has1, df_other, np.nan)
    post = model.posterior(np.where(live, base - emitter[:, None], np.nan))
    conf = np.where(has1, post[rows, first], np.nan)
    out["assign_confidence"] = conf
    status = np.full(n, "no_emitter", dtype=object)
    status[ok_emitter & (n_live == 0)] = "no_live_track"
    status[has1 & (d1 > config.tolerance_hz)] = "beyond_tolerance"
    assigned = has1 & (d1 <= config.tolerance_hz)
    status[assigned] = "assigned"
    out["assign_status"] = status
    out["track_id"] = np.where(assigned, ids[first], np.nan)
    out["track_hz"] = np.where(assigned, f1, np.nan)
    out["emitter_track_df_hz"] = np.where(assigned, d1, np.nan)
    return out
