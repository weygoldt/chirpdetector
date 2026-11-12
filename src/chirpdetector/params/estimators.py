"""Non-learned spectrogram estimators of chirp parameters (baselines).

`box_features`
    Raw box arithmetic: box top minus baseline, box duration, baseline
    minus box bottom. `fit_box_offsets` learns the constant smear to
    subtract (median residual on a training set), giving the "naive box"
    estimator.

`ridge_estimate`
    Classical ridge tracking on a rectified crop (`chirpdetector.params.
    crops`): per frame, the spectral peak (parabolic interpolation) between
    -60 and +830 Hz in a short-window channel is the emitter's frequency;
    the run of frames above +20 Hz nearest the crop centre is the chirp;
    height is its maximum, FWHM and 10 % width the run's extent above 50 %
    and 10 % of the height. No handling of other fish: a louder neighbour
    captures the peak (by design, this is the plain classical baseline). The undershoot is the deepest frequency of the peak near
    baseline (-125..+20 Hz) in the 51.2 ms channel after the chirp, smoothed
    over 3 frames, relative to the same peak's median before the chirp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from chirpdetector.params.crops import RectifiedConfig

__all__ = ["RidgeConfig", "box_features", "fit_box_offsets", "ridge_estimate"]

FloatArray = npt.NDArray[np.float64]
PARAMS = ("height_hz", "fwhm_s", "width10_s", "undershoot_hz")


def box_features(meta: pd.DataFrame) -> pd.DataFrame:
    """Raw box quantities (Hz, s) per chirp."""
    return pd.DataFrame(
        {
            "height_hz": meta["box_f1_hz"] - meta["baseline_hz"],
            "fwhm_s": meta["box_t1_s"] - meta["box_t0_s"],
            "width10_s": meta["box_t1_s"] - meta["box_t0_s"],
            "undershoot_hz": meta["baseline_hz"] - meta["box_f0_hz"],
        }
    )


def fit_box_offsets(meta: pd.DataFrame) -> dict[str, float]:
    """Median ``box - truth`` per parameter (the smear to subtract)."""
    feats = box_features(meta)
    return {p: float(np.nanmedian(feats[p] - meta[p])) for p in PARAMS}


def box_estimate(meta: pd.DataFrame, offsets: dict[str, float]) -> pd.DataFrame:
    """Naive estimator: box quantity minus the fitted smear (floored)."""
    feats = box_features(meta)
    out = pd.DataFrame({p: feats[p] - offsets[p] for p in PARAMS})
    out["height_hz"] = out["height_hz"].clip(lower=5.0)
    out["fwhm_s"] = out["fwhm_s"].clip(lower=0.002)
    out["width10_s"] = out["width10_s"].clip(lower=0.002)
    out["undershoot_hz"] = out["undershoot_hz"].clip(lower=0.0)
    return out


@dataclass(frozen=True, slots=True)
class RidgeConfig:
    """Channels and thresholds of the ridge tracker (normalised units: 1 = 60 dB)."""

    chirp_channel: int = 2
    undershoot_channel: int = 1
    min_delta_f_hz: float = 20.0
    search_hz: tuple[float, float] = (-60.0, 830.0)
    near_baseline_hz: tuple[float, float] = (-125.0, 20.0)


def _parabolic(values: FloatArray, k: int) -> float:
    if 0 < k < values.size - 1:
        a, b, c = values[k - 1], values[k], values[k + 1]
        den = a - 2 * b + c
        if den < 0:
            return k + 0.5 * (a - c) / den
    return float(k)


def _level_extent(f: FloatArray, run: np.ndarray, level: float, dt: float) -> float:
    above = run[f[run] >= level]
    if above.size == 0:
        return dt
    return float(above[-1] - above[0] + 1) * dt


def ridge_estimate(
    crop: npt.NDArray[np.uint8] | FloatArray,
    span_s: float,
    rcfg: RectifiedConfig,
    cfg: RidgeConfig = RidgeConfig(),
) -> dict[str, float]:
    """Ridge-tracking estimate on one rectified crop ``(C, F, T)``.

    Returns NaN for every parameter if no frame lies above ``min_delta_f_hz``.
    """
    raw = np.asarray(crop)
    x = raw.astype(np.float64) / 255.0 if raw.dtype == np.uint8 else raw.astype(np.float64)
    df = rcfg.delta_f_hz
    dt = span_s / (rcfg.n_times - 1)
    band = (df >= cfg.search_hz[0]) & (df <= cfg.search_hz[1])
    ex = x[cfg.chirp_channel][band]
    k = ex.argmax(axis=0)
    f = np.array(
        [df[band][0] + rcfg.df_step_hz * _parabolic(ex[:, t], int(k[t])) for t in range(ex.shape[1])]
    )
    active = f > cfg.min_delta_f_hz
    nan = {p: float("nan") for p in PARAMS}
    if not active.any():
        return nan
    # runs of active frames; pick the one closest to the centre
    edges = np.flatnonzero(np.diff(np.concatenate([[0], active.astype(int), [0]])))
    runs = list(zip(edges[::2], edges[1::2], strict=True))
    centre = (rcfg.n_times - 1) / 2
    start, stop = min(runs, key=lambda r: 0 if r[0] <= centre < r[1] else min(abs(r[0] - centre), abs(r[1] - 1 - centre)))
    run = np.arange(start, stop)
    height = float(f[run].max())
    out = {
        "height_hz": height,
        "fwhm_s": _level_extent(f, run, 0.5 * height, dt),
        "width10_s": _level_extent(f, run, 0.1 * height, dt),
    }
    # undershoot: peak near baseline in the longer window, after vs before the run
    lo, hi = cfg.near_baseline_hz
    near = (df >= lo) & (df <= hi)
    u = x[cfg.undershoot_channel][near]
    fu = np.array([df[near][0] + rcfg.df_step_hz * _parabolic(u[:, t], int(u[:, t].argmax())) for t in range(u.shape[1])])
    before = fu[: max(start, 1)]
    after = fu[stop:]
    ref = float(np.median(before)) if before.size else 0.0
    if after.size >= 3:
        smooth = np.convolve(after, np.ones(3) / 3, mode="valid")
        out["undershoot_hz"] = max(0.0, ref - float(smooth.min()))
    else:
        out["undershoot_hz"] = 0.0
    return out
