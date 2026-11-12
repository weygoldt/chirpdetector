"""Ground-truth chirp parameters on the per-EOD-period frequency.

The ground truth is the noise-free instantaneous frequency the simulator
integrated for the emitter (``gridtools.sim.render.SceneModel.phase``:
baseline + drift + ramps + shared drift + the placed chirp traces, trapezoid
rule at the scene rate), turned into one value per EOD period: cycle
boundaries ``t_m`` are the times where the unwrapped phase crosses ``2 pi m``
(linear interpolation between samples), and each period ``[t_m, t_m+1)``
gets the period-averaged frequency ``1 / T_m`` at its midpoint. The local
baseline of that period is the period average of the chirp-free baseline
frequency (baseline + drift at the chirp), so

``delta_k = 1 / T_k - mean(f_base over period k)``

is exactly the period average of the chirp's frequency excursion. Nothing is
taken from a spectrogram.

Definitions on ``delta_k`` (midpoints ``tau_k``):

height_hz
    ``max_k delta_k`` (peak minus local baseline).
fwhm_s
    Time between the two crossings of ``height / 2`` that bracket the peak
    (contiguous run of periods at or above the level; crossing times
    interpolated linearly between period midpoints).
width10_s
    The same at 10 % of the height.
undershoot_hz
    ``max(0, -min_{tau_k > tau_peak} delta_k)``: depth of the deepest trough
    below baseline after the peak (positive Hz).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["ChirpParams", "measure_period_params", "per_period_trace"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ChirpParams:
    """Parameters of one chirp (see the module docstring).

    ``edge_clipped`` is True if a width crossing was not found inside the
    trace (the width then runs to the trace's end and is a lower bound).
    """

    height_hz: float
    fwhm_s: float
    width10_s: float
    undershoot_hz: float
    edge_clipped: bool = False


def _cumulative_trapezoid(values: FloatArray) -> FloatArray:
    out = np.zeros(values.size, dtype=np.float64)
    out[1:] = np.cumsum(0.5 * (values[:-1] + values[1:]))
    return out


def per_period_trace(
    phase_rad: npt.ArrayLike,
    base_hz: npt.ArrayLike,
    samplerate_hz: float,
    t0_s: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Per-EOD-period frequency excursion over the local baseline.

    Parameters
    ----------
    phase_rad
        Unwrapped phase of the fundamental on consecutive samples (strictly
        increasing).
    base_hz
        Chirp-free baseline frequency on the same samples.
    samplerate_hz
        Sample rate.
    t0_s
        Time of the first sample.

    Returns
    -------
    tau_s, delta_hz
        Period midpoints and period-averaged excursions, one per complete
        period inside the span.
    """
    cycles = np.asarray(phase_rad, dtype=np.float64) / (2.0 * np.pi)
    base = np.asarray(base_hz, dtype=np.float64)
    n = cycles.size
    samples = np.arange(n, dtype=np.float64)
    marks = np.arange(np.ceil(cycles[0]), np.floor(cycles[-1]) + 1)
    if marks.size < 2:
        return np.empty(0), np.empty(0)
    idx = np.interp(marks, cycles, samples)  # fractional sample of each boundary
    period_samples = np.diff(idx)
    base_cycles = np.interp(idx, samples, _cumulative_trapezoid(base))
    base_mean = np.diff(base_cycles) / period_samples
    freq = samplerate_hz / period_samples
    tau = t0_s + 0.5 * (idx[:-1] + idx[1:]) / samplerate_hz
    return tau, freq - base_mean


def _crossing(t_a: float, d_a: float, t_b: float, d_b: float, level: float) -> float:
    if d_b == d_a:
        return 0.5 * (t_a + t_b)
    return t_a + (level - d_a) * (t_b - t_a) / (d_b - d_a)


def _run_width(
    tau: FloatArray, delta: FloatArray, peak: int, level: float
) -> tuple[float, bool]:
    below = delta < level
    left = np.flatnonzero(below[:peak])
    right = np.flatnonzero(below[peak:])
    clipped = False
    if left.size:
        i = int(left[-1])
        t_on = _crossing(tau[i], delta[i], tau[i + 1], delta[i + 1], level)
    else:
        t_on, clipped = float(tau[0]), True
    if right.size:
        j = peak + int(right[0])
        t_off = _crossing(tau[j - 1], delta[j - 1], tau[j], delta[j], level)
    else:
        t_off, clipped = float(tau[-1]), True
    return t_off - t_on, clipped


def measure_period_params(tau_s: npt.ArrayLike, delta_hz: npt.ArrayLike) -> ChirpParams:
    """Height, FWHM, 10 % width and undershoot of a per-period trace."""
    tau = np.asarray(tau_s, dtype=np.float64)
    delta = np.asarray(delta_hz, dtype=np.float64)
    if delta.size < 3:
        nan = float("nan")
        return ChirpParams(nan, nan, nan, nan, True)
    peak = int(np.argmax(delta))
    height = float(delta[peak])
    fwhm, c50 = _run_width(tau, delta, peak, 0.5 * height)
    w10, c10 = _run_width(tau, delta, peak, 0.1 * height)
    after = delta[peak + 1 :]
    undershoot = max(0.0, -float(after.min())) if after.size else 0.0
    return ChirpParams(height, fwhm, w10, undershoot, c50 or c10)
