"""Ground-truth chirp parameters: the per-EOD-period frequency of the emitter.

This module is the single definition of the targets of the chirp-parameter
decoder. Nothing here is read from a spectrogram.

Per-period frequency
--------------------
The ground truth is the noise-free instantaneous frequency the simulator
integrated for the emitter (`gridtools.sim.render.SceneModel.phase`:
baseline, drift, ramps, shared drift and the placed chirp trace, trapezoid
rule at the scene rate), turned into one value per EOD period. A chirp's
truth is measured on the phase with only that chirp
(``phase(..., chirps=<the emitter's other chirps>, phase="natural")``): a
neighbouring chirp of the same fish that falls inside the analysed window
(its support plus `TRUTH_PAD_S`, e.g. in a burst) would otherwise lend it
its peak or its undershoot.

* cycle boundaries ``t_m`` are the times where the unwrapped phase crosses
  ``2 pi m`` (linear interpolation between samples);
* period ``k`` spans ``[t_k, t_k+1)``, its duration is ``T_k`` and its
  midpoint ``tau_k``; ``1 / T_k`` is the time average of the instantaneous
  frequency over the period (one cycle per period);
* the local baseline of the period is the time average, over the same
  period, of the chirp-free baseline (`SceneModel.base_frequency`:
  baseline + drift + ramps, times the shared drift), so

  ``delta_k = 1 / T_k - mean(f_base over period k)``

  is exactly the period average of the chirp's excursion.

Parameters on ``delta_k``
-------------------------
height_hz
    ``max_k delta_k``: the largest per-period excursion above the local
    baseline (the author's definition: the maximum, also on rippled
    plateaus).
fwhm_s
    Full width at half height: the time between the crossings of
    ``height / 2`` that bracket the peak. The run of periods at or above the
    level is the contiguous one containing the peak; each crossing is
    interpolated linearly between the midpoints of the last period inside
    and the first period outside the run.
width10_s
    The same at 10 % of the height.
undershoot_hz
    ``max(0, -min_{k > peak} delta_k)``: depth of the deepest per-period
    drop below the local baseline after the peak (positive Hz; 0 if the
    frequency never falls below baseline after the peak).

A crossing that is not found inside the analysed trace makes the width a
lower bound; `ChirpParams.edge_clipped` is then True. For scene chirps the
trace spans the chirp's support plus `TRUTH_PAD_S` on each side.

Two properties make the definition exact on simple trajectories (and are
tested): for a frequency that is linear in time over a period, ``1 / T_k``
equals the frequency at ``tau_k``, so on linear flanks the interpolated
crossings are exact; and a plateau of ``H`` Hz lasting at least one whole
period gives ``height_hz = H``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = [
    "PARAMETERS",
    "TRUTH_COLUMNS",
    "TRUTH_PAD_S",
    "TRUTH_VERSION",
    "ChirpParams",
    "chirp_truth",
    "excursion_params",
    "measure_period_params",
    "per_period_trace",
    "scene_truth",
]

FloatArray = npt.NDArray[np.float64]

PARAMETERS: tuple[str, ...] = ("height_hz", "fwhm_s", "width10_s", "undershoot_hz")
"""The decoded parameters, in model order."""

TRUTH_PAD_S = 0.02
"""Time added on both sides of a chirp's support when its truth is measured."""

TRUTH_VERSION = 2
"""Version of the definition (2: each chirp measured without the emitter's
other chirps); sample sets record it and are not reused across versions."""

TRUTH_COLUMNS: tuple[str, ...] = (*PARAMETERS, "edge_clipped", "t_peak_s")
"""Columns `scene_truth` adds to a scene's chirp table."""


@dataclass(frozen=True, slots=True)
class ChirpParams:
    """Parameters of one chirp (see the module docstring).

    Attributes
    ----------
    height_hz, fwhm_s, width10_s, undershoot_hz
        The parameters; NaN if the trace has fewer than three periods or
        never rises above baseline.
    edge_clipped
        A width crossing was not found inside the trace: the width runs to
        the trace's end and is a lower bound.
    t_peak_s
        Midpoint of the period with the largest excursion.
    """

    height_hz: float
    fwhm_s: float
    width10_s: float
    undershoot_hz: float
    edge_clipped: bool = False
    t_peak_s: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        """Field name -> value."""
        return asdict(self)


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
    """Per-EOD-period excursion over the local baseline.

    Parameters
    ----------
    phase_rad
        Unwrapped phase of the fundamental on consecutive samples
        (increasing).
    base_hz
        Chirp-free baseline frequency on the same samples.
    samplerate_hz
        Sample rate.
    t0_s
        Time of the first sample.

    Returns
    -------
    tau_s, delta_hz
        Period midpoints and period-averaged excursions ``1 / T_k -
        mean(base over period k)``, one per complete period inside the
        samples. Empty if fewer than two cycle boundaries lie inside.

    Raises
    ------
    ValueError
        If the arrays are not 1-D of equal length or the phase does not
        increase.
    """
    cycles = np.asarray(phase_rad, dtype=np.float64) / (2.0 * np.pi)
    base = np.asarray(base_hz, dtype=np.float64)
    if cycles.shape != base.shape or cycles.ndim != 1:
        msg = f"phase {cycles.shape} and baseline {base.shape} must be equal 1-D"
        raise ValueError(msg)
    if cycles.size < 2:
        return np.empty(0), np.empty(0)
    if np.any(np.diff(cycles) <= 0):
        msg = "the phase must increase from sample to sample"
        raise ValueError(msg)
    samples = np.arange(cycles.size, dtype=np.float64)
    marks = np.arange(np.ceil(cycles[0]), np.floor(cycles[-1]) + 1)
    if marks.size < 2:
        return np.empty(0), np.empty(0)
    boundary = np.interp(marks, cycles, samples)  # fractional sample per boundary
    period_samples = np.diff(boundary)
    base_cycles = np.interp(boundary, samples, _cumulative_trapezoid(base))
    base_mean = np.diff(base_cycles) / period_samples
    freq = samplerate_hz / period_samples
    tau = t0_s + 0.5 * (boundary[:-1] + boundary[1:]) / samplerate_hz
    return tau, freq - base_mean


def _crossing(t_a: float, d_a: float, t_b: float, d_b: float, level: float) -> float:
    """Time where the line through ``(t_a, d_a)`` and ``(t_b, d_b)`` hits ``level``."""
    if d_b == d_a:
        return 0.5 * (t_a + t_b)
    return t_a + (level - d_a) * (t_b - t_a) / (d_b - d_a)


def _run_width(
    tau: FloatArray, delta: FloatArray, peak: int, level: float
) -> tuple[float, bool]:
    """Width of the run of periods >= ``level`` around ``peak``; clipped flag."""
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
    """Height, FWHM, 10 % width and undershoot of a per-period trace.

    Parameters
    ----------
    tau_s, delta_hz
        Output of `per_period_trace`.

    Returns
    -------
    ChirpParams
        All NaN (and ``edge_clipped``) if there are fewer than three periods
        or the excursion never rises above baseline.
    """
    tau = np.asarray(tau_s, dtype=np.float64)
    delta = np.asarray(delta_hz, dtype=np.float64)
    nan = float("nan")
    if delta.size < 3 or not np.isfinite(delta).all():
        return ChirpParams(nan, nan, nan, nan, True)
    peak = int(np.argmax(delta))
    height = float(delta[peak])
    if height <= 0:
        return ChirpParams(nan, nan, nan, nan, True, float(tau[peak]))
    fwhm, clipped_50 = _run_width(tau, delta, peak, 0.5 * height)
    width10, clipped_10 = _run_width(tau, delta, peak, 0.1 * height)
    after = delta[peak + 1 :]
    undershoot = max(0.0, -float(after.min())) if after.size else 0.0
    return ChirpParams(
        height_hz=height,
        fwhm_s=fwhm,
        width10_s=width10,
        undershoot_hz=undershoot,
        edge_clipped=clipped_50 or clipped_10,
        t_peak_s=float(tau[peak]),
    )


def excursion_params(
    delta_f_hz: npt.ArrayLike, samplerate_hz: float, baseline_hz: float
) -> ChirpParams:
    """Per-period parameters of an excursion trace on a constant baseline.

    For traces measured from recordings (e.g. the harvested chirps of the
    template library): the instantaneous frequency is ``baseline_hz +
    delta_f_hz``, its phase is integrated with the trapezoid rule (as the
    simulator does) and measured with `per_period_trace` and
    `measure_period_params`.

    Parameters
    ----------
    delta_f_hz
        Excursion over the baseline on consecutive samples.
    samplerate_hz
        Sample rate of the trace.
    baseline_hz
        Baseline EOD frequency (sets the period grid).
    """
    freq = float(baseline_hz) + np.asarray(delta_f_hz, dtype=np.float64)
    cycles = _cumulative_trapezoid(freq) / samplerate_hz
    tau, delta = per_period_trace(
        2.0 * np.pi * cycles, np.full(freq.size, float(baseline_hz)), samplerate_hz
    )
    return measure_period_params(tau, delta)


def chirp_truth(
    model: Any,
    emitter_id: int,
    support_s: tuple[float, float],
    *,
    pad_s: float = TRUTH_PAD_S,
    chirp_id: int | None = None,
) -> ChirpParams:
    """Ground truth of one simulated chirp.

    Parameters
    ----------
    model
        The scene's `gridtools.sim.render.SceneModel`
        (`gridtools.sim.scene.load_scene_model`).
    emitter_id
        Fish id of the emitter.
    support_s
        ``(t_support_start_s, t_support_stop_s)`` of the chirp.
    pad_s
        Added on both sides of the support (clipped to the scene).
    chirp_id
        The chirp: the emitter's other chirps are left out of the phase
        (module docstring). None measures the phase with all of them.

    Returns
    -------
    ChirpParams
        Measured on the emitter's per-period frequency (module docstring).
    """
    fs = float(model.samplerate_hz)
    start = max(0, round((support_s[0] - pad_s) * fs))
    stop = min(int(model.n_samples), round((support_s[1] + pad_s) * fs))
    if stop - start < 3:
        nan = float("nan")
        return ChirpParams(nan, nan, nan, nan, True)
    if chirp_id is None:
        phase = model.phase(int(emitter_id), start, stop, chirps="all")
    else:
        others = [
            c for c in model.fish[int(emitter_id)].chirp_ids if c != int(chirp_id)
        ]
        if len(others) == len(model.fish[int(emitter_id)].chirp_ids):
            msg = f"chirp {chirp_id} is not a chirp of fish {emitter_id}"
            raise ValueError(msg)
        phase = model.phase(
            int(emitter_id), start, stop, chirps=others, phase="natural"
        )
    base = model.base_frequency(int(emitter_id), start, stop)
    tau, delta = per_period_trace(phase, base, fs, start / fs)
    return measure_period_params(tau, delta)


def scene_truth(
    scene_dir: str | Path,
    chirps: pd.DataFrame | None = None,
    *,
    pad_s: float = TRUTH_PAD_S,
) -> pd.DataFrame:
    """Ground truth of every chirp of a simulated scene.

    Parameters
    ----------
    scene_dir
        Scene directory written by ``gridtools sim run`` (needs
        ``scene.json``, ``paths.npz`` and ``chirp_traces.npz``).
    chirps
        The scene's ``chirps.csv`` (read if None).
    pad_s
        See `chirp_truth`.

    Returns
    -------
    DataFrame
        ``chirp_id`` and `TRUTH_COLUMNS`, one row per chirp, in the order
        of ``chirps``.
    """
    from gridtools.sim.scene import load_scene_model

    scene_dir = Path(scene_dir)
    if chirps is None:
        chirps = pd.read_csv(scene_dir / "chirps.csv")
    columns: Sequence[str] = ("chirp_id", *TRUTH_COLUMNS)
    if len(chirps) == 0:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in columns})
    model = load_scene_model(scene_dir)
    rows = []
    for chirp_id, emitter, t0, t1 in zip(
        chirps["chirp_id"].to_numpy(dtype=np.int64),
        chirps["emitter_id"].to_numpy(dtype=np.int64),
        chirps["t_support_start_s"].to_numpy(dtype=np.float64),
        chirps["t_support_stop_s"].to_numpy(dtype=np.float64),
        strict=True,
    ):
        params = chirp_truth(
            model,
            int(emitter),
            (float(t0), float(t1)),
            pad_s=pad_s,
            chirp_id=int(chirp_id),
        )
        rows.append({"chirp_id": int(chirp_id), **params.to_dict()})
    return pd.DataFrame(rows).loc[:, list(columns)]
