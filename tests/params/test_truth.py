"""Ground truth on synthetic phase trajectories with known parameters."""

from __future__ import annotations

import numpy as np
import pytest

from chirpdetector.params.truth import (
    excursion_params,
    measure_period_params,
    per_period_trace,
)

FS = 20_000.0


def _phase(freq_hz: np.ndarray) -> np.ndarray:
    """Trapezoid-rule phase of an instantaneous frequency (as the simulator)."""
    cycles = np.zeros(freq_hz.size)
    cycles[1:] = np.cumsum(0.5 * (freq_hz[:-1] + freq_hz[1:])) / FS
    return 2 * np.pi * cycles


def _trapezoid(
    t: np.ndarray, t0: float, rise: float, top: float, fall: float
) -> np.ndarray:
    """1 on ``[t0 + rise, t0 + rise + top]``, linear ramps of ``rise``/``fall``."""
    up = np.clip((t - t0) / rise, 0, 1)
    down = np.clip((t0 + rise + top + fall - t) / fall, 0, 1)
    return np.minimum(up, down)


def test_constant_frequency_has_zero_excursion() -> None:
    base = np.full(4000, 800.0)
    tau, delta = per_period_trace(_phase(base), base, FS)
    assert tau.size == pytest.approx(4000 / FS * 800, abs=1)
    np.testing.assert_allclose(delta, 0.0, atol=1e-6)


def test_drifting_baseline_is_removed_period_by_period() -> None:
    t = np.arange(8000) / FS
    base = 700.0 + 50.0 * t + 20.0 * np.sin(2 * np.pi * 3 * t)
    _, delta = per_period_trace(_phase(base), base, FS)
    np.testing.assert_allclose(delta, 0.0, atol=1e-6)


def test_trapezoid_chirp_has_exact_parameters() -> None:
    """Linear flanks and flat plateaus: the per-period definition is exact.

    On a linear flank, 1 / T_k equals the frequency at the period's midpoint,
    so crossings interpolated between midpoints are exact; plateaus longer
    than a period give the exact height and undershoot. Every crossing lies
    at least 1.5 periods (2 ms at 700 Hz) away from a kink, so that both
    periods that bracket it lie on one line.
    """
    t = np.arange(int(0.3 * FS)) / FS
    height, undershoot = 240.0, 12.0
    rise, top, fall = 0.030, 0.006, 0.040
    chirp = height * _trapezoid(t, 0.05, rise, top, fall)
    trough = -undershoot * _trapezoid(
        t, 0.05 + rise + top + fall + 0.004, 0.01, 0.02, 0.03
    )
    base = 700.0 + 30.0 * t  # a drifting baseline must not matter
    tau, delta = per_period_trace(_phase(base + chirp + trough), base, FS)
    p = measure_period_params(tau, delta)
    assert p.height_hz == pytest.approx(height, rel=1e-6)
    # level crossings on the flanks: fraction q of the height on a ramp
    for q, width in ((0.5, p.fwhm_s), (0.1, p.width10_s)):
        expected = top + (1 - q) * rise + (1 - q) * fall
        assert width == pytest.approx(expected, abs=2e-6)
    assert p.undershoot_hz == pytest.approx(undershoot, rel=1e-6)
    assert not p.edge_clipped
    assert 0.05 + rise <= p.t_peak_s <= 0.05 + rise + top


def test_gaussian_chirp_is_period_averaged() -> None:
    t = np.arange(int(0.2 * FS)) / FS
    sigma = 0.01
    excursion = 300.0 * np.exp(-0.5 * ((t - 0.1) / sigma) ** 2)
    trough = -10.0 * np.exp(-0.5 * ((t - 0.15) / 0.01) ** 2)
    base = np.full(t.size, 900.0)
    tau, delta = per_period_trace(_phase(base + excursion + trough), base, FS)
    p = measure_period_params(tau, delta)
    assert p.height_hz < 300.0  # the peak period averages over its flanks
    assert p.height_hz == pytest.approx(300.0, rel=0.01)
    assert p.fwhm_s == pytest.approx(2 * np.sqrt(2 * np.log(2)) * sigma, rel=0.02)
    assert p.width10_s == pytest.approx(2 * np.sqrt(2 * np.log(10)) * sigma, rel=0.02)
    assert p.undershoot_hz == pytest.approx(10.0, rel=0.05)


def test_no_undershoot_is_zero() -> None:
    t = np.arange(int(0.2 * FS)) / FS
    chirp = 100.0 * _trapezoid(t, 0.05, 0.01, 0.01, 0.01)
    base = np.full(t.size, 650.0)
    p = measure_period_params(*per_period_trace(_phase(base + chirp), base, FS))
    assert p.undershoot_hz == pytest.approx(0.0, abs=1e-6)


def test_a_cut_chirp_is_edge_clipped() -> None:
    t = np.arange(int(0.2 * FS)) / FS
    chirp = 100.0 * _trapezoid(t, 0.05, 0.01, 0.01, 0.01)
    base = np.full(t.size, 650.0)
    phase = _phase(base + chirp)
    cut = int(0.075 * FS)  # inside the falling flank
    p = measure_period_params(*per_period_trace(phase[:cut], base[:cut], FS))
    assert p.edge_clipped
    assert p.width10_s < 0.01 + 0.01 + 0.009


def test_degenerate_traces_are_nan() -> None:
    base = np.full(30, 800.0)
    p = measure_period_params(*per_period_trace(_phase(base), base, FS))
    assert np.isnan(p.height_hz)
    assert p.edge_clipped
    with pytest.raises(ValueError, match="increase"):
        per_period_trace(np.zeros(10), np.zeros(10), FS)


def test_excursion_params_of_a_trace() -> None:
    t = np.arange(int(0.3 * FS)) / FS
    trace = 180.0 * _trapezoid(t, 0.1, 0.02, 0.01, 0.02)
    p = excursion_params(trace, FS, 750.0)
    assert p.height_hz == pytest.approx(180.0, rel=1e-6)
    assert p.fwhm_s == pytest.approx(0.01 + 0.01 + 0.01, abs=2e-6)
