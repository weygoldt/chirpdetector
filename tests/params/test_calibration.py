"""Calibration: monotone bias maps, their inverse, sigma coverage, prior robustness."""

from __future__ import annotations

import numpy as np
import pytest

from chirpdetector.params.calibration import (
    CalibrationConfig,
    MonotoneMap,
    ParamCalibration,
    fit_bias_map,
    fit_calibration,
)
from chirpdetector.params.truth import PARAMETERS


def test_monotone_map_inverts_and_extrapolates_with_slope_one() -> None:
    h = MonotoneMap((0.0, 1.0, 3.0), (0.5, 1.0, 2.0))
    x = np.linspace(-2, 5, 71)
    np.testing.assert_allclose(h.inverse(h(x)), x, atol=1e-12)
    assert h(-1.0) == pytest.approx(-0.5)  # 0.5 + (-1 - 0)
    assert h(4.0) == pytest.approx(3.0)  # 2.0 + (4 - 3)
    np.testing.assert_allclose(
        h.slope(np.array([-1.0, 0.5, 2.0, 4.0])), [1, 0.5, 0.5, 1]
    )
    assert np.all(np.diff(h(x)) > 0)
    identity = MonotoneMap()
    np.testing.assert_array_equal(identity(x), x)
    with pytest.raises(ValueError, match="strictly increasing"):
        MonotoneMap((0.0, 1.0), (1.0, 1.0))
    assert MonotoneMap.from_dict(h.to_dict()) == h


def test_bias_map_removes_shrinkage() -> None:
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 5, 20_000)
    pred = 0.8 * y + 0.5 + rng.normal(0, 0.2, y.size)  # regression to the mean
    h = fit_bias_map(y, pred, n_bins=40, min_count=50)
    assert np.all(np.diff(h.y) > 0)
    calibrated = h.inverse(pred)
    slope_raw = np.polyfit(y, pred, 1)[0]
    slope_cal = np.polyfit(y, calibrated, 1)[0]
    assert slope_raw == pytest.approx(0.8, abs=0.01)
    assert slope_cal == pytest.approx(1.0, abs=0.02)
    # median-unbiased in every bin of the true value
    for lo in np.arange(0.5, 4.5, 0.5):
        sel = (y >= lo) & (y < lo + 0.5)
        assert np.median(calibrated[sel] - y[sel]) == pytest.approx(0.0, abs=0.03)


def test_bias_map_is_monotone_with_a_minimum_slope() -> None:
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 1, 5000)
    pred = np.where(y < 0.5, 0.3, 0.3 + y - 0.5) + rng.normal(0, 0.01, y.size)
    h = fit_bias_map(y, pred, n_bins=20, min_count=50, min_slope=0.2)
    slopes = np.diff(h.y) / np.diff(h.x)
    assert slopes.min() >= 0.2 - 1e-9
    assert fit_bias_map(y[:10], pred[:10], min_count=50) == MonotoneMap()


def test_calibration_is_robust_to_the_population_prior() -> None:
    """Fitted on a flat population, unbiased on a skewed one.

    The predictions shrink toward the training mean (as a posterior mean
    does). The inverse bias map depends on the measurement given the true
    value, not on how often each value occurs, so it stays unbiased on a
    population dominated by small values; regressing the truth on the
    prediction (fitted on the flat population) does not.
    """
    rng = np.random.default_rng(2)

    def measure(y: np.ndarray) -> np.ndarray:
        return 0.75 * y + 0.25 * 2.5 + rng.normal(0, 0.3, y.size)

    flat = rng.uniform(0, 5, 40_000)
    h = fit_bias_map(flat, measure(flat))
    skewed = rng.exponential(0.8, 40_000)
    skewed = skewed[skewed < 5]
    pred = measure(skewed)
    inverse = h.inverse(pred)
    forward = np.polyfit(measure(flat), flat, 1)  # E[y | pred] on the flat set
    regressed = np.polyval(forward, pred)
    small = skewed < 1.0
    assert np.median(inverse[small] - skewed[small]) == pytest.approx(0.0, abs=0.05)
    assert np.median(regressed[small] - skewed[small]) > 0.1


def test_sigma_is_rescaled_to_68_and_95_percent() -> None:
    rng = np.random.default_rng(3)
    n = 20_000
    true = rng.uniform(1, 5, (n, len(PARAMETERS)))
    sigma = rng.uniform(0.05, 0.3, (n, len(PARAMETERS)))
    mean = true + rng.normal(0, 1, true.shape) * sigma * 1.7  # sigmas 1.7x too small
    cal = fit_calibration(true, mean, sigma, CalibrationConfig(method="none"))
    _, cal_sigma = cal.apply(mean, sigma)
    z = np.abs(mean - true) / cal_sigma
    for j, p in enumerate(PARAMETERS):
        assert cal.sigma_scale[p] == pytest.approx(1.7, rel=0.05)
        assert np.mean(z[:, j] <= 1) == pytest.approx(0.683, abs=0.01)
        assert np.mean(z[:, j] <= cal.k95[p]) == pytest.approx(0.95, abs=0.01)
        assert cal.k95[p] == pytest.approx(1.96, abs=0.1)
    assert ParamCalibration.from_dict(cal.to_dict()).sigma_scale == cal.sigma_scale


def test_sigma_follows_the_inverse_map() -> None:
    """A map with slope 0.5 doubles the sigma of the calibrated estimate."""
    maps = {p: MonotoneMap((0.0, 10.0), (0.0, 5.0)) for p in PARAMETERS}
    cal = ParamCalibration(method="inverse-median", maps=maps)
    mean = np.full((3, len(PARAMETERS)), 2.0)
    sigma = np.full((3, len(PARAMETERS)), 0.1)
    m, s = cal.apply(mean, sigma)
    np.testing.assert_allclose(m, 4.0)
    np.testing.assert_allclose(s, 0.2)


def test_config_validates() -> None:
    with pytest.raises(ValueError, match="method"):
        CalibrationConfig(method="isotonic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_slope"):
        CalibrationConfig(min_slope=0.0)
