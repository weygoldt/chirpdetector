"""Calibration: monotone bias maps, their inverse, sigma coverage, prior robustness."""

from __future__ import annotations

import itertools

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


def test_a_point_mass_is_not_spread_into_a_mode() -> None:
    """Undershoot: many chirps have none. Below `fit_from` the map is a shift."""
    rng = np.random.default_rng(4)
    n = 20_000
    undershoot = np.where(rng.random(n) < 0.4, 0.0, rng.uniform(1.0, 30.0, n))
    true = np.tile(np.log(undershoot + 1.0)[:, None], (1, len(PARAMETERS)))
    pred = true + 0.4 * np.exp(-true) + rng.normal(0, 0.15, true.shape)
    sigma = np.full(true.shape, 0.15)
    j = PARAMETERS.index("undershoot_hz")
    zero = undershoot == 0
    for fit_from, spread_kept in (({"undershoot_hz": 1.0}, True), ({}, False)):
        config = CalibrationConfig(fit_from=fit_from)
        cal, _ = fit_calibration(true, pred, sigma, config).apply(pred, sigma)
        iqr_raw = np.subtract(*np.quantile(pred[zero, j], [0.75, 0.25]))
        iqr_cal = np.subtract(*np.quantile(cal[zero, j], [0.75, 0.25]))
        assert bool(iqr_cal == pytest.approx(iqr_raw, rel=0.05)) is spread_kept
    with pytest.raises(ValueError, match="unknown parameters"):
        CalibrationConfig(fit_from={"depth_hz": 1.0})


def test_bias_curve_ignores_the_noise_of_small_bins() -> None:
    """2,000 val chirps in 40 bins: the curve must follow the shrinkage, not
    the noise of each ~50-chirp median (its inverse would amplify it)."""
    rng = np.random.default_rng(5)
    n = 2000
    y = rng.uniform(3.0, 6.0, n)
    # shrinkage toward 4.5 that saturates at the ends (as a posterior mean)
    mean = 4.5 + 1.9 * np.tanh((y - 4.5) / 2.0)
    sigma = rng.uniform(0.05, 0.25, n)  # heteroscedastic, known per chirp
    pred = mean + sigma * rng.normal(size=n)
    noisy = fit_bias_map(
        y, pred, n_bins=40, min_count=50, min_slope=0.2, smooth_fraction=0
    )
    smooth = fit_bias_map(y, pred)  # defaults: smoothed, slope >= 0.5
    slopes = np.diff(smooth.y) / np.diff(smooth.x)
    true_slopes = np.diff(4.5 + 1.9 * np.tanh((np.asarray(smooth.x) - 4.5) / 2.0))
    true_slopes /= np.diff(smooth.x)
    assert np.max(np.abs(slopes - true_slopes)) < 0.3
    noisy_slopes = np.diff(noisy.y) / np.diff(noisy.x)
    assert np.ptp(noisy_slopes) > 2 * np.ptp(slopes)
    # median-unbiased per decile on fresh chirps, also at the ends
    y2 = rng.uniform(3.0, 6.0, 40_000)
    s2 = rng.uniform(0.05, 0.25, y2.size)
    p2 = 4.5 + 1.9 * np.tanh((y2 - 4.5) / 2.0) + s2 * rng.normal(size=y2.size)
    cal = smooth.inverse(p2)
    edges = np.quantile(y2, np.linspace(0, 1, 11))
    for lo, hi in itertools.pairwise(edges):
        sel = (y2 >= lo) & (y2 <= hi)
        assert abs(np.median(cal[sel] - y2[sel])) < 0.03
    # the propagated sigma keeps its rank agreement with the error
    from scipy.stats import spearmanr

    def rank(h: MonotoneMap) -> float:
        c = h.inverse(p2)
        return float(np.asarray(spearmanr(s2 / h.slope(c), np.abs(c - y2))[0]))

    raw_error = np.abs(p2 - (4.5 + 1.9 * np.tanh((y2 - 4.5) / 2.0)))
    raw_rank = float(np.asarray(spearmanr(s2, raw_error)[0]))
    assert rank(smooth) > rank(noisy) + 0.05
    assert rank(smooth) > 0.8 * raw_rank


def test_curves_per_sigma_stratum_remove_visibility_dependent_shrinkage() -> None:
    """Poorly seen chirps (large sigma) shrink more. One pooled curve leaves
    them under-corrected and the well seen ones over-corrected; a curve per
    sigma tercile, chosen by the decoder's own sigma, does not."""
    rng = np.random.default_rng(6)

    def simulate(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = rng.uniform(3.0, 6.0, n)
        sigma = np.exp(rng.uniform(np.log(0.04), np.log(0.35), n))
        slope = 1.0 - sigma  # 0.96 for the best seen chirps, 0.65 for the worst
        pred = 4.5 + slope * (y - 4.5) + sigma * rng.normal(size=n)
        columns = (np.tile(v[:, None], (1, len(PARAMETERS))) for v in (y, pred, sigma))
        return tuple(columns)  # type: ignore[return-value]

    y, pred, sigma = simulate(6000)
    pooled = fit_calibration(y, pred, sigma, CalibrationConfig(sigma_strata=1))
    strata = fit_calibration(y, pred, sigma, CalibrationConfig())
    assert len(strata.stratum_maps["height_hz"]) == 3
    assert strata.fitted_on["sigma_strata"]["height_hz"] == 3
    y2, pred2, sigma2 = simulate(60_000)
    small = y2[:, 0] < 3.5
    sharp, blurred = sigma2[:, 0] < 0.07, sigma2[:, 0] > 0.2

    def biases(cal: ParamCalibration) -> tuple[float, float]:
        err = cal.apply(pred2, sigma2)[0][:, 0] - y2[:, 0]
        return (
            float(np.median(err[small & sharp])),
            float(np.median(err[small & blurred])),
        )

    pooled_sharp, pooled_blurred = biases(pooled)
    assert pooled_sharp < -0.05  # over-corrected
    assert pooled_blurred > 0.1  # under-corrected
    sharp_bias, blurred_bias = biases(strata)
    assert abs(sharp_bias) < 0.5 * abs(pooled_sharp)
    assert abs(blurred_bias) < 0.5 * abs(pooled_blurred)
    restored = ParamCalibration.from_dict(strata.to_dict())
    np.testing.assert_allclose(
        restored.apply(pred2[:100], sigma2[:100])[0],
        strata.apply(pred2[:100], sigma2[:100])[0],
    )
    # a bundle written before strata existed still loads (pooled curve)
    old = strata.to_dict()
    old.pop("sigma_edges")
    old.pop("stratum_maps")
    legacy = ParamCalibration.from_dict(old)
    np.testing.assert_allclose(
        legacy.apply(pred2[:100], sigma2[:100])[0][:, 0],
        strata.maps["height_hz"].inverse(pred2[:100, 0]),
    )
    # too few chirps for three strata: fewer
    few = fit_calibration(y[:300], pred[:300], sigma[:300], CalibrationConfig())
    assert few.fitted_on["sigma_strata"]["height_hz"] < 3
