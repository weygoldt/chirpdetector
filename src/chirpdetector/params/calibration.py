"""Post-hoc calibration of the decoder: remove its shrinkage, rescale its sigma.

The problem
-----------
A regressor trained to minimise a (Gaussian) loss estimates something
close to the *posterior* mean of a target under the training population:
it shrinks toward the middle of the training distribution. The
feasibility study measured this as a signed error that falls with the
true value (Spearman -0.17 height, -0.23 FWHM, -0.28 undershoot). Such an
estimate has two properties that matter for a population analysis
(manuscript figure ``chirp_space``):

* its distribution is narrower than the true one (differences between
  groups of chirps are compressed), and
* how much it shrinks depends on the *training* population. The
  training chirps were drawn from an equalised template mix (many big and
  long chirps); real recordings are dominated by small chirps.

Choice: inverse (classical) calibration, not regression of true on predicted
-----------------------------------------------------------------------------
Regressing the true value on the prediction (e.g. isotonic ``E[y | yhat]``)
gives the best estimate *for the population it was fitted on*: it is
again a posterior mean, so it keeps (or even sharpens) the shrinkage
and inherits the val population's prior. It does not change a
well-trained regressor much, because ``E[y | yhat] = yhat`` is what such a
regressor already approximates.

`fit_calibration` fits the opposite direction, per parameter in model
space (logs): ``h(y) = median(yhat | y)``, the bias curve, as a monotone
increasing, piecewise-linear map through the medians of quantile bins of
the true value, slope 1 beyond the outermost bins. The calibrated
estimate is ``h^-1(yhat)``, so that
``median(h^-1(yhat) | y) = y`` for every ``y``: the estimate is
conditionally median-unbiased. ``h`` is a property of the measurement
given the true value; it does not depend on how often each true value
occurs in the calibration set (that only sets how precisely it is
known), so the calibrated estimate keeps its meaning on a population
with a different parameter distribution, such as the natural one. Its
distribution over a population is the true distribution blurred by the
measurement error (slightly *wider*, by a few per cent of the parameter),
not compressed toward the training population.

Fitting the curve
-----------------
1. Quantile bins of the true value (``n_bins``, at least ``min_count``
   chirps each); knots at the bins' median true value and median estimate.
2. **Smoothing**: each knot's estimate is replaced by a weighted local
   linear fit through it and its neighbours (``smooth_knots`` knots,
   weights = bin sizes; the window is cut at the ends, where a local
   linear fit, unlike a running mean, has no boundary bias, so the
   strongest shrinkage at the smallest and largest values is kept).
3. Pooled-adjacent-violators makes it monotone, and every slope is at least
   ``min_slope`` (0.5: the inverse amplifies an error at most twofold).

Without step 2 the curve follows the sampling noise of ~50-chirp bin
medians: on the first plumbing decoder (2,087 val chirps, 40 bins) the
slopes between adjacent knots ranged from the 0.2 floor to 3. Its inverse
then stretched some intervals fivefold and compressed others, which
widened per-chirp errors, put ripples into the calibrated distributions,
and, through the slope in the sigma propagation, made the calibrated sigma
far less informative about the error than the raw one (Spearman of sigma
and |error| 0.42 -> 0.22 for height, 0.35 -> 0.07 for undershoot) and its
coverage uneven across sigma quintiles. With the smoothing the same
decoder keeps the decile-wise bias at most ~2 % (height) while those
side effects largely disappear.

Uncertainty
-----------
The network's sigma (model space) is propagated through ``h^-1``
(divided by the local slope of ``h``) and multiplied by one factor per
parameter, fitted on the same val samples so that ``|error| <= sigma``
holds for 68.3 % of them. ``k95`` records the multiple of sigma that
covers 95 % on val (1.96 for Gaussian errors; larger with heavy tails).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import isotonic_regression

from chirpdetector.params.truth import PARAMETERS

__all__ = [
    "CalibrationConfig",
    "MonotoneMap",
    "ParamCalibration",
    "fit_bias_map",
    "fit_calibration",
]

FloatArray = npt.NDArray[np.float64]
Method = Literal["inverse-median", "none"]
_COVERAGE_68 = 0.682689492137


@dataclass(frozen=True)
class CalibrationConfig:
    """How the calibration is fitted.

    Parameters
    ----------
    method
        ``"inverse-median"`` (module docstring) or ``"none"`` (identity
        maps; sigma is still rescaled).
    n_bins
        Quantile bins of the true value.
    min_count
        Fewer bins are used if a bin would hold fewer samples.
    min_slope
        Smallest slope of the bias curve ``h`` (its inverse amplifies
        errors by at most ``1 / min_slope``).
    smooth_knots
        Knots in the local linear smoothing of the bin medians (odd; 1: no
        smoothing). See "Fitting the curve".
    flag_quantile
        Calibrated sigma above this quantile of the val sigmas flags a
        chirp as ``uncertain`` (`chirpdetector.params.inference`).
    fit_from
        Per parameter, the smallest true value (natural units) the bias
        curve is fitted on; below its first knot the curve continues with
        slope 1. The default excludes undershoots below 1 Hz: many chirps
        have none (a point mass at 0), and inverting a curve through a
        point mass would spread those chirps into an artificial mode.
    """

    method: Method = "inverse-median"
    n_bins: int = 40
    min_count: int = 50
    min_slope: float = 0.5
    smooth_knots: int = 11
    flag_quantile: float = 0.995
    fit_from: Mapping[str, float] = field(
        default_factory=lambda: {"undershoot_hz": 1.0}
    )

    def __post_init__(self) -> None:
        unknown = sorted(set(self.fit_from) - set(PARAMETERS))
        if unknown:
            msg = f"calibration.fit_from: unknown parameters {unknown}"
            raise ValueError(msg)
        object.__setattr__(
            self, "fit_from", {k: float(v) for k, v in self.fit_from.items()}
        )
        if self.method not in ("inverse-median", "none"):
            msg = f"calibration.method must be 'inverse-median' or 'none', got {self.method!r}"
            raise ValueError(msg)
        if self.n_bins < 2 or self.min_count < 2:
            msg = "calibration.n_bins and min_count must be >= 2"
            raise ValueError(msg)
        if self.smooth_knots < 1 or self.smooth_knots % 2 != 1:
            msg = f"calibration.smooth_knots must be an odd number >= 1, got {self.smooth_knots}"
            raise ValueError(msg)
        if not 0 < self.min_slope <= 1:
            msg = f"calibration.min_slope must lie in (0, 1], got {self.min_slope}"
            raise ValueError(msg)
        if not 0.5 < self.flag_quantile <= 1:
            msg = f"calibration.flag_quantile must lie in (0.5, 1], got {self.flag_quantile}"
            raise ValueError(msg)


@dataclass(frozen=True)
class MonotoneMap:
    """A strictly increasing piecewise-linear map with slope 1 outside its knots.

    Attributes
    ----------
    x, y
        Knots, both strictly increasing. One knot (or none) is a shift
        (or the identity).
    """

    x: tuple[float, ...] = ()
    y: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        x, y = np.asarray(self.x, float), np.asarray(self.y, float)
        if x.shape != y.shape or x.ndim != 1:
            msg = "x and y must be 1-D of equal length"
            raise ValueError(msg)
        if x.size > 1 and (np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0)):
            msg = "knots must be strictly increasing in x and y"
            raise ValueError(msg)

    @staticmethod
    def _apply(v: npt.ArrayLike, a: FloatArray, b: FloatArray) -> FloatArray:
        values = np.asarray(v, dtype=np.float64)
        flat = values.reshape(-1)
        if a.size == 0:
            return values.copy()
        out = np.interp(flat, a, b)
        left, right = flat < a[0], flat > a[-1]
        out[left] = b[0] + (flat[left] - a[0])
        out[right] = b[-1] + (flat[right] - a[-1])
        return out.reshape(values.shape)

    def __call__(self, v: npt.ArrayLike) -> FloatArray:
        """``h(v)``."""
        return self._apply(v, np.asarray(self.x, float), np.asarray(self.y, float))

    def inverse(self, v: npt.ArrayLike) -> FloatArray:
        """``h^-1(v)``."""
        return self._apply(v, np.asarray(self.y, float), np.asarray(self.x, float))

    def slope(self, v: npt.ArrayLike) -> FloatArray:
        """``h'(v)`` (1 outside the knots; the right-hand slope at a knot)."""
        values = np.asarray(v, dtype=np.float64)
        flat = values.reshape(-1)
        x, y = np.asarray(self.x, float), np.asarray(self.y, float)
        out = np.ones_like(flat)
        if x.size >= 2:
            slopes = np.diff(y) / np.diff(x)
            inside = (flat >= x[0]) & (flat < x[-1])
            k = np.searchsorted(x, flat[inside], side="right") - 1
            out[inside] = slopes[np.clip(k, 0, slopes.size - 1)]
        return out.reshape(values.shape)

    def to_dict(self) -> dict[str, list[float]]:
        """JSON form."""
        return {"x": list(self.x), "y": list(self.y)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Sequence[float]]) -> MonotoneMap:
        """Inverse of `to_dict`."""
        return cls(
            tuple(float(v) for v in data["x"]), tuple(float(v) for v in data["y"])
        )


def _local_linear(
    x: FloatArray, y: FloatArray, weights: FloatArray, window: int
) -> FloatArray:
    """Each ``y[k]`` replaced by a weighted linear fit through ``window`` knots
    around ``k`` (cut at the ends), evaluated at ``x[k]``."""
    half = window // 2
    out = y.copy()
    for k in range(x.size):
        lo, hi = max(0, k - half), min(x.size, k + half + 1)
        if hi - lo < 3:
            continue
        dx = x[lo:hi] - x[k]
        w = weights[lo:hi]
        design = np.stack([np.ones_like(dx), dx], axis=1) * np.sqrt(w)[:, None]
        coef = np.linalg.lstsq(design, y[lo:hi] * np.sqrt(w), rcond=None)[0]
        out[k] = coef[0]
    return out


def fit_bias_map(
    true: npt.ArrayLike,
    pred: npt.ArrayLike,
    *,
    n_bins: int = 40,
    min_count: int = 50,
    min_slope: float = 0.5,
    smooth_knots: int = 11,
) -> MonotoneMap:
    """Monotone bias curve ``h(y) = median(pred | true = y)`` (module docstring).

    Parameters
    ----------
    true, pred
        True values and predictions (model space); non-finite pairs are
        ignored.
    n_bins
        Quantile bins of ``true`` (fewer if a bin would hold fewer than
        ``min_count`` pairs).
    min_count
        See above.
    min_slope
        Smallest slope of the curve.
    smooth_knots
        Knots in the local linear smoothing of the bin medians (1: none).

    Returns
    -------
    MonotoneMap
        Knots at the bins' median true values. With fewer than
        ``2 * min_count`` pairs: the identity.
    """
    t = np.asarray(true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]
    bins = min(n_bins, t.size // min_count)
    if bins < 2:
        return MonotoneMap()
    order = np.argsort(t, kind="stable")
    groups = np.array_split(order, bins)
    x = np.array([np.median(t[g]) for g in groups])
    y = np.array([np.median(p[g]) for g in groups])
    counts = np.array([g.size for g in groups], dtype=np.float64)
    keep = np.concatenate([[True], np.diff(x) > 0])  # ties in x: merge
    if not keep.all():
        x, y, counts = x[keep], y[keep], counts[keep]
    if smooth_knots > 1:
        y = _local_linear(x, y, counts, smooth_knots)
    y = isotonic_regression(y, weights=counts, increasing=True).x
    for k in range(1, x.size):  # minimum slope, so that h can be inverted
        y[k] = max(y[k], y[k - 1] + min_slope * (x[k] - x[k - 1]))
    return MonotoneMap(tuple(float(v) for v in x), tuple(float(v) for v in y))


@dataclass(frozen=True)
class ParamCalibration:
    """Per-parameter bias maps and sigma factors (module docstring).

    Attributes
    ----------
    method
        How the maps were fitted.
    maps
        Parameter -> bias curve ``h`` in model space.
    sigma_scale
        Parameter -> factor on the propagated sigma.
    k95
        Parameter -> multiple of the calibrated sigma covering 95 % on val.
    fitted_on
        Provenance (split, number of samples per parameter, selection).
    """

    method: str = "none"
    maps: Mapping[str, MonotoneMap] = field(default_factory=dict)
    sigma_scale: Mapping[str, float] = field(default_factory=dict)
    k95: Mapping[str, float] = field(default_factory=dict)
    fitted_on: Mapping[str, Any] = field(default_factory=dict)

    def apply(
        self, mean_log: npt.ArrayLike, sigma_log: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        """Calibrated mean and sigma (model space), ``(n, 4)`` each."""
        mean = np.asarray(mean_log, dtype=np.float64).reshape(-1, len(PARAMETERS))
        sigma = np.asarray(sigma_log, dtype=np.float64).reshape(mean.shape)
        out_m, out_s = mean.copy(), sigma.copy()
        for j, name in enumerate(PARAMETERS):
            h = self.maps.get(name, MonotoneMap())
            out_m[:, j] = h.inverse(mean[:, j])
            out_s[:, j] = (
                sigma[:, j]
                / h.slope(out_m[:, j])
                * float(self.sigma_scale.get(name, 1.0))
            )
        return out_m, out_s

    def to_dict(self) -> dict[str, Any]:
        """JSON form."""
        return {
            "method": self.method,
            "maps": {k: v.to_dict() for k, v in self.maps.items()},
            "sigma_scale": dict(self.sigma_scale),
            "k95": dict(self.k95),
            "fitted_on": dict(self.fitted_on),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ParamCalibration:
        """Inverse of `to_dict`."""
        return cls(
            method=str(data.get("method", "none")),
            maps={k: MonotoneMap.from_dict(v) for k, v in data.get("maps", {}).items()},
            sigma_scale={k: float(v) for k, v in data.get("sigma_scale", {}).items()},
            k95={k: float(v) for k, v in data.get("k95", {}).items()},
            fitted_on=dict(data.get("fitted_on", {})),
        )


def fit_calibration(
    true_log: npt.ArrayLike,
    mean_log: npt.ArrayLike,
    sigma_log: npt.ArrayLike,
    config: CalibrationConfig | None = None,
    *,
    fitted_on: Mapping[str, Any] | None = None,
) -> ParamCalibration:
    """Fit bias maps and sigma factors on validation predictions.

    Parameters
    ----------
    true_log, mean_log, sigma_log
        ``(n, 4)`` true values, predicted means and sigmas, model space
        (`chirpdetector.params.model.to_model_space`), `PARAMETERS` order.
        Rows with a non-finite true value are ignored per parameter.
    config
        Fitting options.
    fitted_on
        Provenance stored with the result.

    Returns
    -------
    ParamCalibration
    """
    config = config or CalibrationConfig()
    y = np.asarray(true_log, dtype=np.float64).reshape(-1, len(PARAMETERS))
    m = np.asarray(mean_log, dtype=np.float64).reshape(y.shape)
    s = np.asarray(sigma_log, dtype=np.float64).reshape(y.shape)
    from chirpdetector.params.model import LOG_OFFSETS

    maps: dict[str, MonotoneMap] = {}
    for j, name in enumerate(PARAMETERS):
        if config.method == "inverse-median":
            lowest = config.fit_from.get(name)
            use = np.ones(len(y), dtype=bool)
            if lowest is not None:
                use = y[:, j] >= np.log(lowest + LOG_OFFSETS[name])
            maps[name] = fit_bias_map(
                y[use, j],
                m[use, j],
                n_bins=config.n_bins,
                min_count=config.min_count,
                min_slope=config.min_slope,
                smooth_knots=config.smooth_knots,
            )
        else:
            maps[name] = MonotoneMap()
    raw = ParamCalibration(method=config.method, maps=maps)
    cal_m, cal_s = raw.apply(m, s)
    scale, k95, counts = {}, {}, {}
    for j, name in enumerate(PARAMETERS):
        ok = np.isfinite(y[:, j]) & np.isfinite(cal_m[:, j]) & (cal_s[:, j] > 0)
        counts[name] = int(ok.sum())
        if ok.sum() < 2:
            scale[name], k95[name] = 1.0, 1.96
            continue
        z = np.abs(cal_m[ok, j] - y[ok, j]) / cal_s[ok, j]
        scale[name] = float(np.quantile(z, _COVERAGE_68))
        k95[name] = float(np.quantile(z / scale[name], 0.95))
    provenance = {"n": counts, **dict(fitted_on or {})}
    return ParamCalibration(
        method=config.method,
        maps=maps,
        sigma_scale=scale,
        k95=k95,
        fitted_on=provenance,
    )
