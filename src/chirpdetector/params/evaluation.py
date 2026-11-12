"""Evaluate a chirp-parameter decoder against the simulated ground truth.

`run_evaluation` (``chirpdetector evaluate params``) predicts the samples
of the test scenes (`chirpdetector.params.samples`, boxes of the decoder's
own detector) and computes, for the *uncalibrated* ("raw") and the
*calibrated* estimates:

Accuracy (`accuracy`, `breakdown`)
    Median absolute and relative error, p90, median signed error (bias)
    per parameter, by chirp family, SNR, ΔF to the nearest other fish,
    overlap with another chirp, another fish in the chirp's band, scene set
    and decile of the true value. The *shrinkage slope* is the slope of the
    regression of the estimate on the true value in model space (logs):
    1 for an estimate that is unbiased at every true value, below 1 for one
    that regresses to the mean (also as the slope through the medians of
    the deciles, which outliers do not move).
Uncertainty (`coverage`)
    Fraction of chirps whose error is within ±1 sd and ±k95 sd, and the
    rank correlation of sd and |error|; calibration by sd quintile.
Distribution fidelity (`distribution_fidelity`, `joint_histograms`)
    The manuscript compares *distributions* of chirp parameters, so the
    estimated population must look like the true one: per parameter, the
    two-sample Kolmogorov-Smirnov distance and the Wasserstein-1 distance
    (natural units and logs) of estimated vs true values, and their
    quantiles; for the joint height x FWHM distribution, the total
    variation distance of 2-D histograms on log bins. As a scale, the same
    distances between the true values of two random halves of the test
    chirps (sampling noise at this n).
Prior sensitivity (`natural_prior_weights`)
    The simulator drew templates from an equalised mix (many big and long
    chirps, `gridtools.templates.TemplateLibrary.cycler`); real recordings
    are dominated by small chirps. Each test chirp is weighted by
    ``p_natural / p_equalised`` of its template's excursion x duration
    equalisation bin (natural = uniform over the eligible pool, as
    ``TemplateLibrary.sample(..., equalize=False)``), turning the test
    population into one with the natural mix of chirp sizes, and the
    distribution fidelity is recomputed with these weights.
Cluster structure (`cluster_tests`)
    Populations resampled from the test chirps by their true parameters:
    two well-separated types (small/short and big/short) and a continuum
    along the segment between them. Along the axis through the two types'
    centres (log height, log FWHM), the density's valley-to-peak ratio
    shows whether separated clusters stay separated and whether a
    continuum stays free of a valley.
Real-data agreement (`real_agreement`)
    The harvested long chirps of the competition recordings (template
    library v4, ``source == "competition"``): the detector and the decoder
    run on the real recording around each chirp, and the decoder's
    estimates are compared with the per-period parameters of the chirp's
    independently extracted frequency trace. This is agreement with
    another estimate, not an error against ground truth.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from chirpdetector.bundle import Bundle, load_bundle
from chirpdetector.params.bundle import (
    ParamsBundle,
    attach_params_metrics,
    load_params_bundle,
)
from chirpdetector.params.inference import ParamsEstimator
from chirpdetector.params.model import from_model_space, natural_sd, to_model_space
from chirpdetector.params.samples import (
    BoxSource,
    SampleSet,
    discover_scenes,
    extract_samples,
    load_samples,
)
from chirpdetector.params.truth import PARAMETERS

__all__ = [
    "ParamsEvaluation",
    "accuracy",
    "breakdown",
    "cluster_tests",
    "coverage",
    "distribution_fidelity",
    "joint_histograms",
    "natural_prior_weights",
    "predict_samples",
    "real_agreement",
    "run_evaluation",
]

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
ESTIMATES = ("raw", "cal")
"""Uncalibrated and calibrated estimates."""

SNR_BINS = (-np.inf, -10.0, 0.0, 10.0, 20.0, np.inf)
DF_BINS = (0.0, 5.0, 20.0, 50.0, 150.0, np.inf)
SCORE_BINS = (0.0, 0.5, 0.7, 0.85, 1.0)
JOINT_BINS = {"height_hz": (10.0, 1000.0, 30), "fwhm_s": (0.002, 0.5, 30)}
"""Log-spaced bins ``(lo, hi, n)`` of the joint height x FWHM histograms."""

TWO_TYPES = {
    "small_short": {"height_hz": (40.0, 90.0), "fwhm_s": (0.006, 0.014)},
    "big_short": {"height_hz": (250.0, 500.0), "fwhm_s": (0.006, 0.020)},
}
"""True-parameter boxes of the two chirp types of the cluster test."""

UNITS = {"height_hz": "Hz", "fwhm_s": "s", "width10_s": "s", "undershoot_hz": "Hz"}


# ---------------------------------------------------------------- predictions


def predict_samples(
    estimator: ParamsEstimator, samples: SampleSet, rows: pd.DataFrame
) -> pd.DataFrame:
    """Raw and calibrated estimates for sample rows.

    Returns
    -------
    DataFrame
        ``rows`` plus, per parameter ``p``: ``raw_p`` and ``raw_sd_p``
        (uncalibrated), ``cal_p`` and ``cal_sd_p`` (calibrated), model-space
        ``mean_log_p``/``sigma_log_p`` and ``cal_log_p``/``cal_sigma_log_p``;
        and ``params_ok``, ``params_flag``.
    """
    inputs = samples.inputs(rows)
    mean, sigma = estimator.raw(inputs)
    cal_mean, cal_sigma = estimator.bundle.calibration.apply(mean, sigma)
    raw_values, cal_values = from_model_space(mean), from_model_space(cal_mean)
    raw_sd, cal_sd = natural_sd(raw_values, sigma), natural_sd(cal_values, cal_sigma)
    ok, flag = estimator.flags(inputs, cal_sigma)
    out = rows.reset_index(drop=True).copy()
    for j, p in enumerate(PARAMETERS):
        out[f"raw_{p}"] = raw_values[:, j]
        out[f"raw_sd_{p}"] = raw_sd[:, j]
        out[f"cal_{p}"] = cal_values[:, j]
        out[f"cal_sd_{p}"] = cal_sd[:, j]
        out[f"mean_log_{p}"] = mean[:, j]
        out[f"sigma_log_{p}"] = sigma[:, j]
        out[f"cal_log_{p}"] = cal_mean[:, j]
        out[f"cal_sigma_log_{p}"] = cal_sigma[:, j]
    out["params_ok"] = ok
    out["params_flag"] = flag
    return out


def _to_log(values: npt.ArrayLike, parameter: str) -> FloatArray:
    """One parameter in model space (`chirpdetector.params.model.LOG_OFFSETS`)."""
    from chirpdetector.params.model import LOG_OFFSETS

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.asarray(values, dtype=np.float64) + LOG_OFFSETS[parameter])


def _true_log(pred: pd.DataFrame) -> FloatArray:
    return to_model_space(pred[[f"true_{p}" for p in PARAMETERS]].to_numpy(float))


def _est_log(pred: pd.DataFrame, estimate: str) -> FloatArray:
    if estimate == "raw":
        return pred[[f"mean_log_{p}" for p in PARAMETERS]].to_numpy(float)
    return pred[[f"cal_log_{p}" for p in PARAMETERS]].to_numpy(float)


# ---------------------------------------------------------------- accuracy


def _slope(x: FloatArray, y: FloatArray) -> float:
    if x.size < 3 or np.var(x) == 0:
        return float("nan")
    return float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))


def _decile_slope(x: FloatArray, y: FloatArray) -> float:
    if x.size < 20:
        return float("nan")
    order = np.argsort(x)
    groups = np.array_split(order, 10)
    xm = np.array([np.median(x[g]) for g in groups])
    ym = np.array([np.median(y[g]) for g in groups])
    return float(np.polyfit(xm, ym, 1)[0])


def accuracy(pred: pd.DataFrame, estimate: str = "cal") -> dict[str, dict[str, float]]:
    """Error statistics of one estimate per parameter (module docstring)."""
    true_log, est_log = _true_log(pred), _est_log(pred, estimate)
    out: dict[str, dict[str, float]] = {}
    for j, p in enumerate(PARAMETERS):
        t = pred[f"true_{p}"].to_numpy(float)
        e = pred[f"{estimate}_{p}"].to_numpy(float)
        ok = np.isfinite(t) & np.isfinite(e)
        err = e[ok] - t[ok]
        rel = np.abs(err) / np.maximum(t[ok], 1e-12)
        out[p] = {
            "n": int(ok.sum()),
            "median_abs_error": float(np.median(np.abs(err))) if ok.any() else np.nan,
            "median_rel_error": float(np.median(rel)) if ok.any() else np.nan,
            "p90_abs_error": float(np.quantile(np.abs(err), 0.9))
            if ok.any()
            else np.nan,
            "median_error": float(np.median(err)) if ok.any() else np.nan,
            "median_abs_log_error": float(
                np.median(np.abs(est_log[ok, j] - true_log[ok, j]))
            )
            if ok.any()
            else np.nan,
            "shrinkage_slope": _slope(true_log[ok, j], est_log[ok, j]),
            "shrinkage_slope_deciles": _decile_slope(true_log[ok, j], est_log[ok, j]),
            "spearman_error_vs_true": float(np.asarray(stats.spearmanr(t[ok], err)[0]))
            if ok.sum() > 2
            else np.nan,
        }
    return out


def box_readout(pred: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Errors of reading the parameters off the detector box (for reference).

    height = box top minus the true baseline; FWHM and 10 % width = box
    duration; undershoot = true baseline minus box bottom (floored at 0).
    """
    baseline = pred["baseline_hz"].to_numpy(float)
    duration = pred["t_end_s"].to_numpy(float) - pred["t_start_s"].to_numpy(float)
    readout = {
        "height_hz": pred["f_high_hz"].to_numpy(float) - baseline,
        "fwhm_s": duration,
        "width10_s": duration,
        "undershoot_hz": np.maximum(baseline - pred["f_low_hz"].to_numpy(float), 0.0),
    }
    out = {}
    for p, values in readout.items():
        t = pred[f"true_{p}"].to_numpy(float)
        ok = np.isfinite(t) & np.isfinite(values)
        err = values[ok] - t[ok]
        out[p] = {
            "n": int(ok.sum()),
            "median_abs_error": float(np.median(np.abs(err))) if ok.any() else np.nan,
            "median_rel_error": float(np.median(np.abs(err) / np.maximum(t[ok], 1e-12)))
            if ok.any()
            else np.nan,
            "median_error": float(np.median(err)) if ok.any() else np.nan,
        }
    return out


def _bin_labels(
    values: npt.ArrayLike, edges: Sequence[float], missing: str
) -> npt.NDArray[np.str_]:
    """``(lo, hi]`` labels of values in bins; ``missing`` for NaN."""
    v = np.asarray(values, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    k = np.clip(np.searchsorted(e, v, side="left") - 1, 0, e.size - 2)
    labels = np.array([f"({e[i]:g}, {e[i + 1]:g}]" for i in k], dtype=object)
    labels[~np.isfinite(v)] = missing
    return labels.astype(str)


def _groups(pred: pd.DataFrame) -> dict[str, npt.NDArray[np.str_]]:
    """Grouping variables of `breakdown` (one label per row)."""

    def flag(column: str, yes: str, no: str) -> npt.NDArray[np.str_]:
        return np.where(pred[column].to_numpy(bool), yes, no)

    scene_set = [
        name.rsplit("_", 1)[0] if name.endswith(("_train", "_val", "_test")) else name
        for name in pred["scene_name"].astype(str)
    ]
    return {
        "family": pred["family"].astype(str).to_numpy(),
        "snr_db": _bin_labels(pred["snr_db"], SNR_BINS, "unknown"),
        "df_live_hz": _bin_labels(pred["min_df_live_hz"], DF_BINS, "single fish"),
        "overlap": flag("overlaps_other_chirp", "overlap", "no overlap"),
        "fish_in_band": flag("fish_in_band", "fish in band", "band free"),
        "visible": flag("visible", "visible", "invisible"),
        "scene_set": np.asarray(scene_set, dtype=str),
        "score": _bin_labels(pred["score"], SCORE_BINS, "no score"),
        "params_ok": flag("params_ok", "ok", "flagged"),
    }


def breakdown(pred: pd.DataFrame, estimate: str = "cal") -> pd.DataFrame:
    """Accuracy per group (module docstring); one row per group and parameter.

    Rows with ``by = "decile_<p>"`` hold both estimates (``estimate``
    column) per decile of the true value of ``p``.
    """
    rows = []
    for by, labels in _groups(pred).items():
        for label in sorted(set(labels.tolist())):
            sub = pred.loc[labels == label]
            for p, values in accuracy(sub, estimate).items():
                rows.append(
                    {
                        "by": by,
                        "group": label,
                        "parameter": p,
                        "estimate": estimate,
                        **values,
                    }
                )
    for p in PARAMETERS:
        t = pred[f"true_{p}"].to_numpy(float)
        order = np.argsort(t, kind="stable")
        for k, index in enumerate(np.array_split(order, 10)):
            if index.size == 0:
                continue
            sub = pred.iloc[index]
            label = f"{k + 1}: ({t[index].min():.3g}, {t[index].max():.3g})"
            for estimate_name in ESTIMATES:
                rows.append(
                    {
                        "by": f"decile_{p}",
                        "group": label,
                        "parameter": p,
                        "estimate": estimate_name,
                        "true_median": float(np.median(t[index])),
                        **accuracy(sub, estimate_name)[p],
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- uncertainty


def coverage(pred: pd.DataFrame, bundle: ParamsBundle) -> pd.DataFrame:
    """Coverage of the sd intervals and sd-|error| agreement per parameter."""
    true_log = _true_log(pred)
    rows = []
    for estimate in ESTIMATES:
        est_log = _est_log(pred, estimate)
        sigma_cols = (
            [f"sigma_log_{p}" for p in PARAMETERS]
            if estimate == "raw"
            else [f"cal_sigma_log_{p}" for p in PARAMETERS]
        )
        sigma = pred[sigma_cols].to_numpy(float)
        for j, p in enumerate(PARAMETERS):
            ok = np.isfinite(true_log[:, j]) & np.isfinite(est_log[:, j])
            err = np.abs(est_log[ok, j] - true_log[ok, j])
            s = sigma[ok, j]
            k95 = (
                float(bundle.calibration.k95.get(p, 1.96))
                if estimate == "cal"
                else 1.96
            )
            z = err / s
            order = np.argsort(s, kind="stable")
            rel = [
                float(np.sqrt(np.mean(z[q] ** 2)))
                for q in np.array_split(order, 5)
                if q.size
            ]
            rows.append(
                {
                    "estimate": estimate,
                    "parameter": p,
                    "n": int(ok.sum()),
                    "cover_1sd": float(np.mean(z <= 1.0)),
                    "cover_1.96sd": float(np.mean(z <= 1.96)),
                    "k95": k95,
                    "cover_k95sd": float(np.mean(z <= k95)),
                    "spearman_sd_abs_error": float(
                        np.asarray(stats.spearmanr(s, err)[0])
                    ),
                    "rms_z_by_sd_quintile": rel,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- distributions


def _wecdf(x: FloatArray, w: FloatArray, grid: FloatArray) -> FloatArray:
    order = np.argsort(x)
    cum = np.cumsum(w[order]) / w.sum()
    idx = np.searchsorted(x[order], grid, side="right")
    return np.where(idx > 0, cum[np.maximum(idx - 1, 0)], 0.0)


def weighted_ks(
    x: FloatArray,
    y: FloatArray,
    wx: FloatArray | None = None,
    wy: FloatArray | None = None,
) -> float:
    """Two-sample KS distance with optional weights."""
    wx = np.ones_like(x) if wx is None else wx
    wy = np.ones_like(y) if wy is None else wy
    grid = np.concatenate([x, y])
    return float(np.max(np.abs(_wecdf(x, wx, grid) - _wecdf(y, wy, grid))))


def weighted_quantile(
    x: FloatArray, w: FloatArray | None, q: Sequence[float]
) -> FloatArray:
    """Quantiles of a weighted sample (midpoint rule)."""
    w = np.ones_like(x) if w is None else w
    order = np.argsort(x)
    cum = (np.cumsum(w[order]) - 0.5 * w[order]) / w.sum()
    return np.interp(np.asarray(q, dtype=float), cum, x[order])


def _halves(
    values: FloatArray,
    weights: FloatArray | None,
    rng: np.random.Generator,
    n: int = 50,
) -> tuple[float, float]:
    """Median KS and W1 between the values of two random halves (sampling noise)."""
    ks, w1 = [], []
    for _ in range(n):
        mask = rng.random(values.size) < 0.5
        wa = None if weights is None else weights[mask]
        wb = None if weights is None else weights[~mask]
        ks.append(weighted_ks(values[mask], values[~mask], wa, wb))
        w1.append(stats.wasserstein_distance(values[mask], values[~mask], wa, wb))
    return float(np.median(ks)), float(np.median(w1))


def distribution_fidelity(
    pred: pd.DataFrame,
    weights: FloatArray | None = None,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """Estimated vs true marginal distributions (module docstring).

    Parameters
    ----------
    pred
        `predict_samples` output.
    weights
        Optional per-row weights (e.g. `natural_prior_weights`).
    seed
        Seed of the half-split reference.

    Returns
    -------
    DataFrame
        One row per parameter and estimate: KS, W1 (natural and log), the
        sampling-noise reference, quantiles of true and estimated values.
    """
    rng = np.random.default_rng(seed)
    qs = (0.05, 0.25, 0.5, 0.75, 0.95)
    rows = []
    for p in PARAMETERS:
        t = pred[f"true_{p}"].to_numpy(float)
        ok = np.isfinite(t)
        w = None if weights is None else np.asarray(weights, float)[ok]
        t = t[ok]
        t_log = _to_log(t, p)
        ref_ks, ref_w1 = _halves(t, w, rng)
        for estimate in ESTIMATES:
            e = pred[f"{estimate}_{p}"].to_numpy(float)[ok]
            e_log = _to_log(e, p)
            row: dict[str, Any] = {
                "parameter": p,
                "estimate": estimate,
                "n": int(ok.sum()),
                "ess": float(w.sum() ** 2 / (w**2).sum())
                if w is not None
                else float(ok.sum()),
                "ks": weighted_ks(e, t, w, w),
                "w1": float(stats.wasserstein_distance(e, t, w, w)),
                "w1_log": float(stats.wasserstein_distance(e_log, t_log, w, w)),
                "ks_halves": ref_ks,
                "w1_halves": ref_w1,
            }
            for q, tv, ev in zip(
                qs,
                weighted_quantile(t, w, qs),
                weighted_quantile(e, w, qs),
                strict=True,
            ):
                row[f"true_q{int(100 * q):02d}"] = float(tv)
                row[f"est_q{int(100 * q):02d}"] = float(ev)
            rows.append(row)
    return pd.DataFrame(rows)


def _joint_edges() -> tuple[FloatArray, FloatArray]:
    (h0, h1, nh), (f0, f1, nf) = JOINT_BINS["height_hz"], JOINT_BINS["fwhm_s"]
    return np.geomspace(h0, h1, int(nh) + 1), np.geomspace(f0, f1, int(nf) + 1)


def joint_histograms(
    pred: pd.DataFrame, weights: FloatArray | None = None, *, seed: int = 0
) -> dict[str, Any]:
    """Normalised height x FWHM histograms (log bins) and their TV distances."""
    h_edges, f_edges = _joint_edges()
    w = np.ones(len(pred)) if weights is None else np.asarray(weights, float)

    def hist(h: FloatArray, f: FloatArray, ww: FloatArray) -> FloatArray:
        ok = np.isfinite(h) & np.isfinite(f)
        counts, _, _ = np.histogram2d(
            np.clip(h[ok], h_edges[0], h_edges[-1] * 0.999999),
            np.clip(f[ok], f_edges[0], f_edges[-1] * 0.999999),
            bins=(h_edges, f_edges),
            weights=ww[ok],
        )
        return counts / max(counts.sum(), 1e-12)

    true = hist(
        pred["true_height_hz"].to_numpy(float), pred["true_fwhm_s"].to_numpy(float), w
    )
    out: dict[str, Any] = {
        "height_edges_hz": h_edges,
        "fwhm_edges_s": f_edges,
        "true": true,
    }
    for estimate in ESTIMATES:
        est = hist(
            pred[f"{estimate}_height_hz"].to_numpy(float),
            pred[f"{estimate}_fwhm_s"].to_numpy(float),
            w,
        )
        out[estimate] = est
        out[f"tv_{estimate}"] = float(0.5 * np.abs(est - true).sum())
    rng = np.random.default_rng(seed)
    tvs = []
    for _ in range(50):
        mask = rng.random(len(pred)) < 0.5
        a = hist(
            pred["true_height_hz"].to_numpy(float)[mask],
            pred["true_fwhm_s"].to_numpy(float)[mask],
            w[mask],
        )
        b = hist(
            pred["true_height_hz"].to_numpy(float)[~mask],
            pred["true_fwhm_s"].to_numpy(float)[~mask],
            w[~mask],
        )
        tvs.append(0.5 * np.abs(a - b).sum())
    out["tv_halves"] = float(np.median(tvs))
    out["spearman_true"] = float(
        np.asarray(stats.spearmanr(pred["true_height_hz"], pred["true_fwhm_s"])[0])
    )
    for estimate in ESTIMATES:
        out[f"spearman_{estimate}"] = float(
            np.asarray(
                stats.spearmanr(
                    pred[f"{estimate}_height_hz"], pred[f"{estimate}_fwhm_s"]
                )[0]
            )
        )
    return out


# ---------------------------------------------------------------- prior


def natural_prior_weights(
    template_ids: npt.ArrayLike,
    scene_dirs: Sequence[str | Path],
    *,
    level: str = "bin",
) -> tuple[FloatArray, dict[str, Any]]:
    """Importance weights from the equalised to the natural template mix.

    Rebuilds the simulator's eligible template pool and sampling options
    from the scenes' ``scene.json`` (`gridtools.sim.chirps.TemplatePool`).
    The natural mix is uniform over the pool (``TemplateLibrary.sample(...,
    equalize=False)``).

    Parameters
    ----------
    template_ids
        Template of every sample.
    scene_dirs
        Scenes the samples come from (all must share library and options).
    level
        ``"bin"`` (default): ``p_natural(bin) / p_equalised(bin)`` of the
        template's excursion x duration equalisation bin
        (`EqualizationBins.bin_probabilities`), which is what the
        equalisation changes; ``"template"``: ``p_natural / p_equalised``
        per template (`EqualizationBins.template_probabilities`, which also
        undoes the within-bin balancing of individuals; much more variable
        weights, so a far smaller effective sample).

    Returns
    -------
    weights, info
        Weights normalised to mean 1, and the pool, options and effective
        sample sizes of both levels.

    Raises
    ------
    ValueError
        If the scenes used different libraries or sampling options, or a
        template is not in the pool.
    """
    from gridtools.sim.chirps import TemplatePool
    from gridtools.sim.config import ChirpConfig

    if level not in ("bin", "template"):
        msg = f"level must be 'bin' or 'template', got {level!r}"
        raise ValueError(msg)
    infos = {
        json.dumps(
            json.loads((Path(d) / "scene.json").read_text())["templates"],
            sort_keys=True,
        )
        for d in sorted(set(map(str, scene_dirs)))
    }
    if len(infos) != 1:
        msg = f"the scenes used {len(infos)} template libraries or sampling schemes"
        raise ValueError(msg)
    templates = json.loads(next(iter(infos)))
    options = dict(templates.get("sampling_options", {}))
    ids = np.asarray(template_ids, dtype=np.int64)
    if not options.get("equalize", True):
        return np.ones(ids.size), {"equalized": False}
    pool = TemplatePool.open(ChirpConfig(library=str(templates["path"])))
    if pool.view.n_templates != int(templates.get("n_eligible", pool.view.n_templates)):
        msg = (
            f"the rebuilt pool has {pool.view.n_templates} templates, the scenes "
            f"{templates.get('n_eligible')}"
        )
        raise ValueError(msg)
    bins = tuple(options.get("bins", pool.bins))
    alpha = float(options.get("alpha", 0.6))
    eq = pool.view.equalization_bins(bins, spacing=options.get("spacing", "log"))
    position = {int(t): k for k, t in enumerate(eq.template_ids.tolist())}
    missing = sorted({int(i) for i in ids} - set(position))
    if missing:
        msg = f"templates {missing[:5]} are not in the equalised pool"
        raise ValueError(msg)
    index = np.array([position[int(i)] for i in ids], dtype=np.int64)
    p_bin_eq = eq.bin_probabilities(alpha)
    p_bin_nat = eq.counts.ravel() / eq.counts.sum()
    bin_of = eq.bin_of[index]
    by_bin = p_bin_nat[bin_of] / p_bin_eq[bin_of]
    p_template = eq.template_probabilities(alpha, options.get("balance", "individual"))
    by_template = (1.0 / eq.template_ids.size) / p_template[index]

    def ess(w: FloatArray) -> float:
        return float(w.sum() ** 2 / (w**2).sum()) if w.size else 0.0

    weights = by_bin if level == "bin" else by_template
    weights = weights / weights.mean() if weights.size else weights
    info = {
        "library": str(templates["path"]),
        "sampling_options": options,
        "pool_size": int(eq.template_ids.size),
        "level": level,
        "ess": ess(weights),
        "ess_bin": ess(by_bin),
        "ess_template": ess(by_template),
        "n": int(ids.size),
    }
    return weights, info


# ---------------------------------------------------------------- clusters


@dataclass
class ClusterPopulation:
    """A resampled population and its projection on the type axis."""

    name: str
    rows: npt.NDArray[np.int64]
    u_true: FloatArray
    u_est: dict[str, FloatArray] = field(default_factory=dict)


def _in_box(
    pred: pd.DataFrame, box: Mapping[str, tuple[float, float]]
) -> npt.NDArray[np.bool_]:
    mask = np.ones(len(pred), dtype=bool)
    for p, (lo, hi) in box.items():
        v = pred[f"true_{p}"].to_numpy(float)
        mask &= (v >= lo) & (v <= hi)
    return mask


def _log_xy(pred: pd.DataFrame, prefix: str) -> FloatArray:
    return np.log(pred[[f"{prefix}_height_hz", f"{prefix}_fwhm_s"]].to_numpy(float))


def _valley_ratio(u: FloatArray, bandwidth: float = 0.06) -> float:
    """Density valley in u in [0.3, 0.7] over the smaller peak near 0 and 1."""
    if u.size < 10:
        return float("nan")
    grid = np.linspace(-0.3, 1.3, 321)
    density = stats.gaussian_kde(u, bw_method=bandwidth / max(np.std(u), 1e-9))(grid)
    peak_a = density[(grid >= -0.2) & (grid <= 0.3)].max()
    peak_b = density[(grid >= 0.7) & (grid <= 1.2)].max()
    valley = density[(grid >= 0.3) & (grid <= 0.7)].min()
    return float(valley / min(peak_a, peak_b))


def cluster_tests(
    pred: pd.DataFrame, *, seed: int = 0, n_per_type: int | None = None
) -> tuple[pd.DataFrame, list[ClusterPopulation]]:
    """Two separated types and a continuum, resampled by true values.

    The type axis runs from the median (log height, log FWHM) of the
    ``small_short`` chirps (u = 0) to that of the ``big_short`` chirps
    (u = 1) (`TWO_TYPES`). ``two_types`` takes equally many chirps of both
    types; ``continuum`` takes chirps whose true position projects onto
    u in [0, 1] within 0.35 (log units) of the axis, equally many per tenth
    of u. Each population's true and estimated u are compared by their
    valley-to-peak ratio, the fraction of chirps on the right side of
    u = 0.5 (two types) and the KS distance of u.

    Returns
    -------
    table, populations
    """
    rng = np.random.default_rng(seed)
    true_xy = _log_xy(pred, "true")
    masks = {k: _in_box(pred, box) for k, box in TWO_TYPES.items()}
    if min(m.sum() for m in masks.values()) < 10:
        log.warning("too few chirps of the two types for the cluster test")
        return pd.DataFrame(), []
    centre_a = np.median(true_xy[masks["small_short"]], axis=0)
    centre_b = np.median(true_xy[masks["big_short"]], axis=0)
    axis = centre_b - centre_a

    def project(xy: FloatArray) -> FloatArray:
        return (xy - centre_a) @ axis / (axis @ axis)

    u_all = project(true_xy)
    perp = np.abs(
        (true_xy - centre_a) @ np.array([-axis[1], axis[0]]) / np.linalg.norm(axis)
    )
    n = n_per_type or int(min(m.sum() for m in masks.values()))
    two = np.concatenate(
        [
            rng.choice(np.flatnonzero(m), n, replace=bool(m.sum() < n))
            for m in masks.values()
        ]
    )
    band = (u_all >= 0) & (u_all <= 1) & (perp <= 0.35)
    tenth = np.minimum((u_all * 10).astype(int), 9)
    per_bin = [np.flatnonzero(band & (tenth == k)) for k in range(10)]
    m = min(len(b) for b in per_bin)
    populations = []
    rows = []
    if m >= 5:
        continuum = np.concatenate([rng.choice(b, m, replace=False) for b in per_bin])
    else:
        log.warning(
            "too few intermediate chirps for the continuum test (%d per tenth)", m
        )
        continuum = np.zeros(0, dtype=np.int64)
    for name, index in (("two_types", two), ("continuum", continuum)):
        if index.size == 0:
            continue
        population = ClusterPopulation(name=name, rows=index, u_true=u_all[index])
        row: dict[str, Any] = {
            "population": name,
            "n": int(index.size),
            "valley_true": _valley_ratio(population.u_true),
        }
        for estimate in ESTIMATES:
            u = project(_log_xy(pred.iloc[index], estimate))
            population.u_est[estimate] = u
            row[f"valley_{estimate}"] = _valley_ratio(u)
            row[f"ks_u_{estimate}"] = weighted_ks(u, population.u_true)
            if name == "two_types":
                side_true = population.u_true > 0.5
                row[f"same_side_{estimate}"] = float(np.mean((u > 0.5) == side_true))
        populations.append(population)
        rows.append(row)
    return pd.DataFrame(rows), populations


# ---------------------------------------------------------------- real data


def real_agreement(
    estimator: ParamsEstimator,
    detector: Bundle,
    library_dir: str | Path,
    recordings_dir: str | Path,
    *,
    device: str | None = None,
    span_s: float = 3.2,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Decoder estimates vs trace estimates of the harvested real chirps.

    For each chirp of the library with ``source == "competition"``, the
    detector and decoder run on ``span_s`` of its recording centred on the
    chirp (`chirpdetector.inference.predict_span`); the detection matching
    the chirp (centre within 100 ms of ``align_abs_s``, frequency range
    overlapping ``baseline - undershoot`` to ``baseline + excursion``) is
    compared with the per-period parameters of the chirp's extracted trace
    (`chirpdetector.params.truth.excursion_params`).

    Returns
    -------
    DataFrame
        One row per harvested chirp: reference ``ref_*``, decoder
        estimates (NaN if not detected), sds, flags, ``detected``.
    """
    from gridtools.io import open_recording
    from gridtools.templates.library import load_library

    from chirpdetector.inference import predict_span
    from chirpdetector.params.samples import match_predictions
    from chirpdetector.params.truth import excursion_params
    from chirpdetector.prediction import load_detector

    say = progress or log.info
    library = load_library(library_dir, mmap=True)
    meta = library.metadata
    harvested = meta.loc[meta["source"].astype(str).to_numpy() == "competition"]
    harvested = harvested.reset_index(drop=True)
    ids = harvested["template_id"].to_numpy(np.int64)
    rows_of = library.rows_of(ids)
    recordings = harvested["recording"].astype(str).to_numpy()
    baselines = harvested["baseline_hz"].to_numpy(np.float64)
    centres = harvested["align_abs_s"].to_numpy(np.float64)
    snrs = (
        harvested["band_snr_db"].to_numpy(np.float64)
        if "band_snr_db" in harvested.columns
        else np.full(len(harvested), np.nan)
    )
    model = load_detector(detector, device=device)
    out = []
    for name in sorted(set(recordings.tolist())):
        recording = open_recording(Path(recordings_dir) / name)
        members = np.flatnonzero(recordings == name)
        for k in members:
            trace = np.asarray(library.delta_f_hz[rows_of[k]], dtype=np.float64)
            ref = excursion_params(trace, library.samplerate_hz, float(baselines[k]))
            centre = float(centres[k])
            row: dict[str, Any] = {
                "template_id": int(ids[k]),
                "recording": name,
                "t_center_s": centre,
                "baseline_hz": float(baselines[k]),
                "band_snr_db": float(snrs[k]),
                **{f"ref_{p}": getattr(ref, p) for p in PARAMETERS},
            }
            detections = predict_span(
                detector,
                recording,
                centre - span_s / 2,
                centre + span_s / 2,
                name=name,
                detector=model,
                device=device,
                params=estimator,
            )
            target = pd.DataFrame(
                {
                    "t_center_s": [centre],
                    "f_low_hz": [float(baselines[k]) - max(ref.undershoot_hz, 5.0)],
                    "f_high_hz": [float(baselines[k]) + ref.height_hz],
                }
            )
            index, _ = match_predictions(detections, target)
            hit = np.flatnonzero(index == 0)
            row["detected"] = bool(hit.size)
            if hit.size:
                det = detections.iloc[int(hit[0])].to_dict()
                for column in (
                    "t_start_s",
                    "t_end_s",
                    "f_low_hz",
                    "f_high_hz",
                    "score",
                    *estimator_columns(),
                ):
                    row[column] = det[column]
            out.append(row)
        say(f"real check: {name} ({members.size} chirps)")
    return pd.DataFrame(out)


def estimator_columns() -> tuple[str, ...]:
    """The decoder's output columns (`chirpdetector.params.PARAM_COLUMNS`)."""
    from chirpdetector.params.inference import PARAM_COLUMNS

    return PARAM_COLUMNS


def agreement_summary(real: pd.DataFrame) -> dict[str, Any]:
    """Median |difference|, relative difference, bias and correlations."""
    detected = (
        real["detected"].to_numpy(bool)
        if "detected" in real.columns
        else np.zeros(len(real), dtype=bool)
    )
    found = real.loc[detected]
    out: dict[str, Any] = {"n": len(real), "detected": len(found)}
    for p in PARAMETERS:
        if p not in found.columns:
            continue
        ref = found[f"ref_{p}"].to_numpy(float)
        est = found[p].to_numpy(float)
        ok = np.isfinite(ref) & np.isfinite(est)
        if ok.sum() < 3:
            continue
        diff = est[ok] - ref[ok]
        out[p] = {
            "n": int(ok.sum()),
            "median_abs_difference": float(np.median(np.abs(diff))),
            "median_rel_difference": float(
                np.median(np.abs(diff) / np.maximum(ref[ok], 1e-12))
            ),
            "median_difference": float(np.median(diff)),
            "pearson_r": float(np.corrcoef(ref[ok], est[ok])[0, 1]),
            "spearman_r": float(np.asarray(stats.spearmanr(ref[ok], est[ok])[0])),
            "ref_median": float(np.median(ref[ok])),
            "ref_iqr": [float(v) for v in np.quantile(ref[ok], [0.25, 0.75])],
        }
    return out


# ---------------------------------------------------------------- driver


@dataclass
class ParamsEvaluation:
    """Everything `run_evaluation` computed (tables and metrics)."""

    bundle: ParamsBundle
    predictions: pd.DataFrame
    unmatched: pd.DataFrame
    metrics: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    joint: dict[str, Any]
    joint_natural: dict[str, Any]
    weights: FloatArray
    clusters: list[ClusterPopulation]
    real: pd.DataFrame | None


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _test_samples(
    bundle: ParamsBundle,
    scenes: Sequence[str | Path],
    split: str,
    samples_dir: Path | None,
    out: Path,
    detector: Bundle | None,
    workers: int,
    device: str,
) -> SampleSet:
    """A sample set with the split's scenes, made like the bundle's training set."""
    background = str(bundle.training.get("config", {}).get("scene_background", "any"))
    refs = [
        r for r in discover_scenes(scenes, background=background) if r.split == split
    ]
    if not refs:
        msg = f"no {split} scenes under {list(map(str, scenes))}"
        raise FileNotFoundError(msg)
    candidates = [samples_dir] if samples_dir is not None else []
    trained_on = bundle.training.get("samples_path")
    if samples_dir is None and trained_on:
        candidates.append(Path(trained_on))
    wanted = {r.source_id for r in refs}
    for candidate in candidates:
        if candidate is None or not (candidate / "manifest.json").is_file():
            continue
        samples = load_samples(candidate)
        have = {
            s["source_id"] for s in samples.manifest["scenes"] if s["split"] == split
        }
        same = (
            samples.manifest["crop"] == bundle.crop.to_dict()
            and samples.manifest["detector"] == bundle.detector
        )
        if same and wanted <= have:
            log.info("using the %s samples of %s", split, candidate)
            return samples
    boxes = BoxSource.from_dict(bundle.training["config"]["boxes"])
    if detector is None and bundle.detector is not None:
        detector = load_bundle(bundle.detector["path"])
    if (
        detector is not None
        and bundle.detector is not None
        and detector.weights_sha256 != bundle.detector["weights_sha256"]
    ):
        msg = "the detector differs from the one the decoder was trained with"
        raise ValueError(msg)
    return extract_samples(
        refs,
        samples_dir or out / "samples",
        crop=bundle.crop,
        boxes=boxes,
        detector=detector,
        spectrogram=bundle.spectrogram,
        preprocessing=bundle.preprocessing,
        workers=workers,
        device=device,
    )


def evaluate_params(
    bundle: ParamsBundle | str | Path,
    samples: SampleSet,
    *,
    split: str = "test",
    min_score: float | None = None,
    device: str | None = None,
    seed: int = 0,
) -> ParamsEvaluation:
    """All simulated-data metrics of a bundle on a sample set's split."""
    bundle = bundle if isinstance(bundle, ParamsBundle) else load_params_bundle(bundle)
    estimator = ParamsEstimator(bundle, device=device)
    threshold = min_score
    if threshold is None and bundle.detector is not None:
        threshold = float(bundle.detector.get("score_threshold", 0.0))
    rows = samples.select(split, usable=True, min_score=threshold)
    if len(rows) < 20:
        msg = f"only {len(rows)} usable {split} samples"
        raise ValueError(msg)
    pred = predict_samples(estimator, samples, rows)
    unmatched_rows = samples.select(split, min_score=threshold)
    unmatched_rows = unmatched_rows.loc[~unmatched_rows["matched"].to_numpy(bool)]
    unmatched = predict_samples(estimator, samples, unmatched_rows)
    weights, prior = natural_prior_weights(
        pred["template_id"].to_numpy(np.int64), pred["scene_dir"].unique().tolist()
    )
    pred["natural_weight"] = weights
    tables = {
        "breakdown": breakdown(pred),
        "coverage": coverage(pred, bundle),
        "fidelity": distribution_fidelity(pred, seed=seed),
        "fidelity_natural": distribution_fidelity(pred, weights, seed=seed),
    }
    clusters_table, clusters = cluster_tests(pred, seed=seed)
    tables["clusters"] = clusters_table
    joint = joint_histograms(pred, seed=seed)
    joint_natural = joint_histograms(pred, weights, seed=seed)
    flagged = ~pred["params_ok"].to_numpy(bool)
    metrics: dict[str, Any] = {
        "split": split,
        "min_score": threshold,
        "n": len(pred),
        "n_unmatched": len(unmatched),
        "accuracy": {e: accuracy(pred, e) for e in ESTIMATES},
        "accuracy_ok_only": accuracy(pred.loc[~flagged], "cal"),
        "box_readout": box_readout(pred),
        "flags": {
            "fraction_flagged": float(flagged.mean()),
            "fraction_flagged_unmatched": float(
                (~unmatched["params_ok"].to_numpy(bool)).mean()
            )
            if len(unmatched)
            else None,
            "reasons": pred["params_flag"]
            .str.split(",")
            .explode()
            .value_counts()
            .to_dict(),
        },
        "coverage": tables["coverage"]
        .drop(columns="rms_z_by_sd_quintile")
        .to_dict("records"),
        "fidelity": tables["fidelity"].to_dict("records"),
        "fidelity_natural": tables["fidelity_natural"].to_dict("records"),
        "natural_prior": prior,
        "natural_conditional_bias": {
            e: {
                p: float(
                    weighted_quantile(
                        (pred[f"{e}_{p}"] - pred[f"true_{p}"]).to_numpy(float),
                        weights,
                        [0.5],
                    )[0]
                )
                for p in PARAMETERS
            }
            for e in ESTIMATES
        },
        "joint": {k: v for k, v in joint.items() if not isinstance(v, np.ndarray)},
        "joint_natural": {
            k: v for k, v in joint_natural.items() if not isinstance(v, np.ndarray)
        },
        "clusters": clusters_table.to_dict("records"),
    }
    return ParamsEvaluation(
        bundle=bundle,
        predictions=pred,
        unmatched=unmatched,
        metrics=_jsonable(metrics),
        tables=tables,
        joint=joint,
        joint_natural=joint_natural,
        weights=weights,
        clusters=clusters,
        real=None,
    )


def run_evaluation(
    bundle: str | Path | ParamsBundle,
    scenes: Sequence[str | Path],
    *,
    out: str | Path | None = None,
    samples_dir: str | Path | None = None,
    detector: str | Path | Bundle | None = None,
    split: str = "test",
    real_library: str | Path | None = None,
    real_recordings: str | Path | None = None,
    workers: int = 4,
    device: str = "auto",
    attach: bool = True,
) -> Path:
    """Evaluate a params bundle and write its report (module docstring).

    Returns
    -------
    Path
        ``report.md`` in ``out`` (default ``BUNDLE/eval``).
    """
    from chirpdetector.params.report import write_report

    bundle = bundle if isinstance(bundle, ParamsBundle) else load_params_bundle(bundle)
    out_dir = Path(out) if out is not None else bundle.path / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    detector_bundle = (
        detector
        if isinstance(detector, Bundle) or detector is None
        else load_bundle(detector)
    )
    samples = _test_samples(
        bundle,
        scenes,
        split,
        None if samples_dir is None else Path(samples_dir),
        out_dir,
        detector_bundle,
        workers,
        device,
    )
    torch_device = None if device == "auto" else device
    result = evaluate_params(bundle, samples, split=split, device=torch_device)
    if real_library is not None and Path(real_library).is_dir():
        if detector_bundle is None and bundle.detector is not None:
            detector_bundle = load_bundle(bundle.detector["path"])
        if detector_bundle is None:
            log.warning("no detector: skipping the real-data check")
        else:
            estimator = ParamsEstimator(bundle, device=torch_device)
            real = real_agreement(
                estimator,
                detector_bundle,
                real_library,
                real_recordings or "/mnt/data2/2019_tube_competition",
                device=torch_device,
            )
            result.real = real
            result.metrics["real_agreement"] = _jsonable(agreement_summary(real))
    report = write_report(result, out_dir)
    if attach:
        attach_params_metrics(
            bundle,
            {
                "summary": summary_lines(result.metrics),
                "metrics": result.metrics,
                "samples": str(samples.directory),
                "report": str(report),
            },
        )
    return report


def summary_lines(metrics: Mapping[str, Any]) -> dict[str, str]:
    """One line per parameter: calibrated median |error| and relative error."""
    out = {}
    acc = metrics.get("accuracy", {}).get("cal", {})
    for p, values in acc.items():
        unit = UNITS[p]
        scale = 1e3 if unit == "s" else 1.0
        shown = "ms" if unit == "s" else unit
        out[p] = (
            f"{scale * values['median_abs_error']:.2g} {shown} "
            f"({100 * values['median_rel_error']:.0f} %)"
        )
    return out
