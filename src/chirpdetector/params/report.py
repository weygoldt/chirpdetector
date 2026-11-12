"""Figures, tables and the markdown report of a decoder evaluation.

`write_report` writes, into the evaluation directory::

    report.md                  the report (tables inline, figures linked)
    metrics.json               all metrics (`chirpdetector.params.evaluation`)
    predictions.parquet        per test sample: truth, raw and calibrated estimates
    breakdown.csv, coverage.csv, fidelity.csv, fidelity_natural.csv, clusters.csv
    real_agreement.csv         if the real-data check ran
    fig_*.png                  figures (needs matplotlib)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from chirpdetector.params.truth import PARAMETERS

if TYPE_CHECKING:
    from chirpdetector.params.evaluation import ParamsEvaluation

__all__ = ["write_report"]

_LABELS = {
    "height_hz": "height (Hz)",
    "fwhm_s": "FWHM (ms)",
    "width10_s": "10 % width (ms)",
    "undershoot_hz": "undershoot (Hz)",
}
_SCALE = {"height_hz": 1.0, "fwhm_s": 1e3, "width10_s": 1e3, "undershoot_hz": 1.0}
_COLOURS = {"true": "0.2", "raw": "tab:orange", "cal": "tab:blue"}
_NAMES = {"true": "true", "raw": "uncalibrated", "cal": "calibrated"}


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    if isinstance(value, float | np.floating):
        return f"{value:.{digits}g}"
    return str(value)


def _table(df: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    columns = list(columns or df.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def _err(value: float, p: str) -> str:
    scale = _SCALE[p]
    unit = "ms" if scale != 1.0 else "Hz"
    return f"{scale * value:.3g} {unit}"


# ---------------------------------------------------------------- figures


def _figure(rows: int, cols: int, size: tuple[float, float]) -> tuple[Any, Any]:
    from matplotlib.figure import Figure

    fig = Figure(figsize=size, layout="constrained")
    axes = fig.subplots(rows, cols, squeeze=False)
    return fig, axes


def _axis_scale(ax: Any, p: str, which: str = "both") -> None:
    if p == "undershoot_hz":
        if which in ("x", "both"):
            ax.set_xscale("symlog", linthresh=1.0)
        if which in ("y", "both"):
            ax.set_yscale("symlog", linthresh=1.0)
    else:
        if which in ("x", "both"):
            ax.set_xscale("log")
        if which in ("y", "both"):
            ax.set_yscale("log")


def _fig_pred_vs_true(pred: pd.DataFrame, path: Path) -> Path:
    fig, axes = _figure(2, 4, (13, 6.2))
    for r, estimate in enumerate(("raw", "cal")):
        for c, p in enumerate(PARAMETERS):
            ax = axes[r, c]
            t = _SCALE[p] * pred[f"true_{p}"].to_numpy(float)
            e = _SCALE[p] * pred[f"{estimate}_{p}"].to_numpy(float)
            ax.scatter(t, e, s=1.5, alpha=0.25, color=_COLOURS[estimate], lw=0)
            lo = max(
                np.nanmin(np.concatenate([t, e])), 0.0 if p == "undershoot_hz" else 1e-3
            )
            hi = np.nanmax(np.concatenate([t, e]))
            ax.plot([lo, hi], [lo, hi], color="k", lw=0.7)
            _axis_scale(ax, p)
            ax.set_xlabel(f"true {_LABELS[p]}")
            ax.set_ylabel(f"{_NAMES[estimate]} {_LABELS[p]}")
    fig.suptitle("Decoder estimate vs per-period ground truth (test chirps)")
    fig.savefig(path, dpi=130)
    return path


def _fig_bias(pred: pd.DataFrame, path: Path) -> Path:
    from chirpdetector.params.evaluation import _to_log

    fig, axes = _figure(1, 4, (14, 3.6))
    for c, p in enumerate(PARAMETERS):
        ax = axes[0, c]
        t = pred[f"true_{p}"].to_numpy(float)
        order = np.argsort(t)
        groups = np.array_split(order, 10)
        centres = _SCALE[p] * np.array([np.median(t[g]) for g in groups])
        for estimate in ("raw", "cal"):
            err = _to_log(pred[f"{estimate}_{p}"], p) - _to_log(t, p)
            med = np.array([np.median(err[g]) for g in groups])
            q1 = np.array([np.quantile(err[g], 0.25) for g in groups])
            q3 = np.array([np.quantile(err[g], 0.75) for g in groups])
            ax.fill_between(centres, q1, q3, color=_COLOURS[estimate], alpha=0.15, lw=0)
            ax.plot(
                centres,
                med,
                "o-",
                ms=3,
                color=_COLOURS[estimate],
                label=_NAMES[estimate],
            )
        ax.axhline(0, color="k", lw=0.7)
        _axis_scale(ax, p, "x")
        ax.set_xlabel(f"true {_LABELS[p]} (decile medians)")
        ax.set_ylabel(
            "log(estimate / true)" if p != "undershoot_hz" else "log((est+1)/(true+1))"
        )
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Signed error by decile of the true value: median and IQR")
    fig.savefig(path, dpi=130)
    return path


def _log_bins(values: np.ndarray, p: str, n: int = 50) -> np.ndarray:
    v = values[np.isfinite(values)]
    if p == "undershoot_hz":
        hi = np.quantile(v, 0.999) + 1
        return np.concatenate([[0.0], np.geomspace(0.1, hi, n)])
    return np.geomspace(max(np.quantile(v, 0.0005), 1e-6), np.quantile(v, 0.9995), n)


def _fig_marginals(pred: pd.DataFrame, weights: np.ndarray, path: Path) -> Path:
    fig, axes = _figure(2, 4, (14, 6.2))
    for r, (w, title) in enumerate(
        (
            (None, "test population (equalised templates)"),
            (weights, "re-weighted to the natural template mix"),
        )
    ):
        for c, p in enumerate(PARAMETERS):
            ax = axes[r, c]
            t = _SCALE[p] * pred[f"true_{p}"].to_numpy(float)
            bins = _log_bins(t, p)
            for which in ("true", "raw", "cal"):
                values = (
                    t
                    if which == "true"
                    else _SCALE[p] * pred[f"{which}_{p}"].to_numpy(float)
                )
                ax.hist(
                    values,
                    bins=bins,
                    weights=w,
                    density=True,
                    histtype="step",
                    lw=1.4 if which == "true" else 1.0,
                    color=_COLOURS[which],
                    label=_NAMES[which],
                )
            _axis_scale(ax, p, "x")
            ax.set_xlabel(_LABELS[p])
            if c == 0:
                ax.set_ylabel(f"density\n{title}", fontsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Marginal distributions: true vs estimated")
    fig.savefig(path, dpi=130)
    return path


def _fig_joint(result: ParamsEvaluation, path: Path) -> Path:
    from matplotlib.colors import LogNorm

    fig, axes = _figure(2, 3, (13, 7.6))
    for r, (joint, title) in enumerate(
        (
            (result.joint, "test population"),
            (result.joint_natural, "natural template mix"),
        )
    ):
        h_edges, f_edges = joint["height_edges_hz"], 1e3 * joint["fwhm_edges_s"]
        vmax = max(joint[k].max() for k in ("true", "raw", "cal"))
        mesh = None
        for c, which in enumerate(("true", "raw", "cal")):
            ax = axes[r, c]
            hist = np.where(joint[which] > 0, joint[which], np.nan)
            mesh = ax.pcolormesh(
                f_edges,
                h_edges,
                hist,
                norm=LogNorm(vmin=vmax * 1e-4, vmax=vmax),
                cmap="magma_r",
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("FWHM (ms)")
            ax.set_ylabel("height (Hz)")
            tv = "" if which == "true" else f", TV {joint[f'tv_{which}']:.3f}"
            ax.set_title(f"{_NAMES[which]} ({title}{tv})", fontsize=9)
        if mesh is not None:
            fig.colorbar(mesh, ax=axes[r, 2], label="fraction of chirps")
    fig.suptitle(
        f"Joint height x FWHM; TV between random halves of the truth: "
        f"{result.joint['tv_halves']:.3f} (test), {result.joint_natural['tv_halves']:.3f} (natural)"
    )
    fig.savefig(path, dpi=130)
    return path


def _fig_clusters(result: ParamsEvaluation, path: Path) -> Path | None:
    from scipy import stats

    pops = result.clusters
    if not pops:
        return None
    pred = result.predictions
    fig, axes = _figure(len(pops), 3, (13, 3.8 * len(pops)))
    for r, pop in enumerate(pops):
        sub = pred.iloc[pop.rows]
        both_f = 1e3 * np.concatenate([sub["true_fwhm_s"], sub["cal_fwhm_s"]])
        both_h = np.concatenate([sub["true_height_hz"], sub["cal_height_hz"]])
        limits = (
            (np.nanmin(both_f) * 0.9, np.nanmax(both_f) * 1.1),
            (np.nanmin(both_h) * 0.9, np.nanmax(both_h) * 1.1),
        )
        for c, which in enumerate(("true", "cal")):
            ax = axes[r, c]
            ax.scatter(
                1e3 * sub[f"{which}_fwhm_s"],
                sub[f"{which}_height_hz"],
                s=3,
                alpha=0.4,
                lw=0,
                color=_COLOURS[which],
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(*limits[0])
            ax.set_ylim(*limits[1])
            ax.set_xlabel("FWHM (ms)")
            ax.set_ylabel("height (Hz)")
            ax.set_title(f"{pop.name}: {_NAMES[which]}", fontsize=9)
        ax = axes[r, 2]
        grid = np.linspace(-0.3, 1.3, 321)
        for which, u in (("true", pop.u_true), *pop.u_est.items()):
            density = stats.gaussian_kde(u, bw_method=0.06 / max(np.std(u), 1e-9))(grid)
            ax.plot(grid, density, color=_COLOURS[which], label=_NAMES[which])
        ends = {"height": "small/short -> big/short", "fwhm": "big/short -> big/long"}
        ax.set_xlabel(f"position on the {ends.get(pop.axis, pop.axis)} axis")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle("Cluster structure: populations resampled by their true parameters")
    fig.savefig(path, dpi=130)
    return path


def _fig_coverage(result: ParamsEvaluation, path: Path) -> Path:
    table = result.tables["coverage"]
    fig, axes = _figure(1, 2, (11, 3.6))
    ax = axes[0, 0]
    x = np.arange(len(PARAMETERS))
    for k, (estimate, column, offset) in enumerate(
        (
            ("raw", "cover_1sd", -0.3),
            ("cal", "cover_1sd", -0.1),
            ("raw", "cover_1.96sd", 0.1),
            ("cal", "cover_k95sd", 0.3),
        )
    ):
        values = [
            float(
                table.loc[
                    (table["estimate"] == estimate) & (table["parameter"] == p), column
                ].iloc[0]
            )
            for p in PARAMETERS
        ]
        ax.bar(
            x + offset,
            values,
            width=0.18,
            color=_COLOURS[estimate],
            alpha=1.0 if k % 2 else 0.5,
            label=f"{_NAMES[estimate]} {column}",
        )
    ax.axhline(0.683, color="k", lw=0.7, ls="--")
    ax.axhline(0.95, color="k", lw=0.7, ls="--")
    ax.set_xticks(x, [_LABELS[p] for p in PARAMETERS], fontsize=8)
    ax.set_ylabel("coverage")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax = axes[0, 1]
    for p in PARAMETERS:
        row = table.loc[(table["estimate"] == "cal") & (table["parameter"] == p)].iloc[
            0
        ]
        ax.plot(
            range(1, len(row["rms_z_by_sd_quintile"]) + 1),
            row["rms_z_by_sd_quintile"],
            "o-",
            label=_LABELS[p],
        )
    ax.axhline(1, color="k", lw=0.7)
    ax.set_xlabel("quintile of the calibrated sd")
    ax.set_ylabel("RMS of error / sd")
    ax.legend(fontsize=8)
    fig.suptitle("Uncertainty calibration (test chirps)")
    fig.savefig(path, dpi=130)
    return path


def _fig_real(real: pd.DataFrame, path: Path) -> Path | None:
    detected = (
        real["detected"].to_numpy(bool)
        if "detected" in real.columns
        else np.zeros(len(real), dtype=bool)
    )
    found = real.loc[detected]
    if len(found) < 3:
        return None
    fig, axes = _figure(1, 4, (14, 3.6))
    ok = found["params_ok"].to_numpy(bool)
    for c, p in enumerate(PARAMETERS):
        ax = axes[0, c]
        ref = _SCALE[p] * found[f"ref_{p}"].to_numpy(float)
        est = _SCALE[p] * found[p].to_numpy(float)
        ax.scatter(
            ref[ok], est[ok], s=6, alpha=0.6, color="tab:blue", lw=0, label="in domain"
        )
        ax.scatter(
            ref[~ok], est[~ok], s=6, alpha=0.6, color="tab:red", lw=0, label="flagged"
        )
        both = np.concatenate([ref, est])
        lo, hi = np.nanmin(both), np.nanmax(both)
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.7)
        ax.set_xlabel(f"trace estimate {_LABELS[p]}")
        ax.set_ylabel(f"decoder {_LABELS[p]}")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Real long chirps: decoder vs independent trace estimate (agreement, not error)"
    )
    fig.savefig(path, dpi=130)
    return path


# ---------------------------------------------------------------- markdown


def _accuracy_table(metrics: dict[str, Any]) -> str:
    lines = [
        "| parameter | estimate | n | median abs. error | median rel. error | p90 abs. error | median signed error | shrinkage slope (OLS, log) | slope (decile medians) | Spearman(error, true) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in PARAMETERS:
        for estimate in ("raw", "cal"):
            a = metrics["accuracy"][estimate][p]
            lines.append(
                f"| {p} | {_NAMES[estimate]} | {a['n']} | {_err(a['median_abs_error'], p)} | "
                f"{100 * a['median_rel_error']:.1f} % | {_err(a['p90_abs_error'], p)} | "
                f"{_err(a['median_error'], p)} | {_fmt(a['shrinkage_slope'])} | "
                f"{_fmt(a['shrinkage_slope_deciles'])} | {_fmt(a['spearman_error_vs_true'], 2)} |"
            )
    return "\n".join(lines)


def _breakdown_section(breakdown: pd.DataFrame) -> list[str]:
    out: list[str] = []
    records = breakdown.to_dict("records")
    plain = [r for r in records if not str(r["by"]).startswith("decile_")]
    for by in dict.fromkeys(str(r["by"]) for r in plain):
        out += [f"**By {by}** (calibrated; median abs. error, relative for height)", ""]
        rows = []
        mine = [r for r in plain if r["by"] == by]
        for group in dict.fromkeys(str(r["group"]) for r in mine):
            values = {str(r["parameter"]): r for r in mine if r["group"] == group}
            h = values["height_hz"]
            rows.append(
                {
                    "group": group,
                    "n": int(h["n"]),
                    "height": f"{_err(h['median_abs_error'], 'height_hz')} "
                    f"({100 * h['median_rel_error']:.0f} %)",
                    "FWHM": _err(values["fwhm_s"]["median_abs_error"], "fwhm_s"),
                    "10 % width": _err(
                        values["width10_s"]["median_abs_error"], "width10_s"
                    ),
                    "undershoot": _err(
                        values["undershoot_hz"]["median_abs_error"], "undershoot_hz"
                    ),
                    "height bias": _err(h["median_error"], "height_hz"),
                }
            )
        out += [_table(pd.DataFrame(rows)), ""]
    out += [
        "**By decile of the true value** (median signed error, raw -> calibrated; "
        "this is where shrinkage shows)",
        "",
    ]
    for p in PARAMETERS:
        mine = [r for r in records if r["by"] == f"decile_{p}"]
        raw = [r for r in mine if r["estimate"] == "raw"]
        cal = [r for r in mine if r["estimate"] == "cal"]
        rows = [
            {
                "decile (true range)": a["group"],
                "n": int(a["n"]),
                "raw bias": _err(a["median_error"], p),
                "calibrated bias": _err(b["median_error"], p),
                "calibrated abs. error": _err(b["median_abs_error"], p),
                "rel.": f"{100 * b['median_rel_error']:.0f} %",
            }
            for a, b in zip(raw, cal, strict=True)
        ]
        out += [f"*{p}*", "", _table(pd.DataFrame(rows)), ""]
    return out


def _fidelity_table(table: pd.DataFrame) -> str:
    rows = []
    for r in table.to_dict("records"):
        p = str(r["parameter"])
        scale = _SCALE[p]
        rows.append(
            {
                "parameter": p,
                "estimate": _NAMES[str(r["estimate"])],
                "n (ESS)": f"{int(r['n'])} ({r['ess']:.0f})",
                "KS": f"{r['ks']:.3f}",
                "KS halves": f"{r['ks_halves']:.3f}",
                "W1": _err(r["w1"], p),
                "W1 halves": _err(r["w1_halves"], p),
                "W1 (log)": f"{r['w1_log']:.3f}",
                "KS blur": f"{r['ks_blur']:.3f}",
                "true q05/q50/q95": " / ".join(
                    f"{scale * r[f'true_q{q:02d}']:.3g}" for q in (5, 50, 95)
                ),
                "estimate q05/q50/q95": " / ".join(
                    f"{scale * r[f'est_q{q:02d}']:.3g}" for q in (5, 50, 95)
                ),
            }
        )
    return _table(pd.DataFrame(rows))


def write_report(result: ParamsEvaluation, out_dir: str | Path) -> Path:
    """Write tables, figures, ``metrics.json`` and ``report.md`` (module docstring)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = result.metrics
    pred = result.predictions
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=1, allow_nan=False) + "\n"
    )
    keep = [c for c in pred.columns if c != "scene_dir"]
    pred.loc[:, keep].to_parquet(out / "predictions.parquet", index=False)
    for name, table in result.tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    if result.real is not None:
        result.real.to_csv(out / "real_agreement.csv", index=False)
    figures = {
        "pred_vs_true": _fig_pred_vs_true(pred, out / "fig_pred_vs_true.png"),
        "bias": _fig_bias(pred, out / "fig_bias_vs_true.png"),
        "marginals": _fig_marginals(pred, result.weights, out / "fig_marginals.png"),
        "joint": _fig_joint(result, out / "fig_joint.png"),
        "clusters": _fig_clusters(result, out / "fig_clusters.png"),
        "coverage": _fig_coverage(result, out / "fig_coverage.png"),
        "real": None
        if result.real is None
        else _fig_real(result.real, out / "fig_real.png"),
    }
    bundle = result.bundle
    detector = bundle.detector or {}
    acc_cal = metrics["accuracy"]["cal"]
    lines = [
        "# Chirp-parameter decoder: evaluation",
        "",
        f"Bundle `{bundle.path}` (weights `{bundle.weights_sha256[:16]}`), trained on the "
        f"boxes of detector `{str(detector.get('weights_sha256'))[:16]}` "
        f"(`{detector.get('path')}`). Calibration: {bundle.calibration.method}, fitted on "
        f"val boxes with score >= {bundle.calibration.fitted_on.get('min_score')}.",
        "",
        f"Test population: {metrics['n']} {metrics['split']} chirps matched to a detector "
        f"box with score >= {metrics['min_score']} and a usable per-period truth "
        f"({metrics['n_unmatched']} further boxes matched no chirp). Ground truth: the "
        "emitter's noise-free per-EOD-period frequency (`chirpdetector.params.truth`).",
        "",
        "## Summary",
        "",
        *[
            f"- **{p}**: median abs. error {_err(acc_cal[p]['median_abs_error'], p)} "
            f"({100 * acc_cal[p]['median_rel_error']:.1f} %), shrinkage slope "
            f"{metrics['accuracy']['raw'][p]['shrinkage_slope']:.3f} -> "
            f"{acc_cal[p]['shrinkage_slope']:.3f} (calibrated)"
            for p in PARAMETERS
        ],
        f"- Flagged outside the validated domain: {100 * metrics['flags']['fraction_flagged']:.1f} % "
        f"of matched test boxes, {100 * (metrics['flags']['fraction_flagged_unmatched'] or 0):.1f} % "
        "of unmatched ones.",
        "",
        "## Accuracy before and after calibration",
        "",
        _accuracy_table(metrics),
        "",
        "For reference, reading the parameters off the detector box (box top minus "
        "the true baseline, box duration, true baseline minus box bottom):",
        "",
        "| parameter | median abs. error | median rel. error | median signed error |",
        "|---|---|---|---|",
        *[
            f"| {p} | {_err(v['median_abs_error'], p)} | {100 * v['median_rel_error']:.0f} % | {_err(v['median_error'], p)} |"
            for p, v in metrics["box_readout"].items()
        ],
        "",
        "![estimate vs truth](fig_pred_vs_true.png)",
        "",
        "![bias by decile](fig_bias_vs_true.png)",
        "",
        "## Breakdowns",
        "",
        *_breakdown_section(result.tables["breakdown"]),
        "## Uncertainty calibration",
        "",
        _table(
            result.tables["coverage"].drop(columns="rms_z_by_sd_quintile"),
        ),
        "",
        "![coverage](fig_coverage.png)",
        "",
        "## Distribution fidelity",
        "",
        "KS and Wasserstein-1 (W1) distances between estimated and true values of the "
        "same chirps; 'halves': the same distances between the true values of two "
        "random halves of the test chirps (sampling noise at this n); 'KS blur': the "
        "KS distance an ideal, conditionally unbiased estimate with the calibrated "
        "error spread would have (the truth plus its decile-wise centred, shuffled "
        "residuals). A calibrated KS near 'KS blur' means the remaining distortion "
        "is blur by measurement error, not a pull toward the training mix.",
        "",
        "**Test population** (templates drawn from the equalised mix)",
        "",
        _fidelity_table(result.tables["fidelity"]),
        "",
        f"**Natural template mix** (importance weights p_natural / p_equalised of each "
        f"template's excursion x duration bin; effective sample size {metrics['natural_prior'].get('ess', float('nan')):.0f} "
        f"of {metrics['natural_prior'].get('n')})",
        "",
        _fidelity_table(result.tables["fidelity_natural"]),
        "",
        "Weighted median signed error under the natural mix: "
        + "; ".join(
            f"{p} {_err(metrics['natural_conditional_bias']['raw'][p], p)} raw -> "
            f"{_err(metrics['natural_conditional_bias']['cal'][p], p)} calibrated"
            for p in PARAMETERS
        ),
        "",
        "![marginals](fig_marginals.png)",
        "",
        f"Joint height x FWHM (30 x 30 log bins): total variation distance to the "
        f"truth {metrics['joint']['tv_raw']:.3f} raw, {metrics['joint']['tv_cal']:.3f} "
        f"calibrated (halves of the truth: {metrics['joint']['tv_halves']:.3f}); natural "
        f"mix {metrics['joint_natural']['tv_raw']:.3f} / {metrics['joint_natural']['tv_cal']:.3f} "
        f"(halves {metrics['joint_natural']['tv_halves']:.3f}). Spearman(height, FWHM): "
        f"true {metrics['joint']['spearman_true']:.3f}, raw {metrics['joint']['spearman_raw']:.3f}, "
        f"calibrated {metrics['joint']['spearman_cal']:.3f}.",
        "",
        "![joint](fig_joint.png)",
        "",
        "## Cluster structure",
        "",
        "Valley-to-peak ratio of the density along the axis from one type to the "
        "other (height axis: small/short -> big/short; fwhm axis: big/short -> "
        "big/long; 0: separated, ~1: no valley), for all chirps and for in-domain "
        "chirps only ('_ok'); 'same side': fraction of two-type chirps whose "
        "estimate falls on the same side of the midpoint as their truth; 'ks_u': "
        "KS distance of estimated vs true positions on the axis.",
        "",
        _table(result.tables["clusters"].drop(columns="counts_per_tenth"))
        if len(result.tables["clusters"])
        else "(not enough chirps)",
        "",
        "Test chirps per tenth of each axis (within the band): "
        + "; ".join(
            f"{r['population']}: {r['counts_per_tenth']}"
            for r in result.tables["clusters"].to_dict("records")
            if str(r["population"]).endswith("two_types")
        )
        if len(result.tables["clusters"])
        else "",
        "",
        "![clusters](fig_clusters.png)" if figures["clusters"] else "",
        "",
    ]
    if result.real is not None:
        real = metrics.get("real_agreement", {})
        rows = [
            {
                "parameter": p,
                "n": v["n"],
                "median abs. difference": _err(v["median_abs_difference"], p),
                "median rel. difference": f"{100 * v['median_rel_difference']:.0f} %",
                "median difference": _err(v["median_difference"], p),
                "Pearson r": f"{v['pearson_r']:.2f}",
                "Spearman r": f"{v['spearman_r']:.2f}",
                "reference median": _err(v["ref_median"], p),
            }
            for p, v in real.items()
            if isinstance(v, dict)
        ]
        lines += [
            "## Real data: agreement with an independent trace estimate",
            "",
            f"{real.get('detected')} of {real.get('n')} harvested long chirps of the "
            "competition recordings were detected (and matched) by the detector. The "
            "reference is the per-period parameters of each chirp's extracted "
            "frequency trace; differences are agreement, not error. These chirps are "
            "also templates of the simulation library, and all are big chirps: real "
            "small chirps are not checked here.",
            "",
            _table(pd.DataFrame(rows)) if rows else "(too few detected)",
            "",
            "![real](fig_real.png)" if figures["real"] else "",
            "",
        ]
    lines += [
        "## Caveats",
        "",
        "- Validated on simulations (template chirps on simulated and hybrid "
        "backgrounds); on real data only the big harvested chirps are checked.",
        "- The decoder is only valid with the detector whose boxes it was trained on "
        "(weight hash recorded in the bundle).",
        "- Estimates are calibrated to be median-unbiased at every true value; their "
        "population distribution is the true one blurred by the measurement error.",
        "",
    ]
    path = out / "report.md"
    path.write_text("\n".join(lines))
    return path
