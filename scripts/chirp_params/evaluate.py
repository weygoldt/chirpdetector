"""Evaluate decoders and baselines against the per-period ground truth (test seeds).

Methods: network runs (``RUNS/<name>/test_predictions.parquet``), the naive
box estimator (smear offsets fitted on val), the ridge tracker on the
rectified crops B (electrode sum), C (3 strongest electrodes) and K (clean
emitter-only render), each raw and bias-corrected (median log ratio on val).

Writes ``metrics.csv``, ``breakdown.csv``, ``calibration.csv``,
``predictions.parquet`` and figures into OUT.

Usage: evaluate.py CACHE RUNS OUT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from chirpdetector.params.crops import RectifiedConfig
from chirpdetector.params.estimators import box_estimate, fit_box_offsets, ridge_estimate

PARAMS = ("height_hz", "fwhm_s", "width10_s", "undershoot_hz")
LABEL = {"height_hz": "height (Hz)", "fwhm_s": "FWHM (ms)", "width10_s": "10% width (ms)", "undershoot_hz": "undershoot (Hz)"}
SCALE = {"height_hz": 1.0, "fwhm_s": 1e3, "width10_s": 1e3, "undershoot_hz": 1.0}
RCFG = RectifiedConfig()
SNR_BINS = [-np.inf, -10, 0, 10, 20, np.inf]
DF_BINS = [0, 5, 20, 50, 150, np.inf]


def ridge_table(cache: Path, split: str, variant: str, meta: pd.DataFrame) -> pd.DataFrame:
    x = np.load(cache / split / f"{variant}.npy", mmap_mode="r")
    span = (meta["span_t1_s"] - meta["span_t0_s"]).to_numpy()
    rows = [ridge_estimate(np.asarray(x[int(r)]), span[i], RCFG) for i, r in enumerate(meta["row"])]
    return pd.DataFrame(rows)


def log_bias(est: pd.DataFrame, meta: pd.DataFrame) -> dict[str, float]:
    out = {}
    for p in PARAMS:
        off = 1.0 if p == "undershoot_hz" else 0.0
        r = np.log(est[p] + off) - np.log(meta[p] + off)
        out[p] = float(np.nanmedian(r[meta["visible"].to_numpy()]))
    return out


def apply_bias(est: pd.DataFrame, bias: dict[str, float]) -> pd.DataFrame:
    out = est.copy()
    for p in PARAMS:
        off = 1.0 if p == "undershoot_hz" else 0.0
        out[p] = np.exp(np.log(est[p] + off) - bias[p]) - off
    return out


def metrics(true: np.ndarray, pred: np.ndarray, param: str) -> dict[str, float]:
    ok = np.isfinite(true)
    n_all = int(ok.sum())
    fail = np.isfinite(true) & ~np.isfinite(pred)
    ok &= np.isfinite(pred)
    t, p = true[ok] * SCALE[param], pred[ok] * SCALE[param]
    e = p - t
    out = {
        "n": n_all,
        "fail_frac": float(fail.sum() / max(n_all, 1)),
        "med_abs": float(np.median(np.abs(e))) if e.size else np.nan,
        "p90_abs": float(np.quantile(np.abs(e), 0.9)) if e.size else np.nan,
        "bias": float(np.median(e)) if e.size else np.nan,
    }
    if param != "undershoot_hz":
        rel = np.abs(e) / t
        out["med_rel"] = float(np.median(rel)) if e.size else np.nan
        out["frac_within_10pct"] = float(np.mean(rel <= 0.1)) if e.size else np.nan
        out["r_log"] = float(np.corrcoef(np.log(t), np.log(np.clip(p, 1e-6, None)))[0, 1]) if e.size > 2 else np.nan
    else:
        out["frac_within_5hz"] = float(np.mean(np.abs(e) <= 5)) if e.size else np.nan
        out["r_log"] = float(np.corrcoef(np.log(t + 1), np.log(np.clip(p, 0, None) + 1))[0, 1]) if e.size > 2 else np.nan
    return out


def covariates(meta: pd.DataFrame) -> pd.DataFrame:
    cov = pd.DataFrame(index=meta.index)
    cov["family"] = meta["family"]
    cov["height_bin"] = pd.cut(meta["height_hz"], [0, 75, 150, 300, np.inf]).astype(str)
    cov["fwhm_bin"] = pd.cut(meta["fwhm_s"] * 1e3, [0, 5, 10, 20, 50, np.inf]).astype(str)
    cov["snr_bin"] = pd.cut(meta["snr_db"], SNR_BINS).astype(str)
    df = meta["min_df_all_hz"].fillna(np.inf)
    cov["df_bin"] = np.where(np.isinf(df), "single fish", pd.cut(df, DF_BINS).astype(str))
    cov["fish_in_band"] = meta["fish_in_band"].map({True: "fish in band", False: "band free"})
    cov["overlap"] = meta["overlaps_other_chirp"].map({True: "overlap", False: "no overlap"})
    cov["visible"] = meta["visible"].map({True: "visible", False: "invisible"})
    return cov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path)
    ap.add_argument("runs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--no-ridge", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    test = pd.read_parquet(args.cache / "test" / "meta.parquet")
    val = pd.read_parquet(args.cache / "val" / "meta.parquet")
    test = test[np.isfinite(test["height_hz"]) & np.isfinite(test["fwhm_s"])].reset_index(drop=True)
    val = val[np.isfinite(val["height_hz"]) & np.isfinite(val["fwhm_s"])].reset_index(drop=True)
    vis = test["visible"].to_numpy()

    est: dict[str, pd.DataFrame] = {}
    offsets = fit_box_offsets(val[val["visible"]])
    est["box (naive)"] = box_estimate(test, offsets)
    # the footprint box comes from the clean render (an oracle detector);
    # jitter it like the training boxes to mimic a real detector's scatter
    jrng = np.random.default_rng(3)
    jit = test.copy()
    for col, sd in (("box_t0_s", 0.01), ("box_t1_s", 0.01), ("box_f0_hz", 5.0), ("box_f1_hz", 5.0)):
        jit[col] = jit[col] + jrng.normal(0, sd, len(jit))
    est["box (naive) [jittered box]"] = box_estimate(jit, offsets)
    if not args.no_ridge:
        for variant in ("B", "C", "K"):
            print("ridge", variant, file=sys.stderr, flush=True)
            cached = {sp: args.out / f"ridge_{variant}_{sp}.parquet" for sp in ("test", "val")}
            if all(f.exists() for f in cached.values()):
                r_test, r_val = (pd.read_parquet(cached[sp]) for sp in ("test", "val"))
            else:
                r_test = ridge_table(args.cache, "test", variant, test)
                r_val = ridge_table(args.cache, "val", variant, val)
                r_test.to_parquet(cached["test"])
                r_val.to_parquet(cached["val"])
            name = {"B": "ridge B (sum)", "C": "ridge C (top-3)", "K": "ridge K (clean, ceiling)"}[variant]
            est[name] = r_test
            est[name + " +bias corr."] = apply_bias(r_test, log_bias(r_val, val))
    for run in sorted(p for p in args.runs.iterdir() if (p / "test_predictions.parquet").exists()):
        pred = pd.concat(
            [pd.read_parquet(f) for f in [run / "test_predictions.parquet", *sorted(run.glob("pred_*.parquet"))]],
            ignore_index=True,
        )
        for cond, g in pred.groupby("condition"):
            g = g.set_index("chirp_row").reindex(test["row"])
            name = f"net {run.name}" + ("" if cond == "true_baseline" else f" [{cond}]")
            df = pd.DataFrame({p: g[f"pred_{p}"].to_numpy() for p in PARAMS})
            for p in PARAMS:
                df[f"sigma_{p}"] = g[f"sigma_log_{p}"].to_numpy()
            est[name] = df

    rows = []
    for name, e in est.items():
        for p in PARAMS:
            rows.append({"method": name, "param": p, "subset": "visible", **metrics(test[p].to_numpy()[vis], e[p].to_numpy()[vis], p)})
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "metrics.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(table.pivot_table(index="method", columns="param", values="med_abs").round(3))
        print(table.pivot_table(index="method", columns="param", values="med_rel").round(3))

    cov = covariates(test)
    rows = []
    for name, e in est.items():
        if "[" in name or "+bias" in name:
            continue
        for col in cov.columns:
            for level, idx in cov.groupby(col).groups.items():
                idx = np.asarray(idx)
                if col != "visible":
                    idx = idx[vis[idx]]
                for p in PARAMS:
                    rows.append({"method": name, "covariate": col, "level": level, "param": p, **metrics(test[p].to_numpy()[idx], e[p].to_numpy()[idx], p)})
    bd = pd.DataFrame(rows)
    bd.to_csv(args.out / "breakdown.csv", index=False)

    # family x SNR for the main table
    rows = []
    for name, e in est.items():
        if "[" in name or "+bias" in name:
            continue
        for (fam, snr), g in pd.DataFrame({"family": test["family"], "snr": cov["snr_bin"]})[vis].groupby(["family", "snr"]):
            idx = g.index.to_numpy()
            for p in PARAMS:
                rows.append({"method": name, "family": fam, "snr_bin": snr, "param": p, **metrics(test[p].to_numpy()[idx], e[p].to_numpy()[idx], p)})
    pd.DataFrame(rows).to_csv(args.out / "breakdown_family_snr.csv", index=False)

    # uncertainty calibration and selective prediction for networks
    rows = []
    for name, e in est.items():
        if not name.startswith("net") or "[" in name:
            continue
        for p in PARAMS:
            off = 1.0 if p == "undershoot_hz" else 0.0
            t = np.log(test[p].to_numpy() + off)[vis]
            q = np.log(np.clip(e[p].to_numpy(), 0, None) + off)[vis]
            s = e[f"sigma_{p}"].to_numpy()[vis]
            z = (q - t) / s
            row = {"method": name, "param": p, "cover_68": float(np.mean(np.abs(z) < 1)), "cover_95": float(np.mean(np.abs(z) < 1.96)), "spearman_sigma_abs_err": float(pd.Series(s).corr(pd.Series(np.abs(q - t)), method="spearman"))}
            for keep in (0.5, 0.8):
                sel = s <= np.quantile(s, keep)
                row[f"med_abs_keep{int(keep * 100)}"] = metrics(test[p].to_numpy()[vis][sel], e[p].to_numpy()[vis][sel], p)["med_abs"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(args.out / "calibration.csv", index=False)

    pred_out = test[["row", "seed", "chirp_id", "family", "snr_db", "min_df_all_hz", "visible", *PARAMS]].copy()
    for name, e in est.items():
        for p in PARAMS:
            pred_out[f"{name}::{p}"] = e[p].to_numpy()
    pred_out.to_parquet(args.out / "predictions.parquet")
    figures(test, est, bd, args.out)


def figures(test: pd.DataFrame, est: dict[str, pd.DataFrame], bd: pd.DataFrame, out: Path) -> None:
    vis = test["visible"].to_numpy()
    show = [m for m in est if ("[" not in m and "+bias" not in m)]
    fig, axes = plt.subplots(len(PARAMS), len(show), figsize=(2.6 * len(show), 2.6 * len(PARAMS)), squeeze=False)
    colors = test["family"].map({"small": "tab:blue", "big": "tab:orange"}).to_numpy()[vis]
    for j, m in enumerate(show):
        for i, p in enumerate(PARAMS):
            ax = axes[i, j]
            t = test[p].to_numpy()[vis] * SCALE[p]
            q = est[m][p].to_numpy()[vis] * SCALE[p]
            ax.scatter(t, q, s=1, c=colors, alpha=0.3, rasterized=True)
            lim = (np.nanquantile(t, 0.001), np.nanquantile(t, 0.999))
            if p == "undershoot_hz":
                ax.set_xscale("symlog", linthresh=1)
                ax.set_yscale("symlog", linthresh=1)
                lim = (0, lim[1])
            else:
                ax.set_xscale("log")
                ax.set_yscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.plot(lim, lim, "k-", lw=0.6)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            if i == 0:
                ax.set_title(m, fontsize=7)
            if j == 0:
                ax.set_ylabel("pred " + LABEL[p], fontsize=7)
            if i == len(PARAMS) - 1:
                ax.set_xlabel("true " + LABEL[p], fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle("Predicted vs per-period ground truth, visible test chirps (blue small, orange big)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_pred_vs_true.png", dpi=110)
    plt.close(fig)

    for cov, fname in (("snr_bin", "fig_error_vs_snr.png"), ("df_bin", "fig_error_vs_df.png"), ("height_bin", "fig_error_vs_height.png"), ("fwhm_bin", "fig_error_vs_fwhm.png")):
        sub = bd[bd["covariate"] == cov]
        levels = sorted(sub["level"].unique(), key=lambda s: (s == "single fish", float(s.split(",")[0].strip("(")) if s.startswith("(") else 0))
        fig, axes = plt.subplots(1, len(PARAMS), figsize=(4 * len(PARAMS), 3.3))
        for ax, p in zip(axes, PARAMS, strict=True):
            for m in show:
                g = sub[(sub["method"] == m) & (sub["param"] == p)].set_index("level").reindex(levels)
                ax.plot(range(len(levels)), g["med_abs"], marker="o", ms=3, label=m)
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels(levels, rotation=40, fontsize=7, ha="right")
            ax.set_yscale("log")
            ax.set_title("median |error| " + LABEL[p], fontsize=8)
        axes[0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
