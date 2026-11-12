"""Agreement of decoder predictions with the independent trace estimates (real data).

Usage: analyze_real.py REAL_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARAMS = ("height_hz", "fwhm_s", "width10_s", "undershoot_hz")
SCALE = {"height_hz": 1.0, "fwhm_s": 1e3, "width10_s": 1e3, "undershoot_hz": 1.0}
LABEL = {"height_hz": "height (Hz)", "fwhm_s": "FWHM (ms)", "width10_s": "10% width (ms)", "undershoot_hz": "undershoot (Hz)"}


def main() -> None:
    d = Path(sys.argv[1])
    meta = pd.read_parquet(d / "meta.parquet")
    pred = pd.read_parquet(d / "predictions.parquet")
    rows = []
    runs = sorted(pred["run"].unique())
    fig, axes = plt.subplots(len(PARAMS), len(runs), figsize=(2.6 * len(runs), 2.6 * len(PARAMS)), squeeze=False)
    for j, run in enumerate(runs):
        g = pred[pred["run"] == run].set_index("template_id").reindex(meta["template_id"])
        for i, p in enumerate(PARAMS):
            ref = meta[p].to_numpy() * SCALE[p]
            q = g[f"pred_{p}"].to_numpy() * SCALE[p]
            e = q - ref
            row = {"run": run, "param": p, "n": len(ref), "med_abs_diff": float(np.median(np.abs(e))), "p90_abs_diff": float(np.quantile(np.abs(e), 0.9)), "median_diff": float(np.median(e)), "pearson_r": float(np.corrcoef(ref, q)[0, 1])}
            if p != "undershoot_hz":
                row["med_rel_diff"] = float(np.median(np.abs(e) / ref))
            rows.append(row)
            ax = axes[i, j]
            ax.scatter(ref, q, s=3, alpha=0.4)
            lo, hi = np.quantile(np.concatenate([ref, q]), [0.005, 0.995])
            ax.plot([lo, hi], [lo, hi], "k-", lw=0.6)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(run, fontsize=7)
            if j == 0:
                ax.set_ylabel("decoder " + LABEL[p], fontsize=7)
            if i == len(PARAMS) - 1:
                ax.set_xlabel("trace estimate " + LABEL[p], fontsize=7)
    fig.suptitle(f"Real long chirps (n = {len(meta)}): decoder vs independent trace estimate", fontsize=9)
    fig.tight_layout()
    fig.savefig(d / "fig_real_agreement.png", dpi=110)
    table = pd.DataFrame(rows)
    table.to_csv(d / "agreement.csv", index=False)
    with pd.option_context("display.width", 200):
        print(table.round(3).to_string())


if __name__ == "__main__":
    main()
