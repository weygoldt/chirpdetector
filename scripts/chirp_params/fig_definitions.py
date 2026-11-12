"""Figure: per-period ground truth and the parameter definitions on two chirps.

Usage: fig_definitions.py CACHE OUT_PNG
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gridtools.sim.scene import load_scene_model

from chirpdetector.params.truth import measure_period_params, per_period_trace

SCENES = Path("/mnt/data1/chirp_v2/scenes/train_v2/test")


def main() -> None:
    cache, out = Path(sys.argv[1]), Path(sys.argv[2])
    meta = pd.read_parquet(cache / "test" / "meta.parquet")
    picks = [
        meta[(meta.family == "small") & meta.visible & (meta.undershoot_hz > 5)].iloc[0],
        meta[(meta.family == "big") & meta.visible & (meta.fwhm_s > 0.08) & (meta.undershoot_hz > 5)].iloc[0],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, row in zip(axes, picks, strict=True):
        model = load_scene_model(SCENES / f"{int(row.seed):06d}")
        fs = model.samplerate_hz
        fish = int(row.emitter_id)
        s0 = int(round((row.t_support_start_s - 0.02) * fs))
        s1 = int(round((row.t_support_stop_s + 0.02) * fs))
        base = model.base_frequency(fish, s0, s1)
        inst = model.frequency(fish, s0, s1) - base
        tau, delta = per_period_trace(model.phase(fish, s0, s1), base, fs, s0 / fs)
        p = measure_period_params(tau, delta)
        t = np.arange(s0, s1) / fs
        t_ref = tau[np.argmax(delta)]
        ax.plot((t - t_ref) * 1e3, inst, color="0.7", lw=1, label="sample-level excursion (template)")
        ax.plot((tau - t_ref) * 1e3, delta, ".-", ms=2.5, lw=0.8, label="per-EOD-period excursion (truth)")
        ax.axhline(p.height_hz, color="tab:red", lw=0.6)
        ax.axhline(p.height_hz / 2, color="tab:green", lw=0.6, ls="--")
        ax.axhline(p.height_hz / 10, color="tab:purple", lw=0.6, ls=":")
        ax.axhline(-p.undershoot_hz, color="tab:brown", lw=0.6)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(
            f"{row.family}: height {p.height_hz:.0f} Hz, FWHM {p.fwhm_s * 1e3:.1f} ms, "
            f"10% width {p.width10_s * 1e3:.1f} ms, undershoot {p.undershoot_hz:.1f} Hz",
            fontsize=7.5,
        )
        ax.set_xlabel("time re peak (ms)")
        ax.set_ylabel("EODf - local baseline (Hz)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


if __name__ == "__main__":
    main()
