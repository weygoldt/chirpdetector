"""Print the key tables of the chirp-parameter study (from evaluate.py outputs).

Usage: summarize.py OUT
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PARAMS = ("height_hz", "fwhm_s", "width10_s", "undershoot_hz")


def fmt(row: pd.Series, p: str) -> str:
    if p == "undershoot_hz":
        return f"{row.med_abs:.1f} Hz (p90 {row.p90_abs:.0f})"
    unit = "Hz" if p == "height_hz" else "ms"
    return f"{row.med_abs:.1f} {unit} / {100 * row.med_rel:.0f}%"


def main() -> None:
    out = Path(sys.argv[1])
    m = pd.read_csv(out / "metrics.csv")
    print("## main table (visible test chirps): median |error| (abs / relative)")
    lines = ["| method | height | FWHM | 10% width | undershoot |", "|---|---|---|---|---|"]
    for method, g in m.groupby("method", sort=False):
        g = g.set_index("param")
        lines.append(f"| {method} | " + " | ".join(fmt(g.loc[p], p) for p in PARAMS) + " |")
    print("\n".join(lines))
    bd = pd.read_csv(out / "breakdown.csv")
    for cov in ("family", "height_bin", "fwhm_bin", "snr_bin", "df_bin", "fish_in_band", "overlap", "visible"):
        print(f"\n## by {cov}: median |error| height Hz / FWHM ms / undershoot Hz (median rel. height)")
        sub = bd[bd.covariate == cov]
        piv = {}
        for (method, level), g in sub.groupby(["method", "level"]):
            g = g.set_index("param")
            piv.setdefault(method, {})[level] = (
                f"{g.loc['height_hz', 'med_abs']:.0f}/{g.loc['fwhm_s', 'med_abs']:.1f}/"
                f"{g.loc['undershoot_hz', 'med_abs']:.1f} ({100 * g.loc['height_hz', 'med_rel']:.0f}%) n={int(g.loc['height_hz', 'n'])}"
            )
        print(pd.DataFrame(piv).T.to_string())
    fs = pd.read_csv(out / "breakdown_family_snr.csv")
    print("\n## family x SNR: median relative height error / median |FWHM error| ms")
    for method in ("net B_sum", "net C_top3", "net A_detector", "box (naive)"):
        g = fs[fs.method == method]
        if g.empty:
            continue
        h = g[g.param == "height_hz"].pivot(index="family", columns="snr_bin", values="med_rel")
        w = g[g.param == "fwhm_s"].pivot(index="family", columns="snr_bin", values="med_abs")
        print(method)
        print((100 * h).round(0).astype("Int64").astype(str).add("%").to_string())
        print(w.round(1).to_string())
    cal = out / "calibration.csv"
    if cal.exists():
        print("\n## calibration")
        print(pd.read_csv(cal).round(3).to_string())


if __name__ == "__main__":
    main()
