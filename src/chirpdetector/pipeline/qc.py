"""Per-recording QC report: one PNG and a short Markdown summary.

Panels: chirp rate over time per track; score histogram (kept vs flagged by
post-processing, operating threshold marked); distance of the predicted
emitter frequency to the nearest live track (tolerance marked); assignment
confidence of assigned chirps; assignment status and post-processing
counts; decoder flags; click removal over time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chirpdetector.pipeline.assign import ATTRIBUTED

__all__ = ["write_qc"]

_TOP_TRACKS = 6


def _rate_panel(ax: Any, table: pd.DataFrame, duration_s: float, bin_s: float) -> None:
    chirps = table.loc[table["chirp"].to_numpy(dtype=bool)]
    edges = np.arange(0.0, max(duration_s, bin_s) + bin_s, bin_s)
    centres = 0.5 * (edges[:-1] + edges[1:]) / 3600
    attributed = np.isin(chirps["assign_status"].to_numpy(dtype=object), ATTRIBUTED)
    assigned = chirps.loc[attributed]
    counts = assigned["track_id"].value_counts()
    top = counts.index[:_TOP_TRACKS].tolist()
    for tid in top:
        t = assigned.loc[assigned["track_id"] == tid, "t_center_s"].to_numpy()
        h, _ = np.histogram(t, edges)
        ax.plot(
            centres, h / (bin_s / 60), lw=1.2, label=f"track {tid:g} (n={counts[tid]})"
        )
    rest = assigned.loc[~assigned["track_id"].isin(top), "t_center_s"].to_numpy()
    if len(rest):
        h, _ = np.histogram(rest, edges)
        ax.plot(
            centres,
            h / (bin_s / 60),
            lw=1,
            color="0.5",
            label=f"other tracks (n={len(rest)})",
        )
    un = chirps.loc[~attributed, "t_center_s"].to_numpy()
    if len(un):
        h, _ = np.histogram(un, edges)
        ax.plot(
            centres,
            h / (bin_s / 60),
            lw=1,
            ls="--",
            color="k",
            label=f"unassigned (n={len(un)})",
        )
    ax.set_xlabel("time (h)")
    ax.set_ylabel(f"chirps / min ({bin_s / 60:g}-min bins)")
    n_again = int(
        np.sum(chirps["assign_status"].to_numpy(dtype=object) == "reattributed")
    )
    ax.set_title(
        "chirp rate per track (chirps = operating & kept; "
        f"{n_again} re-attributed harmonics included)"
    )
    if len(chirps):
        ax.legend(fontsize=7, frameon=False)


def write_qc(
    directory: Path,
    table: pd.DataFrame,
    chunks: pd.DataFrame,
    meta: dict[str, Any],
    duration_s: float,
    tolerance_hz: float | None = None,
) -> dict[str, Any]:
    """Write ``qc.png`` and ``qc.md`` into ``directory``; return a summary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    op = float(meta.get("operating_threshold", 0.5))
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), layout="constrained")
    name = meta.get("recording", "")
    fig.suptitle(
        f"{name}: {meta.get('model', '')} {meta.get('weights_short', '')}, run {meta.get('run', '')}",
        fontsize=11,
    )
    if len(table):
        _rate_panel(axes[0, 0], table, duration_s, 300.0)
        # score histogram
        ax = axes[0, 1]
        bins = np.linspace(0, 1, 41)
        kept = table["kept"].to_numpy(dtype=bool)
        ax.hist(
            [table.loc[kept, "score"], table.loc[~kept, "score"]],
            bins=bins,
            stacked=True,
            label=["kept", "flagged (post-processing)"],
            color=["C0", "C3"],
        )
        ax.axvline(op, color="k", ls="--", lw=1, label=f"operating {op:g}")
        ax.set_yscale("log")
        ax.set_xlabel("score")
        ax.set_ylabel("detections")
        ax.set_title("score (all stored detections)")
        ax.legend(fontsize=7, frameon=False)
        # emitter-track distance
        ax = axes[0, 2]
        chirp = table["chirp"].to_numpy(dtype=bool)
        d = table.loc[chirp, "nearest_df_hz"].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        if len(d):
            ax.hist(np.clip(d, 0, 100), bins=np.linspace(0, 100, 101), color="C0")
            ax.set_yscale("log")
        tol = tolerance_hz
        if tol is None:
            tol = meta.get("assignment_tolerance_hz") or None
        if tol:
            ax.axvline(tol, color="k", ls="--", lw=1, label=f"tolerance {tol:g} Hz")
            ax.legend(fontsize=7, frameon=False)
        ax.set_xlabel("|emitter_hz - nearest live track| (Hz, clipped at 100)")
        ax.set_ylabel("chirps")
        ax.set_title(f"emitter to nearest track (n={len(d)})")
        # confidence
        ax = axes[1, 0]
        conf = table.loc[
            chirp & (table["assign_status"] == "assigned"), "assign_confidence"
        ]
        conf = conf.to_numpy(dtype=float)
        conf = conf[np.isfinite(conf)]
        if len(conf):
            ax.hist(conf, bins=np.linspace(0, 1, 21), color="C2")
        ax.set_xlabel("assign_confidence")
        ax.set_ylabel("assigned chirps")
        ax.set_title("assignment confidence")
        # status / postprocess counts
        ax = axes[1, 1]
        status = table.loc[chirp, "assign_status"].value_counts()
        dropped = table.loc[
            table["operating"].to_numpy(dtype=bool) & ~kept, "pp_drop_reason"
        ].value_counts()
        labels = [f"chirps: {k}" for k in status.index] + [
            f"flagged: {k}" for k in dropped.index
        ]
        values = list(status.to_numpy()) + list(dropped.to_numpy())
        ax.barh(labels, values, color=["C0"] * len(status) + ["C3"] * len(dropped))
        for i, v in enumerate(values):
            ax.text(v, i, f" {v}", va="center", fontsize=8)
        ax.set_title(f"operating threshold {op:g}: status and post-processing")
        # ΔF of the assigned track to the closest other
        ax = axes[1, 2]
        dfo = table.loc[chirp, "track_df_other_hz"].to_numpy(dtype=float)
        dfo = dfo[np.isfinite(dfo)]
        if len(dfo):
            ax.hist(np.clip(dfo, 0, 300), bins=np.linspace(0, 300, 61), color="C4")
        ax.set_xlabel("ΔF nearest track to closest other live track (Hz, clipped 300)")
        ax.set_ylabel("chirps")
        ax.set_title("ΔF at the chirp")
        # decoder flags
        ax = axes[2, 0]
        if "params_flag" in table.columns:
            flags = table.loc[chirp, "params_flag"].astype(str)
            names = ["ok", "edge", "band", "size", "uncertain"]
            vals = [int(table.loc[chirp, "params_ok"].sum())] + [
                int(flags.str.contains(f).sum()) for f in names[1:]
            ]
            ax.bar(names, vals, color="C5")
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=8)
            ax.set_title("decoder flags (chirps)")
        else:
            ax.set_title("no decoder")
        # decoder parameters
        ax = axes[2, 1]
        if "height_hz" in table.columns:
            ok = chirp & table["params_ok"].to_numpy(dtype=bool)
            ax.scatter(
                table.loc[ok, "fwhm_s"] * 1000,
                table.loc[ok, "height_hz"],
                s=3,
                alpha=0.4,
            )
            if ok.any():  # a log axis without data cannot be drawn
                ax.set_xscale("log")
                ax.set_yscale("log")
            ax.set_xlabel("FWHM (ms)")
            ax.set_ylabel("height (Hz)")
            ax.set_title(f"decoder parameters (params_ok chirps, n={int(ok.sum())})")
    # click removal
    ax = axes[2, 2]
    if len(chunks) and chunks["click_removal"].to_numpy(dtype=bool).any():
        ax.plot(
            0.5 * (chunks["t_start_s"] + chunks["t_end_s"]) / 3600,
            100 * chunks["click_fraction"],
            lw=1,
        )
        ax.set_ylabel("% samples repaired")
        ax.set_title(
            f"click removal ({int(chunks['n_click_runs'].to_numpy().sum())} runs)"
        )
    else:
        ax.set_title("click removal not applied")
    ax.set_xlabel("time (h)")
    png = directory / "qc.png"
    fig.savefig(png, dpi=80)
    plt.close(fig)
    summary = _summary(table, chunks, op)
    (directory / "qc.md").write_text(_markdown(name, meta, summary))
    return {"png": "qc/qc.png", "md": "qc/qc.md", **summary}


def _summary(table: pd.DataFrame, chunks: pd.DataFrame, op: float) -> dict[str, Any]:
    if not len(table):
        return {"stored": 0}
    chirp = table["chirp"].to_numpy(dtype=bool)
    operating = table["operating"].to_numpy(dtype=bool)
    out: dict[str, Any] = {
        "stored": len(table),
        "operating": int(operating.sum()),
        "chirps": int(chirp.sum()),
        "flagged": {
            str(k): int(v)
            for k, v in table.loc[
                operating & ~table["kept"].to_numpy(dtype=bool), "pp_drop_reason"
            ]
            .value_counts()
            .items()
        },
        "status": {
            str(k): int(v)
            for k, v in table.loc[chirp, "assign_status"].value_counts().items()
        },
        "reattributed": int(
            np.sum(table["pp_reattributed"].to_numpy(dtype=object) == "harmonic")
        )
        if "pp_reattributed" in table.columns
        else 0,
    }
    d = table.loc[chirp, "nearest_df_hz"].to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    if len(d):
        out["nearest_df_hz_quantiles"] = {
            str(q): float(np.quantile(d, q)) for q in (0.5, 0.9, 0.95, 0.99)
        }
    if len(chunks):
        out["click_fraction_mean"] = float(
            np.mean(chunks["click_fraction"].to_numpy(dtype=float))
        )
    return out


def _markdown(name: str, meta: dict[str, Any], s: dict[str, Any]) -> str:
    lines = [
        f"# QC: {name}",
        "",
        f"Run `{meta.get('run')}`, model {meta.get('model')} `{meta.get('weights_short')}`, "
        f"operating threshold {meta.get('operating_threshold')}, store threshold "
        f"{meta.get('store_threshold')}.",
        "",
        "![qc](qc.png)",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| detections stored | {s.get('stored', 0)} |",
        f"| above operating threshold | {s.get('operating', 0)} |",
        f"| chirps (operating, not flagged) | {s.get('chirps', 0)} |",
    ]
    for k, v in (s.get("flagged") or {}).items():
        lines.append(f"| flagged as {k} (operating) | {v} |")
    for k, v in (s.get("status") or {}).items():
        lines.append(f"| chirps {k} | {v} |")
    lines.append(
        f"| harmonic copies re-attributed to their fish | {s.get('reattributed', 0)} |"
    )
    for q, v in (s.get("nearest_df_hz_quantiles") or {}).items():
        lines.append(f"| nearest-track distance q{q} | {v:.2f} Hz |")
    if "click_fraction_mean" in s:
        lines.append(
            f"| samples repaired (clicks) | {100 * s['click_fraction_mean']:.3f} % |"
        )
    return "\n".join(lines) + "\n"
