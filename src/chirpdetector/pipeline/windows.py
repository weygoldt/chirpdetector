"""Annotated spectrogram windows: a stratified sample per recording, and on demand.

v1 saved every predicted window as an annotated plot. Here a run saves a
small stratified sample per recording, and `render_window` re-renders any
window of any recording from the run's outputs, recomputing the
spectrogram from the raw data (with the detector's front end and click
removal), so nothing else needs to be stored.

Every window is written as ``<stem>.png`` and ``<stem>.npz``. The npz holds
what is needed to redraw it (e.g. for a paper figure):

``spec_db``
    Reference-window power in dB (float16), shape ``(n_freqs, n_times)``,
    cropped to the window's frequency range.
``times_s``, ``freqs_hz``
    Axes (recording clock).
``boxes``
    Structured array: the window's detections (`BOX_FIELDS`).
``track_ids``, ``track_offsets``, ``track_times_s``, ``track_freqs_hz``
    The tracks in the window: track ``k`` has the samples
    ``track_offsets[k]:track_offsets[k + 1]``.
``meta``
    JSON string: recording, run, window, model, front end.

Sampling (`sample_windows`): about 40 % uniformly random windows, 30 %
centred on the densest chirping (most chirps per window), 30 % centred on
uncertain chirps (not assigned, assignment confidence < 0.8, or score
within 0.15 of the operating threshold); windows do not overlap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gridtools.io import Tracks

from chirpdetector.bundle import Bundle
from chirpdetector.inference.spans import span_spectrogram

__all__ = [
    "BOX_FIELDS",
    "Window",
    "render_window",
    "sample_windows",
    "write_windows",
]

BOX_FIELDS = (
    "chirp_id",
    "t_start_s",
    "t_end_s",
    "f_low_hz",
    "f_high_hz",
    "score",
    "emitter_hz",
    "track_id",
    "assign_confidence",
    "operating",
    "kept",
    "chirp",
    "reattributed",
    "pp_harmonic_f_low_hz",
    "pp_harmonic_f_high_hz",
    "pp_harmonic_emitter_hz",
)
"""Fields of the ``boxes`` array of a window's npz (``reattributed``: 1 for
a re-attributed harmonic copy, whose detected box is in ``pp_harmonic_*``
and whose ``f_*``/``emitter_hz`` are halved)."""


class _Row:
    """Attribute access to a table row (a dict)."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name) from None


@dataclass(frozen=True)
class Window:
    """A window to render."""

    t0_s: float
    t1_s: float
    stratum: str


def _no_overlap(t0: float, t1: float, taken: list[Window]) -> bool:
    return all(t1 <= w.t0_s or t0 >= w.t1_s for w in taken)


def sample_windows(
    table: pd.DataFrame,
    span_s: float | tuple[float, float],
    *,
    n: int,
    window_s: float,
    seed: int,
    operating: float | None = None,
) -> list[Window]:
    """A stratified, non-overlapping sample of windows (module docstring)."""
    rng = np.random.default_rng(seed)
    lo, hi = (0.0, float(span_s)) if not isinstance(span_s, tuple) else span_s
    hi = max(lo + window_s, hi)
    taken: list[Window] = []
    chirp = table["chirp"].to_numpy(dtype=bool) if len(table) else np.zeros(0, bool)
    centres = table["t_center_s"].to_numpy(dtype=float) if len(table) else np.zeros(0)
    n_dense = round(0.3 * n)
    n_low = round(0.3 * n)

    def add(centre: float, stratum: str) -> bool:
        t0 = min(max(lo, centre - window_s / 2), hi - window_s)
        t1 = t0 + window_s
        if _no_overlap(t0, t1, taken):
            taken.append(Window(round(t0, 3), round(t1, 3), stratum))
            return True
        return False

    # dense: chirps per window around every chirp
    c = np.sort(centres[chirp])
    if len(c) and n_dense:
        counts = np.searchsorted(c, c + window_s / 2) - np.searchsorted(
            c, c - window_s / 2
        )
        for i in np.argsort(-counts, kind="stable"):
            if sum(w.stratum == "dense" for w in taken) >= n_dense:
                break
            add(float(c[i]), "dense")
    # uncertain chirps
    if len(table) and n_low:
        op = operating if operating is not None else 0.5
        uncertain = (
            (
                table["assign_status"].to_numpy(dtype=object) != "assigned"
            )  # incl. re-attributed
            | ~(table["assign_confidence"].fillna(0).to_numpy(dtype=float) >= 0.8)
            | (np.abs(table["score"].to_numpy(dtype=float) - op) <= 0.15)
        ) & (table["score"].to_numpy(dtype=float) >= max(0.0, op - 0.15))
        idx = np.flatnonzero(uncertain)
        rng.shuffle(idx)
        for i in idx:
            if sum(w.stratum == "uncertain" for w in taken) >= n_low:
                break
            add(float(centres[i]), "uncertain")
    # uniform fill
    tries = 0
    while len(taken) < n and tries < 50 * n:
        tries += 1
        add(float(rng.uniform(lo + window_s / 2, hi - window_s / 2)), "uniform")
    return sorted(taken, key=lambda w: w.t0_s)


def _tracks_in(
    tracks: Tracks | None, t0: float, t1: float
) -> tuple[list[float], list[np.ndarray], list[np.ndarray]]:
    ids: list[float] = []
    times: list[np.ndarray] = []
    freqs: list[np.ndarray] = []
    if tracks is None:
        return ids, times, freqs
    for track in tracks:
        if track.stop_s < t0 or track.start_s > t1:
            continue
        sel = (track.times_s >= t0) & (track.times_s <= t1)
        if sel.any():
            ids.append(float(track.ident))
            times.append(track.times_s[sel])
            freqs.append(track.freqs_hz[sel])
    return ids, times, freqs


def _colours(ids: list[float]) -> dict[float, Any]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    return {tid: cmap(k % 10) for k, tid in enumerate(sorted(ids))}


def render_window(
    path_stem: Path,
    recording: Any,
    bundle: Bundle,
    table: pd.DataFrame,
    tracks: Tracks | None,
    t0_s: float,
    t1_s: float,
    *,
    title: str = "",
    meta: dict[str, Any] | None = None,
    device: str | None = None,
    pad_hz: float = 60.0,
) -> dict[str, Any]:
    """Render one window to ``<stem>.png`` and ``<stem>.npz``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fs = float(recording.samplerate_hz)
    start = max(0, int(np.floor(t0_s * fs)))
    stop = min(int(recording.n_samples), int(np.ceil(t1_s * fs)) + 1)
    spec = span_spectrogram(bundle, recording, start, stop, device=device)
    power = spec.power[0].detach().float().cpu().numpy()
    freqs = spec.freqs_hz
    times = spec.times_s
    in_win = (
        table.loc[(table["t_end_s"] >= t0_s) & (table["t_start_s"] <= t1_s)]
        if len(table)
        else table
    )
    ids, ttimes, tfreqs = _tracks_in(tracks, t0_s, t1_s)
    f_values = [*(np.concatenate(tfreqs).tolist() if tfreqs else [])]
    if len(in_win):
        f_values += in_win["f_low_hz"].tolist() + in_win["f_high_hz"].tolist()
    if f_values:
        f_lo = max(freqs[0], min(f_values) - pad_hz)
        f_hi = min(freqs[-1], max(f_values) + pad_hz)
        if f_hi - f_lo < 200:
            mid = 0.5 * (f_lo + f_hi)
            f_lo, f_hi = max(freqs[0], mid - 100), min(freqs[-1], mid + 100)
    else:
        f_lo, f_hi = freqs[0], freqs[-1]
    rows = (freqs >= f_lo) & (freqs <= f_hi)
    db = 10 * np.log10(np.maximum(power[rows], 1e-30))
    top = float(np.quantile(db, 0.999)) if db.size else 0.0
    fig, ax = plt.subplots(figsize=(14, 5), layout="constrained")
    ax.imshow(
        db,
        origin="lower",
        aspect="auto",
        extent=(times[0], times[-1], freqs[rows][0], freqs[rows][-1]),
        cmap="magma",
        vmin=top - 50,
        vmax=top,
        interpolation="none",
    )
    assigned_ids = (
        sorted({float(v) for v in in_win["track_id"].dropna()}) if len(in_win) else []
    )
    colours = _colours(sorted(set(ids) | set(assigned_ids)))
    for tid, tt, ff in zip(ids, ttimes, tfreqs, strict=True):
        main = tid in assigned_ids
        ax.plot(
            tt,
            ff,
            color=colours[tid],
            lw=1.5 if main else 0.6,
            alpha=0.9 if main else 0.5,
        )
        ax.text(tt[0], ff[0], f" {tid:g}", color=colours[tid], fontsize=7, va="bottom")
    for rec in in_win.to_dict("records"):
        again = rec.get("pp_reattributed") == "harmonic"
        if again:  # draw the detected box, at the harmonic
            rec = {
                **rec,
                "f_low_hz": rec["pp_harmonic_f_low_hz"],
                "f_high_hz": rec["pp_harmonic_f_high_hz"],
                "emitter_hz": rec["pp_harmonic_emitter_hz"],
            }
        row = _Row(rec)
        tid = getattr(row, "track_id", np.nan)
        assigned = np.isfinite(tid)
        colour = colours.get(float(tid), "w") if assigned else "w"
        style = "-" if row.chirp else (":" if not row.kept else "--")
        lw = 1.6 if row.operating else 0.7
        alpha = 1.0 if row.operating else 0.6
        ax.add_patch(
            Rectangle(
                (row.t_start_s, row.f_low_hz),
                row.t_end_s - row.t_start_s,
                row.f_high_hz - row.f_low_hz,
                fill=False,
                edgecolor="red" if not row.kept else colour,
                ls=style,
                lw=lw,
                alpha=alpha,
            )
        )
        emitter = getattr(row, "emitter_hz", np.nan)
        if np.isfinite(emitter):
            ax.plot(
                [row.t_start_s, row.t_end_s],
                [emitter, emitter],
                color=colour,
                lw=0.8,
                ls=":",
            )
        conf = getattr(row, "assign_confidence", np.nan)
        label = f"{row.score:.2f}"
        if again:
            label += f" 2nd harmonic of id {tid:g}"
        elif assigned:
            label += f" id {tid:g}"
            if np.isfinite(conf):
                label += f" p={conf:.2f}"
        elif row.operating:
            label += f" {row.assign_status}"
        if not row.kept:
            label += f" [{row.pp_drop_reason}]"
        ax.text(
            row.t_start_s,
            row.f_high_hz,
            label,
            color="red" if not row.kept else colour,
            fontsize=6 if not row.operating else 7,
            va="bottom",
            alpha=alpha,
        )
    ax.set_xlim(t0_s, t1_s)
    ax.set_ylim(freqs[rows][0], freqs[rows][-1])
    ax.set_xlabel("time (s, recording clock)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(
        title
        + "  boxes: solid = chirp, dashed = below operating threshold, red dotted = "
        "flagged; colour = assigned track; dotted line = predicted emitter EODf",
        fontsize=8,
    )
    # the stem may contain dots (times): append the suffix, never replace one
    png = path_stem.with_name(f"{path_stem.name}.png")
    npz = path_stem.with_name(f"{path_stem.name}.npz")
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=90)
    plt.close(fig)
    boxes = np.zeros(len(in_win), dtype=[(f, "f8") for f in BOX_FIELDS])
    for f in BOX_FIELDS:
        if f == "reattributed" and "pp_reattributed" in in_win.columns:
            boxes[f] = in_win["pp_reattributed"].to_numpy(dtype=object) == "harmonic"
        elif f in in_win.columns:
            boxes[f] = in_win[f].to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            boxes[f] = np.nan
    offsets = np.cumsum([0, *(len(t) for t in ttimes)]).astype(np.int64)
    info = {
        "t0_s": t0_s,
        "t1_s": t1_s,
        "front_end": bundle.spectrogram.to_dict(),
        "channel": "reference window, power summed over electrodes, dB re 1",
        "click_removal": bundle.preprocessing.get("click_removal") is not None,
        **(meta or {}),
    }
    np.savez_compressed(
        npz,
        spec_db=db.astype(np.float16),
        times_s=times,
        freqs_hz=freqs[rows],
        boxes=boxes,
        track_ids=np.asarray(ids, dtype=np.float64),
        track_offsets=offsets,
        track_times_s=np.concatenate(ttimes) if ttimes else np.zeros(0),
        track_freqs_hz=np.concatenate(tfreqs) if tfreqs else np.zeros(0),
        meta=np.asarray(json.dumps(info, default=str)),
    )
    return {"png": png.name, "npz": npz.name, "n_boxes": len(in_win)}


def write_windows(
    directory: Path,
    windows: list[Window],
    recording: Any,
    bundle: Bundle,
    table: pd.DataFrame,
    tracks: Tracks | None,
    *,
    title: str = "",
    device: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render sampled windows into ``directory`` with an index ``windows.csv``."""
    rows = []
    for k, w in enumerate(windows):
        stem = directory / f"w{k:02d}_{w.stratum}_{w.t0_s:09.1f}s"
        info = render_window(
            stem,
            recording,
            bundle,
            table,
            tracks,
            w.t0_s,
            w.t1_s,
            title=f"{title} {w.stratum} window {w.t0_s:.1f}-{w.t1_s:.1f} s",
            meta={"stratum": w.stratum, **(meta or {})},
            device=device,
        )
        rows.append(
            {"window": k, "stratum": w.stratum, "t0_s": w.t0_s, "t1_s": w.t1_s, **info}
        )
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / "windows.csv", index=False)
    return {
        "n": len(rows),
        "strata": {
            s: sum(r["stratum"] == s for r in rows)
            for s in ("uniform", "dense", "uncertain")
        },
    }
