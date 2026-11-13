"""Side-by-side pictures of detectors on hard snippets of real recordings.

Metrics hide *how* a detector fails. A showcase renders the same short
snippets of real recordings for every model tried, so that models can be
compared by eye and the iterations tracked: one panel per snippet and
model, collected in a contact sheet (rows = snippets, columns = models, v1
first).

Each panel shows

* the electrode-summed power spectrogram with the 0.2048 s reference window
  (500-1500 Hz by default), on a fixed dB range per snippet (the same
  background in every column);
* the wavetracker tracks, one colour per fish;
* the model's boxes with their scores, in the colour of the assigned fish
  if the model assigns an emitter, grey otherwise;
* the emitter's EOD frequency at the chirp centre as a marker, when the
  model predicts it (``emitter_hz``; matched to the nearest live track with
  `assign_by_frequency`) or, for v1, its assigned emitter;
* hand labels (dashed, white) where the snippet lies in a labelled region;
* with a params bundle (`bundle_column(..., params=...)`), each box's
  decoded height and FWHM (``chirpdetector.params``), flagged boxes
  marked with ``?``.

The output directory keeps ``showcase.json`` (snippets and columns so far)
and one PNG per panel; running again with more bundles adds columns and
renders only the missing panels, so the sheet grows with the iterations.

Snippets come from a TOML file::

    [[snippet]]
    name = "close-df"
    recording = "/mnt/data2/2019_tube_competition/2020-03-16-10_00"
    t_start_s = 1234.0
    t_end_s = 1243.0
    f_min_hz = 500.0          # optional
    f_max_hz = 1500.0         # optional
    labels = "..."            # optional; default: the recording's audian labels
    reason = "two fish 6 Hz apart, both chirping"

Needs matplotlib (``uv sync --extra label``).
"""

from __future__ import annotations

import json
import logging
import math
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from chirpdetector.evaluation.assignment import assign_by_frequency
from chirpdetector.evaluation.predictions import V1_DETECTIONS_FILE, read_predictions
from chirpdetector.evaluation.schema import read_ground_truth

__all__ = [
    "Column",
    "Snippet",
    "bundle_column",
    "read_snippets",
    "render_showcase",
    "v1_column",
]

log = logging.getLogger(__name__)

LABEL_FILE = "traces-grid1-editable-labels.csv"
INDEX_FILE = "showcase.json"
EMITTER_TOLERANCE_HZ = 10.0
"""Largest distance of a predicted emitter frequency to the track it is assigned to."""
_PANEL_SIZE_IN = (6.4, 3.8)
_DPI = 110
_DYNAMIC_RANGE_DB = 60.0
_HEADROOM_DB = 10.0
_GREY = "0.6"


@dataclass(frozen=True)
class Snippet:
    """A short stretch of a real recording to look at."""

    name: str
    recording: str
    t_start_s: float
    t_end_s: float
    f_min_hz: float = 500.0
    f_max_hz: float = 1500.0
    reason: str = ""
    labels: str | None = None

    def __post_init__(self) -> None:
        if not self.t_end_s > self.t_start_s:
            msg = f"snippet {self.name}: t_end_s must exceed t_start_s"
            raise ValueError(msg)
        if not self.f_max_hz > self.f_min_hz:
            msg = f"snippet {self.name}: f_max_hz must exceed f_min_hz"
            raise ValueError(msg)

    @property
    def label_file(self) -> Path | None:
        """Hand labels of the recording, if any."""
        path = Path(self.labels) if self.labels else Path(self.recording) / LABEL_FILE
        return path if path.is_file() else None


def read_snippets(path: str | Path) -> list[Snippet]:
    """Snippets of a TOML file (``[[snippet]]`` tables, see the module docstring)."""
    with Path(path).open("rb") as f:
        data = tomllib.load(f)
    snippets = [Snippet(**entry) for entry in data.get("snippet", [])]
    names = [s.name for s in snippets]
    if len(set(names)) != len(names):
        msg = f"{path}: snippet names must be unique, got {names}"
        raise ValueError(msg)
    if not snippets:
        msg = f"{path}: no [[snippet]] tables"
        raise ValueError(msg)
    return snippets


@dataclass(frozen=True)
class Column:
    """A model in the showcase: v1's production detections or a bundle."""

    id: str
    label: str
    threshold: float
    bundle: str | None = None
    params: str | None = None


def v1_column(threshold: float = 0.5) -> Column:
    """v1's production detections (``chirpdetector_bboxes.csv``), its emitters as reference."""
    return Column(id="v1", label="v1 production (YOLOv8l)", threshold=threshold)


def bundle_column(
    bundle: Any, threshold: float | None = None, *, params: Any = None
) -> Column:
    """A bundle; its id is the first 12 hex digits of its weight hash.

    ``params`` (a params bundle or its directory) adds the decoded chirp
    parameters to the boxes; the id then also carries the first 6 hex
    digits of the decoder's weight hash.
    """
    from chirpdetector.bundle import Bundle, load_bundle

    bundle = bundle if isinstance(bundle, Bundle) else load_bundle(bundle)
    arch = bundle.model.get("architecture", bundle.model_type)
    run = bundle.training.get("run_id", "?")
    column_id, label, params_path = bundle.weights_sha256[:12], "", None
    if params is not None:
        from chirpdetector.params.bundle import ParamsBundle, load_params_bundle

        decoder = (
            params if isinstance(params, ParamsBundle) else load_params_bundle(params)
        )
        column_id += f"-p{decoder.weights_sha256[:6]}"
        label = f" + params {decoder.weights_sha256[:8]}"
        params_path = str(decoder.path)
    return Column(
        id=column_id,
        label=f"{arch} {bundle.weights_sha256[:8]} (run {run}){label}",
        threshold=bundle.score_threshold if threshold is None else float(threshold),
        bundle=str(bundle.path),
        params=params_path,
    )


# ---------------------------------------------------------------- data


@dataclass
class _Context:
    """What every panel of one snippet shares."""

    snippet: Snippet
    recording: Any
    power_db: NDArray[np.float64]
    extent: tuple[float, float, float, float]
    vmin: float
    vmax: float
    tracks: Any
    colours: dict[int, Any]
    labels: pd.DataFrame


def _background(snippet: Snippet, recording: Any, device: str | None) -> tuple:
    from chirpdetector.spectrogram import (
        SpectrogramConfig,
        compute_spectrogram_span,
        plan_span,
        reference_level_db,
    )

    config = SpectrogramConfig(
        extra_windows_s=(), f_min_hz=snippet.f_min_hz, f_max_hz=snippet.f_max_hz
    )
    fs = float(recording.samplerate_hz)
    start = max(0, math.floor(snippet.t_start_s * fs))
    stop = min(recording.n_samples, math.ceil(snippet.t_end_s * fs))
    plan = plan_span(start, stop, recording.n_samples, fs, config)
    samples = recording.read(plan.read_start, plan.read_stop)
    spec = compute_spectrogram_span(samples, plan, config, device=device)
    power = spec.power[0].double().cpu().numpy()
    power_db = 10 * np.log10(np.maximum(power, np.finfo(np.float64).tiny))
    ref = reference_level_db(spec) or float(np.median(power_db))
    vmax = ref + _HEADROOM_DB
    dt, df = spec.layout.hop_s, spec.layout.df_hz
    extent = (
        float(spec.times_s[0] - dt / 2),
        float(spec.times_s[-1] + dt / 2),
        float(spec.freqs_hz[0] - df / 2),
        float(spec.freqs_hz[-1] + df / 2),
    )
    return power_db, extent, vmax - _DYNAMIC_RANGE_DB, vmax


def _labels(snippet: Snippet) -> pd.DataFrame:
    path = snippet.label_file
    if path is None:
        return pd.DataFrame(columns=["t_start_s", "t_end_s", "f_low_hz", "f_high_hz"])
    truth = read_ground_truth(path)
    labels = truth.labels
    keep = (
        labels["region"].notna().to_numpy()
        & (labels["label"].to_numpy() == "chirp")
        & (labels["t_end_s"].to_numpy() >= snippet.t_start_s)
        & (labels["t_start_s"].to_numpy() <= snippet.t_end_s)
    )
    return labels.loc[keep].reset_index(drop=True)


def _context(snippet: Snippet, device: str | None) -> _Context:
    from gridtools.io import load_tracks, open_recording
    from matplotlib import colormaps

    recording = open_recording(snippet.recording)
    power_db, extent, vmin, vmax = _background(snippet, recording, device)
    try:
        tracks = load_tracks(snippet.recording)
    except (FileNotFoundError, ValueError):
        tracks = None
    colours: dict[int, Any] = {}
    if tracks is not None:
        palette = colormaps["tab10"]
        t = np.linspace(snippet.t_start_s, snippet.t_end_s, 50)
        live = [
            int(i)
            for i in tracks.ids
            if np.isfinite(tracks.track(int(i)).frequency_at(t)).any()
        ]
        colours = {ident: palette(k % 10) for k, ident in enumerate(live)}
    return _Context(
        snippet=snippet,
        recording=recording,
        power_db=power_db,
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        tracks=tracks,
        colours=colours,
        labels=_labels(snippet),
    )


def _detections(
    column: Column, ctx: _Context, detectors: dict[str, Any], device: str | None
) -> pd.DataFrame:
    snippet = ctx.snippet
    name = Path(snippet.recording).name
    if column.bundle is None:
        path = Path(snippet.recording) / V1_DETECTIONS_FILE
        if not path.is_file():
            return pd.DataFrame(
                columns=["t_start_s", "t_end_s", "f_low_hz", "f_high_hz", "score"]
            )
        preds = read_predictions(path)
    else:
        from chirpdetector.bundle import load_bundle
        from chirpdetector.inference import predict_span
        from chirpdetector.prediction import load_detector

        bundle = load_bundle(column.bundle)
        if column.id not in detectors:
            detectors[column.id] = load_detector(bundle, device=device)
        preds = predict_span(
            bundle,
            ctx.recording,
            snippet.t_start_s,
            snippet.t_end_s,
            name=name,
            score_threshold=column.threshold,
            detector=detectors[column.id],
            device=device,
            params=column.params,
        )
        if "emitter_hz" in preds.columns and ctx.tracks is not None:
            preds = assign_by_frequency(
                preds, {name: ctx.tracks}, tolerance_hz=EMITTER_TOLERANCE_HZ
            )
    centre = 0.5 * (preds["t_start_s"].to_numpy() + preds["t_end_s"].to_numpy())
    keep = (
        (preds["score"].to_numpy() >= column.threshold)
        & (centre >= snippet.t_start_s)
        & (centre <= snippet.t_end_s)
    )
    return preds.loc[keep].reset_index(drop=True)


# -------------------------------------------------------------- drawing


def _boxes(df: pd.DataFrame) -> NDArray[np.float64]:
    """``(n, 4)`` array of ``t_start_s, t_end_s, f_low_hz, f_high_hz``."""
    columns = ["t_start_s", "t_end_s", "f_low_hz", "f_high_hz"]
    return df[columns].to_numpy(dtype=np.float64).reshape(-1, 4)


def _render_panel(
    ctx: _Context, column: Column, preds: pd.DataFrame, path: Path
) -> None:
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    snippet = ctx.snippet
    fig = Figure(figsize=_PANEL_SIZE_IN, dpi=_DPI, layout="constrained")
    ax = fig.add_subplot()
    ax.imshow(
        ctx.power_db,
        origin="lower",
        aspect="auto",
        extent=ctx.extent,
        cmap="magma",
        vmin=ctx.vmin,
        vmax=ctx.vmax,
        interpolation="nearest",
    )
    if ctx.tracks is not None:
        t = np.linspace(snippet.t_start_s, snippet.t_end_s, 400)
        for ident, colour in ctx.colours.items():
            f = ctx.tracks.track(ident).frequency_at(t)
            ax.plot(t, f, color=colour, lw=0.8, alpha=0.9)
    for t0, t1, f0, f1 in _boxes(ctx.labels):
        ax.add_patch(
            Rectangle(
                (t0, f0),
                t1 - t0,
                f1 - f0,
                fill=False,
                ec="white",
                lw=0.9,
                ls=(0, (3, 2)),
            )
        )
    emitters = (
        preds["emitter_id"].to_numpy(dtype=object)
        if "emitter_id" in preds.columns
        else np.full(len(preds), None, dtype=object)
    )
    emitter_hz = (
        preds["emitter_hz"].to_numpy(dtype=np.float64)
        if "emitter_hz" in preds.columns
        else np.full(len(preds), np.nan)
    )
    scores = preds["score"].to_numpy(dtype=np.float64)
    for k, (t0, t1, f0, f1) in enumerate(_boxes(preds)):
        ident = emitters[k]
        if (
            ident is None
            or ident is pd.NA
            or (isinstance(ident, float) and math.isnan(ident))
        ):
            colour = _GREY
        else:
            colour = ctx.colours.get(int(float(ident)), _GREY)
        ax.add_patch(
            Rectangle((t0, f0), t1 - t0, f1 - f0, fill=False, ec=colour, lw=1.1)
        )
        text = f"{scores[k]:.2f}"
        if "height_hz" in preds.columns:
            row = preds.iloc[k]
            mark = "" if bool(row["params_ok"]) else "?"
            text += f" {row['height_hz']:.0f} Hz {1e3 * row['fwhm_s']:.0f} ms{mark}"
        ax.text(
            t0,
            f1 + 4,
            text,
            color=colour,
            fontsize=5.5,
            va="bottom",
        )
        if np.isfinite(emitter_hz[k]):
            centre = 0.5 * (t0 + t1)
            ax.plot(
                centre, emitter_hz[k], "o", ms=3.5, mfc=colour, mec="white", mew=0.5
            )
    ax.set_xlim(snippet.t_start_s, snippet.t_end_s)
    ax.set_ylim(snippet.f_min_hz, snippet.f_max_hz)
    ax.set_xlabel("time (s)", fontsize=7)
    ax.set_ylabel("frequency (Hz)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(
        f"{column.label}, threshold {column.threshold:g}: {len(preds)} detections\n"
        f"{snippet.name}: {snippet.reason}",
        fontsize=6.5,
        loc="left",
    )
    fig.savefig(path)


def _compose(panels: list[list[Path]], path: Path) -> None:
    from PIL import Image

    images = [[Image.open(p) for p in row] for row in panels]
    width = max(im.width for row in images for im in row)
    height = max(im.height for row in images for im in row)
    sheet = Image.new(
        "RGB",
        (width * max(len(r) for r in images), height * len(images)),
        "white",
    )
    for r, row in enumerate(images):
        for c, image in enumerate(row):
            sheet.paste(image.convert("RGB"), (c * width, r * height))
    sheet.save(path)


def _write_index(out: Path, snippets: list[Snippet], columns: list[Column]) -> None:
    lines = [
        "# Detector showcase",
        "",
        "Rows: snippets; columns: models (v1 first, then in the order added). "
        "Background: electrode-summed power, 0.2048 s window, fixed dB range per "
        "snippet. Tracks: thin lines, one colour per fish. Boxes: detections at the "
        "column's threshold, coloured by the assigned fish (grey: no emitter). "
        "Dots: the emitter's EOD frequency at the chirp centre. Dashed white: hand "
        "labels (only inside labelled regions). With a params bundle, each box "
        "also shows the decoded height and FWHM ('?': outside the decoder's "
        "validated domain).",
        "",
        "![contact sheet](contact_sheet.png)",
        "",
        "## Columns",
        "",
        "| id | model | threshold | bundle |",
        "|---|---|---|---|",
        *[
            f"| {c.id} | {c.label} | {c.threshold:g} | {c.bundle or '-'} |"
            for c in columns
        ],
        "",
        "## Snippets",
        "",
        "| name | recording | span (s) | band (Hz) | why |",
        "|---|---|---|---|---|",
        *[
            f"| {s.name} | {Path(s.recording).name} | {s.t_start_s:g}-{s.t_end_s:g} | "
            f"{s.f_min_hz:g}-{s.f_max_hz:g} | {s.reason} |"
            for s in snippets
        ],
        "",
    ]
    (out / "index.md").write_text("\n".join(lines))


def render_showcase(
    columns: Sequence[Column],
    snippets: Sequence[Snippet],
    out_dir: str | Path,
    *,
    device: str | None = None,
    force: bool = False,
) -> Path:
    """Render panels and the contact sheet (module docstring).

    Parameters
    ----------
    columns
        Models to add (`v1_column`, `bundle_column`); columns already in
        ``out_dir`` are kept, new ones are appended (v1 always first).
    snippets
        The rows; they replace the stored ones.
    out_dir
        Output directory (``showcase.json``, ``panels/``, ``contact_sheet.png``,
        ``index.md``).
    device
        Torch device for spectrograms and models.
    force
        Re-render panels that exist.

    Returns
    -------
    Path
        The contact sheet.
    """
    out = Path(out_dir)
    (out / "panels").mkdir(parents=True, exist_ok=True)
    index_path = out / INDEX_FILE
    stored: list[Column] = []
    if index_path.is_file():
        stored = [Column(**c) for c in json.loads(index_path.read_text())["columns"]]
    merged = list(stored)
    for column in columns:
        known = [k for k, c in enumerate(merged) if c.id == column.id]
        if known:
            merged[known[0]] = column
        else:
            merged.append(column)
    merged.sort(key=lambda c: c.id != "v1")  # stable: v1 first, others as added
    snippets = list(snippets)
    detectors: dict[str, Any] = {}
    grid: list[list[Path]] = []
    for snippet in snippets:
        ctx: _Context | None = None
        row = []
        for column in merged:
            path = out / "panels" / f"{snippet.name}__{column.id}.png"
            if force or not path.is_file():
                if ctx is None:
                    ctx = _context(snippet, device)
                preds = _detections(column, ctx, detectors, device)
                _render_panel(ctx, column, preds, path)
                log.info("%s / %s: %d detections", snippet.name, column.id, len(preds))
            row.append(path)
        grid.append(row)
    sheet = out / "contact_sheet.png"
    _compose(grid, sheet)
    index_path.write_text(
        json.dumps(
            {
                "columns": [asdict(c) for c in merged],
                "snippets": [asdict(s) for s in snippets],
            },
            indent=1,
        )
        + "\n"
    )
    _write_index(out, snippets, merged)
    return sheet
