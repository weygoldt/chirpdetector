"""Command-line interface: ``chirpdetector <command>``."""

from __future__ import annotations

import getpass
import logging
from pathlib import Path
from typing import Annotated

import typer

import chirpdetector

__all__ = ["app"]

DEFAULT_LABEL_DIR = Path("/mnt/data1/chirp_v2/labels/emitters")

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Detect chirps of wave-type weakly electric fish and assign them to their emitters.",
)


@app.callback()
def main(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Debug logging.")
    ] = False,
) -> None:
    """chirpdetector command-line tools."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_grid(grid: str) -> tuple[int, int]:
    try:
        rows, cols = (int(v) for v in grid.lower().split("x"))
    except ValueError:
        msg = f"--grid must look like 4x6, got {grid!r}"
        raise typer.BadParameter(msg) from None
    if rows < 1 or cols < 1:
        msg = "--grid needs at least one row and column"
        raise typer.BadParameter(msg)
    return rows, cols


def _parse_bins(bins: str) -> tuple[float, ...]:
    try:
        return tuple(float(v) for v in bins.split(","))
    except ValueError:
        msg = f"--bins must be comma-separated numbers, got {bins!r}"
        raise typer.BadParameter(msg) from None


@app.command("label-emitters")
def label_emitters(
    trial_dirs: Annotated[
        list[Path],
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Trial directories with chirpdetector_bboxes.csv, tracks and the recording.",
        ),
    ],
    out: Annotated[
        Path, typer.Option(help="Output directory (queue, event log, label CSVs).")
    ] = DEFAULT_LABEL_DIR,
    per_bin: Annotated[
        int, typer.Option(help="Detections sampled per trial and ΔF bin.")
    ] = 20,
    seed: Annotated[int, typer.Option(help="Sampling seed.")] = 0,
    labeler: Annotated[
        str, typer.Option(help="Name stored with every label.")
    ] = getpass.getuser(),
    grid: Annotated[str, typer.Option(help="Cutouts per page, ROWSxCOLS.")] = "4x6",
    bins: Annotated[
        str, typer.Option(help="Lower edges of the ΔF bins in Hz; the last is open.")
    ] = "0,10,25,50,100",
    pad_s: Annotated[float, typer.Option(help="Time padding around the box.")] = 0.15,
    pad_hz: Annotated[
        float, typer.Option(help="Frequency padding around tracks and box.")
    ] = 40.0,
    tolerance_hz: Annotated[
        float,
        typer.Option(help="Max. distance of a left click from a track baseline."),
    ] = 15.0,
    nfft: Annotated[
        int,
        typer.Option(
            help=(
                "Spectrogram window in samples: 4096 (205 ms at 20 kHz) separates "
                "close baselines; 2048 or 1024 show the chirp's shape better."
            )
        ),
    ] = 4096,
) -> None:
    """Label chirp emitters on a grid of spectrogram cutouts.

    Left click on the emitter's baseline = chirp from that fish; right
    click = not a chirp; middle click = chirp, emitter unclear. Space for
    the next page, u to undo, h for help, q to quit. Every click is saved
    immediately; running the same command again resumes the same queue.
    """
    from chirpdetector.labeling.queue import QueueConfig, load_or_build_queue
    from chirpdetector.labeling.render import CutoutConfig
    from chirpdetector.labeling.store import LabelStore

    out = out.resolve()
    trials = [d.resolve() for d in trial_dirs]
    for trial in trials:
        if out.is_relative_to(trial):
            msg = f"--out must not be inside a trial directory ({trial})"
            raise typer.BadParameter(msg)
    rows_cols = _parse_grid(grid)
    config = QueueConfig(per_bin=per_bin, seed=seed, bin_edges_hz=_parse_bins(bins))
    cutout_config = CutoutConfig(pad_s=pad_s, pad_hz=pad_hz, nfft=nfft)
    try:
        from chirpdetector.labeling.gui import run
    except ImportError as err:
        typer.echo(
            f"the labeller needs matplotlib (uv sync --extra label): {err}", err=True
        )
        raise typer.Exit(code=1) from err

    queue = load_or_build_queue(trials, config, out)
    try:
        with LabelStore(out, labeler, chirpdetector.__version__) as store:
            run(
                queue,
                {t.name: t for t in trials},
                store,
                config=cutout_config,
                grid=rows_cols,
                tolerance_hz=tolerance_hz,
            )
    except RuntimeError as err:  # no display, or another session on --out
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
