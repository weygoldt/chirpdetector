"""Command-line interface: ``chirpdetector <command>``."""

from __future__ import annotations

import getpass
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import typer

import chirpdetector
from chirpdetector.cli_data import data_app

if TYPE_CHECKING:
    from chirpdetector.evaluation.matching import MatchMethod

__all__ = ["app"]

DEFAULT_LABEL_DIR = Path("/mnt/data1/chirp_v2/labels/emitters")

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Detect chirps of wave-type weakly electric fish and assign them to their emitters.",
)

app.add_typer(data_app, name="data")


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
    window_s: Annotated[
        float,
        typer.Option(
            help=(
                "Spectrogram window in seconds, at any sample rate: 0.2048 "
                "(4.9 Hz resolution; 4096 samples at 20 kHz) separates close "
                "baselines; 0.1024 or 0.0512 show the chirp's shape better."
            )
        ),
    ] = 0.2048,
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
    cutout_config = CutoutConfig(pad_s=pad_s, pad_hz=pad_hz, window_s=window_s)
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


_TRACK_FILES = ("times.npy", "fund_v.npy", "ident_v.npy", "idx_v.npy")


def _has_tracks(directory: Path) -> bool:
    return all((directory / name).exists() for name in _TRACK_FILES)


def _is_simulation(path: Path) -> bool:
    if path.is_dir():
        return (path / "chirps.csv").exists()
    import pandas as pd

    return "t_onset_s" in pd.read_csv(path, nrows=0).columns


@app.command("evaluate")
def evaluate_command(
    predictions: Annotated[
        list[Path],
        typer.Argument(
            exists=True,
            help=(
                "Prediction files: v2 tables (CSV/Parquet), v1 "
                "chirpdetector_bboxes.csv (or its recording directory), or v1 "
                "boxes (t1, f1, t2, f2, score)."
            ),
        ),
    ],
    labels: Annotated[
        list[Path],
        typer.Option(
            "--labels",
            exists=True,
            help=(
                "Ground truth, repeatable: audian label files in recording "
                "directories, label-emitters CSVs, or simulated scenes "
                "(chirps.csv or the scene directory)."
            ),
        ),
    ],
    out: Annotated[Path, typer.Option(help="Output directory.")],
    thresholds: Annotated[
        str, typer.Option(help="Score thresholds: START:STOP:STEP or a list a,b,c.")
    ] = "0.25:0.95:0.05",
    reference_threshold: Annotated[
        float, typer.Option(help="Threshold of CIs, assignment and item tables.")
    ] = 0.5,
    tolerance_s: Annotated[
        float, typer.Option(help="Max. difference of centre times for a match.")
    ] = 0.1,
    method: Annotated[
        str, typer.Option(help="One-to-one matching: hungarian or greedy.")
    ] = "hungarian",
    band_filter: Annotated[
        bool,
        typer.Option(help="Drop predictions outside a region's band (harmonics)."),
    ] = True,
    edge_tolerance: Annotated[
        bool,
        typer.Option(help="Let predictions just outside a region match its chirps."),
    ] = False,
    nms_iou: Annotated[
        float | None,
        typer.Option(
            help="Apply v1's NMS (per batch) at this IoU first, e.g. 0.2 for "
            "pre-NMS v1 boxes."
        ),
    ] = None,
    tracks: Annotated[
        list[Path] | None,
        typer.Option(
            "--tracks",
            exists=True,
            file_okay=False,
            help=(
                "wavetracker directory, repeatable; its name is the recording. "
                "Default: the recording directories of labels and predictions."
            ),
        ),
    ] = None,
    n_bootstrap: Annotated[int, typer.Option(help="Bootstrap resamples.")] = 5000,
    seed: Annotated[int, typer.Option(help="Bootstrap seed.")] = 0,
    plot: Annotated[bool, typer.Option(help="Also draw pr_curve.png.")] = False,
) -> None:
    """Score predictions against hand labels or simulated ground truth.

    Writes metrics.json and CSV tables (summary, confidence intervals, PR
    curve, items, assignment, end to end) to --out. The default matching
    rule is detector_benchmark's: centre times within 100 ms, overlapping
    frequency ranges, one-to-one.
    """
    import pandas as pd
    from gridtools.io import load_tracks

    from chirpdetector.evaluation import (
        EvaluationConfig,
        GroundTruth,
        MatchConfig,
        evaluate,
        non_max_suppression,
        parse_thresholds,
        read_ground_truth,
        read_predictions,
        sha256_file,
        simulation_ground_truth,
    )
    from chirpdetector.evaluation.predictions import V1_DETECTIONS_FILE

    log = logging.getLogger("chirpdetector.evaluate")
    try:
        config = EvaluationConfig(
            match=MatchConfig(
                time_tolerance_s=tolerance_s,
                band_filter=band_filter,
                edge_tolerance=edge_tolerance,
                method=cast("MatchMethod", method),  # validated by MatchConfig
            ),
            thresholds=parse_thresholds(thresholds),
            reference_threshold=reference_threshold,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=seed,
        )
        inputs, candidates = [], []
        parts = []
        for path in labels:
            if _is_simulation(path):
                parts.append(simulation_ground_truth(path))
                candidates.append(path if path.is_dir() else path.parent)
            else:
                parts.append(read_ground_truth(path))
                candidates.append(path.parent)
            source = path / "chirps.csv" if path.is_dir() else path
            inputs.append({"role": "labels", "path": str(source.resolve())})
        truth = GroundTruth.concat(parts)
        frames = []
        for path in predictions:
            frames.append(read_predictions(path))
            source = path / V1_DETECTIONS_FILE if path.is_dir() else path
            candidates.append(source.parent)
            inputs.append({"role": "predictions", "path": str(source.resolve())})
        preds = pd.concat(frames, ignore_index=True)
        if nms_iou is not None:
            before = len(preds)
            preds = non_max_suppression(preds, nms_iou)
            log.info("NMS at IoU %g kept %d of %d boxes", nms_iou, len(preds), before)
        for entry in inputs:
            entry["sha256"] = sha256_file(entry["path"])
        recordings = set(truth.recordings)
        track_dirs = {d.name: d for d in candidates if _has_tracks(d)}
        track_dirs.update({d.name: d for d in tracks or []})
        loaded = {
            name: load_tracks(d) for name, d in track_dirs.items() if name in recordings
        }
        result = evaluate(
            preds,
            truth,
            config=config,
            tracks=loaded,
            provenance={
                "inputs": inputs,
                "tracks": {k: str(v.resolve()) for k, v in track_dirs.items()},
                "nms_iou": nms_iou,
            },
        )
        written = result.write(out, plot=plot)
    except (ValueError, FileNotFoundError, ImportError) as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    pooled = result.metrics()["detection"].get("all")
    if pooled:
        p_ci, r_ci = pooled["precision_ci"], pooled["recall_ci"]

        def fmt(v: float | None) -> str:
            return "nan" if v is None else f"{v:.3f}"

        typer.echo(
            f"threshold {reference_threshold:g}: "
            f"precision {fmt(pooled['precision'])} [{fmt(p_ci[0])}, {fmt(p_ci[1])}], "
            f"recall {fmt(pooled['recall'])} [{fmt(r_ci[0])}, {fmt(r_ci[1])}], "
            f"F1 {fmt(pooled['f1'])} ({pooled['labels']} labels, "
            f"{pooled.get('n_regions', 0)} regions)"
        )
    typer.echo(f"wrote {len(written)} files to {out}")
