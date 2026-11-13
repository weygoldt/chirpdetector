"""Command-line interface: ``chirpdetector <command>``."""

from __future__ import annotations

import getpass
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, cast

import typer
from typer.core import TyperGroup

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
        list[Path] | None,
        typer.Argument(
            exists=True,
            file_okay=False,
            show_default=False,
            help=(
                "Trial directories with chirpdetector_bboxes.csv, tracks and the "
                "recording (v1 detections are sampled per ΔF bin). Not with "
                "--candidates."
            ),
        ),
    ] = None,
    candidates: Annotated[
        Path | None,
        typer.Option(
            "--candidates",
            exists=True,
            dir_okay=False,
            help=(
                "CSV or Parquet of boxes to label instead of v1 detections. "
                "Required columns: recording, t_start_s, t_end_s, f_low_hz, "
                "f_high_hz; optional: candidate_id, score, emitter_hz, track_id, "
                "source, tracks (see the README)."
            ),
        ),
    ] = None,
    out: Annotated[
        Path, typer.Option(help="Output directory (queue, event log, label CSVs).")
    ] = DEFAULT_LABEL_DIR,
    per_bin: Annotated[
        int, typer.Option(help="Detections sampled per trial and ΔF bin (v1).")
    ] = 20,
    seed: Annotated[int, typer.Option(help="Sampling and shuffling seed.")] = 0,
    shuffle: Annotated[
        bool,
        typer.Option(
            "--shuffle/--keep-order",
            help="Shuffle the candidates with --seed, or keep the file's order.",
        ),
    ] = True,
    labeler: Annotated[
        str, typer.Option(help="Name stored with every label.")
    ] = getpass.getuser(),
    view: Annotated[
        str,
        typer.Option(
            help=(
                "single: one candidate per maximised window with a baseline "
                "zoom; grid: pages of small cutouts (matplotlib only)."
            )
        ),
    ] = "single",
    grid: Annotated[
        str, typer.Option(help="Cutouts per page with --view grid, ROWSxCOLS.")
    ] = "4x6",
    bins: Annotated[
        str, typer.Option(help="Lower edges of the ΔF bins in Hz; the last is open.")
    ] = "0,10,25,50,100",
    span_s: Annotated[
        float, typer.Option(help="Time span of the single view, centred on the box.")
    ] = 2.0,
    span_hz: Annotated[
        float,
        typer.Option(help="Frequency span of the single view's overview panel."),
    ] = 500.0,
    pad_s: Annotated[float, typer.Option(help="Time padding around the box.")] = 0.25,
    pad_hz: Annotated[
        float, typer.Option(help="Frequency padding around tracks and box.")
    ] = 40.0,
    tolerance_hz: Annotated[
        float,
        typer.Option(
            help="Max. distance of a click from a track for it to snap to it."
        ),
    ] = 15.0,
    window_s: Annotated[
        float,
        typer.Option(
            help=(
                "Spectrogram window in seconds, at any sample rate: 0.2048 "
                "(4.9 Hz resolution; 4096 samples at 20 kHz) separates close "
                "baselines; 0.0512 shows the chirp's shape (w toggles them in "
                "the overview; the baseline zoom always uses 0.2048)."
            )
        ),
    ] = 0.2048,
    track_gap_s: Annotated[
        float,
        typer.Option(
            help=(
                "Show a track if its detections around the box centre are at "
                "most this far apart (wavetracker tracks often have gaps of "
                "seconds; ΔF bins keep the evaluation's 1 s)."
            )
        ),
    ] = 10.0,
    serve: Annotated[
        bool,
        typer.Option(
            "--serve",
            help="Serve the browser labeller (for an SSH port-forward) instead "
            "of opening a window.",
        ),
    ] = False,
    host: Annotated[
        str, typer.Option(help="Interface the browser labeller binds to.")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port of the browser labeller.")] = 8765,
) -> None:
    """Label chirps and their emitters, one candidate at a time.

    Click on the baseline of the fish the chirp comes from (the click snaps
    to the highlighted track) or press its number; 0 = chirp, emitter
    unclear; x = not a chirp; s = skip; u = undo; w = window; e = model's
    prediction; h = help. Every label is saved immediately; running the
    same command again (either frontend) resumes the same queue.
    """
    from chirpdetector.labeling.candidates import (
        load_or_build_candidate_queue,
        read_candidates,
    )
    from chirpdetector.labeling.queue import QueueConfig, load_or_build_queue
    from chirpdetector.labeling.store import LabelStore

    if (candidates is None) == (not trial_dirs):
        msg = "give either trial directories or --candidates (not both)"
        raise typer.BadParameter(msg)
    if view not in ("single", "grid"):
        raise typer.BadParameter(f"--view must be single or grid, got {view!r}")
    if serve and view == "grid":
        raise typer.BadParameter("--serve shows one candidate at a time (no grid)")
    out = out.resolve()
    rows_cols = _parse_grid(grid)
    config = QueueConfig(per_bin=per_bin, seed=seed, bin_edges_hz=_parse_bins(bins))
    if candidates is not None:
        try:
            table = read_candidates(candidates)
        except (ValueError, FileNotFoundError) as err:
            typer.echo(f"error: {err}", err=True)
            raise typer.Exit(code=1) from err
        protected = {Path(p) for p in table["recording"]} | {
            Path(p) for p in table["tracks"]
        }
        overrides = None
    else:
        protected = {d.resolve() for d in trial_dirs or []}
        overrides = {d.name: d for d in protected}
    for directory in sorted(protected):
        if out.is_relative_to(directory):
            msg = f"--out must not be inside a trial directory ({directory})"
            raise typer.BadParameter(msg)
    try:
        from chirpdetector.labeling.gui import run
        from chirpdetector.labeling.session import LabelSession, TrialSources
        from chirpdetector.labeling.single import run_single
    except ImportError as err:
        typer.echo(
            f"the labeller needs matplotlib (uv sync --extra label): {err}", err=True
        )
        raise typer.Exit(code=1) from err

    if candidates is not None:
        queue = load_or_build_candidate_queue(candidates, config, out, shuffle=shuffle)
    else:
        queue = load_or_build_queue(sorted(protected), config, out)
    if len(queue) == 0:
        typer.echo("nothing to label: the queue is empty")
        raise typer.Exit(code=0)
    sources = TrialSources.from_queue(queue, overrides)
    try:
        with LabelStore(out, labeler, chirpdetector.__version__) as store:
            if view == "grid":
                from chirpdetector.labeling.render import CutoutConfig

                run(
                    queue,
                    sources,
                    store,
                    config=CutoutConfig(pad_s=pad_s, pad_hz=pad_hz, window_s=window_s),
                    grid=rows_cols,
                    tolerance_hz=tolerance_hz,
                )
                return
            from chirpdetector.labeling.view import ViewConfig

            session = LabelSession(
                queue,
                store,
                config=ViewConfig(
                    span_s=span_s,
                    span_hz=span_hz,
                    pad_s=pad_s,
                    pad_hz=pad_hz,
                    window_s=window_s,
                    max_gap_s=track_gap_s,
                ),
                tolerance_hz=tolerance_hz,
                frontend="browser" if serve else "mpl",
                sources=sources,
            )
            if serve:
                from chirpdetector.labeling.server import serve_until_interrupted

                serve_until_interrupted(session, host, port, out, typer.echo)
            else:
                run_single(session)
    except (RuntimeError, ValueError) as err:  # no display, busy --out, bad ids
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err


@app.command("label-summary")
def label_summary(
    out: Annotated[
        Path, typer.Argument(help="Output directory of label-emitters.")
    ] = DEFAULT_LABEL_DIR,
    labeler: Annotated[
        str | None, typer.Option(help="Only this labeller's labels.")
    ] = None,
    queue: Annotated[
        str | None,
        typer.Option(
            help="Only labels of this queue (id or JSON file); counts open items."
        ),
    ] = None,
) -> None:
    """Count emitter labels by decision, by candidate source and by ΔF bin."""
    from chirpdetector.labeling.summary import summarize

    typer.echo(summarize(out, labeler=labeler, queue=queue).to_text())


_TRACK_FILES = ("times.npy", "fund_v.npy", "ident_v.npy", "idx_v.npy")


def _has_tracks(directory: Path) -> bool:
    return all((directory / name).exists() for name in _TRACK_FILES)


def _is_simulation(path: Path) -> bool:
    if path.is_dir():
        return (path / "chirps.csv").exists()
    import pandas as pd

    return "t_onset_s" in pd.read_csv(path, nrows=0).columns


class _DefaultCommandGroup(TyperGroup):
    """A group whose first argument, if not a subcommand, selects ``detections``.

    ``chirpdetector evaluate PREDICTIONS --labels ...`` (the original
    command) therefore keeps working next to ``chirpdetector evaluate
    params BUNDLE ...``.
    """

    default_command = "detections"

    def parse_args(self, ctx: typer.Context, args: list[str]) -> list[str]:  # type: ignore[override]
        if args and args[0] not in self.commands and args[0] not in ("--help", "-h"):
            args = [self.default_command, *args]
        return super().parse_args(ctx, args)


evaluate_app = typer.Typer(
    cls=_DefaultCommandGroup,
    no_args_is_help=True,
    help=(
        "Evaluate detections (default: chirpdetector evaluate PREDICTIONS "
        "--labels ...) or a chirp-parameter decoder (evaluate params)."
    ),
)
app.add_typer(evaluate_app, name="evaluate")


@evaluate_app.command("detections")
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


# ----------------------------------------------------------------- training

train_app = typer.Typer(no_args_is_help=True, help="Train models.")
app.add_typer(train_app, name="train")


def _parse_fold(fold: str | None) -> int | Literal["final"] | None:
    if fold is None:
        return None
    if fold == "final":
        return "final"
    try:
        return int(fold)
    except ValueError:
        msg = f"--fold must be a fold number or 'final', got {fold!r}"
        raise typer.BadParameter(msg) from None


@train_app.command("detector")
def train_detector_command(
    config: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Training config (TOML)."),
    ],
    dataset: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Detector dataset directory."),
    ],
    run_dir: Annotated[Path, typer.Option(help="Run directory (created).")],
    fold: Annotated[
        str | None,
        typer.Option(
            help="Train only this fold (0..k-1) or 'final'; default: all folds, then the final fit."
        ),
    ] = None,
    check_files: Annotated[
        bool, typer.Option(help="Hash and decode every dataset file first.")
    ] = True,
) -> None:
    """Train chirp detectors: grouped k-fold CV and a final fit, with bundles.

    Run it through scripts/guarded on a shared GPU (caps RAM and sets
    CHIRP_MAX_VRAM_FRACTION).
    """
    from chirpdetector.data import DatasetError
    from chirpdetector.training import train_detector

    try:
        result = train_detector(
            dataset, config, run_dir, fold=_parse_fold(fold), check_files=check_files
        )
    except (DatasetError, ValueError, FileExistsError, KeyError) as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    for fit in result.fits:
        metrics = ", ".join(
            f"{k.split('/')[-1]} {v:.3f}"
            for k, v in fit.metrics.items()
            if k.startswith("metrics/")
        )
        typer.echo(f"{fit.name}: best epoch {fit.best_epoch}, {metrics}")
    if result.bundle is not None:
        typer.echo(f"bundle: {result.bundle}")
    typer.echo(f"run {result.run_id} in {result.run_dir}")


@train_app.command("params")
def train_params_command(
    config: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Training config (TOML)."),
    ],
    scenes: Annotated[
        list[Path],
        typer.Option(
            "--scenes",
            exists=True,
            file_okay=False,
            help=(
                "Simulated scenes, repeatable: a scene, a directory of scenes, or "
                "a directory of train/val/test scene directories."
            ),
        ),
    ],
    run_dir: Annotated[Path, typer.Option(help="Run directory (created).")],
    detector: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            help=(
                "Detector bundle whose boxes the decoder learns on (required for "
                "boxes.mode = 'detector', the default)."
            ),
        ),
    ] = None,
    samples: Annotated[
        Path | None,
        typer.Option(
            help="Sample-set directory (default RUN/samples; reused if it matches)."
        ),
    ] = None,
) -> None:
    """Train a chirp-parameter decoder (height, widths, undershoot) and its bundle.

    Run it through scripts/guarded on a shared GPU (caps RAM and sets
    CHIRP_MAX_VRAM_FRACTION).
    """
    from chirpdetector.bundle import BundleError
    from chirpdetector.params.training import train_params

    try:
        result = train_params(
            config, scenes, run_dir, detector=detector, samples_dir=samples
        )
    except (BundleError, ValueError, FileExistsError, FileNotFoundError) as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    for name, values in result.validation.items():
        typer.echo(
            f"{name}: median |log error| {values['median_abs_log_error_raw']:.3f} raw, "
            f"{values['median_abs_log_error_calibrated']:.3f} calibrated (val)"
        )
    typer.echo(f"best epoch {result.best_epoch} of {result.epochs_run}")
    typer.echo(f"bundle: {result.bundle}")


# ------------------------------------------------------------------ bundles

bundle_app = typer.Typer(no_args_is_help=True, help="Model bundles.")
app.add_typer(bundle_app, name="bundle")


@bundle_app.command("inspect")
def bundle_inspect(
    path: Annotated[
        Path, typer.Argument(exists=True, help="Bundle directory or its bundle.json.")
    ],
) -> None:
    """Verify a bundle's weight hash and print what it holds.

    Works for detector bundles and chirp-parameter (params) bundles.
    """
    from chirpdetector.bundle import BundleError, describe_bundle, load_bundle
    from chirpdetector.params.bundle import (
        describe_params_bundle,
        is_params_bundle,
        load_params_bundle,
    )

    try:
        if is_params_bundle(path):
            text = describe_params_bundle(load_params_bundle(path))
        else:
            text = describe_bundle(load_bundle(path))
    except (BundleError, FileNotFoundError) as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    typer.echo(text)


@evaluate_app.command("params")
def evaluate_params_command(
    bundle: Annotated[
        Path, typer.Argument(exists=True, help="Params bundle (directory).")
    ],
    scenes: Annotated[
        list[Path],
        typer.Option(
            "--scenes",
            exists=True,
            file_okay=False,
            help="Simulated scenes, repeatable (their test split is evaluated).",
        ),
    ],
    out: Annotated[
        Path | None, typer.Option(help="Output directory (default BUNDLE/eval).")
    ] = None,
    samples: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Sample-set directory holding the test scenes (default: the "
                "training sample set if it matches, else OUT/samples)."
            )
        ),
    ] = None,
    detector: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            help="Detector bundle (default: the one recorded in the params bundle).",
        ),
    ] = None,
    split: Annotated[str, typer.Option(help="Split to evaluate.")] = "test",
    real: Annotated[
        bool,
        typer.Option(
            help="Also check agreement on the harvested real long chirps (library v4)."
        ),
    ] = True,
    library: Annotated[
        Path, typer.Option(help="Template library with the harvested chirps.")
    ] = Path("/mnt/data1/chirp_v2/templates/v4"),
    recordings: Annotated[
        Path, typer.Option(help="Directory of the competition recordings.")
    ] = Path("/mnt/data2/2019_tube_competition"),
    workers: Annotated[int, typer.Option(help="Extraction processes.")] = 4,
    device: Annotated[
        str, typer.Option(help="Torch device (auto, cpu, cuda).")
    ] = "auto",
    attach: Annotated[
        bool, typer.Option(help="Write metrics.json into the bundle.")
    ] = True,
) -> None:
    """Evaluate a chirp-parameter decoder against the simulated ground truth.

    Accuracy by family, SNR, ΔF, overlap and decile; uncertainty
    calibration; distribution fidelity (marginals, joint, natural template
    prior, cluster structure); real-data agreement. Writes figures, tables,
    metrics.json and report.md.
    """
    from chirpdetector.bundle import BundleError
    from chirpdetector.params.evaluation import run_evaluation

    try:
        report = run_evaluation(
            bundle,
            scenes,
            out=out,
            samples_dir=samples,
            detector=detector,
            split=split,
            real_library=library if real else None,
            real_recordings=recordings,
            workers=workers,
            device=device,
            attach=attach,
        )
    except (BundleError, ValueError, FileNotFoundError) as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    typer.echo(f"report: {report}")


@app.command("showcase")
def showcase_command(
    snippets: Annotated[
        Path, typer.Option(exists=True, help="TOML file with [[snippet]] entries.")
    ],
    out: Annotated[Path, typer.Option(help="Output directory (sheet, panels, index).")],
    bundle: Annotated[
        list[Path] | None,
        typer.Option(help="Model bundle; repeat for several (one column each)."),
    ] = None,
    v1: Annotated[
        bool, typer.Option("--v1", help="Add v1's production detections as a column.")
    ] = False,
    threshold: Annotated[
        float | None,
        typer.Option(help="Score threshold for the bundles (default: each bundle's)."),
    ] = None,
    device: Annotated[
        str | None, typer.Option(help="Torch device (e.g. cpu, cuda).")
    ] = None,
    force: Annotated[bool, typer.Option(help="Re-render existing panels.")] = False,
    params: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            help=(
                "Params bundle: label every box of the --bundle columns with its "
                "decoded height and FWHM (the bundles must be its detector)."
            ),
        ),
    ] = None,
) -> None:
    """Render models side by side on hard snippets of real recordings.

    One panel per snippet and model (tracks, boxes, scores, predicted emitter
    EODf, hand labels, optionally decoded chirp parameters), collected in a
    contact sheet. Columns already in ``--out`` are kept, so the sheet grows
    with every model tried.
    """
    from chirpdetector.evaluation.showcase import (
        bundle_column,
        read_snippets,
        render_showcase,
        v1_column,
    )

    columns = [v1_column()] if v1 else []
    columns += [bundle_column(path, threshold, params=params) for path in bundle or []]
    sheet = render_showcase(
        columns, read_snippets(snippets), out, device=device, force=force
    )
    typer.echo(f"contact sheet: {sheet}")


# ----------------------------------------------------------------- datasets

dataset_app = typer.Typer(no_args_is_help=True, help="Detector datasets.")
app.add_typer(dataset_app, name="dataset")


@dataset_app.command("validate")
def dataset_validate(
    directory: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="Dataset directory.")
    ],
    check_files: Annotated[
        bool, typer.Option(help="Hash and decode every image and label file.")
    ] = True,
) -> None:
    """Check a detector dataset against the contract."""
    from chirpdetector.data import DatasetError, validate_dataset

    try:
        manifest = validate_dataset(directory, check_files=check_files)
    except DatasetError as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    counts = {s: len(manifest.image_ids(s)) for s in ("train", "val", "test")}
    typer.echo(
        f"valid: {len(manifest.images)} images ({counts}), "
        f"{len(manifest.splits)} groups, content sha256 {manifest.content_sha256}"
    )
