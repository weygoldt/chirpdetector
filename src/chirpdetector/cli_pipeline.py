"""CLI of the production pipeline: ``run``, ``render``, ``check-reference``."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

__all__ = ["check_reference_command", "render_command", "run_command"]

log = logging.getLogger(__name__)


def run_command(
    inputs: Annotated[
        list[Path] | None,
        typer.Argument(
            metavar="[CONFIG.toml] [RECORDING]...",
            help="Run config (TOML; optional: without it, the pretrained models "
            "and every default, see --out), then recording directories (fishgrid "
            "raw or a single wav).",
            show_default=False,
        ),
    ] = None,
    dataset: Annotated[
        Path | None,
        typer.Option(
            "--dataset",
            file_okay=False,
            help="Process every recording directory in ROOT.",
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Output root (overrides the config's run.out; required without a config).",
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help="Run name (overrides the config's run.name; 'default' without a config).",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Quick check: only --limit-s (default 300 s) of each recording, "
            "into run '<name>_dryrun'.",
        ),
    ] = False,
    limit_s: Annotated[
        float | None, typer.Option("--limit-s", help="Process only this many seconds.")
    ] = None,
    start_s: Annotated[
        float, typer.Option("--start-s", help="Start of the --limit-s span.")
    ] = 0.0,
    redo: Annotated[
        bool, typer.Option("--redo", help="Recompute recordings that are complete.")
    ] = False,
    device: Annotated[
        str | None,
        typer.Option(
            "--device", help="Torch device (overrides the config's compute.device)."
        ),
    ] = None,
    vram: Annotated[
        float | None,
        typer.Option(
            "--vram", help="GPU memory fraction (overrides compute.max_vram_fraction)."
        ),
    ] = None,
    list_only: Annotated[
        bool,
        typer.Option(
            "--list", help="List the recordings and their status; run nothing."
        ),
    ] = False,
) -> None:
    """Detect, decode and assign chirps in whole recordings (resumable).

    Outputs go to OUT/<name>/<recording>/ (chirps_<model>.parquet, meta.json,
    legacy chirp_times/chirp_ids arrays, QC report, sample windows), with a
    run manifest (run.json), a log (run.log) and validation candidates.
    Recordings that are complete are skipped; a failing recording is logged
    and the run continues (exit status 1 at the end).

    Without CONFIG.toml (`chirpdetector run --out OUT RECORDING...`), the
    pretrained detector and decoder (downloaded on first use) and every
    default setting are used, on CUDA if available.
    """
    from chirpdetector.pipeline.config import default_config, load_config
    from chirpdetector.pipeline.run import _done, discover_recordings, run_pipeline

    paths_given = list(inputs or [])
    config_file: Path | None = None
    if paths_given and paths_given[0].suffix.lower() == ".toml":
        config_file = paths_given.pop(0)
        if not config_file.is_file():
            typer.echo(f"config {config_file} does not exist", err=True)
            raise typer.Exit(2)
    recordings = paths_given
    if config_file is not None:
        config = load_config(config_file)
    elif out is None:
        typer.echo(
            "give a run config (CONFIG.toml) or --out OUT for the default config",
            err=True,
        )
        raise typer.Exit(2)
    else:
        config = default_config(out.resolve())
    run_name = name or config.name
    if dry_run:
        limit_s = 300.0 if limit_s is None else limit_s
        if name is None:
            run_name = f"{config.name}_dryrun"
    compute: dict[str, object] = {}
    if device is not None:
        compute["device"] = device
    if vram is not None:
        compute["max_vram_fraction"] = vram
    config = config.with_overrides(
        name=run_name,
        out=out.resolve() if out is not None else None,
        **({"compute": compute} if compute else {}),
    )
    paths = discover_recordings(recordings or [], dataset)
    if not paths:
        typer.echo("no recordings given (RECORDINGS... or --dataset ROOT)", err=True)
        raise typer.Exit(2)
    t_range = None if limit_s is None else (start_s, start_s + limit_s)
    if list_only:
        dhash = config.detection_hash()
        for p in paths:
            done = _done(
                config.run_dir / p.name,
                dhash,
                None if t_range is None else list(t_range),
            )
            typer.echo(f"{'done' if done else 'todo'}  {p}")
        typer.echo(f"{len(paths)} recordings -> {config.run_dir}")
        return
    summary = run_pipeline(
        config, paths, config_file=config_file, t_range_s=t_range, redo=redo
    )
    typer.echo(
        f"{config.run_dir}: {len(summary.done)} done, {len(summary.skipped)} skipped, "
        f"{len(summary.failed)} failed"
    )
    if summary.failed:
        for k, v in summary.failed.items():
            typer.echo(f"FAILED {k}: {v}", err=True)
        raise typer.Exit(1)


def render_command(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Run directory OUT/<name>."),
    ],
    recording: Annotated[str, typer.Argument(help="Recording name (directory name).")],
    t0: Annotated[float, typer.Argument(help="Window start (s, recording clock).")],
    t1: Annotated[float, typer.Argument(help="Window end (s).")],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out", help="Output directory (default RUN_DIR/RECORDING/renders)."
        ),
    ] = None,
    device: Annotated[str, typer.Option("--device", help="Torch device.")] = "cpu",
) -> None:
    """Re-render a window of a recording from a run's outputs (PNG + NPZ).

    The spectrogram is recomputed from the raw recording with the run's
    detector front end and click removal; boxes, assignments and tracks come
    from the run's chirp table and the recording's tracks.
    """
    from gridtools.io import open_recording

    from chirpdetector.bundle import load_bundle
    from chirpdetector.pipeline.outputs import read_chirps
    from chirpdetector.pipeline.run import _load_tracks
    from chirpdetector.pipeline.windows import render_window

    if not t1 > t0:
        typer.echo("T1 must be larger than T0", err=True)
        raise typer.Exit(2)
    rec_dir = run_dir / recording
    meta = json.loads((rec_dir / "meta.json").read_text())
    table, file_meta = read_chirps(rec_dir / meta["outputs"]["chirps"]["file"])
    bundle = load_bundle(meta["models"]["detector"]["path"])
    if bundle.weights_sha256 != meta["models"]["detector"]["weights_sha256"]:
        typer.echo("warning: the detector bundle changed since the run", err=True)
    rec = open_recording(meta["inputs"]["recording_dir"])
    tdir = (meta.get("tracks") or {}).get("dir")
    tracks, _ = _load_tracks(Path(tdir) if tdir else None)
    target = out or rec_dir / "renders"
    stem = target / f"{recording}_{t0:09.2f}-{t1:09.2f}s"
    info = render_window(
        stem,
        rec,
        bundle,
        table,
        tracks,
        t0,
        t1,
        title=f"{recording} ({meta['run']})",
        meta={
            "run": meta["run"],
            "recording": recording,
            "model": file_meta.get("model"),
        },
        device=device,
    )
    typer.echo(f"{stem}.png, {stem}.npz ({info['n_boxes']} boxes)")


def check_reference_command(
    config_file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Run config (TOML).")
    ],
    recording: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="Recording directory.")
    ],
    start_s: Annotated[float, typer.Option("--start-s")] = 3600.0,
    duration_s: Annotated[float, typer.Option("--duration-s")] = 300.0,
    device: Annotated[str | None, typer.Option("--device")] = None,
    chunk_tiles: Annotated[
        int | None,
        typer.Option("--chunk-tiles", help="Chunk size (default: the config's)."),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Write the report as JSON.")
    ] = None,
) -> None:
    """Compare the chunked pipeline with predict_span on a part of a recording."""
    from gridtools.io import open_recording

    from chirpdetector.pipeline.check import compare_with_reference
    from chirpdetector.pipeline.config import load_config
    from chirpdetector.pipeline.run import load_models

    config = load_config(config_file)
    if device is not None:
        config = config.with_overrides(compute={"device": device})
    models = load_models(config)
    report = compare_with_reference(
        models.bundle,
        open_recording(recording),
        start_s,
        duration_s,
        detector=models.detector,
        estimator=models.estimator,
        store_threshold=config.detection.store_threshold,
        chunk_tiles=chunk_tiles or config.compute.chunk_tiles,
        click_workers=config.compute.click_workers,
        prefetch=config.compute.prefetch,
        clean_processes=config.compute.clean_processes,
        device=config.compute.device,
    )
    report["recording"] = str(recording)
    text = json.dumps(report, indent=1)
    typer.echo(text)
    if out is not None:
        out.write_text(text + "\n")
    if not report.get("same_count"):
        sys.exit(1)
