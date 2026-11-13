"""``chirpdetector label-benchmark``: human against model on simulated scenes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

__all__ = ["benchmark_app"]

benchmark_app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Human against model on simulated scenes: build blind candidates and a "
        "separate truth file, then score the labels (chirpdetector.labeling.benchmark)."
    ),
)


@benchmark_app.command("build")
def build(
    scenes: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=False, help="Directory of simulated test scenes."
        ),
    ],
    detections: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="The model's detections per scene (<scene>.parquet, predict_span with tracks, post-filter on).",
        ),
    ],
    candidates: Annotated[
        Path, typer.Option(help="Candidates file to write (what the labeller sees).")
    ],
    truth: Annotated[
        Path, typer.Option(help="Truth file to write (another directory; never shown).")
    ],
    seed: Annotated[int, typer.Option(help="Sampling and shuffling seed.")] = 0,
    n_chirps: Annotated[int, typer.Option(help="Visible true chirps.")] = 150,
    n_ambiguous: Annotated[
        int,
        typer.Option(
            help="Of these, from the lowest SNR quartile with excursion < 60 Hz."
        ),
    ] = 75,
    n_hard: Annotated[
        int, typer.Option(help="Model detections without a chirp.")
    ] = 100,
    n_easy: Annotated[int, typer.Option(help="Baseline windows without a chirp.")] = 50,
    n_invisible: Annotated[
        int, typer.Option(help="Invisible chirps (reported, not scored).")
    ] = 15,
    min_hard_score: Annotated[
        float, typer.Option(help="Least score of a hard negative.")
    ] = 0.05,
) -> None:
    """Sample the benchmark items and write candidates and truth."""
    from chirpdetector.labeling.benchmark import (
        BenchmarkConfig,
        build_benchmark,
        read_detections,
        write_benchmark,
    )

    config = BenchmarkConfig(
        seed=seed,
        n_chirps=n_chirps,
        n_ambiguous=n_ambiguous,
        n_hard=n_hard,
        n_easy=n_easy,
        n_invisible=n_invisible,
        min_hard_score=min_hard_score,
    )
    dets = read_detections(detections)
    scene_dirs = sorted(d for d in scenes.iterdir() if d.is_dir() and d.name in dets)
    if not scene_dirs:
        typer.echo("error: no scene has detections", err=True)
        raise typer.Exit(code=1)
    cands, table, summary = build_benchmark(scene_dirs, dets, config)
    summary["detections"] = str(detections.resolve())
    summary["scenes"] = str(scenes.resolve())
    try:
        write_benchmark(cands, table, summary, candidates, truth)
    except ValueError as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    typer.echo(
        json.dumps(
            {k: summary[k] for k in ("n_items", "per_stratum", "pools")}, indent=1
        )
    )
    typer.echo(f"candidates: {candidates}\ntruth: {truth}")


@benchmark_app.command("score")
def score(
    truth: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Truth file of build.")
    ],
    labels: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="label-emitters --out of the benchmark session.",
        ),
    ],
    out: Annotated[Path, typer.Option(help="Report directory.")],
    labeler: Annotated[
        str | None, typer.Option(help="Only this labeller's labels.")
    ] = None,
    threshold: Annotated[float, typer.Option(help="Model score threshold.")] = 0.5,
    n_boot: Annotated[int, typer.Option(help="Bootstrap resamples.")] = 5000,
    seed: Annotated[int, typer.Option(help="Bootstrap seed.")] = 0,
) -> None:
    """Score the human's labels and the model on the benchmark items."""
    import pandas as pd

    from chirpdetector.labeling.benchmark import score_benchmark, write_report
    from chirpdetector.labeling.summary import read_label_files

    table = pd.read_parquet(truth)
    sidecar = truth.with_suffix(".json")
    summary = json.loads(sidecar.read_text()) if sidecar.exists() else None
    try:
        result = score_benchmark(
            table,
            read_label_files(labels),
            threshold=threshold,
            labeler=labeler,
            n_boot=n_boot,
            seed=seed,
        )
    except ValueError as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    path = write_report(result, out, summary=summary)
    typer.echo(path.read_text().split("\n## ")[0])
    typer.echo(f"report: {path}")


@benchmark_app.command("gallery")
def gallery(
    labels: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="label-emitters --out with the real labels.",
        ),
    ],
    candidates: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="The real candidates file those labels came from.",
        ),
    ],
    scenes: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="Simulated test scenes.")
    ],
    detections: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Detections with decoder columns per scene (<scene>.parquet).",
        ),
    ],
    bundle: Annotated[
        Path,
        typer.Option(
            exists=True, help="Detector bundle (its front end feeds the decoder)."
        ),
    ],
    params: Annotated[
        Path, typer.Option(exists=True, help="Chirp-parameter decoder bundle.")
    ],
    out: Annotated[Path, typer.Option(help="Gallery directory.")],
    exclude: Annotated[
        list[Path] | None,
        typer.Option(
            exists=True,
            help="Candidates files whose scenes must not be shown (the benchmark's).",
        ),
    ] = None,
    label: Annotated[str, typer.Option(help="Real label to show.")] = "not_chirp",
    labeler: Annotated[
        str | None, typer.Option(help="Only this labeller's labels.")
    ] = None,
    device: Annotated[str, typer.Option(help="Torch device of the decoder.")] = "cpu",
) -> None:
    """Real boxes with a label (default not_chirp) next to similar simulated chirps."""
    import pandas as pd
    from gridtools.io import Recording, open_recording

    from chirpdetector.labeling.benchmark import read_detections
    from chirpdetector.labeling.candidates import read_candidates
    from chirpdetector.labeling.gallery import (
        GalleryConfig,
        box_contrast_db,
        estimate_boxes,
        real_cases,
        sim_pool,
        write_gallery,
    )
    from chirpdetector.labeling.summary import read_label_files

    config = GalleryConfig()
    cases = real_cases(
        read_label_files(labels), read_candidates(candidates), label, labeler
    )
    if not len(cases):
        typer.echo(f"no candidates labelled {label!r}")
        raise typer.Exit(code=0)
    recordings = {
        str(n): open_recording(r)
        for n, r in zip(cases["_name"], cases["recording"], strict=True)
    }
    estimates = estimate_boxes(cases, recordings, bundle, params, device=device)
    cases = pd.concat([cases, estimates], axis=1)
    cases["contrast_db"] = [
        box_contrast_db(
            recordings[str(row["_name"])],
            [row[c] for c in ("t_start_s", "t_end_s", "f_low_hz", "f_high_hz")],
        )
        for row in cases.to_dict("records")
    ]
    skip: set[str] = set()
    for path in exclude or []:
        skip |= {
            Path(str(r)).name
            for r in pd.read_parquet(path, columns=["recording"])["recording"]
        }
    dets = read_detections(detections)
    scene_dirs = sorted(d for d in scenes.iterdir() if d.is_dir() and d.name in dets)
    pool = sim_pool(scene_dirs, dets, exclude=skip, tolerance_s=config.tolerance_s)
    typer.echo(
        f"{len(cases)} real cases; {len(pool)} simulated chirps from {pool['scene'].nunique()} scenes (excluded {len(skip)})"
    )
    opened: dict[str, Recording] = {}
    contrast = []
    for row in pool.to_dict("records"):
        rec = opened.setdefault(row["recording"], open_recording(row["recording"]))
        contrast.append(
            box_contrast_db(
                rec, [row[c] for c in ("t_start_s", "t_end_s", "f_low_hz", "f_high_hz")]
            )
        )
    pool["contrast_db"] = contrast
    out.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(out / "sim_pool.parquet", index=False)
    cases.drop(columns=["_name"]).to_parquet(out / "real_cases.parquet", index=False)
    index = write_gallery(cases, pool, out, config=config)
    typer.echo(f"gallery: {index}")
