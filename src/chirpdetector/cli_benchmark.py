"""``chirpdetector label-benchmark``: human against model on simulated scenes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

if TYPE_CHECKING:
    import pandas as pd

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
    out: Annotated[Path, typer.Option(help="Report directory.")],
    truth: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Truth file (single set)."),
    ] = None,
    labels: Annotated[
        Path | None,
        typer.Option(
            exists=True, file_okay=False, help="label-emitters --out of that set."
        ),
    ] = None,
    set_: Annotated[
        list[str] | None,
        typer.Option(
            "--set", help="NAME=TRUTH:LABELS_DIR, repeatable: merge several sets."
        ),
    ] = None,
    model: Annotated[
        list[str] | None,
        typer.Option(
            help="NAME=DIR[:v1], repeatable: score these models on every item."
        ),
    ] = None,
    features_dir: Annotated[
        Path | None,
        typer.Option(
            "--features",
            exists=True,
            file_okay=False,
            help="Scene features (fill in old items).",
        ),
    ] = None,
    retest: Annotated[
        list[str] | None,
        typer.Option(
            help="NAME=REPEAT_TRUTH:REPEAT_LABELS:ORIGINAL_TRUTH:ORIGINAL_LABELS, repeatable: "
            "test-retest of a repeat session against the session it repeats."
        ),
    ] = None,
    labeler: Annotated[
        str | None, typer.Option(help="Only this labeller's labels.")
    ] = None,
    threshold: Annotated[float, typer.Option(help="Model score threshold.")] = 0.5,
    n_boot: Annotated[int, typer.Option(help="Bootstrap resamples.")] = 5000,
    seed: Annotated[int, typer.Option(help="Bootstrap seed.")] = 0,
) -> None:
    """Score the human's labels and the models on the benchmark items.

    One set with --truth/--labels and the model scores in the truth file
    (the v1 report), or several --set and --model (the v2 report: every
    stratum, negative type, harmonic copies, emitter by ΔF, test-retest).
    """
    import pandas as pd

    from chirpdetector.labeling.summary import read_label_files

    if set_:
        from chirpdetector.labeling.benchmark_report import (
            attach_features,
            merge_sets,
            retest_pairs,
            score_v2,
            write_report_v2,
        )
        from chirpdetector.labeling.benchmark_v2 import model_outcomes, read_model_dir

        sets = []
        for value in set_:
            name, _, rest = value.partition("=")
            t_path, _, l_path = rest.rpartition(":")
            if not (name and t_path and l_path):
                raise typer.BadParameter(
                    f"--set wants NAME=TRUTH:LABELS_DIR, got {value!r}"
                )
            sets.append((name, pd.read_parquet(t_path), read_label_files(l_path)))
        items = merge_sets(sets, labeler)
        if features_dir is not None:
            feats = {
                name: chirps
                for name, (_, chirps, _) in _load_features(features_dir).items()
            }
            items = attach_features(items, feats)  # pyright: ignore[reportArgumentType]
        models = {
            name: read_model_dir(path, kind)
            for name, (path, kind) in _model_option(model).items()
        }
        outcomes = model_outcomes(items, models)
        result = score_v2(
            items, outcomes, threshold=threshold, n_boot=n_boot, seed=seed
        )
        retests = {}
        for value in retest or []:
            name, _, rest = value.partition("=")
            parts = rest.split(":")
            if len(parts) != 4:
                raise typer.BadParameter(
                    f"--retest wants NAME=RT:RL:OT:OL, got {value!r}"
                )
            rt, rl, ot, ol = parts
            retests[name] = retest_pairs(
                pd.read_parquet(rt),
                read_label_files(rl),
                pd.read_parquet(ot),
                read_label_files(ol),
                labeler,
            )
        path = write_report_v2(result, out, retests=retests)
        typer.echo(path.read_text().split("\n## Where")[0])
        typer.echo(f"report: {path}")
        return
    if truth is None or labels is None:
        raise typer.BadParameter("give --truth and --labels, or --set")
    from chirpdetector.labeling.benchmark import score_benchmark, write_report

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


def _features_one(args: tuple[str, str, str]) -> str:
    """Worker: features of one scene into OUT/<scene>.parquet and .json."""
    import pandas as pd

    from chirpdetector.labeling.benchmark_v2 import scene_features

    scene, out, templates_path = args
    target = Path(out) / f"{Path(scene).name}.parquet"
    if target.exists():
        return Path(scene).name
    templates = pd.read_parquet(templates_path) if templates_path else None
    chirps, info = scene_features(scene, templates=templates)
    chirps.to_parquet(target.with_suffix(".tmp"), index=False)
    target.with_suffix(".tmp").replace(target)
    target.with_suffix(".json").write_text(json.dumps(info))
    return Path(scene).name


@benchmark_app.command("features")
def features(
    scenes: Annotated[
        list[Path],
        typer.Argument(exists=True, file_okay=False, help="Scene directories."),
    ],
    out: Annotated[
        Path, typer.Option(help="Feature directory (<scene>.parquet + .json).")
    ],
    templates: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, help="Template table (templates.parquet)."
        ),
    ] = None,
    workers: Annotated[int, typer.Option(help="Processes.")] = 8,
) -> None:
    """Per-chirp features of scenes (chirpdetector.labeling.benchmark_v2), cached."""
    from concurrent.futures import ProcessPoolExecutor

    out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(s), str(out), str(templates) if templates else "") for s in sorted(scenes)
    ]
    with ProcessPoolExecutor(workers) as pool:
        for k, name in enumerate(pool.map(_features_one, jobs), start=1):
            if k % 50 == 0 or k == len(jobs):
                typer.echo(f"{k}/{len(jobs)} {name}")


def _load_features(
    directory: Path, names: set[str] | None = None
) -> dict[str, tuple[Path, pd.DataFrame, dict[str, Any]]]:
    """Cached scene features: ``{scene: (path, chirps, info)}``.

    ``drift`` is re-derived from ``drift_hz`` with the current threshold.
    """
    import numpy as np
    import pandas as pd

    from chirpdetector.labeling.benchmark_v2 import BuildV2Config

    threshold = BuildV2Config().drift_hz
    out: dict[str, tuple[Path, pd.DataFrame, dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.parquet")):
        if names is not None and path.stem not in names:
            continue
        info = json.loads(path.with_suffix(".json").read_text())
        chirps = pd.read_parquet(path)
        if len(chirps):
            chirps["drift"] = (
                np.nan_to_num(chirps["drift_hz"].to_numpy(dtype=float)) >= threshold
            )
        out[path.stem] = (Path(info.get("path", "")), chirps, info)
    return out


@benchmark_app.command("gaps")
def gaps(
    scenes: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="Scene root.")
    ],
    features_dir: Annotated[
        Path,
        typer.Option(
            "--features", exists=True, file_okay=False, help="Feature directory."
        ),
    ],
    out: Annotated[Path, typer.Option(help="Root of the gap copies.")],
    n: Annotated[int, typer.Option(help="Gap copies (one chirp each).")] = 20,
    seed: Annotated[int, typer.Option(help="Seed.")] = 2,
) -> None:
    """Copies of scenes whose emitter track breaks around one chirp each."""
    from chirpdetector.labeling.benchmark_v2 import BuildV2Config, plan_gaps

    found = {
        name: (scenes / name, chirps, info)
        for name, (_, chirps, info) in _load_features(features_dir).items()
        if (scenes / name).is_dir()
    }
    config = BuildV2Config()
    made = plan_gaps(
        found,
        n,
        seed=seed,
        margin_s=config.margin_s,
        target_root=out,
        half_s=config.gap_half_s,
    )  # pyright: ignore[reportArgumentType]
    for alias, (path, chirps, info) in made.items():
        chirps.to_parquet(features_dir / f"{alias}.parquet", index=False)
        (features_dir / f"{alias}.json").write_text(
            json.dumps({**info, "path": str(path)})
        )
    typer.echo(f"{len(made)} gap copies in {out}")


def _model_option(values: list[str] | None) -> dict[str, tuple[Path, str]]:
    out = {}
    for value in values or []:
        name, _, rest = value.partition("=")
        path, _, kind = rest.partition(":")
        if not name or not path:
            raise typer.BadParameter(f"--model wants NAME=DIR[:v1], got {value!r}")
        out[name] = (Path(path), kind or "v2")
    return out


@benchmark_app.command("build-v2")
def build_v2_command(
    scenes: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="Root of the new scenes.")
    ],
    features_dir: Annotated[
        Path,
        typer.Option(
            "--features", exists=True, file_okay=False, help="Feature directory."
        ),
    ],
    candidates: Annotated[Path, typer.Option(help="Candidates file to write.")],
    truth: Annotated[
        Path, typer.Option(help="Truth file to write (another directory).")
    ],
    model: Annotated[
        list[str] | None,
        typer.Option(help="NAME=DIR[:v1], repeatable (detections per scene)."),
    ] = None,
    gap_root: Annotated[
        Path | None, typer.Option(help="Root of the gap copies.")
    ] = None,
    repeat_from: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, help="Truth of labelled items to repeat."
        ),
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed.")] = 2,
) -> None:
    """Sample the v2 items (chirpdetector.labeling.benchmark_v2)."""
    import pandas as pd

    from chirpdetector.labeling.benchmark import write_benchmark
    from chirpdetector.labeling.benchmark_v2 import (
        BuildV2Config,
        build_v2,
        model_outcomes,
        read_model_dir,
    )

    models = {
        name: read_model_dir(path, kind)
        for name, (path, kind) in _model_option(model).items()
    }
    feats = _load_features(features_dir)
    new = {
        n: (scenes / n, c, i) for n, (_, c, i) in feats.items() if (scenes / n).is_dir()
    }
    gap = (
        {
            n: (gap_root / n, c, i)
            for n, (_, c, i) in feats.items()
            if (gap_root / n).is_dir()
        }
        if gap_root
        else {}
    )
    exclude = {(str(i["gap_of"]), int(i["gap_chirp_id"])) for _, _, i in gap.values()}
    cands, table, summary = build_v2(
        new,  # pyright: ignore[reportArgumentType]
        models,
        BuildV2Config(seed=seed),
        repeat_pool=pd.read_parquet(repeat_from) if repeat_from else None,
        gap_scenes=gap,  # pyright: ignore[reportArgumentType]
        exclude=exclude,
    )
    outcomes = model_outcomes(table, models)
    for name in models:
        sel = outcomes[outcomes["model"] == name].set_index("item_id")
        table[f"model_{name}_score"] = sel["score"].reindex(table["item_id"]).to_numpy()
    write_benchmark(cands, table, summary, candidates, truth)
    typer.echo(
        json.dumps(
            {k: summary[k] for k in ("n_items", "per_stratum", "pools")}, indent=1
        )
    )
