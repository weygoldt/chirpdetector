"""``chirpdetector data``: build detector datasets."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

__all__ = ["data_app"]

data_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Build detector datasets from simulated scenes and hand-labelled recordings.",
)


@data_app.command("build-scenes")
def build_scenes(
    out: Annotated[Path, typer.Option(help="Output dataset directory.")],
    scenes: Annotated[
        str | None,
        typer.Argument(
            help=(
                "Scene directories: a glob (quote it), e.g. "
                "'/mnt/data1/chirp_v2/scenes/train_v1/*', or a directory of "
                "scenes. Optional with --split-dir (default: all scenes in "
                "the split directories)."
            )
        ),
    ] = None,
    workers: Annotated[int, typer.Option(help="Worker processes.")] = 12,
    threads: Annotated[int, typer.Option(help="Torch threads per worker.")] = 2,
    split: Annotated[
        str,
        typer.Option(
            help=(
                "Seed ranges (train=0:1600,val=1600:1800,test=1800:2000) or "
                "fractions with a seeded shuffle (train=0.8,val=0.1,test=0.1)."
            )
        ),
    ] = "train=0:1600,val=1600:1800,test=1800:2000",
    split_seed: Annotated[int, typer.Option(help="Shuffle seed of fractions.")] = 0,
    split_dir: Annotated[
        list[str] | None,
        typer.Option(
            "--split-dir",
            help=(
                "SPLIT=DIRECTORY, repeatable: the scenes in DIRECTORY go to "
                "SPLIT (replaces --split), e.g. --split-dir "
                "train=/mnt/data1/chirp_v2/scenes/train_v2/train"
            ),
        ),
    ] = None,
    device: Annotated[
        str, typer.Option(help="STFT device: cpu, cuda or auto.")
    ] = "cpu",
    limit: Annotated[
        int | None, typer.Option(help="Use only the first N scenes (trial builds).")
    ] = None,
    overwrite: Annotated[
        bool, typer.Option(help="Replace an existing dataset.")
    ] = False,
) -> None:
    """Tile simulated scenes into a detector dataset (footprint labels)."""
    from chirpdetector.data.scenes import (
        SceneDatasetConfig,
        SplitRule,
        build_scene_dataset,
        find_scenes,
    )

    if device not in ("cpu", "cuda", "auto"):
        msg = f"--device must be cpu, cuda or auto, got {device!r}"
        raise typer.BadParameter(msg)
    try:
        if split_dir:
            rule = SplitRule.from_roots(split_dir)
            roots = (rule.roots or {}).values()
            dirs = (
                find_scenes(scenes)
                if scenes is not None
                else sorted(d for root in roots for d in find_scenes(root))
            )
        elif scenes is None:
            msg = "give SCENES or --split-dir"
            raise ValueError(msg)
        else:
            rule = SplitRule.parse(split, seed=split_seed)
            dirs = find_scenes(scenes)
    except (ValueError, FileNotFoundError) as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    if limit is not None:
        dirs = dirs[:limit]
    config = SceneDatasetConfig(
        split=rule,
        workers=workers,
        threads_per_worker=threads,
        device=device,  # pyright: ignore[reportArgumentType]
    )
    dataset = build_scene_dataset(
        dirs,
        out,
        config,
        overwrite=overwrite,
        provenance={
            "command": sys.argv,
            "scenes_glob": scenes,
            "split_dirs": split_dir,
        },
    )
    manifest = dataset.manifest
    counts = {s: len(manifest.image_ids(split=s)) for s in ("train", "val", "test")}
    typer.echo(
        f"wrote {len(manifest.images)} images of {len(dirs)} scenes to {out} "
        f"({counts}); content sha256 {manifest.content_sha256[:12]}"
    )


@data_app.command("build-real")
def build_real(
    out: Annotated[Path, typer.Option(help="Output dataset directory.")],
    recording: Annotated[
        list[Path] | None,
        typer.Option(
            "--recording",
            exists=True,
            file_okay=False,
            help="Labelled recording directory, repeatable (default: the four).",
        ),
    ] = None,
    device: Annotated[str, typer.Option(help="STFT device: cpu or cuda.")] = "cpu",
    overwrite: Annotated[
        bool, typer.Option(help="Replace an existing dataset.")
    ] = False,
) -> None:
    """Tile the hand-labelled regions of real recordings into the test set."""
    from chirpdetector.data.real import (
        DEFAULT_RECORDINGS,
        RealDatasetConfig,
        build_real_dataset,
    )

    dirs = list(recording) if recording else list(DEFAULT_RECORDINGS)
    dataset = build_real_dataset(
        out,
        dirs,
        RealDatasetConfig(device=device),
        overwrite=overwrite,
        provenance={"command": sys.argv},
    )
    manifest = dataset.manifest
    n_chirps = sum(r.n_chirps for r in manifest.images)
    typer.echo(
        f"wrote {len(manifest.images)} images ({n_chirps} chirp boxes) of "
        f"{len(dirs)} recordings to {out}; content sha256 "
        f"{manifest.content_sha256[:12]}"
    )
