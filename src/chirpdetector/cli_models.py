"""CLI of the pretrained models: ``chirpdetector models list | download | path``."""

from __future__ import annotations

from typing import Annotated

import typer

from chirpdetector import pretrained

__all__ = ["models_app"]

models_app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Pretrained models: list, download, locate. They are downloaded on "
        "first use; set CHIRPDETECTOR_CACHE to move the cache and "
        "CHIRPDETECTOR_OFFLINE=1 to forbid downloads."
    ),
)


def _model(name: str) -> pretrained.PretrainedModel:
    try:
        return pretrained.get_model(name)
    except KeyError as err:
        typer.echo(err.args[0], err=True)
        raise typer.Exit(2) from None


@models_app.command("list")
def list_command() -> None:
    """List the published models, their aliases and whether they are cached."""
    aliases = {v: k for k, v in pretrained.ALIASES.items()}
    rows = [
        (
            m.name,
            m.kind,
            aliases.get(m.name, "-"),
            f"{m.size / 1e6:.1f} MB",
            "cached" if pretrained.is_cached(m.name) else "not downloaded",
            m.description,
        )
        for m in pretrained.list_models()
    ]
    header = ("NAME", "KIND", "ALIAS", "SIZE", "STATUS")
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        typer.echo("  ".join(v.ljust(w) for v, w in zip(row, widths, strict=False)))
        if row is not header:
            typer.echo(f"  {row[5]}")
    typer.echo(f"cache: {pretrained.cache_dir()}")


@models_app.command("download")
def download_command(
    names: Annotated[
        list[str] | None,
        typer.Argument(
            help="Model names or aliases (default: every published model).",
            show_default=False,
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Download again even if cached.")
    ] = False,
) -> None:
    """Download models into the cache (checked against their published sha256)."""
    models = [_model(n) for n in names] if names else pretrained.list_models()
    for model in models:
        try:
            directory = pretrained.download(model.name, force=force)
        except pretrained.PretrainedError as err:
            typer.echo(str(err), err=True)
            raise typer.Exit(1) from None
        typer.echo(f"{model.name}: {directory}")


@models_app.command("path")
def path_command(
    name: Annotated[str, typer.Argument(help="Model name or alias.")],
) -> None:
    """Print a model's bundle directory (downloading it first if missing)."""
    model = _model(name)
    try:
        typer.echo(str(pretrained.path(model.name)))
    except pretrained.PretrainedError as err:
        typer.echo(str(err), err=True)
        raise typer.Exit(1) from None
