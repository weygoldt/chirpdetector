"""The command line works on any machine: no default is a path of the author's."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from chirpdetector.cli import app
from chirpdetector.pipeline.config import config_from_dict, load_config

runner = CliRunner()

MACHINE = ("/mnt/", "/home/", "/run/media/", "chirp_v2")


def _commands(
    command: Any, path: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], Any]]:
    yield path, command
    for name, sub in getattr(command, "commands", {}).items():
        yield from _commands(sub, (*path, name))


def test_no_default_is_a_path_of_the_development_machine() -> None:
    for path, command in _commands(typer.main.get_command(app)):
        for param in getattr(command, "params", []):
            default = str(param.default)
            assert not any(m in default for m in MACHINE), (path, param.name, default)


def test_help_texts_survive_rich_markup() -> None:
    # rich drops "[word]" as a style tag: "[run] out" -> " out"
    tag = re.compile(r"(?<!\\)\[[a-z#/@][^\[\]]*\]")
    for path, command in _commands(typer.main.get_command(app)):
        texts = [getattr(command, "help", None)]
        texts += [getattr(p, "help", None) for p in getattr(command, "params", [])]
        for text in filter(None, texts):
            assert not tag.search(text), (path, text)


@pytest.mark.parametrize(
    ("args", "missing"),
    [
        (["label-emitters"], "--out"),
        (["label-summary"], "OUT"),
        (["data", "build-real", "--out", "dataset"], "--recording"),
    ],
)
def test_outputs_and_inputs_are_given_explicitly(
    args: list[str], missing: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # nothing is written next to the tests
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert missing in result.output


def test_labeler_defaults_to_the_login_name_at_run_time() -> None:
    command = typer.main.get_command(app).commands["label-emitters"]  # type: ignore[attr-defined]
    labeler = next(p for p in command.params if p.name == "labeler")
    assert labeler.default is None


def test_real_check_needs_the_recordings(tmp_path: Path) -> None:
    for name in ("bundle", "scenes", "library"):
        (tmp_path / name).mkdir()
    result = runner.invoke(
        app,
        [
            "evaluate",
            "params",
            str(tmp_path / "bundle"),
            "--scenes",
            str(tmp_path / "scenes"),
            "--library",
            str(tmp_path / "library"),
        ],
    )
    assert result.exit_code == 2
    assert "--recordings" in result.output


def test_run_output_defaults_next_to_the_config(tmp_path: Path) -> None:
    config = config_from_dict({"run": {"name": "x"}}, base_dir=tmp_path)
    assert config.out == tmp_path / "detections"
    path = tmp_path / "sub" / "run.toml"
    path.parent.mkdir()
    path.write_text('[run]\nname = "x"\n')
    assert load_config(path).out == path.parent.resolve() / "detections"
