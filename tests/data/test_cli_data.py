"""The ``chirpdetector data`` commands."""

from __future__ import annotations

import shutil
from pathlib import Path

from typer.testing import CliRunner

from chirpdetector.cli import app
from chirpdetector.data import load_dataset

runner = CliRunner()


def test_help_lists_the_commands_and_options() -> None:
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    assert "build-scenes" in result.output
    assert "build-real" in result.output
    result = runner.invoke(app, ["data", "build-scenes", "--help"])
    assert result.exit_code == 0
    for option in ("--out", "--workers", "--split", "--device", "--limit"):
        assert option in result.output


def test_build_scenes_from_a_glob(scene_dirs: dict[str, Path], tmp_path: Path) -> None:
    root = scene_dirs["grid9"].parent
    out = tmp_path / "dataset"
    result = runner.invoke(
        app,
        [
            "data",
            "build-scenes",
            str(root / "00000[45]"),
            "--out",
            str(out),
            "--workers",
            "1",
            "--split",
            "train=4:5,test=5:6",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "wrote 8 images of 2 scenes" in result.output
    manifest = load_dataset(out).manifest
    assert manifest.splits == {"test-000004": "train", "test-000005": "test"}
    assert manifest.provenance["scenes_glob"] == str(root / "00000[45]")


def test_build_scenes_reports_bad_input(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["data", "build-scenes", str(tmp_path / "none*"), "--out", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "no scene directories" in result.output
    result = runner.invoke(
        app,
        ["data", "build-scenes", "x", "--out", str(tmp_path), "--split", "a=1"],
    )
    assert result.exit_code == 1
    assert "unknown splits" in result.output


def test_build_scenes_with_a_split_per_directory(
    scene_dirs: dict[str, Path], tmp_path: Path
) -> None:
    roots = {"train": tmp_path / "a", "test": tmp_path / "b"}
    shutil.copytree(scene_dirs["grid4"], roots["train"] / "000004")
    shutil.copytree(scene_dirs["clean"], roots["test"] / "000005")
    out = tmp_path / "dataset"
    result = runner.invoke(
        app,
        [
            "data",
            "build-scenes",
            "--out",
            str(out),
            "--workers",
            "1",
            "--split-dir",
            f"train={roots['train']}",
            "--split-dir",
            f"test={roots['test']}",
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = load_dataset(out).manifest
    assert manifest.splits == {"test-000004": "train", "test-000005": "test"}
    assert manifest.provenance["config"]["split"] == {
        "roots": {k: str(v.resolve()) for k, v in roots.items()}
    }
    result = runner.invoke(app, ["data", "build-scenes", "--out", str(out)])
    assert result.exit_code == 1
    assert "SCENES or --split-dir" in result.output
