"""The ``chirpdetector label-emitters`` command line."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from chirpdetector.cli import app

runner = CliRunner()


def test_help_lists_the_options() -> None:
    result = runner.invoke(app, ["label-emitters", "--help"])
    assert result.exit_code == 0
    for option in (
        "--out",
        "--per-bin",
        "--seed",
        "--labeler",
        "--grid",
        "--tolerance-hz",
        "--nfft",
    ):
        assert option in result.output


def test_refuses_to_write_into_a_trial_directory(synthetic_trial: Path) -> None:
    out = synthetic_trial / "labels"
    result = runner.invoke(
        app, ["label-emitters", str(synthetic_trial), "--out", str(out)]
    )
    assert result.exit_code != 0
    assert "inside a trial directory" in result.output
    assert not out.exists()


def test_rejects_a_malformed_grid(synthetic_trial: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "label-emitters",
            str(synthetic_trial),
            "--out",
            str(tmp_path),
            "--grid",
            "4by6",
        ],
    )
    assert result.exit_code != 0
    assert "4x6" in result.output


def test_reports_a_missing_display_cleanly(
    synthetic_trial: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        app, ["label-emitters", str(synthetic_trial), "--out", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "not interactive" in result.output
    assert (tmp_path / "queues").is_dir()
