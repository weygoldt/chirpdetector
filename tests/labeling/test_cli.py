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
        "--window-s",
        "--candidates",
        "--view",
        "--serve",
        "--port",
        "--keep-order",
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


def test_needs_trials_or_candidates_but_not_both(
    synthetic_trial: Path, tmp_path: Path
) -> None:
    result = runner.invoke(app, ["label-emitters", "--out", str(tmp_path)])
    assert result.exit_code != 0
    assert "--candidates" in result.output
    path = tmp_path / "c.csv"
    path.write_text(
        f"recording,t_start_s,t_end_s,f_low_hz,f_high_hz\n{synthetic_trial},3.93,4.07,680,820\n"
    )
    result = runner.invoke(
        app,
        [
            "label-emitters",
            str(synthetic_trial),
            "--candidates",
            str(path),
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0


def test_invalid_candidates_are_reported(synthetic_trial: Path, tmp_path: Path) -> None:
    path = tmp_path / "c.csv"
    path.write_text(
        f"recording,t_start_s,t_end_s,f_low_hz,f_high_hz\n{synthetic_trial},4.07,3.93,680,820\n"
    )
    result = runner.invoke(
        app, ["label-emitters", "--candidates", str(path), "--out", str(tmp_path / "o")]
    )
    assert result.exit_code == 1
    assert "t_end_s <= t_start_s" in result.output
    assert not (tmp_path / "o").exists()


def test_candidates_refuse_to_write_into_a_recording(
    synthetic_trial: Path, tmp_path: Path
) -> None:
    path = tmp_path / "c.csv"
    path.write_text(
        f"recording,t_start_s,t_end_s,f_low_hz,f_high_hz\n{synthetic_trial},3.93,4.07,680,820\n"
    )
    out = synthetic_trial / "labels"
    result = runner.invoke(
        app, ["label-emitters", "--candidates", str(path), "--out", str(out)]
    )
    assert result.exit_code != 0
    assert "inside a trial directory" in result.output
    assert not out.exists()


def test_serve_does_not_take_the_grid(synthetic_trial: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "label-emitters",
            str(synthetic_trial),
            "--out",
            str(tmp_path),
            "--serve",
            "--view",
            "grid",
        ],
    )
    assert result.exit_code != 0
    assert "no grid" in result.output


def test_label_summary_command(tmp_path: Path) -> None:
    result = runner.invoke(app, ["label-summary", str(tmp_path)])
    assert result.exit_code == 0
    assert "by decision" in result.output


def test_an_empty_candidates_file_has_nothing_to_label(tmp_path: Path) -> None:
    path = tmp_path / "c.csv"
    path.write_text("recording,t_start_s,t_end_s,f_low_hz,f_high_hz\n")
    result = runner.invoke(
        app, ["label-emitters", "--candidates", str(path), "--out", str(tmp_path / "o")]
    )
    assert result.exit_code == 0
    assert "nothing to label" in result.output
