"""The ``chirpdetector evaluate`` command line."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from gridtools.io import write_labels
from typer.testing import CliRunner

from chirpdetector.cli import app

runner = CliRunner()


def test_help_lists_the_options() -> None:
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "detections" in result.output
    assert "params" in result.output
    # `evaluate PREDICTIONS ...` is `evaluate detections PREDICTIONS ...`
    result = runner.invoke(app, ["evaluate", "detections", "--help"])
    assert result.exit_code == 0
    for option in (
        "--labels",
        "--out",
        "--thresholds",
        "--nms-iou",
        "--tracks",
        "--method",
    ):
        assert option in result.output


def _hand_labels(trial: Path) -> Path:
    """Label the ramp trial's first 50 detections; region 0-10 s."""
    boxes = pd.read_csv(trial / "chirpdetector_bboxes.csv").iloc[:50]
    rows = [
        {
            "category": "labeled_region",
            "kind": "span",
            "t_start_s": 0.0,
            "t_end_s": 10.0,
            "f_low_hz": 600.0,
            "f_high_hz": 1000.0,
        }
    ] + [
        {
            "category": "chirp",
            "kind": "span",
            "t_start_s": t1,
            "t_end_s": t2,
            "f_low_hz": 690.0,
            "f_high_hz": 800.0,
        }
        for t1, t2 in zip(boxes["t1"], boxes["t2"], strict=True)
    ]
    path = trial / "traces-grid1-editable-labels.csv"
    write_labels(pd.DataFrame(rows), path)
    return path


def test_evaluates_v1_detections_with_tracks(
    make_ramp_trial: Callable[[Path], Path], tmp_path: Path
) -> None:
    trial = make_ramp_trial(tmp_path / "data" / "ramp")
    labels = _hand_labels(trial)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(trial),
            "--labels",
            str(labels),
            "--out",
            str(out),
            "--thresholds",
            "0.5,0.8",
            "--n-bootstrap",
            "20",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "precision 0.500" in result.output
    assert "recall 1.000" in result.output
    data = json.loads((out / "metrics.json").read_text())
    assert data["metrics"]["detection"]["all"]["tp"] == 50
    inputs = data["provenance"]["inputs"]
    assert {i["role"] for i in inputs} == {"labels", "predictions"}
    assert all(len(i["sha256"]) == 64 for i in inputs)
    assert "ramp" in data["provenance"]["tracks"]  # found next to the labels
    items = pd.read_csv(out / "items.csv")
    assert "unknown" not in set(items["df_bin"])
    summary = pd.read_csv(out / "summary.csv")
    assert sorted(set(summary["threshold"])) == [0.5, 0.8]


def test_reports_bad_input_without_a_traceback(
    make_ramp_trial: Callable[[Path], Path], tmp_path: Path
) -> None:
    trial = make_ramp_trial(tmp_path / "ramp")
    labels = _hand_labels(trial)
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(trial),
            "--labels",
            str(labels),
            "--out",
            str(tmp_path / "out"),
            "--thresholds",
            "high",
        ],
    )
    assert result.exit_code == 1
    assert "error: thresholds must look like" in result.output
    assert not (tmp_path / "out").exists()


def test_options_before_the_predictions_still_select_detections(
    make_ramp_trial: Callable[[Path], Path], tmp_path: Path
) -> None:
    """`evaluate` became a group: `evaluate --labels L PRED` (options first)
    and `evaluate PRED ...` must both reach the detection evaluation."""
    trial = make_ramp_trial(tmp_path / "ramp")
    labels = _hand_labels(trial)
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--labels",
            str(labels),
            "--out",
            str(tmp_path / "out"),
            "--thresholds",
            "high",
            str(trial),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "error: thresholds must look like" in result.output
    result = runner.invoke(app, ["evaluate", "params", "--help"])
    assert result.exit_code == 0
    assert "--scenes" in result.output
