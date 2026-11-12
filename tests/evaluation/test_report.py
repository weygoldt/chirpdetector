"""evaluate(), the result object, its files and bundle attachment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gridtools.io import load_tracks

from chirpdetector.evaluation import (
    EvaluationConfig,
    GroundTruth,
    MatchConfig,
    attach_to_bundle,
    evaluate,
    parse_thresholds,
    read_predictions,
    simulation_ground_truth,
)
from chirpdetector.evaluation.schema import Column, as_series


@pytest.fixture
def ramp(make_ramp_trial: Callable[[Path], Path], tmp_path: Path):
    """v1 detections and tracks of the ramp trial, labelled by hand.

    The first 50 detections are chirps, alternately of track 1 (v1's
    emitter for all of them) and track 0; the other 50 are false positives.
    """
    directory = make_ramp_trial(tmp_path / "ramp")
    preds = read_predictions(directory)
    first = preds.iloc[:50]
    labels = pd.DataFrame(
        {
            "recording": "ramp",
            "dataset": "synthetic",
            "t_start_s": first["t_start_s"].to_numpy(),
            "t_end_s": first["t_end_s"].to_numpy(),
            "f_low_hz": 690.0,
            "f_high_hz": 800.0,
            "label": "chirp",
            "emitter_id": pd.array((np.arange(50) % 2 == 0).astype(int), dtype="Int64"),
        }
    )
    regions = pd.DataFrame(
        {
            "recording": ["ramp"],
            "dataset": ["synthetic"],
            "region": [0],
            "t_start_s": [0.0],
            "t_end_s": [10.0],
            "f_low_hz": [600.0],
            "f_high_hz": [1000.0],
        }
    )
    return preds, GroundTruth(regions, labels), {"ramp": load_tracks(directory)}


def test_evaluate_counts_assignment_and_delta_f(ramp) -> None:
    preds, truth, tracks = ramp
    result = evaluate(
        preds, truth, tracks=tracks, config=EvaluationConfig(n_bootstrap=200)
    )
    pooled = result.metrics()["detection"]["all"]
    assert (pooled["labels"], pooled["detections"], pooled["tp"]) == (50, 100, 50)
    assert pooled["precision"] == 0.5
    assert pooled["recall"] == 1.0
    assert pooled["n_regions"] == 1
    items = result.items
    assert _counts(items["outcome"]) == {"tp": 50, "fp": 50}
    assert _counts(items.loc[items["outcome"] == "tp", "df_bin"]) == {
        "single": 2,
        "0-10": 2,
        "10-25": 8,
        "25-50": 12,
        "50-100": 25,
        "100+": 1,
    }
    assignment = result.assignment.set_index("df_bin")
    assert assignment.loc["all", "n"] == 50
    assert assignment.loc["all", "accuracy"] == 0.5  # v1 says track 1 for all
    assert assignment.loc["all", "coverage"] == 1.0
    assert result.coverage_accuracy.empty  # v1 has no emitter probabilities
    e2e = result.metrics()["end_to_end"]["region"]["all"]
    assert (e2e["labels"], e2e["detections"], e2e["tp"]) == (50, 100, 25)


def _counts(values: Column) -> dict:
    return as_series(values).value_counts().to_dict()


def test_result_is_deterministic_and_json_clean(ramp, tmp_path: Path) -> None:
    preds, truth, tracks = ramp
    config = EvaluationConfig(n_bootstrap=100)
    a = evaluate(preds, truth, tracks=tracks, config=config)
    b = evaluate(preds, truth, tracks=tracks, config=config)
    for name in a.TABLES:
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name))
    data = json.loads(a.to_json())  # allow_nan=False: no NaN may leak
    assert data["format"] == 1
    assert data["config"]["match"]["time_tolerance_s"] == 0.1
    assert (
        data["provenance"]["predictions_sha256"] == b.provenance["predictions_sha256"]
    )
    assert data["metrics"]["detection"]["recordings"]["ramp"]["tp"] == 50


def test_write_creates_metrics_tables_and_plot(ramp, tmp_path: Path) -> None:
    preds, truth, tracks = ramp
    result = evaluate(
        preds, truth, tracks=tracks, config=EvaluationConfig(n_bootstrap=50)
    )
    written = result.write(tmp_path / "out", plot=True)
    names = {p.name for p in written}
    assert {
        "metrics.json",
        "summary.csv",
        "intervals.csv",
        "items.csv",
        "pr_curve.png",
    } <= names
    assert all(p.exists() for p in written)
    summary = pd.read_csv(tmp_path / "out" / "summary.csv")
    assert set(summary["level"]) == {"recording", "dataset", "all"}
    intervals = pd.read_csv(tmp_path / "out" / "intervals.csv", index_col=0)
    assert intervals.index.tolist() == ["all", "synthetic"]


def test_attach_to_bundle_records_the_sha256(ramp, tmp_path: Path) -> None:
    preds, truth, _ = ramp
    result = evaluate(preds, truth, config=EvaluationConfig(n_bootstrap=10))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    weights = bundle / "weights.pt"
    weights.write_bytes(b"not really weights")
    path = attach_to_bundle(result, bundle, weights)
    data = json.loads(path.read_text())
    assert data["evaluated"] == [
        {
            "path": "weights.pt",
            "sha256": hashlib.sha256(b"not really weights").hexdigest(),
        }
    ]
    assert data["metrics"] == result.metrics()
    with pytest.raises(FileNotFoundError):
        attach_to_bundle(result, bundle, tmp_path / "missing.csv")


def test_simulated_scene_end_to_end(tmp_path: Path) -> None:
    scene = tmp_path / "scene_001"
    scene.mkdir()
    pd.DataFrame(
        {
            "t_onset_s": [1.0, 3.0, 5.0],
            "t_center_s": [1.02, 3.02, 5.02],
            "t_offset_s": [1.04, 3.04, 5.04],
            "f_low_hz": [600.0, 800.0, 600.0],
            "f_high_hz": [700.0, 900.0, 700.0],
            "emitter_id": [0, 1, 0],
        }
    ).to_csv(scene / "chirps.csv", index=False)
    (scene / "scene.json").write_text(json.dumps({"duration_s": 6.0}))
    truth = simulation_ground_truth(scene)
    preds = pd.DataFrame(
        {
            "recording": "scene_001",
            "t_start_s": [0.98, 2.98, 4.0],
            "t_end_s": [1.06, 3.06, 4.1],
            "f_low_hz": [600.0, 800.0, 600.0],
            "f_high_hz": [700.0, 900.0, 700.0],
            "score": [0.9, 0.8, 0.7],
            "emitter_id": pd.array([0, 0, 1], dtype="Int64"),
            "emitter_prob": [0.95, 0.6, 0.5],
        }
    )
    result = evaluate(preds, truth, config=EvaluationConfig(n_bootstrap=10))
    pooled = result.metrics()["detection"]["all"]
    assert (pooled["tp"], pooled["fp"], pooled["fn"]) == (2, 1, 1)
    e2e = result.metrics()["end_to_end"]["region"]["all"]
    assert (e2e["tp"], e2e["labels"], e2e["detections"]) == (1, 3, 3)
    curve = result.coverage_accuracy
    curve = curve[curve["df_bin"] == "all"]
    assert curve["coverage"].tolist() == [0.5, 1.0]
    assert curve["accuracy"].tolist() == [1.0, 0.5]
    assert set(result.items["df_bin"]) == {"unknown"}  # no tracks given


def test_unknown_recordings_and_missing_predictions(
    truth_factory, predictions_factory
) -> None:
    truth = GroundTruth.concat(
        [
            truth_factory(regions=[(0.0, 10.0, 0, 2000)], labels=[(5.0, 700, 800)]),
            truth_factory(
                regions=[(0.0, 10.0, 0, 2000)], labels=[(5.0, 700, 800)], recording="b"
            ),
        ]
    )
    preds = pd.concat(
        [
            predictions_factory([(5.0, 700, 800, 0.9)]),
            predictions_factory([(5.0, 700, 800, 0.9)], recording="other"),
        ]
    )
    result = evaluate(preds, truth, config=EvaluationConfig(n_bootstrap=10))
    rec = result.metrics()["detection"]["recordings"]
    assert rec["rec"]["tp"] == 1
    assert rec["b"]["tp"] == 0
    assert rec["b"]["fn"] == 1
    assert result.provenance["n_predictions_scored_recordings"] == 1


def test_greedy_and_hungarian_are_configurable(
    truth_factory, predictions_factory
) -> None:
    truth = truth_factory(
        regions=[(0.0, 100.0, 0, 2000)], labels=[(10.0, 700, 800), (10.125, 700, 800)]
    )
    preds = predictions_factory([(10.0625, 700, 800, 0.9), (9.9375, 700, 800, 0.8)])
    tp = {}
    for method in ("hungarian", "greedy"):
        config = EvaluationConfig(
            match=MatchConfig(time_tolerance_s=0.07, method=method), n_bootstrap=10
        )
        tp[method] = evaluate(preds, truth, config=config).metrics()["detection"][
            "all"
        ]["tp"]
    assert tp == {"hungarian": 2, "greedy": 1}


def test_parse_thresholds() -> None:
    assert parse_thresholds("0.25:0.95:0.05") == tuple(
        np.round(np.arange(0.25, 0.951, 0.05), 2).tolist()
    )
    assert parse_thresholds("0.3, 0.5") == (0.3, 0.5)
    for bad in ("a:b:c", "0.1:0.5:0", ""):
        with pytest.raises(ValueError, match=r"thresholds|step"):
            parse_thresholds(bad)


def test_config_always_evaluates_the_reference_threshold() -> None:
    config = EvaluationConfig(thresholds=(0.9, 0.3), reference_threshold=0.5)
    assert config.thresholds == (0.3, 0.9)
    assert config.all_thresholds == (0.3, 0.5, 0.9)
    with pytest.raises(ValueError, match="alpha"):
        EvaluationConfig(alpha=1.5)
