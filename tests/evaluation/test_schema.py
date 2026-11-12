"""Prediction validation, ground truth construction and readers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gridtools.io import write_labels

from chirpdetector.evaluation import (
    GroundTruth,
    read_ground_truth,
    simulation_ground_truth,
    validate_predictions,
)
from chirpdetector.evaluation.schema import UNKNOWN_DATASET, infer_dataset

BOX = {"t_start_s": [1.0], "t_end_s": [1.1], "f_low_hz": [700.0], "f_high_hz": [800.0]}


def test_validate_fills_recording_and_optional_columns() -> None:
    df = validate_predictions(pd.DataFrame({**BOX, "score": [0.9]}), recording="r")
    assert df["recording"].tolist() == ["r"]
    assert df["emitter_id"].dtype == "Int64"
    assert df["emitter_id"].isna().to_numpy().all()
    assert np.isnan(df["emitter_prob"]).all()


def test_validate_converts_float_emitters_to_nullable_ints() -> None:
    df = pd.DataFrame(
        {**{k: v * 2 for k, v in BOX.items()}, "score": [0.9, 0.8]}
    ).assign(emitter_id=[3.0, np.nan], recording="r")
    out = validate_predictions(df)
    assert out["emitter_id"].tolist()[0] == 3
    assert out["emitter_id"].isna().tolist() == [False, True]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"score": [np.nan]}, "finite"),
        ({"t_end_s": [0.5]}, "t_end_s < t_start_s"),
        ({"f_high_hz": [600.0]}, "f_high_hz < f_low_hz"),
        ({"emitter_prob": [1.5]}, r"\[0, 1\]"),
        ({"emitter_id": [1.5]}, "integers"),
    ],
)
def test_validate_rejects_malformed_rows(change: dict, message: str) -> None:
    df = pd.DataFrame({**BOX, "score": [0.9], "recording": ["r"], **change})
    with pytest.raises(ValueError, match=message):
        validate_predictions(df)


def test_validate_needs_a_recording() -> None:
    with pytest.raises(ValueError, match="recording"):
        validate_predictions(pd.DataFrame({**BOX, "score": [0.9]}))
    with pytest.raises(ValueError, match="disagrees"):
        validate_predictions(
            pd.DataFrame({**BOX, "score": [0.9], "recording": ["a"]}), recording="b"
        )
    with pytest.raises(ValueError, match="missing columns"):
        validate_predictions(pd.DataFrame(BOX), recording="r")


def test_labels_belong_to_the_region_containing_their_centre(truth_factory) -> None:
    truth = truth_factory(
        regions=[(10.0, 20.0, 600.0, 900.0)],
        labels=[(10.0, 700, 800), (20.0, 700, 800), (20.5, 700, 800), (5.0, 1, 2)],
    )
    assert truth.labels["region"].isna().tolist() == [True, False, False, True]
    assert truth.labels.loc[1, "t_center_s"] == 10.0  # sorted by centre time


def test_missing_band_edges_are_unbounded() -> None:
    regions = pd.DataFrame(
        {
            "recording": ["r"],
            "dataset": ["d"],
            "region": [0],
            "t_start_s": [0.0],
            "t_end_s": [1.0],
            "f_low_hz": [np.nan],
            "f_high_hz": [np.nan],
        }
    )
    truth = GroundTruth(
        regions, regions.iloc[:0].drop(columns="region").assign(label=[])
    )
    assert truth.regions.loc[0, "f_low_hz"] == -np.inf
    assert truth.regions.loc[0, "f_high_hz"] == np.inf


def test_overlapping_regions_are_rejected(truth_factory) -> None:
    with pytest.raises(ValueError, match="overlap"):
        truth_factory(regions=[(0.0, 10.0, 0, 1), (10.0, 20.0, 0, 1)])


def test_unknown_labels_and_missing_regions_are_rejected(truth_factory) -> None:
    with pytest.raises(ValueError, match="unknown label"):
        truth_factory(labels=[(1.0, 700, 800, "rise")])
    truth = truth_factory(regions=[(0.0, 10.0, 0, 1)], labels=[(1.0, 700, 800)])
    labels = truth.labels.assign(region=pd.array([5], dtype="Int64"))
    with pytest.raises(ValueError, match="missing regions"):
        GroundTruth(truth.regions, labels)


def test_concat_renumbers_regions_and_fills_unknown_datasets(truth_factory) -> None:
    a = truth_factory(regions=[(0.0, 10.0, 0, 1e4)], labels=[(5.0, 700, 800)])
    b = truth_factory(
        regions=[(20.0, 30.0, 0, 1e4)],
        labels=[(25.0, 700, 800), (40.0, 700, 800, "not_chirp")],
        dataset=UNKNOWN_DATASET,
    )
    truth = GroundTruth.concat([a, b])
    assert truth.regions["region"].tolist() == [0, 1]
    assert truth.labels["region"].tolist()[:2] == [0, 1]
    assert pd.isna(truth.labels["region"].tolist()[2])
    assert set(truth.labels["dataset"]) == {"ds"}
    assert truth.datasets == {"rec": "ds"}


def _audian(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_labels(pd.DataFrame(rows), path)
    return path


def _row(category: str, t0: float, t1: float, f0: float, f1: float, **extra) -> dict:
    return {
        "category": category,
        "kind": "span",
        "t_start_s": t0,
        "t_end_s": t1,
        "f_low_hz": f0,
        "f_high_hz": f1,
        **extra,
    }


def test_reads_hand_labels_from_a_recording_directory(tmp_path: Path) -> None:
    path = _audian(
        tmp_path / "2022_tube_competition" / "raw" / "rec1" / "labels.csv",
        [
            _row("labeled_region", 10, 20, 600, 900),
            _row("chirp", 11, 11.1, 700, 800),
            _row("chirp", 30, 30.1, 700, 800),
            _row("rise", 12, 14, 700, 720),
        ],
    )
    truth = read_ground_truth(path)
    assert truth.recordings == ("rec1",)
    assert set(truth.datasets.values()) == {"2022_tube_competition"}
    assert len(truth.regions) == 1
    assert truth.labels["label"].tolist() == ["chirp", "chirp"]
    assert truth.labels["region"].isna().tolist() == [False, True]
    assert truth.labels.loc[0, "t_center_s"] == pytest.approx(11.05)


def test_dataset_is_the_parent_directory_skipping_raw(tmp_path: Path) -> None:
    assert infer_dataset(tmp_path / "ds" / "raw" / "rec") == "ds"
    assert infer_dataset(tmp_path / "ds" / "rec") == "ds"


def test_reads_label_emitters_output_as_spot_labels(tmp_path: Path) -> None:
    rows = [
        _row("chirp", 1, 1.1, 700, 800, label="chirp", emitter_id=3, item_id="tr#4"),
        _row("not_chirp", 2, 2.1, 700, 800, label="not_chirp", item_id="tr#7"),
        _row("chirp", 3, 3.1, 700, 800, label="chirp_emitter_unclear", item_id="tr#9"),
    ]
    truth = read_ground_truth(_audian(tmp_path / "labels" / "tr.csv", rows))
    assert truth.recordings == ("tr",)
    assert truth.labels["label"].tolist() == [
        "chirp",
        "not_chirp",
        "chirp_emitter_unclear",
    ]
    assert truth.labels["emitter_id"].tolist()[0] == 3
    assert truth.labels["emitter_id"].isna().tolist() == [False, True, True]
    assert truth.labels["region"].isna().to_numpy().all()
    assert truth.labels["dataset"].unique().tolist() == [UNKNOWN_DATASET]


def _scene(directory: Path, **extra_columns) -> Path:
    directory.mkdir(parents=True)
    pd.DataFrame(
        {
            "t_onset_s": [1.0, 5.0],
            "t_center_s": [1.02, 5.01],
            "t_offset_s": [1.1, 5.05],
            "f_low_hz": [600.0, 800.0],
            "f_high_hz": [700.0, 900.0],
            "emitter_id": [0, 1],
            **extra_columns,
        }
    ).to_csv(directory / "chirps.csv", index=False)
    return directory


def test_simulation_scene_is_one_region(tmp_path: Path) -> None:
    scene = _scene(tmp_path / "scene_000", visible=[True, False])
    (scene / "scene.json").write_text(json.dumps({"config": {"duration_s": 10.0}}))
    truth = simulation_ground_truth(
        scene, band_hz=(500.0, 1000.0), visible_column="visible"
    )
    region = truth.regions.iloc[0]
    assert (region["t_start_s"], region["t_end_s"]) == (0.0, 10.0)
    assert (region["f_low_hz"], region["f_high_hz"]) == (500.0, 1000.0)
    assert truth.labels["t_center_s"].tolist() == [1.02, 5.01]  # not the midpoint
    assert truth.labels["label"].tolist() == ["chirp", "ignore"]
    assert truth.labels["emitter_id"].tolist() == [0, 1]
    assert truth.recordings == ("scene_000",)
    assert truth.datasets["scene_000"] == "simulation"


def test_simulation_without_duration_is_unbounded(tmp_path: Path) -> None:
    truth = simulation_ground_truth(_scene(tmp_path / "s") / "chirps.csv")
    assert truth.regions.loc[0, "t_end_s"] == np.inf
    assert truth.regions.loc[0, "f_low_hz"] == -np.inf


def test_simulation_needs_its_columns(tmp_path: Path) -> None:
    scene = _scene(tmp_path / "s")
    pd.read_csv(scene / "chirps.csv").drop(columns="emitter_id").to_csv(
        scene / "chirps.csv", index=False
    )
    with pytest.raises(ValueError, match="emitter_id"):
        simulation_ground_truth(scene)
