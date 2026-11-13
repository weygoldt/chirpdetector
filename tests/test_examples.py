"""The example configs in ``examples/`` load and match the documented defaults."""

from __future__ import annotations

from pathlib import Path

from chirpdetector.params.training import ParamsTrainingConfig
from chirpdetector.pipeline.config import load_config
from chirpdetector.training import DetectorTrainingConfig

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_detector_examples_load() -> None:
    dualhead = DetectorTrainingConfig.from_toml(EXAMPLES / "train_dualhead.toml")
    assert dualhead.model_type == "dualhead"
    assert dualhead.model["architecture"] == "rtdetr_v2_r18vd"
    yolo = DetectorTrainingConfig.from_toml(EXAMPLES / "train_yolo26s.toml")
    assert yolo.model_type == "ultralytics-yolo"
    assert yolo.model["architecture"] == "yolo26s"


def test_params_example_loads() -> None:
    config = ParamsTrainingConfig.from_toml(EXAMPLES / "train_params.toml")
    assert config.boxes.mode == "detector"


def test_run_example_loads() -> None:
    config = load_config(EXAMPLES / "run.toml")
    assert config.name == "my_run"
    assert config.out == (EXAMPLES / "detections").resolve()
    assert config.detection.operating_threshold == 0.5
