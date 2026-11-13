"""Training config, grouped folds, and the GPU memory cap."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from chirpdetector.resources import VRAM_ENV, apply_vram_limit, vram_fraction
from chirpdetector.training import (
    DetectorTrainingConfig,
    OptimConfig,
    assign_folds,
    fold_groups,
)

# ------------------------------------------------------------------- config


def test_defaults_resolve_the_model_section() -> None:
    # The dual-head RT-DETRv2 is the default model; YOLO26 is the box-only option.
    config = DetectorTrainingConfig()
    assert config.model_type == "dualhead"
    assert config.model["architecture"] == "rtdetr_v2_r18vd"
    assert config.model["pretrained"] is True
    assert config.augment.time_flip == 0.0
    data = config.to_dict()
    assert data["model"]["type"] == "dualhead"
    assert DetectorTrainingConfig.from_dict(data) == config
    assert DetectorTrainingConfig.from_dict({}) == config


def test_a_model_table_without_type_is_the_dualhead() -> None:
    config = DetectorTrainingConfig.from_dict(
        {"model": {"architecture": "tiny", "pretrained": False}}
    )
    assert config.model_type == "dualhead"
    assert config.model["architecture"] == "tiny"


def test_yolo_is_selected_by_type() -> None:
    config = DetectorTrainingConfig.from_dict({"model": {"type": "ultralytics-yolo"}})
    assert config.model_type == "ultralytics-yolo"
    assert config.model["architecture"] == "yolo26s"


DUALHEAD_OPTIM = {
    "epochs": 30,
    "lr0": 1e-4,
    "lrf": 0.05,
    "warmup_epochs": 0.5,
    "weight_decay": 1e-4,
    "patience": 6,
}


def test_optim_defaults_follow_the_model_type() -> None:
    # without [optim], the dual-head gets the settings it was trained with ...
    dualhead = DetectorTrainingConfig.from_dict({})
    assert dualhead.optim == OptimConfig(**DUALHEAD_OPTIM)  # type: ignore[arg-type]
    assert DetectorTrainingConfig().optim == dualhead.optim
    # ... and YOLO26 keeps the defaults tuned for it
    yolo = DetectorTrainingConfig.from_dict({"model": {"type": "ultralytics-yolo"}})
    assert yolo.optim == OptimConfig()
    assert yolo.optim.lr0 == 0.001
    assert yolo.optim.epochs == 100
    # a partial [optim] keeps the model type's defaults for the other keys
    partial = DetectorTrainingConfig.from_dict({"optim": {"epochs": 2}})
    assert partial.optim == OptimConfig(**{**DUALHEAD_OPTIM, "epochs": 2})  # type: ignore[arg-type]
    for config in (dualhead, yolo, partial):
        assert DetectorTrainingConfig.from_dict(config.to_dict()) == config
    # the released dual-head's example differs only in the machine's workers
    example = DetectorTrainingConfig.from_toml(
        Path(__file__).resolve().parents[2] / "examples" / "train_dualhead.toml"
    )
    assert example.optim == OptimConfig(**{**DUALHEAD_OPTIM, "workers": 8})  # type: ignore[arg-type]


def test_toml_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "train.toml"
    path.write_text(
        """
seed = 3
device = "cpu"
max_vram_fraction = 0.5

[model]
type = "ultralytics-yolo"
architecture = "yolo26n"
pretrained = false

[optim]
epochs = 2
batch = 4

[augment]
time_flip = 0.5

[cv]
n_folds = 3
"""
    )
    config = DetectorTrainingConfig.from_toml(path)
    assert config.model == {
        "architecture": "yolo26n",
        "pretrained": False,
        "weights": None,
        "max_det": 300,
    }
    assert config.optim.epochs == 2
    assert config.cv.n_folds == 3
    assert config.max_vram_fraction == 0.5
    assert DetectorTrainingConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"sed": 1}, "unknown top-level keys"),
        ({"optim": {"epoch": 3}}, r"unknown keys in \[optim\]"),
        ({"augment": {"flipud": 0.5}}, r"unknown keys in \[augment\]"),
        ({"augment": {"time_flip": 2.0}}, "time_flip"),
        ({"model": {"type": "ultralytics-yolo", "mosaic": 1.0}}, "unknown"),
        ({"model": {"architecture": "yolov8l"}}, "architecture"),
        ({"model": {"architecture": "yolo26s"}}, "architecture"),
        ({"model": {"type": "detr"}}, "unknown model type"),
        ({"cv": {"n_folds": 1}}, "n_folds"),
        ({"device": "gpu"}, "device"),
        ({"max_vram_fraction": 1.5}, "max_vram_fraction"),
    ],
)
def test_bad_configs_raise(data: dict, match: str) -> None:
    with pytest.raises((ValueError, KeyError), match=match):
        DetectorTrainingConfig.from_dict(data)


@pytest.mark.parametrize("batch", [-1, 0, 0.7])
def test_autobatch_is_not_available(batch: float) -> None:
    with pytest.raises(ValueError, match="no AutoBatch"):
        OptimConfig(batch=batch)  # type: ignore[arg-type]


# -------------------------------------------------------------------- folds


def test_no_group_is_in_two_folds() -> None:
    sizes = {f"g{k}": 1 + (k * 7) % 5 for k in range(23)}
    assignment = assign_folds(sizes, 5, seed=1)
    assert set(assignment) == set(sizes)
    for fold in range(5):
        train, val = fold_groups(assignment, fold)
        assert not set(train) & set(val)
        assert set(train) | set(val) == set(sizes)
    validated = [g for fold in range(5) for g in fold_groups(assignment, fold)[1]]
    assert sorted(validated) == sorted(sizes)  # every group validated exactly once


def test_folds_are_balanced_and_deterministic() -> None:
    sizes = {f"g{k}": 10 for k in range(10)} | {"big": 30}
    assignment = assign_folds(sizes, 4, seed=0)
    loads = [sum(sizes[g] for g, f in assignment.items() if f == k) for k in range(4)]
    assert max(loads) - min(loads) <= 10
    assert assign_folds(sizes, 4, seed=0) == assignment
    assert assign_folds(sizes, 4, seed=1) != assignment  # ties broken by the seed


def test_too_few_groups_for_the_folds() -> None:
    with pytest.raises(ValueError, match="cannot fill"):
        assign_folds({"a": 1, "b": 2}, 3)


# -------------------------------------------------------------- VRAM cap


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> list[tuple[float, int | None]]:
    calls: list[tuple[float, int | None]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device=None: calls.append((fraction, device)),
    )
    return calls


def test_env_var_caps_gpu_memory(
    monkeypatch: pytest.MonkeyPatch, fake_cuda: list[tuple[float, int | None]]
) -> None:
    monkeypatch.setenv(VRAM_ENV, "0.6")
    assert apply_vram_limit(None, "0") == 0.6
    assert fake_cuda == [(0.6, 0)]


def test_the_smaller_of_env_and_config_wins(
    monkeypatch: pytest.MonkeyPatch, fake_cuda: list[tuple[float, int | None]]
) -> None:
    monkeypatch.setenv(VRAM_ENV, "0.6")
    assert apply_vram_limit(0.4, torch.device("cuda", 1)) == 0.4
    assert apply_vram_limit(0.9, "cuda:0") == 0.6
    assert fake_cuda == [(0.4, 1), (0.6, 0)]


def test_no_cap_and_cpu_do_nothing(
    no_vram_env: None, fake_cuda: list[tuple[float, int | None]]
) -> None:
    assert apply_vram_limit(None, "0") is None
    assert apply_vram_limit(0.5, "cpu") == 0.5
    assert fake_cuda == []


def test_bad_env_value_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(VRAM_ENV, "75%")
    with pytest.raises(ValueError, match=VRAM_ENV):
        vram_fraction()
    monkeypatch.setenv(VRAM_ENV, "1.5")
    with pytest.raises(ValueError, match=VRAM_ENV):
        vram_fraction()
