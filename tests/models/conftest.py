"""Fixtures for the dual-head detector: a synthetic dataset and tiny bundles."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("transformers")

from chirpdetector.bundle import Bundle, load_bundle, write_bundle
from chirpdetector.data import Dataset, Tiling, load_dataset
from chirpdetector.data.synthetic import make_synthetic_dataset
from chirpdetector.models._dualhead_engine import build_model, export_weights
from chirpdetector.models.dualhead import DualHeadModelConfig, input_shape
from chirpdetector.spectrogram import SpectrogramConfig

TINY: dict[str, Any] = {
    "architecture": "tiny",
    "pretrained": False,
    "num_queries": 20,
    "num_denoising": 10,
    "max_det": 20,
    "ema_decay": 0.0,
}
"""``[model]`` table of a tiny, randomly initialised dual-head model."""


@pytest.fixture(scope="session")
def dualhead_dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic dataset: 8 groups (5 train, 2 val, 1 test), 3 images each."""
    directory = tmp_path_factory.mktemp("dualhead") / "synthetic"
    make_synthetic_dataset(
        directory,
        splits=("train", "train", "train", "train", "train", "val", "val", "test"),
        images_per_group=3,
        seed=5,
    )
    return directory


@pytest.fixture(scope="session")
def dualhead_dataset(dualhead_dataset_dir: Path) -> Dataset:
    """The loaded synthetic dataset."""
    return load_dataset(dualhead_dataset_dir)


@pytest.fixture
def make_tiny_bundle(tmp_path: Path) -> Callable[[SpectrogramConfig, Tiling], Bundle]:
    """Bundle of an untrained tiny model for a front end and tiling."""

    def make(spectrogram: SpectrogramConfig, tiling: Tiling) -> Bundle:
        shape = input_shape(tiling.height_px, tiling.width_px)
        model, _ = build_model(
            DualHeadModelConfig(**TINY), emitter_unit=tiling.height_px / shape[0]
        )
        weights = tmp_path / "tiny.pt"
        export_weights(model, weights)
        directory = tmp_path / "bundle"
        write_bundle(
            directory,
            weights=weights,
            model_type="dualhead",
            model={"architecture": "tiny", "input_shape": list(shape), "max_det": 20},
            spectrogram=spectrogram,
            tiling=tiling,
            preprocessing={"click_removal": None},
            score_threshold=0.5,
            training={},
        )
        return load_bundle(directory)

    return make
