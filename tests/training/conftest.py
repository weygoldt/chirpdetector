"""Fixtures for training: a synthetic dataset, a stub detector backend."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from chirpdetector.bundle import Bundle
from chirpdetector.data import Dataset, load_dataset
from chirpdetector.data.synthetic import make_synthetic_dataset
from chirpdetector.models import (
    PIXEL_COLUMNS,
    Detector,
    FitOutput,
    FitRequest,
    register_backend,
)

STUB = "stub"


@dataclass
class StubDetector:
    """Returns the same pixel table for every image."""

    table: pd.DataFrame

    def predict(
        self, images: Sequence[np.ndarray], *, score_threshold: float
    ) -> list[pd.DataFrame]:
        keep = self.table["score"].to_numpy() >= score_threshold
        return [self.table.loc[keep].reset_index(drop=True) for _ in images]


@dataclass
class StubBackend:
    """A backend that records its fits and writes distinct fake weights."""

    model_type: str = STUB
    fits: list[FitRequest] = field(default_factory=list)
    output: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=list(PIXEL_COLUMNS))
    )

    def resolve_model_config(self, params: Mapping[str, Any]) -> dict[str, Any]:
        unknown = set(params) - {"width"}
        if unknown:
            msg = f"unknown stub keys {sorted(unknown)}"
            raise ValueError(msg)
        return {"width": int(params.get("width", 1))}

    def fit(self, request: FitRequest) -> FitOutput:
        self.fits.append(request)
        weights = request.out_dir / "weights.bin"
        weights.write_text(
            json.dumps({"fit": request.name, "train": list(request.train_ids)})
        )
        curves = request.out_dir / "curves.csv"
        curves.write_text("epoch,metrics/mAP50(B)\n1,0.5\n")
        return FitOutput(
            weights=weights,
            model={"kind": "stub", "fit": request.name},
            curves=curves,
            best_epoch=1,
            epochs_run=1,
            metrics={"metrics/mAP50(B)": 0.5 + 0.01 * len(self.fits)},
            details={"n_train_images": len(request.train_ids)},
        )

    def load(self, bundle: Bundle, *, device: str | None = None) -> Detector:
        return StubDetector(self.output)


@pytest.fixture
def stub_backend() -> StubBackend:
    """A fresh stub backend registered as ``stub``."""
    backend = StubBackend()
    register_backend(STUB, lambda: backend, replace=True)
    return backend


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic dataset: 8 groups (5 train, 2 val, 1 test), 3 images each."""
    directory = tmp_path_factory.mktemp("training") / "synthetic"
    make_synthetic_dataset(
        directory,
        splits=("train", "train", "train", "train", "train", "val", "val", "test"),
        images_per_group=3,
        seed=11,
    )
    return directory


@pytest.fixture(scope="session")
def dataset(dataset_dir: Path) -> Dataset:
    """The loaded synthetic dataset."""
    return load_dataset(dataset_dir)


@pytest.fixture
def no_vram_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No GPU memory cap from the environment."""
    monkeypatch.delenv("CHIRP_MAX_VRAM_FRACTION", raising=False)


def cuda_available() -> bool:
    """Whether a CUDA device exists (for ``gpu`` tests)."""
    return torch.cuda.is_available()
