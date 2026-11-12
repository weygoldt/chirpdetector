"""Fixtures of the chirp-parameter tests: tiny scenes, a stub detector, bundles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chirpdetector.bundle import Bundle, load_bundle, write_bundle
from chirpdetector.data.tiles import TileLayout
from chirpdetector.models import PIXEL_COLUMNS, register_backend
from chirpdetector.spectrogram import SpectrogramConfig

from ..data.conftest import (
    clean_single_fish_config,
    scene_config,
    write_template_library,
)

SPLIT_SEEDS = {"train": (0, 1, 2), "val": (1600,), "test": (1800, 1801)}
"""Seeds of the scenes per split (the detector datasets' seed ranges)."""

STUB_TYPE = "params-stub"


@dataclass
class _StubDetector:
    """One fixed box per tile (pixels), whatever the image shows."""

    def predict(
        self, images: Sequence[np.ndarray], *, score_threshold: float
    ) -> list[pd.DataFrame]:
        table = pd.DataFrame(
            [[300.0, 60.0, 330.0, 110.0, 0.9]], columns=list(PIXEL_COLUMNS)
        )
        return [table.loc[table["score"] >= score_threshold] for _ in images]


@dataclass
class _StubBackend:
    model_type: str = STUB_TYPE

    def resolve_model_config(self, params: Mapping[str, Any]) -> dict[str, Any]:
        return dict(params)

    def fit(self, request: Any) -> Any:  # pragma: no cover - never trained
        raise NotImplementedError

    def load(self, bundle: Bundle, *, device: str | None = None) -> _StubDetector:
        return _StubDetector()


register_backend(STUB_TYPE, _StubBackend, replace=True)


@pytest.fixture(scope="session")
def params_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Gaussian chirp templates, 60-400 Hz, 8-25 ms FWHM."""
    return write_template_library(tmp_path_factory.mktemp("params_library"))


@pytest.fixture(scope="session")
def params_scenes(
    params_library: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Six 8-s scenes on 3x3 grids in a ``train``/``val``/``test`` layout."""
    from gridtools.sim.config import ChirpConfig, PopulationConfig
    from gridtools.sim.scene import simulate_scene, write_scene

    root = tmp_path_factory.mktemp("params_scenes")
    config = scene_config(
        params_library,
        layout=(3, 3),
        population=PopulationConfig(n_fish=(2, 2)),
        chirps=ChirpConfig(library=str(params_library), rate_hz=(1.2, 1.4)),
    )
    for split, seeds in SPLIT_SEEDS.items():
        for seed in seeds:
            scene = simulate_scene(replace(config, name=f"params_{split}"), seed)
            write_scene(scene, root / split / f"{seed:06d}", plots=False)
    return root


@pytest.fixture(scope="session")
def clean_scene(params_library: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One fish at 800 Hz, no noise: its chirps are the templates."""
    from gridtools.sim.scene import simulate_scene, write_scene

    scene = simulate_scene(clean_single_fish_config(params_library), 5)
    return write_scene(scene, tmp_path_factory.mktemp("clean") / "000005", plots=False)


@pytest.fixture(scope="session")
def stub_detector(tmp_path_factory: pytest.TempPathFactory) -> Bundle:
    """A detector bundle (default front end, 3.2-s tiles) of the stub backend."""
    directory = tmp_path_factory.mktemp("stub_detector")
    weights = directory / "w.bin"
    weights.write_bytes(b"stub detector")
    config = SpectrogramConfig()
    write_bundle(
        directory / "bundle",
        weights=weights,
        model_type=STUB_TYPE,
        model={},
        spectrogram=config,
        tiling=TileLayout().tiling(config),
        preprocessing={"click_removal": None},
        score_threshold=0.5,
        training={"run_id": "stub"},
    )
    return load_bundle(directory / "bundle")


def footprint_detections(
    scene_dirs: Mapping[str, Path],
) -> Any:
    """A stand-in for `detect_in_spectrogram`: the scene's footprint boxes.

    Visible chirps' footprint boxes (score 0.9) plus one box that matches no
    chirp (score 0.6), so that detector-mode extraction can be tested
    without a trained detector.
    """

    def detect(bundle: Bundle, spec: Any, *, name: str, **_: Any) -> pd.DataFrame:
        chirps = pd.read_csv(scene_dirs[name] / "chirps.csv")
        chirps = chirps.loc[chirps["visible"].to_numpy(bool)]
        boxes = chirps[["box_t0_s", "box_t1_s", "box_f0_hz", "box_f1_hz"]].dropna()
        table = pd.DataFrame(
            boxes.to_numpy(),
            columns=["t_start_s", "t_end_s", "f_low_hz", "f_high_hz"],
        ).assign(score=0.9)
        bogus = pd.DataFrame(
            [[4.0, 4.1, 1300.0, 1350.0, 0.6]],
            columns=["t_start_s", "t_end_s", "f_low_hz", "f_high_hz", "score"],
        )
        return pd.concat([table, bogus], ignore_index=True)

    return detect


TINY_CONFIG = {
    "device": "cpu",
    "workers": 0,
    "crop": {"size": [32, 32]},
    "model": {"stem": 4, "widths": [4, 4, 4], "bottleneck": 4, "hidden": 8},
    "optim": {"epochs": 2, "batch": 16, "patience": 0, "amp": False},
    "calibration": {"n_bins": 4, "min_count": 5},
}
"""A training config small enough for a CPU test."""
