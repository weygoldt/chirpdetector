"""Synthetic detector datasets and helpers to break them."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chirpdetector.data.contract import BOXES_FILE, MANIFEST_FILE, sha256_file
from chirpdetector.data.synthetic import make_synthetic_dataset


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A valid synthetic dataset (6 groups: 4 train, 1 val, 1 test)."""
    directory = tmp_path_factory.mktemp("datasets") / "synthetic"
    make_synthetic_dataset(directory, seed=3)
    return directory


@pytest.fixture
def broken(synthetic_dir: Path, tmp_path: Path) -> Callable[..., Path]:
    """Copy the synthetic dataset and edit its manifest and/or boxes.

    ``broken(manifest=f, boxes=g)``: ``f(dict)`` edits the parsed
    ``dataset.json`` in place, ``g(DataFrame) -> DataFrame`` the box
    table (its sha256 in the manifest is updated). The stored
    ``content_sha256`` is dropped, so that only the edit is wrong.
    """

    def make(
        manifest: Callable[[dict[str, Any]], None] | None = None,
        boxes: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> Path:
        directory = tmp_path / f"broken-{len(list(tmp_path.iterdir()))}"
        shutil.copytree(synthetic_dir, directory)
        data = json.loads((directory / MANIFEST_FILE).read_text())
        data.pop("content_sha256")
        if boxes is not None:
            table = pd.read_parquet(directory / BOXES_FILE)
            boxes(table).to_parquet(directory / BOXES_FILE, index=False)
            data["boxes_sha256"] = sha256_file(directory / BOXES_FILE)
        if manifest is not None:
            manifest(data)
        (directory / MANIFEST_FILE).write_text(json.dumps(data))
        return directory

    return make


# ------------------------------------------------------- simulated scenes

FS = 20_000.0
_N_TEMPLATE = 4000  # 0.2 s, centre at 2000
_TAPER = 200


def _gaussian_template(
    excursion_hz: float, fwhm_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian chirp centred on sample N/2, tapered (as gridtools' tests)."""
    t = (np.arange(_N_TEMPLATE) - _N_TEMPLATE // 2) / FS
    sigma = fwhm_s / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    delta = excursion_hz * np.exp(-0.5 * (t / sigma) ** 2)
    weight = np.zeros(_N_TEMPLATE)
    idx = np.flatnonzero((t > -6 * sigma - 0.005) & (t < 6 * sigma + 0.005))
    weight[idx] = 1.0
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(_TAPER) / _TAPER))
    weight[idx[0] - _TAPER : idx[0]] = ramp
    weight[idx[-1] + 1 : idx[-1] + 1 + _TAPER] = ramp[::-1]
    delta *= weight
    amp = 1.0 - 0.3 * np.clip(delta, 0, None) / excursion_hz
    return delta.astype(np.float32), amp.astype(np.float32)


def write_template_library(path: Path) -> Path:
    """A small template library of Gaussian chirps (60-400 Hz, 6-25 ms)."""
    path.mkdir(parents=True, exist_ok=True)
    deltas, amps, rows = [], [], []
    for i, (exc, fwhm) in enumerate(
        (e, w) for e in (60.0, 120.0, 250.0, 400.0) for w in (0.008, 0.015, 0.025)
    ):
        delta, amp = _gaussian_template(exc, fwhm)
        deltas.append(delta)
        amps.append(amp)
        rows.append(
            {
                "template_id": i,
                "excursion_hz": float(delta.max()),
                "duration_50_s": fwhm,
                "duration_10_s": fwhm * np.sqrt(np.log(10) / np.log(2)),
                "family": "big" if exc >= 200 else "small",
                "valid": True,
                "has_nan": False,
                "duplicate_of": -1,
                "truncated": False,
                "step_flag": False,
                "too_short": False,
                "multi_peak": False,
                "outlier": False,
                "eodf_atypical": False,
                "ripple_flag": False,
                "amp_ripple_flag": False,
                "amp_usable": True,
            }
        )
    np.save(path / "delta_f_hz.npy", np.stack(deltas))
    np.save(path / "amp_rel.npy", np.stack(amps))
    pd.DataFrame(rows).to_parquet(path / "templates.parquet", index=False)
    manifest = {
        "library_format_version": 1,
        "samplerate_hz": FS,
        "n_templates": len(rows),
        "n_samples": _N_TEMPLATE,
        "centre_index": _N_TEMPLATE // 2,
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def scene_config(library: Path, *, layout: tuple[int, int], **changes: Any) -> Any:
    """An 8-s scene on a real (regular) layout, synthetic noise only (no real
    background segment); ``changes`` replace sections."""
    from gridtools.sim.config import (
        ChirpConfig,
        ClickConfig,
        GridConfig,
        HybridConfig,
        NoiseConfig,
        PlotConfig,
        PopulationConfig,
        SceneConfig,
    )

    config = SceneConfig(
        name="test",
        duration_s=8.0,
        background=HybridConfig(prob=0.0),
        grid=GridConfig(layouts=(layout,), real_layout_prob=1.0),
        population=PopulationConfig(n_fish=(3, 3)),
        chirps=ChirpConfig(library=str(library), rate_hz=(0.8, 1.0)),
        clicks=ClickConfig(rate_per_h_median=3600.0, rate_log10_sd=0.0),
        noise=NoiseConfig(clip_limit=None),
        plot=PlotConfig(enabled=False),
    )
    return replace(config, **changes) if changes else config


def clean_single_fish_config(library: Path) -> Any:
    """One fish at 800 Hz, no noise, no clicks: chirps on a black image."""
    from gridtools.sim.config import (
        ChirpConfig,
        ClickConfig,
        NoiseConfig,
        PopulationConfig,
    )

    return scene_config(
        library,
        layout=(3, 3),
        population=PopulationConfig(
            n_fish=(1, 1), baseline_uniform_hz=(800.0, 800.0), baseline_uniform_prob=1.0
        ),
        noise=NoiseConfig(
            colored=False, hum=False, clip_limit=None, dead_prob=0.0, noisy_prob=0.0
        ),
        clicks=ClickConfig(rate_per_h_median=0.0),
        chirps=ChirpConfig(library=str(library), rate_hz=(1.0, 1.0)),
    )


@pytest.fixture(scope="session")
def template_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_template_library(tmp_path_factory.mktemp("library"))


@pytest.fixture(scope="session")
def scene_dirs(
    template_library: Path, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Path]:
    """Three written scenes: 9 electrodes (seed 3), 4 electrodes (seed 4), and
    one clean single fish on 9 electrodes (seed 5)."""
    from gridtools.sim.scene import simulate_scene, write_scene

    root = tmp_path_factory.mktemp("scenes")
    configs = {
        "grid9": (scene_config(template_library, layout=(3, 3)), 3),
        "grid4": (scene_config(template_library, layout=(2, 2)), 4),
        "clean": (clean_single_fish_config(template_library), 5),
    }
    out = {}
    for key, (config, seed) in configs.items():
        scene = simulate_scene(config, seed)
        out[key] = write_scene(scene, root / f"{seed:06d}", plots=False)
    return out
