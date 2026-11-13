"""Fixtures of the pipeline tests: a content-dependent stub detector and bundles."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chirpdetector.bundle import Bundle, load_bundle, write_bundle
from chirpdetector.data import Tiling
from chirpdetector.models import register_backend
from chirpdetector.pipeline.assign import SIM_V4_DUALHEAD
from chirpdetector.pipeline.run import Models
from chirpdetector.spectrogram import SpectrogramConfig

CONTENT_STUB = "pipeline-content-stub"
CONFIG = SpectrogramConfig()
TILING = Tiling(
    tile_duration_s=0.64, overlap_s=0.2, width_px=128, height_px=CONFIG.n_freqs
)


@dataclass
class ContentStub:
    """Boxes that depend on the image: the brightest column and row.

    Two boxes per image: one around the column with the largest mean of the
    short-window channel, one 20 px to its right with a lower score; the
    emitter row is the brightest row of the reference channel. Equal
    images give equal boxes, different ones (almost always) different
    boxes, so chunked and one-pass inference can be compared.
    """

    emitter: bool = True

    def predict(
        self, images: Sequence[np.ndarray], *, score_threshold: float
    ) -> list[pd.DataFrame]:
        out = []
        for image in images:
            height, width = image.shape[:2]
            cols = image[:, :, 2].astype(np.float64).mean(axis=0)
            rows = image[:, :, 0].astype(np.float64).mean(axis=1)
            x = float(np.argmax(cols))
            y = float(np.argmax(rows))
            base = float(np.clip(cols.max() / 255.0, 0.06, 0.99))
            boxes = []
            for dx, factor in ((0.0, 1.0), (20.0, 0.5)):
                cx = min(width - 5.0, x + dx)
                boxes.append(
                    {
                        "x_min": max(0.0, cx - 4),
                        "y_min": max(0.0, y - 10),
                        "x_max": min(float(width), cx + 4),
                        "y_max": min(float(height), y + 30),
                        "score": base * factor,
                        "emitter_y": y,
                    }
                )
            table = pd.DataFrame(boxes)
            if not self.emitter:
                table = table.drop(columns="emitter_y")
            out.append(
                table.loc[table["score"] >= score_threshold].reset_index(drop=True)
            )
        return out


@dataclass
class ContentBackend:
    model_type: str = CONTENT_STUB
    emitter: bool = True

    def resolve_model_config(self, params: Mapping[str, Any]) -> dict[str, Any]:
        return dict(params)

    def fit(self, request: Any) -> Any:  # pragma: no cover - never trained
        raise NotImplementedError

    def load(self, bundle: Bundle, *, device: str | None = None) -> ContentStub:
        return ContentStub(emitter=bool(bundle.model.get("emitter", True)))


register_backend(CONTENT_STUB, ContentBackend, replace=True)


def _bundle(directory: Path, *, emitter: bool) -> Bundle:
    weights = directory / "w.bin"
    directory.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"content stub" + bytes([emitter]))
    write_bundle(
        directory / "bundle",
        weights=weights,
        model_type=CONTENT_STUB,
        model={
            "architecture": "rtdetr_stub" if emitter else "yolo_stub",
            "emitter": emitter,
        },
        spectrogram=CONFIG,
        tiling=TILING,
        preprocessing={"click_removal": None},
        score_threshold=0.5,
        training={"run_id": "stub"},
    )
    return load_bundle(directory / "bundle")


@pytest.fixture(scope="session")
def stub_bundle(tmp_path_factory: pytest.TempPathFactory) -> Bundle:
    """A dual-head-like stub bundle (boxes + emitter row), small tiles."""
    return _bundle(tmp_path_factory.mktemp("pipeline_bundle"), emitter=True)


@pytest.fixture(scope="session")
def box_only_bundle(tmp_path_factory: pytest.TempPathFactory) -> Bundle:
    """A box-only (YOLO-like) stub bundle."""
    return _bundle(tmp_path_factory.mktemp("pipeline_box_bundle"), emitter=False)


def make_models(bundle: Bundle) -> Models:
    """`Models` for a stub bundle (as `load_models` would build them)."""
    emitter = bool(bundle.model.get("emitter", True))
    from chirpdetector.pipeline.outputs import model_tag
    from chirpdetector.prediction import load_detector

    return Models(
        bundle=bundle,
        detector=load_detector(bundle),
        estimator=None,
        tag=model_tag(bundle),
        has_emitter=emitter,
        error_model=SIM_V4_DUALHEAD if emitter else None,
        info={
            "detector": {
                "path": str(bundle.path),
                "architecture": bundle.model.get("architecture"),
                "weights_sha256": bundle.weights_sha256,
                "weights_short": bundle.weights_sha256[:8],
            },
            "decoder": None,
            "tag": model_tag(bundle),
        },
    )


@pytest.fixture
def trial_dirs(
    tmp_path: Path, make_synthetic_trial: Callable[[Path], Path]
) -> list[Path]:
    """Two synthetic recording directories (8 s, 4 electrodes, tracks)."""
    return [
        make_synthetic_trial(tmp_path / "data" / name)
        for name in ("2000-01-01-10_00", "2000-01-02-10_00")
    ]
