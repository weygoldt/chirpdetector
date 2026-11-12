"""YOLO26 (ultralytics) as a chirp detector: model config and backend.

Model choice
------------
The default is ``yolo26s`` (10 M parameters, 23 GFLOPs at 640 x 640). The
task has one class and simple shapes; the inputs are 224 x 640 (205 bins
zero-padded to the stride), a third of a square 640 image. ``yolo26s``
trains at batch 16 with AMP within a few GB, so it fits beside the other
jobs on the shared 16 GB card; ``yolo26m`` (22 M) is the ablation if the
GPU is free. v1 used YOLOv8l (43 M) at 800 px, which this task does not
need.

Inputs
------
Images are never resampled: a spectrogram image (H x W, the dataset's
tiling) is zero-padded at the bottom and right to multiples of the model
stride (32) and fed at that size, in training, validation and inference
alike. One pixel is one frame by one frequency bin, so the time and
frequency scales the model learns are those it sees at inference. Zero is
the bottom of the normalised dB range ("no power"), so the padding looks
like silence, not like a grey band. Channel ``c`` of the model input is
window ``c`` of the front end (the long reference window first).

Head
----
YOLO26 trains a one-to-many and a one-to-one head; we validate and infer
with the one-to-one head (NMS-free, ``nms=False``), so near-simultaneous
chirps of different fish are never merged by non-maximum suppression
(audit N11).

Ignore regions and augmentation: see `chirpdetector.models._yolo_engine`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chirpdetector.bundle import Bundle
    from chirpdetector.models import Detector, FitOutput, FitRequest

__all__ = ["ARCHITECTURES", "MODEL_TYPE", "STRIDE", "YoloBackend", "YoloModelConfig"]

MODEL_TYPE = "ultralytics-yolo"
ARCHITECTURES = ("yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x")
STRIDE = 32
"""Largest stride of the YOLO26 detection heads; inputs are padded to it."""


@dataclass(frozen=True, slots=True)
class YoloModelConfig:
    """The ``[model]`` table of a training config for ``ultralytics-yolo``.

    Parameters
    ----------
    architecture
        One of `ARCHITECTURES`.
    pretrained
        Start from ultralytics' COCO weights of the architecture
        (downloaded once into ``~/.cache/chirpdetector/weights``); else
        from random weights.
    weights
        Path of a starting checkpoint instead (overrides ``pretrained``).
    max_det
        Most detections per image.
    """

    architecture: str = "yolo26s"
    pretrained: bool = True
    weights: str | None = None
    max_det: int = 300

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURES:
            msg = f"architecture must be one of {ARCHITECTURES}, got {self.architecture!r}"
            raise ValueError(msg)
        if not isinstance(self.pretrained, bool):
            msg = f"pretrained must be true or false, got {self.pretrained!r}"
            raise ValueError(msg)
        if self.weights is not None and not str(self.weights).endswith(".pt"):
            msg = f"weights must be a .pt checkpoint, got {self.weights!r}"
            raise ValueError(msg)
        if not isinstance(self.max_det, int) or self.max_det < 1:
            msg = f"max_det must be a positive integer, got {self.max_det!r}"
            raise ValueError(msg)


def input_shape(height_px: int, width_px: int) -> tuple[int, int]:
    """Model input size for images of ``height_px`` x ``width_px``: padded to `STRIDE`."""
    return (
        STRIDE * math.ceil(height_px / STRIDE),
        STRIDE * math.ceil(width_px / STRIDE),
    )


class YoloBackend:
    """``ultralytics-yolo``: YOLO26 trained with the ultralytics trainer."""

    model_type = MODEL_TYPE

    def resolve_model_config(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Validate the ``[model]`` table and fill in defaults."""
        known = {f.name for f in fields(YoloModelConfig)}
        unknown = sorted(set(params) - known)
        if unknown:
            msg = f"unknown [model] keys for {MODEL_TYPE}: {unknown}; known: {sorted(known)}"
            raise ValueError(msg)
        return asdict(YoloModelConfig(**params))

    def fit(self, request: FitRequest) -> FitOutput:
        """Train one fresh YOLO26 model (see `_yolo_engine.fit`)."""
        from chirpdetector.models import _yolo_engine

        return _yolo_engine.fit(request)

    def load(self, bundle: Bundle, *, device: str | None = None) -> Detector:
        """Load a bundle's YOLO weights for inference."""
        from chirpdetector.models import _yolo_engine

        return _yolo_engine.YoloDetector(bundle, device=device)
