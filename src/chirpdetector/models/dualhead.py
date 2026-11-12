"""Dual-head detector: chirp boxes and the emitter's EOD frequency in one pass.

Model type ``dualhead`` (plan, decision 19). An RT-DETRv2 detector
(`transformers`, COCO-pretrained) with one more head on its decoder
queries: besides the class score and the box, every query predicts the
baseline EOD frequency of the fish that emitted the chirp. Assignment then
becomes "the live track nearest to the predicted emitter frequency"
(`chirpdetector.evaluation.assign_by_frequency`).

Why RT-DETRv2
-------------
* *Set prediction, NMS-free.* Hungarian matching trains one query per chirp,
  so near-simultaneous chirps of different fish are never merged by
  non-maximum suppression (audit N11), and each query's emitter belongs to
  exactly one chirp.
* *Extensible without forking.* The loss (`RTDetrLoss`: matcher, varifocal
  and box losses on the last layer, every auxiliary layer, the encoder's
  top-k proposals and the denoising groups) is reused as is; the emitter
  head adds one loss term and the ignore regions mask one (see
  `chirpdetector.models._dualhead_engine`).
* *Fits the shared GPU.* ResNet18-vd RT-DETRv2 (20 M parameters) trains at
  batch 16 on 224 x 640 inputs in about 3 GiB with bf16 autocast.
* *Inputs at native size.* Anchors and position embeddings are built from
  the feature-map sizes, so spectrograms are padded to the stride (205 x
  640 -> 224 x 640) and never resampled, as for YOLO26.

Emitter target
--------------
The emitter is regressed relative to the box, so that the target does not
depend on where in the frequency band the fish sits:

    e = (emitter_hz - f_low_hz) / image frequency span
      = (y_max - y_emitter) / image height        (pixel rows)

``e`` is positive when the emitter's baseline lies above the box's lower
edge (the usual case: footprints reach below the baseline by the window's
frequency smear and the chirp's undershoot) and negative below it. In the
model, ``e`` is measured from the *predicted* box's lower edge and
supervised so that the decoded emitter frequency matches the label
(`encode_emitter` / `decode_emitter` are the exact conversions).

Augmentation
------------
Optional random time shifts within the tile (``time_shift_px``): the image
content moves by up to that many columns, the vacated columns are zero (no
power, as the padding), and chirps that lose more than half their width
become ignore regions. No vertical flips, colour jitter or rescaling (see
`chirpdetector.models._yolo_engine` for why).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from chirpdetector.bundle import Bundle
    from chirpdetector.models import Detector, FitOutput, FitRequest

__all__ = [
    "ARCHITECTURES",
    "MODEL_TYPE",
    "STRIDE",
    "DualHeadBackend",
    "DualHeadModelConfig",
    "decode_emitter",
    "encode_emitter",
    "input_shape",
]

MODEL_TYPE = "dualhead"
STRIDE = 32
"""Largest feature stride of RT-DETRv2; inputs are padded to it."""

ARCHITECTURES: dict[str, str | None] = {
    "rtdetr_v2_r18vd": "PekingU/rtdetr_v2_r18vd",
    "rtdetr_v2_r34vd": "PekingU/rtdetr_v2_r34vd",
    "rtdetr_v2_r50vd": "PekingU/rtdetr_v2_r50vd",
    "tiny": None,
}
"""Architecture -> Hugging Face repository of its COCO weights.

``tiny`` is a randomly initialised miniature for tests; it has no
pretrained weights.
"""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class DualHeadModelConfig:
    """The ``[model]`` table of a training config for ``dualhead``.

    Parameters
    ----------
    architecture
        One of `ARCHITECTURES`.
    pretrained
        Start from the architecture's COCO weights (Hugging Face hub,
        cached in ``HF_HOME``); else from random weights. ``tiny`` has none.
    weights
        Path of a ``dualhead`` weights file (a bundle's ``weights.pt`` or a
        fit's ``dualhead/best.pt``) to continue from instead (overrides
        ``pretrained``); its sha256 is recorded with the fit. It must be of
        ``architecture`` and made for the same image height.
    num_queries
        Decoder queries per image (upper bound on chirps per tile).
    num_denoising
        Denoising queries per image in training (0 disables them).
    max_det
        Most detections per image at inference (<= ``num_queries``).
    emitter_loss_weight
        Weight of the L1 loss on the emitter offset (in units of the image's
        frequency span; 0.01 = 10 Hz at 1000 Hz span).
    backbone_lr_scale
        Learning rate of the backbone relative to ``optim.lr0``.
    grad_clip
        Largest gradient norm (0 disables clipping).
    ema_decay
        Decay of the exponential moving average of the weights that is
        validated and shipped (0 disables it).
    time_shift_px
        Largest random time shift of a training image, in columns (0: none).
    val_max_images
        Validate on at most this many validation images (a fixed, evenly
        spaced subset; 0: all).
    """

    architecture: str = "rtdetr_v2_r18vd"
    pretrained: bool = True
    weights: str | None = None
    num_queries: int = 300
    num_denoising: int = 100
    max_det: int = 300
    emitter_loss_weight: float = 10.0
    backbone_lr_scale: float = 0.1
    grad_clip: float = 0.1
    ema_decay: float = 0.9999
    time_shift_px: int = 0
    val_max_images: int = 0

    def __post_init__(self) -> None:
        _check(
            self.architecture in ARCHITECTURES,
            f"architecture must be one of {sorted(ARCHITECTURES)}, got "
            f"{self.architecture!r}",
        )
        _check(isinstance(self.pretrained, bool), "pretrained must be true or false")
        _check(
            not (
                self.pretrained
                and self.weights is None
                and ARCHITECTURES[self.architecture] is None
            ),
            f"{self.architecture!r} has no pretrained weights; set pretrained = false",
        )
        _check(
            self.weights is None
            or (isinstance(self.weights, str) and self.weights.endswith(".pt")),
            f"weights must be the path of a .pt file, got {self.weights!r}",
        )
        for name in ("num_queries", "max_det"):
            value = getattr(self, name)
            _check(_integer(value) and value >= 1, f"{name} must be a positive integer")
        _check(self.max_det <= self.num_queries, "max_det must not exceed num_queries")
        for name in ("num_denoising", "time_shift_px", "val_max_images"):
            value = getattr(self, name)
            _check(_integer(value) and value >= 0, f"{name} must be an integer >= 0")
        for name in ("emitter_loss_weight", "backbone_lr_scale", "grad_clip"):
            value = getattr(self, name)
            _check(_number(value) and value >= 0, f"{name} must be a number >= 0")
        _check(
            _number(self.ema_decay) and 0 <= self.ema_decay < 1,
            "ema_decay must lie in [0, 1)",
        )

    @property
    def repository(self) -> str | None:
        """Hugging Face repository of the starting weights (None: random or file)."""
        if self.weights is not None or not self.pretrained:
            return None
        return ARCHITECTURES[self.architecture]


def input_shape(height_px: int, width_px: int) -> tuple[int, int]:
    """Model input size for images of ``height_px`` x ``width_px``: padded to `STRIDE`."""
    return (
        STRIDE * math.ceil(height_px / STRIDE),
        STRIDE * math.ceil(width_px / STRIDE),
    )


def encode_emitter(
    emitter_y: ArrayLike, y_max: ArrayLike, height_px: float
) -> NDArray[np.float64]:
    """Emitter offset relative to a box: ``(y_max - emitter_y) / height_px``.

    Parameters
    ----------
    emitter_y
        Row of the emitter's baseline (continuous pixel coordinates, row 0
        at the top = highest frequency).
    y_max
        Lower edge of the box (its ``f_low_hz``) in the same coordinates.
    height_px
        Image height. Rows are linear in frequency, so the offset equals
        ``(emitter_hz - f_low_hz) / image frequency span``.

    Returns
    -------
    ndarray
        The offset; > 0 when the emitter lies above the lower edge.
    """
    if not height_px > 0:
        msg = f"height_px must be positive, got {height_px}"
        raise ValueError(msg)
    return (np.asarray(y_max, np.float64) - np.asarray(emitter_y, np.float64)) / float(
        height_px
    )


def decode_emitter(
    offset: ArrayLike, y_max: ArrayLike, height_px: float
) -> NDArray[np.float64]:
    """Row of the emitter from its offset (inverse of `encode_emitter`)."""
    if not height_px > 0:
        msg = f"height_px must be positive, got {height_px}"
        raise ValueError(msg)
    return np.asarray(y_max, np.float64) - np.asarray(offset, np.float64) * float(
        height_px
    )


class DualHeadBackend:
    """``dualhead``: RT-DETRv2 with an emitter-frequency head."""

    model_type = MODEL_TYPE

    def resolve_model_config(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Validate the ``[model]`` table and fill in defaults."""
        known = {f.name for f in fields(DualHeadModelConfig)}
        unknown = sorted(set(params) - known)
        if unknown:
            msg = f"unknown [model] keys for {MODEL_TYPE}: {unknown}; known: {sorted(known)}"
            raise ValueError(msg)
        return asdict(DualHeadModelConfig(**params))

    def fit(self, request: FitRequest) -> FitOutput:
        """Train one fresh model (see `_dualhead_engine.fit`)."""
        from chirpdetector.models import _dualhead_engine

        return _dualhead_engine.fit(request)

    def load(self, bundle: Bundle, *, device: str | None = None) -> Detector:
        """Load a bundle's weights for inference."""
        from chirpdetector.models import _dualhead_engine

        return _dualhead_engine.DualHeadDetector(bundle, device=device)
