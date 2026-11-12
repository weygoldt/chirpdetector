"""Detector implementations behind one small interface.

A detector implementation (a *backend*) is selected by a ``model_type``
string, stored in the training config (``[model] type``) and in every model
bundle. It has to do three things:

* resolve its part of the training config (``[model]`` table) to a full,
  validated dict (`DetectorBackend.resolve_model_config`);
* train one model on given images of a detector dataset
  (`DetectorBackend.fit`), returning a self-contained weights file;
* load a bundle's weights for inference (`DetectorBackend.load`), giving
  a `Detector` that maps images to boxes in *pixel* coordinates.

Everything else is shared: the dataset contract
(`chirpdetector.data.contract`), grouped cross-validation, run
directories and provenance (`chirpdetector.training`), bundles
(`chirpdetector.bundle`), the pixel -> time/frequency mapping
(`chirpdetector.prediction`) and evaluation. A model with extra heads (e.g.
the emitter's EOD frequency) reads them from the dataset's box table and
reports them as extra pixel columns (``emitter_y``: the row of the
emitter's baseline), which `chirpdetector.prediction` maps to
``emitter_hz``.

Backends register a factory with `register_backend`; ``ultralytics-yolo``
(YOLO26, `chirpdetector.models.yolo`) is built in.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from chirpdetector.bundle import Bundle
    from chirpdetector.data.contract import Dataset
    from chirpdetector.training.config import DetectorTrainingConfig

__all__ = [
    "PIXEL_COLUMNS",
    "Detector",
    "DetectorBackend",
    "FitOutput",
    "FitRequest",
    "backend_types",
    "get_backend",
    "register_backend",
]

PIXEL_COLUMNS = ("x_min", "y_min", "x_max", "y_max", "score")
"""Columns of a `Detector`'s output per image; ``emitter_y`` is optional."""


@runtime_checkable
class Detector(Protocol):
    """A loaded model that detects chirps in spectrogram images."""

    def predict(
        self, images: Sequence[NDArray[np.uint8]], *, score_threshold: float
    ) -> list[pd.DataFrame]:
        """Detect chirps.

        Parameters
        ----------
        images
            ``(H, W, 3)`` uint8 images in window order (as `to_image` and
            `chirpdetector.data.read_image` give), all of the bundle's
            image size.
        score_threshold
            Keep detections with ``score >= score_threshold``.

        Returns
        -------
        list of DataFrame
            One per image, columns `PIXEL_COLUMNS` (continuous pixel
            coordinates, inside ``[0, W] x [0, H]``) and optionally
            ``emitter_y``.
        """
        ...


@dataclass(frozen=True)
class FitRequest:
    """One model to train.

    Attributes
    ----------
    dataset
        The validated dataset.
    train_ids, val_ids
        Images to fit on and to validate (early stopping, model selection)
        on; disjoint by group.
    config
        The full training config; ``config.model`` is this backend's
        resolved model config.
    out_dir
        Directory for everything this fit writes.
    name
        Name of the fit (``fold-0``, ..., ``final``).
    """

    dataset: Dataset
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    config: DetectorTrainingConfig
    out_dir: Path
    name: str


@dataclass(frozen=True)
class FitOutput:
    """What a fit produced.

    Attributes
    ----------
    weights
        Self-contained weights file (copied into the bundle).
    model
        Description stored as ``bundle.json["model"]`` (architecture,
        input shape, head, ...); the backend's `load` reads it back.
    curves
        Per-epoch CSV (losses, validation metrics, learning rates), if any.
    best_epoch
        Epoch (1-based) of the selected weights, if known.
    epochs_run
        Epochs trained.
    metrics
        The backend's validation metrics of the selected weights.
    details
        Anything else worth recording (batch size actually used, device,
        starting weights and their sha256, ...).
    """

    weights: Path
    model: Mapping[str, Any]
    curves: Path | None
    best_epoch: int | None
    epochs_run: int
    metrics: Mapping[str, Any]
    details: Mapping[str, Any] = field(default_factory=dict)


class DetectorBackend(Protocol):
    """A detector implementation (see the module docstring)."""

    model_type: str

    def resolve_model_config(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Validate the ``[model]`` table (without ``type``); fill defaults."""
        ...

    def fit(self, request: FitRequest) -> FitOutput:
        """Train one fresh model."""
        ...

    def load(self, bundle: Bundle, *, device: str | None = None) -> Detector:
        """Load a bundle's (already verified) weights for inference."""
        ...


_FACTORIES: dict[str, Callable[[], DetectorBackend]] = {}


def register_backend(
    model_type: str, factory: Callable[[], DetectorBackend], *, replace: bool = False
) -> None:
    """Make a backend available under ``model_type``.

    Parameters
    ----------
    model_type
        Key used in training configs and bundles.
    factory
        Callable returning the backend; called on first use, so heavy
        imports stay lazy.
    replace
        Allow replacing an existing registration (e.g. in tests).
    """
    if model_type in _FACTORIES and not replace:
        msg = f"a backend is already registered as {model_type!r}"
        raise ValueError(msg)
    _FACTORIES[model_type] = factory


def backend_types() -> tuple[str, ...]:
    """Registered model types."""
    return tuple(sorted(_FACTORIES))


def get_backend(model_type: str) -> DetectorBackend:
    """The backend registered as ``model_type``.

    Raises
    ------
    KeyError
        If no backend has that type.
    """
    if model_type not in _FACTORIES:
        msg = f"unknown model type {model_type!r}; known: {list(backend_types())}"
        raise KeyError(msg)
    return _FACTORIES[model_type]()


def _yolo() -> DetectorBackend:
    from chirpdetector.models.yolo import YoloBackend

    return YoloBackend()


register_backend("ultralytics-yolo", _yolo)
