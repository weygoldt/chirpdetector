"""Typed, TOML-loadable configuration of detector training.

Example (every key optional; these are the defaults)::

    seed = 0
    deterministic = true
    device = "auto"            # "cpu", "cuda", "cuda:1", "0"
    max_vram_fraction = 0.6    # optional; CHIRP_MAX_VRAM_FRACTION also applies
    score_threshold = 0.5      # default threshold stored in the bundle

    [model]
    type = "dualhead"          # selects the implementation; or "ultralytics-yolo"
    architecture = "rtdetr_v2_r18vd"  # further keys belong to the implementation
    pretrained = true

    [optim]
    epochs = 100
    batch = 16                 # fixed; no AutoBatch
    optimizer = "AdamW"
    lr0 = 0.001
    lrf = 0.01                 # final learning rate = lr0 * lrf
    schedule = "cosine"        # or "linear"
    warmup_epochs = 3.0
    weight_decay = 0.0005
    momentum = 0.937
    patience = 20              # early stopping; 0 disables it
    amp = true
    workers = 4

    [augment]
    time_flip = 0.0            # probability of time reversal

    [cv]
    n_folds = 5                # 0: no cross-validation
    final_fit = true           # train on all train groups, stop on val groups

The default model is the dual-head RT-DETRv2 (``dualhead``,
`chirpdetector.models.dualhead`: boxes and the emitter's EOD frequency);
``type = "ultralytics-yolo"`` selects YOLO26 (`chirpdetector.models.yolo`),
which predicts boxes only. The ``[optim]`` defaults were tuned for YOLO26;
the dual-head detector was trained with ``lr0 = 1e-4``, ``lrf = 0.05``,
``warmup_epochs = 0.5``, ``weight_decay = 1e-4``, ``epochs = 30`` and
``patience = 6`` (``examples/train_dualhead.toml`` in the repository).

Unknown keys raise at every level, so a typo never silently falls back to a
default. `DetectorTrainingConfig.to_dict` gives the resolved config (every
value, including the implementation's defaults), which every run directory
stores.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, get_args

from chirpdetector.models import get_backend

__all__ = [
    "DEFAULT_MODEL_TYPE",
    "AugmentConfig",
    "CVConfig",
    "DetectorTrainingConfig",
    "OptimConfig",
]

DEFAULT_MODEL_TYPE = "dualhead"
"""Model type of a config without ``[model] type``: the dual-head RT-DETRv2."""

Schedule = Literal["cosine", "linear"]
OPTIMIZERS = (
    "SGD",
    "MuSGD",
    "Adam",
    "Adamax",
    "AdamW",
    "NAdam",
    "RAdam",
    "RMSProp",
    "auto",
)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _from_table(cls: type, data: Mapping[str, Any], what: str) -> Any:
    if not isinstance(data, Mapping):
        msg = f"[{what}] must be a table, got {data!r}"
        raise ValueError(msg)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        msg = f"unknown keys in [{what}]: {unknown}; known: {sorted(known)}"
        raise ValueError(msg)
    return cls(**data)


@dataclass(frozen=True, slots=True)
class OptimConfig:
    """Optimisation: epochs, batch, optimizer, schedule, early stopping, AMP.

    ``batch`` is a fixed number of images per step, chosen to fit the GPU
    memory cap (see `chirpdetector.resources`); ultralytics' AutoBatch is
    not available. Gradients are accumulated to a nominal batch of 64
    (ultralytics' ``nbs``).
    """

    epochs: int = 100
    batch: int = 16
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    schedule: Schedule = "cosine"
    warmup_epochs: float = 3.0
    weight_decay: float = 0.0005
    momentum: float = 0.937
    patience: int = 20
    amp: bool = True
    workers: int = 4

    def __post_init__(self) -> None:
        _check(_is_int(self.epochs) and self.epochs >= 1, "optim.epochs must be >= 1")
        _check(
            _is_int(self.batch) and self.batch >= 1,
            f"optim.batch must be a positive integer (no AutoBatch), got {self.batch!r}",
        )
        _check(
            self.optimizer in OPTIMIZERS, f"optim.optimizer must be one of {OPTIMIZERS}"
        )
        _check(_is_number(self.lr0) and self.lr0 > 0, "optim.lr0 must be positive")
        _check(
            _is_number(self.lrf) and 0 < self.lrf <= 1, "optim.lrf must lie in (0, 1]"
        )
        _check(
            self.schedule in get_args(Schedule),
            f"optim.schedule must be one of {get_args(Schedule)}",
        )
        _check(
            _is_number(self.warmup_epochs) and self.warmup_epochs >= 0,
            "optim.warmup_epochs must be >= 0",
        )
        _check(
            _is_number(self.weight_decay) and self.weight_decay >= 0,
            "optim.weight_decay must be >= 0",
        )
        _check(
            _is_number(self.momentum) and 0 <= self.momentum < 1,
            "optim.momentum must lie in [0, 1)",
        )
        _check(
            _is_int(self.patience) and self.patience >= 0, "optim.patience must be >= 0"
        )
        _check(isinstance(self.amp, bool), "optim.amp must be true or false")
        _check(
            _is_int(self.workers) and self.workers >= 0, "optim.workers must be >= 0"
        )


@dataclass(frozen=True, slots=True)
class AugmentConfig:
    """Image-space augmentation (see `chirpdetector.models._yolo_engine`).

    Only time reversal is offered; vertical flips, colour, rotation, shear,
    scaling, translation, mosaic and mixup are not, because they break the
    time and frequency semantics of a spectrogram.
    """

    time_flip: float = 0.0

    def __post_init__(self) -> None:
        _check(
            _is_number(self.time_flip) and 0 <= self.time_flip <= 1,
            "augment.time_flip must lie in [0, 1]",
        )


@dataclass(frozen=True, slots=True)
class CVConfig:
    """Grouped k-fold cross-validation over the ``train`` groups.

    ``n_folds`` 0 skips cross-validation. ``final_fit`` trains one more model
    on all ``train`` groups, with the ``val`` groups for early stopping;
    that model becomes the run's bundle.
    """

    n_folds: int = 5
    final_fit: bool = True

    def __post_init__(self) -> None:
        _check(
            _is_int(self.n_folds) and (self.n_folds == 0 or self.n_folds >= 2),
            f"cv.n_folds must be 0 or >= 2, got {self.n_folds!r}",
        )
        _check(isinstance(self.final_fit, bool), "cv.final_fit must be true or false")


_DEVICE_HINT = "device must be 'auto', 'cpu', 'cuda', 'cuda:N' or 'N'"


@dataclass(frozen=True)
class DetectorTrainingConfig:
    """Everything that determines a detector training run (module docstring).

    ``model`` is the implementation's resolved ``[model]`` table, without
    ``type`` (that is ``model_type``).
    """

    model_type: str = DEFAULT_MODEL_TYPE
    model: Mapping[str, Any] = field(default_factory=dict)
    optim: OptimConfig = field(default_factory=OptimConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    score_threshold: float = 0.5
    seed: int = 0
    deterministic: bool = True
    device: str = "auto"
    max_vram_fraction: float | None = None

    def __post_init__(self) -> None:
        resolved = get_backend(self.model_type).resolve_model_config(dict(self.model))
        object.__setattr__(self, "model", resolved)
        _check(
            _is_number(self.score_threshold) and 0 <= self.score_threshold <= 1,
            "score_threshold must lie in [0, 1]",
        )
        _check(
            _is_int(self.seed) and self.seed >= 0, "seed must be a non-negative integer"
        )
        _check(
            isinstance(self.deterministic, bool), "deterministic must be true or false"
        )
        device = str(self.device).strip().lower()
        valid = (
            device in {"auto", "cpu", "cuda"}
            or device.isdigit()
            or (device.startswith("cuda:") and device[5:].isdigit())
        )
        _check(valid, f"{_DEVICE_HINT}, got {self.device!r}")
        if self.max_vram_fraction is not None:
            _check(
                _is_number(self.max_vram_fraction) and 0 < self.max_vram_fraction <= 1,
                "max_vram_fraction must lie in (0, 1]",
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DetectorTrainingConfig:
        """Build from a parsed TOML document (or `to_dict` output)."""
        data = dict(data)
        top = {f.name for f in fields(cls)} - {"model_type"}
        unknown = sorted(set(data) - top)
        if unknown:
            msg = f"unknown top-level keys: {unknown}; known: {sorted(top)}"
            raise ValueError(msg)
        model = dict(data.pop("model", {}) or {})
        model_type = str(model.pop("type", DEFAULT_MODEL_TYPE))
        return cls(
            model_type=model_type,
            model=model,
            optim=_from_table(OptimConfig, data.pop("optim", {}), "optim"),
            augment=_from_table(AugmentConfig, data.pop("augment", {}), "augment"),
            cv=_from_table(CVConfig, data.pop("cv", {}), "cv"),
            **data,
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> DetectorTrainingConfig:
        """Load a TOML file."""
        with Path(path).open("rb") as f:
            return cls.from_dict(tomllib.load(f))

    def to_dict(self) -> dict[str, Any]:
        """The resolved config, in the TOML layout."""
        return {
            "seed": self.seed,
            "deterministic": self.deterministic,
            "device": self.device,
            "max_vram_fraction": self.max_vram_fraction,
            "score_threshold": self.score_threshold,
            "model": {"type": self.model_type, **dict(self.model)},
            "optim": asdict(self.optim),
            "augment": asdict(self.augment),
            "cv": asdict(self.cv),
        }
