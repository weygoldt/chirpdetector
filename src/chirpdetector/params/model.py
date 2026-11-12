"""The chirp-parameter decoder: a compact CNN with heteroscedastic heads.

Targets live in *model space* (`to_model_space`): ``log height``, ``log
FWHM``, ``log 10 % width`` and ``log(undershoot + 1 Hz)``. Errors of all
four scale with their size (study report), so a Gaussian in log space is
the natural noise model; the 1 Hz offset keeps chirps without undershoot
finite. The network predicts, per target, a mean and a log-variance in
standardised model space (z-scores over the training set); the bundle
stores the standardisation (`Standardization`).

Architecture (`ParamNet`, decoder "A" of the feasibility study, ~0.4 M
parameters): two coordinate channels (frequency and time in [-1, 1]) are
appended to the crop so that absolute position (where the baseline and
the peak sit in the crop) survives the convolutions; a stride-2 stem,
three conv blocks with max pooling, a 1x1 bottleneck, *flattened* (not
pooled, for the same reason), concatenated with the scalars, and an MLP
with ``2 * n_targets`` outputs. The log-variance is clamped to [-12, 8].

Loss: beta-NLL (`beta_nll`, Seitzer et al. 2022, beta 0.5), a Gaussian
negative log-likelihood whose per-sample weight ``var**beta`` keeps the
gradients of hard (high-variance) samples from vanishing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from chirpdetector.params.truth import PARAMETERS

__all__ = [
    "LOG_OFFSETS",
    "ModelConfig",
    "ParamNet",
    "Standardization",
    "beta_nll",
    "from_model_space",
    "natural_sd",
    "to_model_space",
]

LOG_OFFSETS: dict[str, float] = {
    "height_hz": 0.0,
    "fwhm_s": 0.0,
    "width10_s": 0.0,
    "undershoot_hz": 1.0,
}
"""Model space of parameter ``p`` is ``log(p + LOG_OFFSETS[p])``."""

FloatArray = npt.NDArray[np.float64]


def to_model_space(values: npt.ArrayLike) -> FloatArray:
    """``(n, 4)`` parameters (`PARAMETERS` order, natural units) -> model space."""
    array = np.asarray(values, dtype=np.float64).reshape(-1, len(PARAMETERS))
    offsets = np.array([LOG_OFFSETS[p] for p in PARAMETERS])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(array + offsets)


def from_model_space(values: npt.ArrayLike) -> FloatArray:
    """Inverse of `to_model_space` (undershoot floored at 0)."""
    array = np.asarray(values, dtype=np.float64).reshape(-1, len(PARAMETERS))
    offsets = np.array([LOG_OFFSETS[p] for p in PARAMETERS])
    out = np.exp(array) - offsets
    return np.maximum(out, 0.0)


def natural_sd(values: npt.ArrayLike, sigma_log: npt.ArrayLike) -> FloatArray:
    """Standard deviation in natural units from one in model space.

    First-order (delta method): ``sd = (value + offset) * sigma_log``.
    """
    array = np.asarray(values, dtype=np.float64).reshape(-1, len(PARAMETERS))
    sigma = np.asarray(sigma_log, dtype=np.float64).reshape(array.shape)
    offsets = np.array([LOG_OFFSETS[p] for p in PARAMETERS])
    return (array + offsets) * sigma


@dataclass(frozen=True)
class Standardization:
    """Means and standard deviations of targets (model space) and scalars."""

    target_mean: tuple[float, ...]
    target_sd: tuple[float, ...]
    scalar_mean: tuple[float, ...]
    scalar_sd: tuple[float, ...]

    @classmethod
    def fit(cls, targets: npt.ArrayLike, scalars: npt.ArrayLike) -> Standardization:
        """From training targets (model space) and scalars."""
        y = np.asarray(targets, dtype=np.float64)
        s = np.asarray(scalars, dtype=np.float64)
        return cls(
            target_mean=tuple(float(v) for v in y.mean(axis=0)),
            target_sd=tuple(float(v) for v in np.maximum(y.std(axis=0), 1e-6)),
            scalar_mean=tuple(float(v) for v in s.mean(axis=0)),
            scalar_sd=tuple(float(v) for v in np.maximum(s.std(axis=0), 1e-6)),
        )

    def scalars(self, scalars: npt.ArrayLike) -> npt.NDArray[np.float32]:
        """Standardised scalars."""
        s = np.asarray(scalars, dtype=np.float64)
        return ((s - self.scalar_mean) / np.asarray(self.scalar_sd)).astype(np.float32)

    def targets(self, targets: npt.ArrayLike) -> npt.NDArray[np.float32]:
        """Standardised targets."""
        y = np.asarray(targets, dtype=np.float64)
        return ((y - self.target_mean) / np.asarray(self.target_sd)).astype(np.float32)

    def unscale(
        self, mean: npt.ArrayLike, log_var: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        """Network outputs -> mean and standard deviation in model space."""
        sd = np.asarray(self.target_sd)
        m = np.asarray(mean, dtype=np.float64) * sd + np.asarray(self.target_mean)
        sigma = np.exp(0.5 * np.asarray(log_var, dtype=np.float64)) * sd
        return m, sigma

    def to_dict(self) -> dict[str, list[float]]:
        """JSON form."""
        return {k: list(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Sequence[float]]) -> Standardization:
        """Inverse of `to_dict`."""
        return cls(
            **{f.name: tuple(float(v) for v in data[f.name]) for f in fields(cls)}
        )


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of `ParamNet`.

    Parameters
    ----------
    stem
        Channels of the stride-2 stem.
    widths
        Channels of the conv blocks (each halves the resolution).
    bottleneck
        Channels of the 1x1 convolution before flattening.
    hidden
        Width of the MLP's hidden layer.
    dropout
        Dropout before the output layer.
    """

    stem: int = 24
    widths: tuple[int, ...] = (32, 48, 64)
    bottleneck: int = 32
    hidden: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(self, "widths", tuple(int(w) for w in self.widths))
        if min(self.stem, self.bottleneck, self.hidden, *self.widths) < 1:
            msg = f"model widths must be positive: {self}"
            raise ValueError(msg)
        if not 0 <= self.dropout < 1:
            msg = f"dropout must lie in [0, 1), got {self.dropout}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """JSON form."""
        out = asdict(self)
        out["widths"] = list(self.widths)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelConfig:
        """Inverse of `to_dict`; unknown keys raise."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            msg = f"unknown model keys {unknown}; known: {sorted(known)}"
            raise ValueError(msg)
        values = dict(data)
        if "widths" in values:
            values["widths"] = tuple(values["widths"])
        return cls(**values)


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ParamNet(nn.Module):
    """CNN regressor with a mean and a log-variance per target (module docstring).

    Parameters
    ----------
    in_channels
        Channels of the crop (the front end's windows).
    input_hw
        ``(H, W)`` of the crop.
    n_scalars
        Number of scalar inputs.
    n_targets
        Number of targets.
    config
        Architecture.
    """

    def __init__(
        self,
        in_channels: int,
        input_hw: tuple[int, int],
        n_scalars: int,
        n_targets: int,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        config = config or ModelConfig()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels + 2, config.stem, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(config.stem),
            nn.ReLU(inplace=True),
        ]
        channels = config.stem
        for w in config.widths:
            layers.append(_block(channels, w))
            channels = w
        self.features = nn.Sequential(
            *layers, nn.Conv2d(channels, config.bottleneck, 1), nn.ReLU(inplace=True)
        )
        h, w = input_hw[0] // 2, input_hw[1] // 2
        for _ in config.widths:
            h, w = h // 2, w // 2
        if h < 1 or w < 1:
            msg = f"input {input_hw} is too small for {len(config.widths)} blocks"
            raise ValueError(msg)
        self.head = nn.Sequential(
            nn.Linear(config.bottleneck * h * w + n_scalars, config.hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, 2 * n_targets),
        )
        self.n_targets = n_targets

    def forward(
        self, x: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and log-variance, each ``(N, n_targets)``, in standardised units."""
        n, _, h, w = x.shape
        fy = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        fx = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        grid = torch.stack(torch.meshgrid(fy, fx, indexing="ij"))
        coords = grid[None].expand(n, -1, -1, -1)
        z = self.features(torch.cat([x, coords], dim=1)).flatten(1)
        out = self.head(torch.cat([z, scalars.to(z.dtype)], dim=1))
        mean, log_var = out[:, : self.n_targets], out[:, self.n_targets :]
        return mean, log_var.clamp(-12.0, 8.0)


def beta_nll(
    mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor, beta: float = 0.5
) -> torch.Tensor:
    """beta-NLL (Seitzer et al. 2022): Gaussian NLL weighted by ``var**beta``.

    ``beta = 0`` is the plain NLL; ``beta = 1`` gives MSE-like gradients for
    the mean. Samples with a non-finite target are ignored.
    """
    finite = torch.isfinite(target)
    target = torch.where(finite, target, mean.detach())
    var = log_var.exp()
    nll = 0.5 * (log_var + (target - mean).square() / var)
    if beta > 0:
        nll = nll * var.detach() ** beta
    nll = torch.where(finite, nll, torch.zeros_like(nll))
    return nll.sum() / finite.sum().clamp_min(1)
