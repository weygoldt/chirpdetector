"""Compact CNN regressor with heteroscedastic Gaussian heads.

Input: ``(N, C, H, W)`` crops in [0, 1] plus ``(N, S)`` scalars (e.g. the
log crop duration). Two coordinate channels (frequency and time, in [-1,
1]) are appended so that absolute position, i.e. delta-f in a rectified
crop, survives the convolutions; the last feature map is flattened, not
pooled, for the same reason. Output: mean and log-variance per target, in
standardised target units.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["ParamNet", "beta_nll"]


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
    """Stride-2 stem, three conv blocks, flatten, MLP; means and log-variances."""

    def __init__(
        self,
        in_channels: int,
        input_hw: tuple[int, int],
        n_scalars: int,
        n_targets: int,
        widths: tuple[int, ...] = (32, 48, 64),
        stem: int = 24,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels + 2, stem, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(stem),
            nn.ReLU(inplace=True),
        ]
        c = stem
        for w in widths:
            layers.append(_block(c, w))
            c = w
        self.features = nn.Sequential(*layers, nn.Conv2d(c, 32, 1), nn.ReLU(inplace=True))
        h, w = input_hw[0] // 2, input_hw[1] // 2
        for _ in widths:
            h, w = h // 2, w // 2
        self.head = nn.Sequential(
            nn.Linear(32 * h * w + n_scalars, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 2 * n_targets),
        )
        self.n_targets = n_targets

    def forward(
        self, x: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, _, h, w = x.shape
        fy = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        fx = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        coords = torch.stack(torch.meshgrid(fy, fx, indexing="ij"))[None].expand(n, -1, -1, -1)
        z = self.features(torch.cat([x, coords], dim=1)).flatten(1)
        out = self.head(torch.cat([z, scalars], dim=1))
        mean, log_var = out[:, : self.n_targets], out[:, self.n_targets :]
        return mean, log_var.clamp(-12, 8)


def beta_nll(
    mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor, beta: float = 0.5
) -> torch.Tensor:
    """beta-NLL (Seitzer et al. 2022): Gaussian NLL weighted by ``var**beta``.

    ``beta = 0`` is the plain NLL; ``beta = 1`` gives MSE-like gradients for
    the mean, which keeps high-variance samples from being ignored.
    """
    var = log_var.exp()
    nll = 0.5 * (log_var + (target - mean).square() / var)
    if beta > 0:
        nll = nll * var.detach() ** beta
    return nll.mean()
