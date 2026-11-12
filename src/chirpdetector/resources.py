"""Limits on shared compute resources (GPU memory).

The GPU is shared with other jobs. Training and inference cap this
process's share of GPU memory with
``torch.cuda.set_per_process_memory_fraction`` before they build a model:
the caching allocator then raises an out-of-memory error inside this
process instead of starving the other jobs. The cap is the smaller of

* the environment variable ``CHIRP_MAX_VRAM_FRACTION`` (set by the
  workspace's ``scripts/guarded`` wrapper, ``--vram``), and
* the config value ``max_vram_fraction`` of the training config.

Batch sizes are fixed numbers chosen to fit under the cap; ultralytics'
AutoBatch (``batch=-1`` or a fraction) is not used, because it sizes the
batch from the memory that is free at that moment, not from the cap.
"""

from __future__ import annotations

import logging
import math
import os

import torch

__all__ = ["VRAM_ENV", "apply_vram_limit", "vram_fraction"]

log = logging.getLogger(__name__)

VRAM_ENV = "CHIRP_MAX_VRAM_FRACTION"
"""Environment variable with the largest fraction of GPU memory to use."""


def _parse(value: object, what: str) -> float:
    try:
        fraction = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as err:
        msg = f"{what} must be a number in (0, 1], got {value!r}"
        raise ValueError(msg) from err
    if not (math.isfinite(fraction) and 0 < fraction <= 1):
        msg = f"{what} must lie in (0, 1], got {value!r}"
        raise ValueError(msg)
    return fraction


def vram_fraction(config_fraction: float | None = None) -> float | None:
    """The GPU memory cap: the smaller of the environment and the config.

    Parameters
    ----------
    config_fraction
        ``max_vram_fraction`` from a config, or None.

    Returns
    -------
    float or None
        The fraction in (0, 1], or None if neither is set.

    Raises
    ------
    ValueError
        If either value is not a number in (0, 1].
    """
    values = []
    env = os.environ.get(VRAM_ENV)
    if env is not None and env.strip():
        values.append(_parse(env, VRAM_ENV))
    if config_fraction is not None:
        values.append(_parse(config_fraction, "max_vram_fraction"))
    return min(values) if values else None


def apply_vram_limit(
    config_fraction: float | None = None, device: torch.device | str | None = None
) -> float | None:
    """Cap this process's GPU memory on ``device`` (see `vram_fraction`).

    Call it before building a model. Without CUDA, or on a CPU device, it
    does nothing but return the fraction.

    Parameters
    ----------
    config_fraction
        ``max_vram_fraction`` from a config, or None.
    device
        Torch device (or ultralytics device string such as ``"0"``); CUDA
        device 0 if None.

    Returns
    -------
    float or None
        The fraction applied (or that would apply), None if uncapped.
    """
    fraction = vram_fraction(config_fraction)
    if fraction is None:
        return None
    if isinstance(device, str) and device.strip().isdigit():
        device = torch.device("cuda", int(device.strip()))
    dev = torch.device("cuda", 0) if device is None else torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        log.debug("no CUDA device; GPU memory cap %.3g not applied", fraction)
        return fraction
    index = 0 if dev.index is None else dev.index
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    log.info("capped GPU memory of cuda:%d at %.0f%%", index, 100 * fraction)
    return fraction
