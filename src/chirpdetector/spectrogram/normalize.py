"""Map linear power to [0, 1] with a fixed dB range and a robust reference.

Steps: power -> dB; one reference level per call, measured on the reference
(longest-window) channel; the fixed interval ``[ref - (range - headroom),
ref + headroom]`` mapped linearly to [0, 1] and clipped. The same reference
is applied to every channel, so relative levels across channels (and across
time within the call) are preserved. There is no min-max scaling: a global
gain shifts every dB value and the reference by the same amount and leaves
the output unchanged.

Reference level
---------------
For each frame, take the largest power across the frequency grid of the
reference channel (the strongest fish's EOD peak in that frame); the
reference is the median of these per-frame peaks, in dB.

* It is anchored on the fish, which are present in every competition
  recording and every simulated scene, so the strongest fish sits about
  ``headroom_db`` below the top of the range and weaker fish and the noise
  floor fall below it. A fixed dB range keeps the contrast between any two
  features independent of the anchor; the anchor only decides what clips.
  Anchoring on the fish clips the noise floor (when the strongest fish is
  more than ``dynamic_range_db - headroom_db`` above it), never the fish.
* The per-frame maximum does not depend on how many fish there are or how
  wide the frequency crop is, unlike a high percentile of all cells, whose
  value moves with the fraction of cells the fish occupy.
* The median over frames ignores transients (clicks, a chirp, movement
  artefacts) in fewer than half of the frames.
* Rejected: the median noise floor. It is degenerate for noise-free
  simulations and would clip the strongest (nearest) fish, whose chirps
  are the most common detection targets.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from chirpdetector.spectrogram.compute import Spectrogram
from chirpdetector.spectrogram.config import SpectrogramConfig

_NORMALIZATION_FIELDS = ("dynamic_range_db", "headroom_db")


def reference_level_db(spec: Spectrogram) -> float | None:
    """Robust reference level of a spectrogram.

    Parameters
    ----------
    spec
        Spectrogram; only channel 0 (the reference window) is used.

    Returns
    -------
    float or None
        Median over frames of the per-frame peak power of the reference
        channel, in dB. Frames that are exactly zero (digital silence) are
        ignored. ``None`` if every frame is silent.
    """
    peaks = spec.power[0].amax(dim=0).double().cpu().numpy()
    peaks = peaks[peaks > 0]
    if peaks.size == 0:
        return None
    return float(10.0 * np.log10(np.median(peaks)))


def normalize(
    spec: Spectrogram,
    config: SpectrogramConfig,
    *,
    reference_db: float | None = None,
) -> torch.Tensor:
    """Map a spectrogram to [0, 1].

    Parameters
    ----------
    spec
        Spectrogram from `compute_spectrogram` or `compute_spectrogram_span`.
    config
        Configuration providing ``dynamic_range_db`` and ``headroom_db``.
        All other fields must match ``spec.config``.
    reference_db
        Reference level to use instead of `reference_level_db(spec)`, e.g.
        one measured over a longer stretch of the recording so that
        adjacent chunks share a scale.

    Returns
    -------
    torch.Tensor
        float32 on ``spec.power``'s device, shape ``(n_windows, n_freqs,
        n_times)``: ``clip((dB - ref + range - headroom) / range, 0, 1)``.
        All zeros if the spectrogram is entirely silent.

    Raises
    ------
    ValueError
        If ``config`` differs from ``spec.config`` in a field other than
        the normalisation parameters.
    """
    reset = {name: getattr(spec.config, name) for name in _NORMALIZATION_FIELDS}
    if dataclasses.replace(config, **reset) != spec.config:
        msg = (
            "config differs from the spectrogram's config in STFT parameters; "
            "only dynamic_range_db and headroom_db may differ"
        )
        raise ValueError(msg)
    ref = reference_level_db(spec) if reference_db is None else float(reference_db)
    if ref is None:
        return torch.zeros_like(spec.power)
    tiny = torch.finfo(torch.float32).tiny
    power_db = 10.0 * torch.log10(spec.power.clamp_min(tiny))
    bottom_db = ref + config.headroom_db - config.dynamic_range_db
    scaled = (power_db - bottom_db) / config.dynamic_range_db
    return scaled.clamp_(0.0, 1.0)
