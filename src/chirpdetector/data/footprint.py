"""chirpdetector's front end as the spectrogram of gridtools' chirp footprints.

`gridtools.sim.footprint.chirp_footprint` derives a chirp's ground-truth
box from where it changes a spectrogram, and takes the spectrogram as a
function (``SpectrogramFn``). Its scenes store footprints computed with a
plain Hann spectrogram (`gridtools.sim.footprint.HannSpectrogram`), which
equals channel 0 of chirpdetector's default front end except for the frame
grid (frames centred ``nfft / 2 + k * hop`` samples after the rendered
span's start rather than on multiples of the hop). `FrontEndSpectrogram`
wraps `chirpdetector.spectrogram.compute_spectrogram` as such a function,
on the absolute frame grid of the recording, so that footprints can be
computed with any front end a detector uses, and their box edges fall on
the dataset images' pixel edges.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chirpdetector.spectrogram import SpectrogramConfig, compute_spectrogram

__all__ = ["FrontEndSpectrogram", "footprints_match"]

# Relative tolerance when comparing a scene's footprint settings with a config.
_RTOL = 1e-9


@dataclass(frozen=True)
class FrontEndSpectrogram:
    """One channel of chirpdetector's front end as a footprint spectrogram.

    Call it as ``fn(signal, samplerate_hz, t0_s)`` with ``signal`` of shape
    ``(n, n_electrodes)`` whose first sample lies at ``t0_s`` (seconds on
    the recording's clock, a whole number of samples after its start). It
    returns ``(power (F, T), freqs_hz (F,), times_s (T,))``: linear power
    of channel ``channel`` aggregated over electrodes as in ``config``, on
    the frames of the whole recording (centred on multiples of the hop)
    whose reference window lies entirely inside ``signal``.

    Attributes
    ----------
    config
        The front end.
    channel
        Window index (0 = reference window).
    device
        Torch device of the STFT.
    """

    config: SpectrogramConfig
    channel: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not 0 <= self.channel < self.config.n_windows:
            msg = (
                f"channel {self.channel} does not exist; the front end has "
                f"{self.config.n_windows} windows"
            )
            raise ValueError(msg)

    @property
    def window_s(self) -> float:
        """Longest analysis window (sets the span `chirp_footprint` renders)."""
        return self.config.reference_window_s

    def _channel_config(self) -> tuple[SpectrogramConfig, int]:
        if self.channel == 0:
            return dataclasses.replace(self.config, extra_windows_s=()), 0
        return self.config, self.channel

    def __call__(
        self, signal: NDArray[np.floating], samplerate_hz: float, t0_s: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Spectrogram of ``signal``; see the class docstring."""
        config, channel = self._channel_config()
        layout = config.layout(samplerate_hz)
        x = np.asarray(signal)
        if x.ndim == 1:
            x = x[:, None]
        first = round(t0_s * samplerate_hz)
        if abs(first - t0_s * samplerate_hz) > 1e-6 * max(1.0, abs(first)):
            msg = f"t0_s = {t0_s!r} s is not on the sample grid at {samplerate_hz} Hz"
            raise ValueError(msg)
        hop, n_fft = layout.hop_samples, layout.n_fft
        freqs = layout.freqs_hz
        n = x.shape[0]
        if n < n_fft:
            return np.zeros((freqs.size, 0)), freqs, np.zeros(0)
        # Pad the front to the previous multiple of the hop, so that frames
        # are centred on the recording's grid; keep only the frames whose
        # window lies inside the given samples.
        pad = first % hop
        if pad:
            x = np.concatenate([np.zeros((pad, x.shape[1]), dtype=x.dtype), x])
        origin = first - pad
        spec = compute_spectrogram(
            x, samplerate_hz, config, t0_s=origin / samplerate_hz, device=self.device
        )
        centres = origin + (spec.frame_start + np.arange(spec.n_times)) * hop
        lo = centres - n_fft // 2
        keep = (lo >= first) & (lo + n_fft <= first + n)
        power = spec.power[channel].double().cpu().numpy()[:, keep]
        return power, np.asarray(spec.freqs_hz), np.asarray(spec.times_s)[keep]


def footprints_match(footprint: Mapping[str, Any], config: SpectrogramConfig) -> bool:
    """Whether a scene's footprint boxes were computed with ``config``'s channel 0.

    Parameters
    ----------
    footprint
        The ``footprint`` section of a scene's ``scene.json`` (the
        ``HannSpectrogram`` settings of `gridtools.sim.footprint`).
    config
        The dataset's front end.

    Returns
    -------
    bool
        True if the scene's footprints used a Hann window of
        ``config.reference_window_s``, hop ``config.hop_s``, the band
        ``config.f_min_hz``-``config.f_max_hz`` and summed power, i.e. the
        same cells as the dataset's reference channel (up to a sub-hop
        offset of the frame grid). False also when the scene has no
        footprints.
    """
    if not footprint or not footprint.get("enabled", False):
        return False
    pairs = (
        ("window_s", config.reference_window_s),
        ("hop_s", config.hop_s),
        ("f_min_hz", config.f_min_hz),
        ("f_max_hz", config.f_max_hz),
    )
    for key, value in pairs:
        stated = footprint.get(key)
        if stated is None or not math.isclose(float(stated), value, rel_tol=_RTOL):
            return False
    return config.window == "hann" and config.aggregation == "sum"
