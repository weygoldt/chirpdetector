"""Decoder inputs: rectified multi-resolution crops and detector-view crops.

Rectified crop (`rectified_crop`)
    Power spectrogram of a short stretch of the recording around a detection
    box, on a frequency axis relative to the emitter's baseline
    (``delta_f = f - baseline``, fixed grid in Hz) and a time axis of a fixed
    number of frames spread evenly over the box shrunk by ``inset_s`` on each
    side (the footprint box pads the chirp's support by ~80 ms of
    reference-window smear). One channel per window duration; every window
    is zero-padded to one FFT length, so all channels share one frequency
    grid (as in `chirpdetector.spectrogram`). Frames are evaluated at
    arbitrary centres (not a fixed hop), so the time axis scales with the box:
    ~1.7 ms per frame for a small chirp, ~7 ms for a 0.6 s big chirp.

Detector-view crop (`detector_view`)
    Bilinear resampling of the detector's normalised image (absolute
    frequency, 5 ms hop) to a fixed size over the box plus a margin, as a
    region-of-interest head on the detector's input would see it.

Both are normalised like the detector: dB relative to the median over frames
of the per-frame peak in channel 0, the interval ``[ref - (range -
headroom), ref + headroom]`` mapped to [0, 1].
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

__all__ = [
    "RectifiedConfig",
    "crop_span",
    "detector_view",
    "framed_power",
    "normalize_db",
    "rectified_crop",
    "shift_frequency",
]


@dataclass(frozen=True, slots=True)
class RectifiedConfig:
    """Geometry of the rectified crop.

    Attributes
    ----------
    windows_s
        Hann window durations, one channel each.
    fft_s
        FFT length in seconds; sets the native grid ``1 / fft_s`` before the
        interpolation onto the delta-f grid.
    df_min_hz, df_step_hz, n_freqs
        Delta-f grid: ``df_min_hz + df_step_hz * k``, ``k < n_freqs``.
    n_times
        Frames per crop.
    inset_s
        The crop spans ``[box_t0 + inset_s, box_t1 - inset_s]``.
    min_span_s
        Shortest crop span.
    dynamic_range_db, headroom_db
        Normalisation (as `chirpdetector.spectrogram.SpectrogramConfig`).
    """

    windows_s: tuple[float, ...] = (0.2048, 0.0512, 0.0128, 0.0064)
    fft_s: float = 0.2048
    df_min_hz: float = -125.0
    df_step_hz: float = 5.0
    n_freqs: int = 192
    n_times: int = 96
    inset_s: float = 0.04
    min_span_s: float = 0.1
    dynamic_range_db: float = 60.0
    headroom_db: float = 10.0

    @property
    def delta_f_hz(self) -> npt.NDArray[np.float64]:
        """Delta-f grid, ascending."""
        return self.df_min_hz + self.df_step_hz * np.arange(self.n_freqs)


def crop_span(
    box_t0_s: float, box_t1_s: float, config: RectifiedConfig
) -> tuple[float, float]:
    """Time span of the rectified crop for a box (centred, at least ``min_span_s``)."""
    t0, t1 = box_t0_s + config.inset_s, box_t1_s - config.inset_s
    if t1 - t0 < config.min_span_s:
        centre = 0.5 * (box_t0_s + box_t1_s)
        t0, t1 = centre - config.min_span_s / 2, centre + config.min_span_s / 2
    return t0, t1


def _hann_padded(length: int, n_fft: int, device: torch.device) -> torch.Tensor:
    window = torch.zeros(n_fft, dtype=torch.float32, device=device)
    start = n_fft // 2 - length // 2
    window[start : start + length] = torch.hann_window(
        length, periodic=True, dtype=torch.float32, device=device
    )
    return window


def framed_power(
    signal: torch.Tensor,
    centres: torch.Tensor,
    window_lengths: Sequence[int],
    n_fft: int,
    bin_range: tuple[int, int],
) -> torch.Tensor:
    """Calibrated one-sided power of frames at arbitrary centre samples.

    Parameters
    ----------
    signal
        ``(n_electrodes, n_samples)``; frames that reach beyond it are
        zero-padded.
    centres
        Centre sample of each frame (int64), ``(n_frames,)``.
    window_lengths
        Even Hann window lengths in samples, ``<= n_fft``; one channel each.
    n_fft
        FFT length (even).
    bin_range
        FFT bins ``[k0, k1)`` to keep.

    Returns
    -------
    torch.Tensor
        ``(n_windows, n_electrodes, k1 - k0, n_frames)``; a sinusoid of
        amplitude ``A`` on a bin reads ``A**2 / 2``.
    """
    half = n_fft // 2
    padded = F.pad(signal, (half, half))
    offsets = torch.arange(-half, half, device=signal.device)
    index = centres[:, None] + half + offsets[None, :]
    frames = padded[:, index]  # (n_el, n_frames, n_fft)
    k0, k1 = bin_range
    out = []
    for length in window_lengths:
        window = _hann_padded(int(length), n_fft, signal.device)
        spec = torch.fft.rfft(frames * window, dim=-1)[..., k0:k1]
        power = spec.real.square() + spec.imag.square()
        scale = 2.0 / float(window.double().sum()) ** 2
        out.append((power * scale).permute(0, 2, 1))
    return torch.stack(out)


def normalize_db(
    power: torch.Tensor, dynamic_range_db: float, headroom_db: float
) -> torch.Tensor:
    """Map ``(n_channels, n_freqs, n_times)`` power to [0, 1] (detector rule)."""
    db = 10.0 * torch.log10(power.clamp_min(1e-30))
    ref = db[0].amax(dim=0).median()
    bottom = ref + headroom_db - dynamic_range_db
    return ((db - bottom) / dynamic_range_db).clamp(0.0, 1.0)


def _interp_bins(power: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Linear interpolation along axis -2 at fractional bin positions."""
    n = power.shape[-2]
    position = position.clamp(0, n - 1)
    lo = position.floor().long().clamp(max=n - 2)
    w = (position - lo).to(power.dtype)[:, None]
    return power[..., lo, :] * (1 - w) + power[..., lo + 1, :] * w


def rectified_crop(
    segment: torch.Tensor,
    segment_t0_s: float,
    samplerate_hz: float,
    baseline_hz: float,
    span_s: tuple[float, float],
    config: RectifiedConfig,
    electrode_sets: Sequence[Sequence[int] | None] = (None,),
) -> list[torch.Tensor]:
    """Rectified crops of one chirp, one per electrode set.

    Parameters
    ----------
    segment
        ``(n_electrodes, n_samples)`` samples starting at ``segment_t0_s``;
        should extend ``fft_s / 2`` beyond the span (otherwise zero-padded).
    segment_t0_s
        Time of the segment's first sample.
    samplerate_hz
        Rate of the segment.
    baseline_hz
        Emitter's baseline EOD frequency (true or estimated).
    span_s
        ``(t_start, t_stop)`` of the crop (`crop_span`).
    config
        Geometry.
    electrode_sets
        Electrode indices whose power is summed for each output crop; None
        sums all.

    Returns
    -------
    list of torch.Tensor
        One ``(n_windows, n_freqs, n_times)`` float32 tensor in [0, 1] per
        electrode set.
    """
    fs = samplerate_hz
    n_fft = 2 * round(fs * config.fft_s / 2)
    lengths = [min(n_fft, 2 * round(fs * w / 2)) for w in config.windows_s]
    times = np.linspace(span_s[0], span_s[1], config.n_times)
    centres = torch.as_tensor(
        np.round((times - segment_t0_s) * fs).astype(np.int64), device=segment.device
    )
    df_native = fs / n_fft
    f_targets = baseline_hz + config.delta_f_hz
    k0 = max(0, math.floor(f_targets[0] / df_native) - 1)
    k1 = math.ceil(f_targets[-1] / df_native) + 2
    power = framed_power(segment, centres, lengths, n_fft, (k0, k1))
    position = torch.as_tensor(
        f_targets / df_native - k0, dtype=torch.float32, device=segment.device
    )
    out = []
    for electrodes in electrode_sets:
        summed = (
            power.sum(dim=1) if electrodes is None else power[:, list(electrodes)].sum(1)
        )
        crop = _interp_bins(summed, position)
        out.append(normalize_db(crop, config.dynamic_range_db, config.headroom_db))
    return out


def detector_view(
    image: torch.Tensor,
    freqs_hz: npt.NDArray[np.float64],
    times_s: npt.NDArray[np.float64],
    box: tuple[float, float, float, float],
    size: tuple[int, int] = (96, 96),
    margin_rel: float = 0.15,
) -> torch.Tensor:
    """Bilinear crop ``(C, n_f, n_t)`` of a normalised image around a box.

    ``box`` is ``(t0, t1, f0, f1)``; the crop covers it plus ``margin_rel``
    of its width/height on every side. Outside the image reads 0.
    """
    t0, t1, f0, f1 = box
    mt, mf = margin_rel * (t1 - t0), margin_rel * (f1 - f0)
    t = np.linspace(t0 - mt, t1 + mt, size[1])
    f = np.linspace(f0 - mf, f1 + mf, size[0])
    # grid_sample coordinates in [-1, 1] over pixel centres (align_corners=True)
    gx = 2 * (t - times_s[0]) / (times_s[-1] - times_s[0]) - 1
    gy = 2 * (f - freqs_hz[0]) / (freqs_hz[-1] - freqs_hz[0]) - 1
    gy_grid, gx_grid = np.meshgrid(gy, gx, indexing="ij")
    grid = torch.as_tensor(
        np.stack([gx_grid, gy_grid], axis=-1)[None], dtype=image.dtype, device=image.device
    )
    return F.grid_sample(
        image[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )[0]


def shift_frequency(x: torch.Tensor, shift_bins: torch.Tensor) -> torch.Tensor:
    """Shift ``(N, C, F, T)`` crops by fractional bins along F (edges replicate).

    A positive shift moves content to higher bins, i.e. emulates a baseline
    estimate that is too low by ``shift_bins * step``.
    """
    n_f = x.shape[2]
    source = torch.arange(n_f, device=x.device, dtype=x.dtype)[None, :] - shift_bins[:, None]
    source = source.clamp(0, n_f - 1)
    lo = source.floor().long().clamp(max=n_f - 2)
    w = (source - lo)[:, None, :, None]
    idx_lo = lo[:, None, :, None].expand(-1, x.shape[1], -1, x.shape[3])
    return torch.gather(x, 2, idx_lo) * (1 - w) + torch.gather(x, 2, idx_lo + 1) * w
