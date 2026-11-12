"""Multi-resolution power spectrogram of multi-electrode recordings.

Every channel (window length) is computed with ``torch.stft`` at the FFT
length of the reference (longest) window. Shorter windows are zero-padded
symmetrically to that length, so

* all channels share the reference window's frequency bins exactly; the
  shorter windows' spectra are evaluated on the fine grid by zero-padding
  (band-limited interpolation of the short-window DTFT), not by
  interpolating between their coarse bins, and
* all windows of a frame are centred on the same sample, ``k * hop``.

Power is calibrated so that a sinusoid of amplitude ``A`` whose frequency
lies on a grid bin reads ``A**2 / 2`` (its mean-square value) in that bin,
in every channel. Tones therefore have the same level in all channels,
whereas the broadband noise floor rises as the window gets shorter (less
processing gain), which is the physical trade-off.

Electrodes are combined in linear power (sum or max) before any dB
conversion.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from chirpdetector.spectrogram.config import SpectrogramConfig
from chirpdetector.spectrogram.framing import (
    SpanPlan,
    complete_frames,
    frame_times_s,
    plan_span,
)

log = logging.getLogger(__name__)

# Upper bound on the complex STFT buffer of one block of frames. Frames are
# processed in blocks to bound memory on long inputs; per-frame results do
# not depend on the blocking.
_BLOCK_BYTES = 128 * 2**20

_WINDOW_FUNCTIONS: dict[str, Callable[..., torch.Tensor]] = {
    "hann": torch.hann_window,
    "hamming": torch.hamming_window,
    "blackman": torch.blackman_window,
}

SignalLike = NDArray[np.floating] | torch.Tensor
DeviceLike = torch.device | str


@dataclass(frozen=True, slots=True, eq=False)
class Spectrogram:
    """Linear power spectrogram on a shared time-frequency grid.

    Attributes
    ----------
    power
        Linear power, float32, shape ``(n_windows, n_freqs, n_times)``.
        Channel ``i`` uses window ``config.window_lengths_samples[i]``;
        channel 0 is the reference window. Frequency ascends along axis 1.
    freqs_hz
        Bin centre frequencies, ascending, shape ``(n_freqs,)``.
    times_s
        Frame centre times, shape ``(n_times,)``. Frame ``k`` sits at
        ``t0_s + k * hop_samples / samplerate_hz``.
    frame_start
        Absolute index ``k`` of the first frame (0 for a whole recording).
    complete
        ``True`` for frames whose reference window lies entirely inside the
        recording, shape ``(n_times,)``.
    config
        Configuration the spectrogram was computed with.
    """

    power: torch.Tensor
    freqs_hz: NDArray[np.float64]
    times_s: NDArray[np.float64]
    frame_start: int
    complete: NDArray[np.bool_]
    config: SpectrogramConfig

    @property
    def n_windows(self) -> int:
        """Number of channels."""
        return int(self.power.shape[0])

    @property
    def n_freqs(self) -> int:
        """Number of frequency bins."""
        return int(self.power.shape[1])

    @property
    def n_times(self) -> int:
        """Number of frames."""
        return int(self.power.shape[2])

    @property
    def frame_stop(self) -> int:
        """Absolute index one past the last frame."""
        return self.frame_start + self.n_times


def compute_spectrogram(
    signal: SignalLike,
    config: SpectrogramConfig,
    *,
    t0_s: float = 0.0,
    device: DeviceLike | None = None,
) -> Spectrogram:
    """Spectrogram of a whole recording (or of a chunk treated as one).

    Parameters
    ----------
    signal
        Samples, shape ``(n_samples, n_electrodes)``, floating point (cast
        to float32). NumPy array or torch tensor.
    config
        Spectrogram configuration.
    t0_s
        Time of the first sample; frame times are ``t0_s + k * hop_s``.
    device
        Torch device to compute on. Defaults to the tensor's device, or CPU
        for NumPy input.

    Returns
    -------
    Spectrogram
        Frames ``k = 0 .. ceil(n_samples / hop_samples) - 1``. The signal is
        treated as zero outside its extent (see `chirpdetector.spectrogram.
        framing`). To process a long recording in chunks that agree with
        the whole, use `plan_span` and `compute_spectrogram_span` instead.
    """
    n_samples = _n_samples(signal)
    plan = plan_span(0, n_samples, n_samples, config)
    return compute_spectrogram_span(signal, plan, config, t0_s=t0_s, device=device)


def compute_spectrogram_span(
    samples: SignalLike,
    plan: SpanPlan,
    config: SpectrogramConfig,
    *,
    t0_s: float = 0.0,
    device: DeviceLike | None = None,
) -> Spectrogram:
    """Spectrogram of the frames of one span of a long recording.

    Parameters
    ----------
    samples
        ``recording[plan.read_start:plan.read_stop]``, shape
        ``(plan.n_read, n_electrodes)``.
    plan
        Output of `plan_span` for this recording and config.
    config
        Spectrogram configuration (the one `plan` was made with).
    t0_s
        Time of sample 0 of the *recording* (not of the span).
    device
        Torch device to compute on. Defaults to the tensor's device, or CPU
        for NumPy input.

    Returns
    -------
    Spectrogram
        Frames ``plan.frame_start .. plan.frame_stop - 1``, equal to the same
        frames of `compute_spectrogram` on the whole recording.
    """
    x = _as_electrode_major(samples, device)
    if x.shape[1] != plan.n_read:
        msg = (
            f"expected {plan.n_read} samples for span plan "
            f"[{plan.read_start}, {plan.read_stop}), got {x.shape[1]}"
        )
        raise ValueError(msg)
    if plan.pad_before or plan.pad_after:
        x = torch.nn.functional.pad(x, (plan.pad_before, plan.pad_after))
    power = _framed_power(x, plan.n_frames, config)
    frames = np.arange(plan.frame_start, plan.frame_stop, dtype=np.int64)
    return Spectrogram(
        power=power,
        freqs_hz=config.freqs_hz,
        times_s=frame_times_s(frames, config, t0_s),
        frame_start=plan.frame_start,
        complete=complete_frames(
            plan.frame_start, plan.frame_stop, plan.n_samples_total, config
        ),
        config=config,
    )


def _n_samples(signal: SignalLike) -> int:
    """Validate the ``(n_samples, n_electrodes)`` layout; return ``n_samples``."""
    shape = tuple(signal.shape)
    if len(shape) != 2:
        msg = f"signal must have shape (n_samples, n_electrodes), got {shape}"
        raise ValueError(msg)
    n_samples, n_electrodes = int(shape[0]), int(shape[1])
    if n_electrodes < 1:
        msg = "signal has no electrodes"
        raise ValueError(msg)
    if n_electrodes > n_samples:
        msg = (
            f"signal has shape {shape}: more electrodes than samples; expected "
            "(n_samples, n_electrodes) - is it transposed?"
        )
        raise ValueError(msg)
    return n_samples


def _as_electrode_major(signal: SignalLike, device: DeviceLike | None) -> torch.Tensor:
    """Return a contiguous float32 ``(n_electrodes, n_samples)`` tensor."""
    _n_samples(signal)
    if isinstance(signal, torch.Tensor):
        if not signal.is_floating_point():
            msg = f"signal must be floating point, got {signal.dtype}"
            raise TypeError(msg)
        target = signal.device if device is None else torch.device(device)
        return signal.detach().to(device=target, dtype=torch.float32).T.contiguous()
    array = np.asarray(signal)
    if not np.issubdtype(array.dtype, np.floating):
        msg = f"signal must be floating point, got {array.dtype}"
        raise TypeError(msg)
    # Always copy: makes the buffer writable (memmaps) and electrode-major.
    tensor = torch.from_numpy(np.array(array.T, dtype=np.float32, order="C"))
    return tensor.to(torch.device("cpu") if device is None else torch.device(device))


def _window(
    config: SpectrogramConfig, length: int, device: torch.device
) -> torch.Tensor:
    make = _WINDOW_FUNCTIONS[config.window]
    return make(length, periodic=True, dtype=torch.float32, device=device)


def _power_scale(window: torch.Tensor, config: SpectrogramConfig) -> torch.Tensor:
    """Per-bin factor turning ``|X|**2`` into calibrated one-sided power."""
    window_sum = float(window.double().sum())
    bins = torch.arange(
        config.freq_bin_start, config.freq_bin_stop, device=window.device
    )
    one_sided = torch.where((bins == 0) | (bins == config.n_fft // 2), 1.0, 2.0).to(
        torch.float32
    )
    return one_sided / window_sum**2


def _framed_power(
    x: torch.Tensor, n_frames: int, config: SpectrogramConfig
) -> torch.Tensor:
    """Aggregated power of frames ``j = 0 .. n_frames - 1`` of ``x``.

    ``x`` is ``(n_electrodes, (n_frames - 1) * hop + n_fft)``; frame ``j``
    covers ``x[:, j * hop : j * hop + n_fft]`` and its windows peak at
    ``j * hop + n_fft / 2``.
    """
    n_electrodes = x.shape[0]
    n_fft, hop = config.n_fft, config.hop_samples
    expected = (n_frames - 1) * hop + n_fft
    if x.shape[1] != expected:  # internal invariant
        msg = f"padded signal has {x.shape[1]} samples, expected {expected}"
        raise RuntimeError(msg)
    k0, k1 = config.freq_bin_start, config.freq_bin_stop
    out = torch.empty(
        (config.n_windows, config.n_freqs, n_frames),
        dtype=torch.float32,
        device=x.device,
    )
    bytes_per_frame = n_electrodes * (n_fft // 2 + 1) * 8
    block = max(1, _BLOCK_BYTES // bytes_per_frame)
    windows = [_window(config, n, x.device) for n in config.window_lengths_samples]
    scales = [_power_scale(w, config)[:, None] for w in windows]
    log.debug(
        "STFT: %d electrodes, %d frames, %d channels, block %d frames",
        n_electrodes,
        n_frames,
        config.n_windows,
        block,
    )
    with torch.no_grad():
        for j0 in range(0, n_frames, block):
            j1 = min(n_frames, j0 + block)
            segment = x[:, j0 * hop : (j1 - 1) * hop + n_fft]
            for channel, (length, window) in enumerate(
                zip(config.window_lengths_samples, windows, strict=True)
            ):
                spec = torch.stft(
                    segment,
                    n_fft=n_fft,
                    hop_length=hop,
                    win_length=length,
                    window=window,
                    center=False,
                    return_complex=True,
                )[:, k0:k1, :]
                power = spec.real.square() + spec.imag.square()
                if config.aggregation == "sum":
                    combined = power.sum(dim=0)
                else:
                    combined = power.amax(dim=0)
                out[channel, :, j0:j1] = combined * scales[channel]
    return out
