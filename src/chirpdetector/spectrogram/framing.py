"""Frame grid of the spectrogram and span planning for long recordings.

Frame convention
----------------
Frame ``k`` (an integer, counted from the first sample of the recording) is
centred on sample ``k * hop_samples``. Every channel's window (length ``n``,
even) covers samples ``[k * hop - n / 2, k * hop + n / 2)``; the periodic
window's peak sits exactly on sample ``k * hop``. The frame's time is that
sample's time, ``t0_s + k * hop_samples / samplerate_hz``.

Recording edges
---------------
The recording is treated as zero outside ``[0, n_samples)``. A recording of
``n_samples`` samples has the frames whose centres lie inside it,
``k = 0 .. ceil(n_samples / hop) - 1``, so the last frame centre is within
one hop of the last sample and nothing at the tail is dropped. Frames whose
reference window reaches past either edge are computed from the zero-padded
signal (their power fades over the first and last half reference window)
and are flagged ``complete == False`` in the result. Zero padding (rather
than reflection) is the only extension that cannot create mirrored
pseudo-chirps, and because it is defined on the recording, not on the
chunk, a span gives the same frames as the whole recording.

Spans
-----
A span ``[start, stop)`` (in samples) owns the frames whose centres lie in
it. Consecutive spans therefore partition the frames of a recording without
gaps or duplicates. `plan_span` says which samples to read and how much zero
padding to add; with those samples, `compute_spectrogram_span` produces the
same frames as computing the whole recording at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chirpdetector.spectrogram.config import SpectrogramConfig


def context_samples(config: SpectrogramConfig) -> int:
    """Samples needed on each side of a frame centre.

    This is half the reference (longest) window. To compute frames whose
    centres lie in ``[start, stop)``, samples from ``start - context`` to
    ``stop + context`` (clipped to the recording) are sufficient.

    Parameters
    ----------
    config
        Spectrogram configuration.

    Returns
    -------
    int
        Half the reference window length, in samples.
    """
    return config.n_fft // 2


def n_frames_total(n_samples: int, config: SpectrogramConfig) -> int:
    """Number of frames of a recording with ``n_samples`` samples.

    Parameters
    ----------
    n_samples
        Length of the recording.
    config
        Spectrogram configuration.

    Returns
    -------
    int
        ``ceil(n_samples / hop_samples)``: one frame per hop, the last one
        centred within one hop of the final sample.
    """
    if n_samples < 1:
        msg = f"n_samples must be >= 1, got {n_samples}"
        raise ValueError(msg)
    return -(-n_samples // config.hop_samples)


def frame_times_s(
    frame_indices: NDArray[np.int64],
    config: SpectrogramConfig,
    t0_s: float = 0.0,
) -> NDArray[np.float64]:
    """Centre times of frames.

    Parameters
    ----------
    frame_indices
        Absolute frame indices ``k``.
    config
        Spectrogram configuration.
    t0_s
        Time of sample 0 of the recording.

    Returns
    -------
    NDArray[np.float64]
        ``t0_s + k * hop_samples / samplerate_hz``.
    """
    sample_positions = np.asarray(frame_indices, dtype=np.int64) * config.hop_samples
    return t0_s + sample_positions / config.samplerate_hz


@dataclass(frozen=True, slots=True)
class SpanPlan:
    """What to read for the frames of one span of a recording.

    Attributes
    ----------
    frame_start, frame_stop
        Absolute frame indices ``[frame_start, frame_stop)`` of the span.
    read_start, read_stop
        Recording samples to read, ``recording[read_start:read_stop]``.
    pad_before, pad_after
        Zeros to add before and after the read samples (non-zero only at the
        recording edges).
    n_samples_total
        Length of the whole recording.
    """

    frame_start: int
    frame_stop: int
    read_start: int
    read_stop: int
    pad_before: int
    pad_after: int
    n_samples_total: int

    @property
    def n_frames(self) -> int:
        """Number of frames in the span."""
        return self.frame_stop - self.frame_start

    @property
    def n_read(self) -> int:
        """Number of samples to read."""
        return self.read_stop - self.read_start

    @property
    def read_slice(self) -> slice:
        """``slice(read_start, read_stop)`` for indexing the recording."""
        return slice(self.read_start, self.read_stop)


def plan_span(
    start_sample: int,
    stop_sample: int,
    n_samples_total: int,
    config: SpectrogramConfig,
) -> SpanPlan:
    """Plan the computation of the frames centred in ``[start, stop)``.

    Parameters
    ----------
    start_sample, stop_sample
        Span of the recording, ``0 <= start_sample < stop_sample <=
        n_samples_total``. The span owns the frames whose centre sample
        lies inside it.
    n_samples_total
        Length of the whole recording.
    config
        Spectrogram configuration.

    Returns
    -------
    SpanPlan
        Frames of the span and the samples (plus edge zero padding) needed
        to compute them.

    Raises
    ------
    ValueError
        If the span is outside the recording or contains no frame centre
        (it is shorter than one hop and falls between two centres).
    """
    start, stop, total = int(start_sample), int(stop_sample), int(n_samples_total)
    if not 0 <= start < stop <= total:
        msg = (
            "need 0 <= start_sample < stop_sample <= n_samples_total, "
            f"got {start}, {stop}, {total}"
        )
        raise ValueError(msg)
    hop = config.hop_samples
    frame_start = -(-start // hop)
    frame_stop = -(-stop // hop)
    if frame_stop <= frame_start:
        msg = (
            f"span [{start}, {stop}) contains no frame centre "
            f"(frame centres are multiples of {hop} samples)"
        )
        raise ValueError(msg)
    half = context_samples(config)
    first_needed = frame_start * hop - half
    stop_needed = (frame_stop - 1) * hop + half
    read_start = max(0, first_needed)
    read_stop = min(total, stop_needed)
    return SpanPlan(
        frame_start=frame_start,
        frame_stop=frame_stop,
        read_start=read_start,
        read_stop=read_stop,
        pad_before=read_start - first_needed,
        pad_after=stop_needed - read_stop,
        n_samples_total=total,
    )


def complete_frames(
    frame_start: int,
    frame_stop: int,
    n_samples_total: int,
    config: SpectrogramConfig,
) -> NDArray[np.bool_]:
    """Flag frames whose reference window lies entirely inside the recording.

    Parameters
    ----------
    frame_start, frame_stop
        Absolute frame indices ``[frame_start, frame_stop)``.
    n_samples_total
        Length of the whole recording.
    config
        Spectrogram configuration.

    Returns
    -------
    NDArray[np.bool_]
        ``True`` where no zero padding entered the frame.
    """
    half = context_samples(config)
    centres = np.arange(frame_start, frame_stop, dtype=np.int64) * config.hop_samples
    return (centres - half >= 0) & (centres + half <= n_samples_total)
