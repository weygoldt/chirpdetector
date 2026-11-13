"""Spectrogram cutouts of single detections.

A cutout is the power summed over electrodes (sum of per-channel power,
then dB) in a long analysis window (fine frequency), around one v1 box with
some time and frequency padding.

Only the displayed band is computed: the snippet is Fourier transformed
once, the band is cut out and shifted to zero, and an inverse FFT of the
band gives a complex baseband signal at a decimated rate. Short-time spectra
of that signal are the in-band part of the full-rate spectrogram (same
window in seconds) at a fraction of the cost.

Window and hop are durations, so a cutout looks the same at any sample
rate. At 20 kHz the defaults are the former 4096-sample window and 32-sample
hop, decimated exactly as before.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.fft as sfft
from gridtools.io import Recording, Tracks
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal.windows import hann

from chirpdetector.labeling.queue import QueueItem

__all__ = [
    "BandSpectrogram",
    "Cutout",
    "CutoutConfig",
    "band_spectrogram",
    "live_baselines",
    "render_cutout",
    "zoom_baseband",
]


@dataclass(frozen=True)
class CutoutConfig:
    """How to render a cutout.

    Parameters
    ----------
    pad_s
        Time shown before the box start and after the box end.
    pad_hz
        Frequency shown below the lowest live track (or box bottom) and
        above the highest live track (or box top).
    window_s
        Analysis window (0.2048 s: 4096 samples at 20 kHz, 4.9 Hz
        resolution). Shorter windows show a chirp's shape better but
        separate close baselines worse.
    hop_s
        Hop between spectra (0.0016 s: 32 samples at 20 kHz).
    fft_pad
        Zero-padding factor of the FFT (smoother display only).
    db_range
        Colour range below the cutout's maximum.
    max_gap_s
        Track liveness gap (as in the queue).
    """

    pad_s: float = 0.15
    pad_hz: float = 40.0
    window_s: float = 0.2048
    hop_s: float = 0.0016
    fft_pad: int = 2
    db_range: float = 45.0
    max_gap_s: float = 1.0

    def __post_init__(self) -> None:
        if not (self.window_s > 0 and self.hop_s > 0) or self.fft_pad < 1:
            msg = "window_s and hop_s must be > 0 (seconds), fft_pad >= 1"
            raise ValueError(msg)
        if self.window_s > 10.0:
            msg = f"window_s is {self.window_s:g} s; is this a sample count?"
            raise ValueError(msg)
        if self.pad_s < 0 or self.pad_hz < 0 or self.db_range <= 0:
            msg = "pad_s and pad_hz must be >= 0, db_range > 0"
            raise ValueError(msg)


@dataclass(frozen=True)
class Cutout:
    """A rendered cutout.

    Parameters
    ----------
    item
        The queue item (which never holds v1's emitter).
    db
        Power in dB, shape ``(n_freq, n_time)``, row 0 the lowest frequency.
    extent
        ``(t0_s, t1_s, f0_hz, f1_hz)`` of the pixel edges, in absolute
        recording time; pass to ``imshow(..., origin="lower")``.
    window
        The requested ``(t0_s, t1_s, f0_hz, f1_hz)``; ``extent`` matches it
        to within one pixel.
    clim
        Colour limits in dB.
    baselines
        ``(track_id, frequency_hz)`` of every track live at the detection
        centre, sorted by frequency.
    render_ms
        Time it took to render.
    """

    item: QueueItem
    db: npt.NDArray[np.float32]
    extent: tuple[float, float, float, float]
    window: tuple[float, float, float, float]
    clim: tuple[float, float]
    baselines: tuple[tuple[int, float], ...]
    render_ms: float


def zoom_baseband(
    x: npt.NDArray[np.floating],
    samplerate_hz: float,
    f_lo_hz: float,
    f_hi_hz: float,
    decimation: int,
) -> tuple[npt.NDArray[np.complex64], float, float]:
    """Complex baseband of a band of a real multi-channel signal.

    Parameters
    ----------
    x
        Real signal, shape ``(n, channels)``.
    samplerate_hz
        Its sampling rate.
    f_lo_hz, f_hi_hz
        Band to keep.
    decimation
        Integer decimation factor; ``samplerate_hz / decimation`` must
        exceed the band width.

    Returns
    -------
    y
        Shape ``(ceil(n / decimation), channels)``; sample ``m`` is at the
        time of input sample ``m * decimation``. A sinusoid of amplitude A
        at frequency f in the band becomes ``A * exp(2j pi (f - fc) t)``.
    rate_hz
        Sampling rate of ``y``.
    fc_hz
        Frequency shifted to zero.

    Raises
    ------
    ValueError
        If the band does not fit the decimated rate.
    """
    n = x.shape[0]
    n_fft = decimation * math.ceil(n / decimation)
    m = n_fft // decimation
    df = samplerate_hz / n_fft
    k_lo = max(0, math.floor(f_lo_hz / df))
    k_hi = min(n_fft // 2, math.ceil(f_hi_hz / df))
    if k_hi - k_lo >= m:
        msg = (
            f"band {f_lo_hz:.1f}-{f_hi_hz:.1f} Hz does not fit a decimation "
            f"of {decimation} at {samplerate_hz:g} Hz"
        )
        raise ValueError(msg)
    k_c = round(0.5 * (k_lo + k_hi))
    spectrum = sfft.rfft(x, n=n_fft, axis=0)
    band = np.zeros((m, x.shape[1]), dtype=np.complex64)
    k = np.arange(k_lo, k_hi + 1)
    band[(k - k_c) % m] = spectrum[k]
    y = np.asarray(sfft.ifft(band, axis=0)) * np.float32(2.0 * m / n_fft)
    return y.astype(np.complex64, copy=False), samplerate_hz / decimation, k_c * df


def _decimation(nfft: int, hop: int, samplerate_hz: float, bandwidth_hz: float) -> int:
    """Largest divisor of nfft and hop whose rate still holds the band."""
    common = math.gcd(nfft, hop)
    fits = [
        d
        for d in range(1, common + 1)
        if common % d == 0 and samplerate_hz / d >= 1.25 * bandwidth_hz
    ]
    return max(fits) if fits else 1


def _sampling(
    window_s: float, hop_s: float, samplerate_hz: float, bandwidth_hz: float
) -> tuple[int, int, int]:
    """Decimation and window/hop lengths at the decimated rate.

    Returns ``(q, n_window, n_hop)``: the snippet is decimated by ``q``,
    spectra use ``n_window`` samples every ``n_hop`` samples at
    ``samplerate_hz / q``.

    The window and hop are first rounded to samples at the recording rate.
    If a common divisor of both gives at least half the largest decimation
    the band allows, it is used and the lengths are exact (always the case
    at 20 kHz with the defaults: 4096 and 32 samples). Otherwise (e.g. at
    48 kHz, 9830 and 77 samples have no common divisor) the snippet is
    decimated by the largest factor the band and the hop allow, and window
    and hop are rounded at the decimated rate; they then differ from the
    requested durations by at most half a decimated sample, which is
    invisible in a display.
    """
    n_window = max(2, round(window_s * samplerate_hz))
    n_hop = max(1, round(hop_s * samplerate_hz))
    q_band = max(1, math.floor(samplerate_hz / (1.25 * bandwidth_hz)))
    q_best = max(1, min(q_band, n_hop))
    q_exact = _decimation(n_window, n_hop, samplerate_hz, bandwidth_hz)
    if 2 * q_exact >= q_best:
        return q_exact, n_window // q_exact, n_hop // q_exact
    rate = samplerate_hz / q_best
    return q_best, max(2, round(window_s * rate)), max(1, round(hop_s * rate))


def _read_padded(rec: Recording, start: int, stop: int) -> npt.NDArray[np.float32]:
    """Samples ``[start, stop)``, zero outside the recording."""
    out = np.zeros((stop - start, rec.n_channels), dtype=np.float32)
    a, b = max(start, 0), min(stop, rec.n_samples)
    if b > a:
        out[a - start : b - start] = rec.read(a, b)
    return out


def live_baselines(
    tracks: Tracks, t_s: float, max_gap_s: float
) -> tuple[tuple[int, float], ...]:
    """``(track_id, frequency_hz)`` of the tracks live at ``t_s``, by frequency."""
    live = tracks.frequencies_at(t_s, max_gap_s)
    return tuple(
        sorted(((int(i), float(f)) for i, f in live.items()), key=lambda p: p[1])
    )


@dataclass(frozen=True)
class BandSpectrogram:
    """Electrode-summed power of one time-frequency window.

    Parameters
    ----------
    db
        Power in dB, shape ``(n_freq, n_time)``, row 0 the lowest frequency.
    extent
        ``(t0_s, t1_s, f0_hz, f1_hz)`` of the pixel *edges*: column ``j``
        spans ``t0 + j dt`` to ``t0 + (j + 1) dt`` and row ``i`` spans
        ``f0 + i df`` to ``f0 + (i + 1) df``.
    """

    db: npt.NDArray[np.float32]
    extent: tuple[float, float, float, float]


def band_spectrogram(
    rec: Recording,
    window: tuple[float, float, float, float],
    *,
    window_s: float,
    hop_s: float,
    fft_pad: int,
) -> BandSpectrogram:
    """Electrode-summed spectrogram of ``window = (t0_s, t1_s, f0_hz, f1_hz)``.

    The spectra are those of the full-rate recording (same window in
    seconds), computed on a decimated complex baseband of the band only
    (:func:`zoom_baseband`). Frames whose centre lies in ``[t0, t1]`` and
    bins whose centre lies in ``[f0, f1]`` are kept, so the extent matches
    the requested window to within one pixel.
    """
    t0, t1, f0, f1 = window
    fs = rec.samplerate_hz
    # compute a slightly wider band so window leakage at the edges is intact
    guard_hz = 4.0 * fs / max(2, round(window_s * fs))
    band_lo, band_hi = max(0.0, f0 - guard_hz), min(0.5 * fs, f1 + guard_hz)
    q, nwin, hop = _sampling(window_s, hop_s, fs, band_hi - band_lo)
    half = nwin * q // 2
    start = (round(t0 * fs) - half - hop * q) // q * q
    stop = round(t1 * fs) + half + hop * q
    x = _read_padded(rec, start, stop)
    y, fs_d, fc = zoom_baseband(x, fs, band_lo, band_hi, q)

    taper = hann(nwin, sym=False).astype(np.float32)
    taper /= taper.sum()
    frames = sliding_window_view(y, nwin, axis=0)[::hop]
    t = start / fs + (np.arange(frames.shape[0]) * hop + nwin / 2) / fs_d
    f = fc + sfft.fftshift(sfft.fftfreq(nwin * fft_pad)) * fs_d
    tsel = np.flatnonzero((t >= t0) & (t <= t1))
    fsel = np.flatnonzero((f >= f0) & (f <= f1))
    # transform only the frames shown; sum channels before shifting
    spectra = sfft.fft(
        frames[tsel[0] : tsel[-1] + 1] * taper, n=nwin * fft_pad, axis=-1
    )
    spectra = np.asarray(spectra, dtype=np.complex64)
    power = np.fft.fftshift((np.abs(spectra) ** 2).sum(axis=1), axes=-1)
    summed = power[:, fsel[0] : fsel[-1] + 1].T
    db = (10.0 * np.log10(summed + np.float32(1e-30))).astype(np.float32)
    dt, df = hop / fs_d, fs_d / (nwin * fft_pad)
    extent = (
        float(t[tsel[0]] - dt / 2),
        float(t[tsel[-1]] + dt / 2),
        float(f[fsel[0]] - df / 2),
        float(f[fsel[-1]] + df / 2),
    )
    return BandSpectrogram(db=db, extent=extent)


def render_cutout(
    rec: Recording, tracks: Tracks, item: QueueItem, config: CutoutConfig
) -> Cutout:
    """Electrode-summed spectrogram around one detection.

    Parameters
    ----------
    rec
        The trial's recording.
    tracks
        The trial's tracks.
    item
        The detection.
    config
        Rendering parameters.

    Returns
    -------
    The cutout. Time window: the box padded by ``pad_s``. Frequency window:
    from the lower of the lowest live baseline and the box bottom, minus
    ``pad_hz``, to the higher of the highest live baseline and the box top,
    plus ``pad_hz``.
    """
    tic = time.perf_counter()
    fs = rec.samplerate_hz
    baselines = live_baselines(tracks, item.t_center_s, config.max_gap_s)
    freqs = [f for _, f in baselines]
    f0 = max(0.0, min([item.f_low_hz, *freqs]) - config.pad_hz)
    f1 = min(0.5 * fs, max([item.f_high_hz, *freqs]) + config.pad_hz)
    t0 = item.t_start_s - config.pad_s
    t1 = item.t_end_s + config.pad_s
    spec = band_spectrogram(
        rec,
        (t0, t1, f0, f1),
        window_s=config.window_s,
        hop_s=config.hop_s,
        fft_pad=config.fft_pad,
    )
    vmax = float(spec.db.max())
    return Cutout(
        item=item,
        db=spec.db,
        extent=spec.extent,
        window=(t0, t1, f0, f1),
        clim=(vmax - config.db_range, vmax),
        baselines=baselines,
        render_ms=1e3 * (time.perf_counter() - tic),
    )
