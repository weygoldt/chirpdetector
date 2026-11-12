"""Configuration of the spectrogram front end, in physical units.

The configuration is stored inside model bundles, so it is frozen, validated
on construction and round-trips through JSON (`to_dict` / `from_dict`). It
holds no sample rate: windows and hop are durations, the frequency grid is a
resolution in Hz, and a bundle trained at one rate applies at any other.
`SpectrogramConfig.layout` turns it into sample counts for the rate of the
recording at hand (`StftLayout`), validating it for that rate.

From seconds to samples (rate ``fs``)
-------------------------------------
* FFT length ``n_fft = round(fs * reference_window_s)``, i.e.
  ``round(fs / df_hz)``: the exact length, not an FFT-friendly one. The grid
  spacing ``fs / n_fft`` then deviates from ``df_hz`` by at most half a
  sample's worth (relative ``0.5 / n_fft``); at 1500 Hz that is <= 0.07 Hz
  for 20-96 kHz. ``scipy.fft.next_fast_len`` would shift the 1500 Hz bin by
  up to 30 Hz (6 bins) at 22.05, 44.1 and 48 kHz, for at most a 1.8x faster
  FFT at those rates.
* Window lengths: the even integer nearest ``fs * duration`` (and at most
  ``n_fft``). Even lengths put the periodic window's peak exactly on the
  frame's centre sample, so frame times are exact; the cost is a deviation
  of up to one sample from ``round(fs * duration)``.
* Hop ``hop_samples = round(fs * hop_s)``; frame ``k`` is centred on sample
  ``k * hop_samples`` with time ``k * hop_samples / fs`` (exact; the actual
  hop differs from ``hop_s`` by at most half a sample).
* Frequency bins: indices ``k`` with ``k * df_hz`` in ``[f_min_hz,
  f_max_hz]``, the same indices at every rate. Their actual frequencies are
  ``k * fs / n_fft``, within the deviation above of ``k * df_hz``, so the
  image height does not depend on the rate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, TypeGuard, cast

import numpy as np
from numpy.typing import NDArray

WindowName = Literal["hann", "hamming", "blackman"]
Aggregation = Literal["sum", "max"]

WINDOW_NAMES: tuple[WindowName, ...] = ("hann", "hamming", "blackman")
AGGREGATIONS: tuple[Aggregation, ...] = ("sum", "max")

# Tolerance (in bins) when deciding whether a crop edge falls on a bin.
_BIN_EDGE_TOLERANCE = 1e-9

# Longest plausible analysis window. Longer values are almost certainly a
# sample count passed where seconds are expected (e.g. 4096 for 0.2048 s).
_MAX_WINDOW_S = 10.0

# Keys of the sample-count configuration (chirpdetector 2.0.0.dev before
# physical units). Bundles that carry them were trained at one fixed rate.
_SAMPLE_COUNT_KEYS = {
    "samplerate_hz": "the rate now comes from the recording",
    "window_lengths_samples": (
        "use reference_window_s and extra_windows_s (seconds): "
        "window_lengths_samples / samplerate_hz, e.g. (4096, 1024, 256) at "
        "20 kHz -> reference_window_s=0.2048, extra_windows_s=(0.0512, 0.0128)"
    ),
    "hop_samples": "use hop_s (seconds): hop_samples / samplerate_hz, e.g. 100 -> 0.005",
}


@dataclass(frozen=True, slots=True)
class StftLayout:
    """A `SpectrogramConfig` in samples, for one sample rate.

    Build it with `SpectrogramConfig.layout`, which validates the config
    for the rate (see the module docstring for the rounding rules).

    Attributes
    ----------
    samplerate_hz
        Sample rate of the recording.
    window_lengths_samples
        Window lengths, one per channel, the reference window first; all
        even.
    n_fft
        FFT length of every channel.
    hop_samples
        Distance between frame centres.
    freq_bin_start, freq_bin_stop
        FFT bins ``[start, stop)`` kept (same at every rate).
    """

    samplerate_hz: float
    window_lengths_samples: tuple[int, ...]
    n_fft: int
    hop_samples: int
    freq_bin_start: int
    freq_bin_stop: int

    @property
    def n_windows(self) -> int:
        """Number of channels."""
        return len(self.window_lengths_samples)

    @property
    def hop_s(self) -> float:
        """Actual distance between frame centres in seconds."""
        return self.hop_samples / self.samplerate_hz

    @property
    def df_hz(self) -> float:
        """Actual spacing of the frequency grid."""
        return self.samplerate_hz / self.n_fft

    @property
    def n_freqs(self) -> int:
        """Number of frequency bins."""
        return self.freq_bin_stop - self.freq_bin_start

    @property
    def freqs_hz(self) -> NDArray[np.float64]:
        """Actual bin centre frequencies, ascending."""
        bins = np.arange(self.freq_bin_start, self.freq_bin_stop, dtype=np.float64)
        return bins * self.samplerate_hz / self.n_fft

    @property
    def context_samples(self) -> int:
        """Samples needed on each side of a frame centre (``n_fft // 2``)."""
        return self.n_fft // 2


@dataclass(frozen=True, slots=True)
class SpectrogramConfig:
    """Parameters of the multi-resolution spectrogram, in seconds and Hz.

    Parameters
    ----------
    reference_window_s
        Duration of the reference window, channel 0. It sets the frequency
        resolution of every channel, ``df_hz = 1 / reference_window_s``
        (0.2048 s: 4.88 Hz), and the normalisation reference is measured on
        it. The long window is kept because its frequency resolution
        separates fish with close EOD frequencies.
    extra_windows_s
        Durations of additional, shorter windows, one channel each, in this
        order. They only add time resolution: each is zero-padded to the
        reference FFT length, so all channels share one frequency grid.
        Empty for a single-window spectrogram.
    window
        Window function, applied in its periodic form.
    hop_s
        Distance between frame centres, shared by all channels. Must be at
        most half the shortest window, so that every sample lies within a
        quarter window of some frame centre (Hann weight >= 0.5).
    f_min_hz, f_max_hz
        Frequency crop, inclusive: bins ``k`` with ``k * df_hz`` inside it.
        The default, 500-1500 Hz, is the band of the v1 production run: EOD
        frequencies of A. leptorhynchus and their chirps, without most
        second harmonics.
    aggregation
        How per-electrode power is combined: ``"sum"`` (total power) or
        ``"max"`` (loudest electrode). Always applied to linear power, never
        to dB.
    dynamic_range_db
        Width of the dB interval mapped to [0, 1] by `normalize`.
    headroom_db
        How far above the reference level the top of that interval lies.
        The interval is ``[ref - (dynamic_range_db - headroom_db),
        ref + headroom_db]``.
    """

    reference_window_s: float = 0.2048
    extra_windows_s: tuple[float, ...] = (0.0512, 0.0128)
    window: WindowName = "hann"
    hop_s: float = 0.005
    f_min_hz: float = 500.0
    f_max_hz: float = 1500.0
    aggregation: Aggregation = "sum"
    dynamic_range_db: float = 60.0
    headroom_db: float = 10.0

    def __post_init__(self) -> None:
        _canonicalize(self)
        _validate(self)

    # ------------------------------------------------------------------ derived

    @property
    def window_durations_s(self) -> tuple[float, ...]:
        """All window durations, one per channel, the reference first."""
        return (self.reference_window_s, *self.extra_windows_s)

    @property
    def n_windows(self) -> int:
        """Number of channels."""
        return 1 + len(self.extra_windows_s)

    @property
    def df_hz(self) -> float:
        """Target spacing of the frequency grid, ``1 / reference_window_s``."""
        return 1.0 / self.reference_window_s

    @property
    def freq_bin_start(self) -> int:
        """Index of the lowest kept FFT bin (inclusive)."""
        return math.ceil(self.f_min_hz / self.df_hz - _BIN_EDGE_TOLERANCE)

    @property
    def freq_bin_stop(self) -> int:
        """Index one past the highest kept FFT bin (exclusive)."""
        return math.floor(self.f_max_hz / self.df_hz + _BIN_EDGE_TOLERANCE) + 1

    @property
    def n_freqs(self) -> int:
        """Number of frequency bins (the same at every rate)."""
        return self.freq_bin_stop - self.freq_bin_start

    @property
    def nominal_freqs_hz(self) -> NDArray[np.float64]:
        """Target bin frequencies ``k * df_hz``, ascending.

        The actual frequencies at a given rate are `StftLayout.freqs_hz`.
        """
        bins = np.arange(self.freq_bin_start, self.freq_bin_stop, dtype=np.float64)
        return bins * self.df_hz

    def layout(self, samplerate_hz: float) -> StftLayout:
        """Sample counts of this config at ``samplerate_hz``.

        Parameters
        ----------
        samplerate_hz
            Sample rate of the recording.

        Returns
        -------
        StftLayout
            FFT length, window lengths, hop and kept bins at that rate.

        Raises
        ------
        ValueError
            If the rate is not positive and finite, the band reaches above
            its Nyquist frequency, or a window or the hop is too short to be
            sampled (fewer than 2 samples per window, a zero hop, windows
            that coincide after rounding, or a hop above half the shortest
            window).
        """
        fs = float(samplerate_hz)
        if not (math.isfinite(fs) and fs > 0):
            msg = f"samplerate_hz must be positive and finite, got {samplerate_hz!r}"
            raise ValueError(msg)
        if self.f_max_hz > fs / 2:
            msg = (
                f"band {self.f_min_hz:g}-{self.f_max_hz:g} Hz reaches above the "
                f"Nyquist frequency ({fs / 2:g} Hz) of a {fs:g} Hz recording"
            )
            raise ValueError(msg)
        n_fft = round(fs * self.reference_window_s)
        largest_even = 2 * (n_fft // 2)
        lengths = tuple(
            min(2 * round(fs * duration / 2), largest_even)
            for duration in self.window_durations_s
        )
        if min(lengths) < 2:
            msg = (
                f"at {fs:g} Hz the shortest window "
                f"({min(self.window_durations_s):g} s) has fewer than 2 samples"
            )
            raise ValueError(msg)
        if len(set(lengths)) != len(lengths):
            msg = (
                f"at {fs:g} Hz windows {self.window_durations_s} s coincide: {lengths}"
            )
            raise ValueError(msg)
        hop = round(fs * self.hop_s)
        if not 1 <= hop <= min(lengths) // 2:
            msg = (
                f"at {fs:g} Hz the hop is {hop} samples; it must be between 1 "
                f"and half the shortest window ({min(lengths)} samples)"
            )
            raise ValueError(msg)
        if self.freq_bin_stop - 1 > n_fft // 2:  # guarded by the Nyquist check
            msg = f"bin {self.freq_bin_stop - 1} is above Nyquist at {fs:g} Hz"
            raise ValueError(msg)
        return StftLayout(
            samplerate_hz=fs,
            window_lengths_samples=lengths,
            n_fft=n_fft,
            hop_samples=hop,
            freq_bin_start=self.freq_bin_start,
            freq_bin_stop=self.freq_bin_stop,
        )

    # ---------------------------------------------------------------- JSON I/O

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of all fields."""
        out = asdict(self)
        out["extra_windows_s"] = list(self.extra_windows_s)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpectrogramConfig:
        """Build a config from `to_dict` output (or parsed JSON).

        Missing keys take their defaults. Unknown keys raise, so that a
        bundle written by a newer version is not silently misread; so do
        the sample-count keys of earlier development versions, with the
        conversion to seconds in the message.
        """
        old = sorted(set(data) & set(_SAMPLE_COUNT_KEYS))
        if old:
            hints = "; ".join(f"{key}: {_SAMPLE_COUNT_KEYS[key]}" for key in old)
            msg = (
                f"SpectrogramConfig is in seconds and Hz and has no sample rate; "
                f"got sample-count keys {old} ({hints})"
            )
            raise ValueError(msg)
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            msg = f"unknown SpectrogramConfig keys: {sorted(unknown)}"
            raise ValueError(msg)
        return cls(**data)


_FLOAT_FIELDS = (
    "reference_window_s",
    "hop_s",
    "f_min_hz",
    "f_max_hz",
    "dynamic_range_db",
    "headroom_db",
)


def _canonicalize(cfg: SpectrogramConfig) -> None:
    """Coerce numeric fields to plain ``float`` and the extras to a tuple.

    Keeps equality, hashing and JSON output independent of whether a value
    arrived as ``1``, ``1.0``, ``np.float32(1)`` or a JSON list. Values that
    cannot be coerced are left alone for `_validate` to reject.
    """
    for name in _FLOAT_FIELDS:
        value = getattr(cfg, name)
        if _is_real(value):
            object.__setattr__(cfg, name, float(value))
    extras: object = cfg.extra_windows_s
    if isinstance(extras, list | tuple):
        items = cast("list[object] | tuple[object, ...]", extras)
        object.__setattr__(
            cfg,
            "extra_windows_s",
            tuple(float(d) if _is_real(d) else d for d in items),
        )


def _is_real(
    value: object,
) -> TypeGuard[int | float | np.integer | np.floating]:
    return (
        isinstance(value, int | float | np.integer | np.floating)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate(cfg: SpectrogramConfig) -> None:
    if not _is_real(cfg.reference_window_s) or cfg.reference_window_s <= 0:
        msg = (
            "reference_window_s must be a positive duration in seconds, "
            f"got {cfg.reference_window_s!r}"
        )
        raise ValueError(msg)
    if cfg.reference_window_s > _MAX_WINDOW_S:
        msg = (
            f"reference_window_s is {cfg.reference_window_s:g} s; durations are in "
            f"seconds (at most {_MAX_WINDOW_S:g} s), is this a sample count? "
            "4096 samples at 20 kHz are 0.2048 s"
        )
        raise ValueError(msg)
    extras = cfg.extra_windows_s
    if not isinstance(extras, tuple) or not all(_is_real(d) and d > 0 for d in extras):
        msg = f"extra_windows_s must be positive durations in seconds, got {extras!r}"
        raise ValueError(msg)
    if any(d >= cfg.reference_window_s for d in extras):
        msg = (
            "extra windows must be shorter than the reference window "
            f"({cfg.reference_window_s} s), got {extras}"
        )
        raise ValueError(msg)
    if len(set(extras)) != len(extras):
        msg = f"extra windows must be distinct, got {extras}"
        raise ValueError(msg)

    if cfg.window not in WINDOW_NAMES:
        msg = f"window must be one of {WINDOW_NAMES}, got {cfg.window!r}"
        raise ValueError(msg)

    shortest_s = min(cfg.window_durations_s)
    if not _is_real(cfg.hop_s) or not 0 < cfg.hop_s <= shortest_s / 2:
        msg = (
            f"hop_s ({cfg.hop_s!r}) must be positive and at most half the "
            f"shortest window ({shortest_s} s); larger hops leave samples "
            "outside the high-weight part of every short window"
        )
        raise ValueError(msg)

    if not (_is_real(cfg.f_min_hz) and _is_real(cfg.f_max_hz)):
        msg = "f_min_hz and f_max_hz must be finite numbers"
        raise ValueError(msg)
    if not 0 <= cfg.f_min_hz < cfg.f_max_hz:
        msg = f"need 0 <= f_min_hz < f_max_hz, got [{cfg.f_min_hz}, {cfg.f_max_hz}]"
        raise ValueError(msg)
    if cfg.n_freqs < 1:
        msg = (
            f"[{cfg.f_min_hz}, {cfg.f_max_hz}] Hz contains no bin of the "
            f"{cfg.df_hz:.4g} Hz grid"
        )
        raise ValueError(msg)

    if cfg.aggregation not in AGGREGATIONS:
        msg = f"aggregation must be one of {AGGREGATIONS}, got {cfg.aggregation!r}"
        raise ValueError(msg)

    if not _is_real(cfg.dynamic_range_db) or cfg.dynamic_range_db <= 0:
        msg = f"dynamic_range_db must be positive, got {cfg.dynamic_range_db!r}"
        raise ValueError(msg)
    if not _is_real(cfg.headroom_db) or not (
        0 <= cfg.headroom_db < cfg.dynamic_range_db
    ):
        msg = (
            "headroom_db must satisfy 0 <= headroom_db < dynamic_range_db, "
            f"got {cfg.headroom_db!r}"
        )
        raise ValueError(msg)
