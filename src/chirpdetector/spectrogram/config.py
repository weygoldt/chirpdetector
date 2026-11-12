"""Configuration of the spectrogram front end.

The configuration is stored inside model bundles, so it is frozen, validated
on construction and round-trips through JSON (`to_dict` / `from_dict`).
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


@dataclass(frozen=True, slots=True)
class SpectrogramConfig:
    """Parameters of the multi-resolution spectrogram.

    Parameters
    ----------
    samplerate_hz
        Sampling rate of the recording.
    window_lengths_samples
        Analysis window lengths, one image channel each. The first entry is
        the reference window: it must be the longest, it sets the FFT length
        and therefore the frequency grid of every channel, and it is the
        channel the normalisation reference is measured on. The long window
        is kept because its frequency resolution separates fish with close
        EOD frequencies; shorter windows only add time resolution as extra
        channels. A single window is valid. All lengths must be even (so that
        every window is centred exactly on its frame's sample position) and
        distinct.
    window
        Window function, applied in its periodic form.
    hop_samples
        Distance between frame centres, shared by all channels. Must be at
        most half the shortest window, so that every sample lies within a
        quarter window of some frame centre (Hann weight >= 0.5).
    f_min_hz, f_max_hz
        Frequency crop, inclusive on both ends: the grid contains the
        reference window's bins ``k * samplerate_hz / n_fft`` that fall
        inside ``[f_min_hz, f_max_hz]``.
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

    samplerate_hz: float = 20_000.0
    window_lengths_samples: tuple[int, ...] = (4096, 1024, 256)
    window: WindowName = "hann"
    hop_samples: int = 100
    f_min_hz: float = 400.0
    f_max_hz: float = 2000.0
    aggregation: Aggregation = "sum"
    dynamic_range_db: float = 60.0
    headroom_db: float = 10.0

    def __post_init__(self) -> None:
        _canonicalize(self)
        _validate(self)

    # ------------------------------------------------------------------ derived

    @property
    def n_windows(self) -> int:
        """Number of channels (one per window length)."""
        return len(self.window_lengths_samples)

    @property
    def n_fft(self) -> int:
        """FFT length of every channel: the reference window length."""
        return self.window_lengths_samples[0]

    @property
    def hop_s(self) -> float:
        """Distance between frame centres in seconds."""
        return self.hop_samples / self.samplerate_hz

    @property
    def df_hz(self) -> float:
        """Spacing of the frequency grid."""
        return self.samplerate_hz / self.n_fft

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
        """Number of frequency bins after cropping."""
        return self.freq_bin_stop - self.freq_bin_start

    @property
    def freqs_hz(self) -> NDArray[np.float64]:
        """Bin centre frequencies of the cropped grid, ascending."""
        bins = np.arange(self.freq_bin_start, self.freq_bin_stop, dtype=np.float64)
        return bins * self.df_hz

    # ---------------------------------------------------------------- JSON I/O

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of all fields."""
        out = asdict(self)
        out["window_lengths_samples"] = list(self.window_lengths_samples)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpectrogramConfig:
        """Build a config from `to_dict` output (or parsed JSON).

        Missing keys take their defaults; unknown keys raise, so that a
        bundle written by a newer version is not silently misread.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            msg = f"unknown SpectrogramConfig keys: {sorted(unknown)}"
            raise ValueError(msg)
        return cls(**data)


_FLOAT_FIELDS = (
    "samplerate_hz",
    "f_min_hz",
    "f_max_hz",
    "dynamic_range_db",
    "headroom_db",
)


def _canonicalize(cfg: SpectrogramConfig) -> None:
    """Coerce numeric fields to plain ``float`` / ``int`` / ``tuple``.

    Keeps equality, hashing and JSON output independent of whether a value
    arrived as ``20000``, ``20000.0``, ``np.float32(2e4)`` or a JSON list.
    Values that cannot be coerced are left alone for `_validate` to reject.
    """
    for name in _FLOAT_FIELDS:
        value = getattr(cfg, name)
        if _is_real(value):
            object.__setattr__(cfg, name, float(value))
    lengths: object = cfg.window_lengths_samples
    if isinstance(lengths, list | tuple):
        items = cast("list[object] | tuple[object, ...]", lengths)
        object.__setattr__(
            cfg,
            "window_lengths_samples",
            tuple(int(n) if _is_int(n) else n for n in items),
        )
    if _is_int(cfg.hop_samples):
        object.__setattr__(cfg, "hop_samples", int(cfg.hop_samples))


def _is_int(value: object) -> TypeGuard[int | np.integer]:
    return isinstance(value, int | np.integer) and not isinstance(value, bool)


def _is_real(
    value: object,
) -> TypeGuard[int | float | np.integer | np.floating]:
    return (
        isinstance(value, int | float | np.integer | np.floating)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate(cfg: SpectrogramConfig) -> None:
    if not _is_real(cfg.samplerate_hz) or cfg.samplerate_hz <= 0:
        msg = (
            f"samplerate_hz must be a positive finite number, got {cfg.samplerate_hz!r}"
        )
        raise ValueError(msg)

    lengths = cfg.window_lengths_samples
    if not isinstance(lengths, tuple) or len(lengths) == 0:
        msg = "window_lengths_samples must be a non-empty tuple of integers"
        raise ValueError(msg)
    if not all(_is_int(n) for n in lengths):
        msg = f"window lengths must be integers, got {lengths!r}"
        raise ValueError(msg)
    if any(n < 2 or n % 2 for n in lengths):
        msg = f"window lengths must be even and >= 2, got {lengths!r}"
        raise ValueError(msg)
    if len(set(lengths)) != len(lengths):
        msg = f"window lengths must be distinct, got {lengths!r}"
        raise ValueError(msg)
    if lengths[0] != max(lengths):
        msg = (
            "the first window is the reference window and must be the longest, "
            f"got {lengths!r}"
        )
        raise ValueError(msg)

    if cfg.window not in WINDOW_NAMES:
        msg = f"window must be one of {WINDOW_NAMES}, got {cfg.window!r}"
        raise ValueError(msg)

    if not _is_int(cfg.hop_samples) or cfg.hop_samples < 1:
        msg = f"hop_samples must be a positive integer, got {cfg.hop_samples!r}"
        raise ValueError(msg)
    shortest = min(lengths)
    if cfg.hop_samples > shortest // 2:
        msg = (
            f"hop_samples ({cfg.hop_samples}) must be at most half the shortest "
            f"window ({shortest}); larger hops leave samples outside the "
            "high-weight part of every short window"
        )
        raise ValueError(msg)

    nyquist_hz = cfg.samplerate_hz / 2
    if not (_is_real(cfg.f_min_hz) and _is_real(cfg.f_max_hz)):
        msg = "f_min_hz and f_max_hz must be finite numbers"
        raise ValueError(msg)
    if not 0 <= cfg.f_min_hz < cfg.f_max_hz <= nyquist_hz:
        msg = (
            "need 0 <= f_min_hz < f_max_hz <= Nyquist "
            f"({nyquist_hz} Hz), got [{cfg.f_min_hz}, {cfg.f_max_hz}]"
        )
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
