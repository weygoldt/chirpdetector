"""Removal of clicks -- short, broadband impulses -- from grid recordings.

Origin
------
Ported from audian's ``audian.clicks`` (https://github.com/bendalab/audian,
fork at ``/home/weygoldt/wrk/tools/claudian``, commit 6fa5274 "Remove clicks
from the filtered trace, and so from the spectrogram", 2026-11-03), which is
itself a numpy port of ``wavetracker.clicks`` (wavetracker commit eaebd4d;
wavetracker by Till Raab and Joerg Henninger, maintained by Patrick Weygoldt
and Jan Benda; audian by Jan Benda, maintained by Patrick Weygoldt). Both
source projects are licensed under the GPL; this module keeps that origin and
is not relicensed by being copied here. Detection and repair, and their
defaults, are unchanged. What changed: a mandatory electrode-coincidence
guard (below), field names that carry units (``max_duration`` became
``max_duration_s`` and so on), a configuration that validates itself and
round-trips through JSON, and block scheduling expressed as spans of a
recording (`plan_clean`, `clean_span`, `read_clean`) to match
`chirpdetector.spectrogram`.

Why
---
Electrodes pick up clicks: a few milliseconds, up to four orders of
magnitude above the noise, on one or several electrodes. In a spectrogram a
click is a vertical stripe across all frequencies, and chirpdetector v1
often took them for chirps.

How
---
Each channel is split by its spectrum over a block:

- Clicks are searched in the top of the spectrum, above ``high_band`` of the
  sampling rate (4 kHz at 20 kHz), where a click reaches and a fish's
  narrowband EOD does not. Loud events longer than ``max_duration_s`` are
  signals and stay.
- Cutting the clicks out is not enough: a hole in a loud EOD is itself a
  click as loud as the EOD. What stands out of the spectral floor (EODs,
  their harmonics, hum) is set aside, the clicks are bridged by straight
  lines in the rest, and the tonal part is added back on top, scaled in each
  hole to what the samples around it hold of it.

Samples outside a click are returned exactly as they came in, and a channel
without a click is returned untouched.

The electrode-coincidence guard (not in audian) is mandatory
------------------------------------------------------------
On grid recordings the higher EOD harmonics of a fish next to an electrode
reach the click band during a big chirp, so audian's per-channel removal
repairs chirps (24 of 335 hand-labelled chirps touched, up to -11 dB). Real
clicks reach most electrodes (8-14 of 15), chirp-triggered events at most
`MAX_CHIRP_ELECTRODES` (4). An event is therefore repaired on all electrodes
only when it reaches at least ``coincidence`` of them (0.5 by default),
otherwise only where it is very strong (``isolated_threshold``). This only
works with enough electrodes: below `ClickRemoval.min_electrodes` (9 for the
default) a loud chirp can pass the coincidence test, and on a single channel
clicks and chirps cannot be told apart at all, so the functions refuse such
recordings (`TooFewElectrodesError`). Pipelines check first with
`can_remove_clicks` and skip click removal, with a warning, where it is
False.

Validation (2026-11-12): /mnt/data1/chirp_v2/preprocessing/clicks/report.md,
produced by ``scripts/click_removal_report.py`` at chirpdetector commit
b1b7d35 (the script compares the then-possible unguarded settings and was
removed with them).

Blocks
------
The recording is cleaned in blocks fixed in absolute time: block ``k``
covers samples ``[k * B, (k + 1) * B)`` with ``B = block_samples``, and is
computed from ``[k * B - M, (k + 1) * B + M)`` (``M = margin_samples``),
clipped to the recording. A sample therefore comes out the same however the
recording is chunked, as long as each chunk is read with the margins
`plan_clean` asks for. Typical use with the spectrogram::

    plan = plan_span(start, stop, recording.n_samples, spec_config)
    samples, mask = read_clean(recording, plan.read_start, plan.read_stop)
    spec = compute_spectrogram_span(samples, plan, spec_config)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from typing import Any, Protocol, TypeGuard

import numpy as np
import scipy.fft
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter1d, rank_filter

log = logging.getLogger(__name__)

FloatArray = NDArray[np.floating[Any]]
BoolArray = NDArray[np.bool_]

# Below this many samples a block is too short to tell anything, and to make
# periodic; it is returned as it is.
_MIN_SAMPLES = 64

#: Most electrodes a chirp-triggered event reached in the validation. The
#: 30 unguarded events inside the 23 labelled chirps (of 43) that triggered
#: in the two 2022 tube trials reached 1-4 of 11 electrodes; no chirp
#: triggered in the 2019 trials, and the two events inside 2020-06-03 chirp
#: boxes are clicks, on 6 and 14 of 15 electrodes. Over the whole 2022
#: recordings (2 x 4 h, 11 electrodes) the events on 5-7 electrodes are the
#: trials' periodic interference (97% with a partner exactly 1 s away) or
#: clicks; none of the 19 others shows a chirp. The guard needs
#: ``coincidence * n_electrodes`` to exceed it.
MAX_CHIRP_ELECTRODES = 4


#: Least ``isolated_threshold`` [robust sd of the high band] that left every
#: chirp intact in the validation (none of 335 labelled chirps touched; 24
#: per channel still repaired 12 of them). Setting ``isolated_threshold`` to
#: ``threshold`` would switch the guard off, so lower values are refused.
MIN_ISOLATED_THRESHOLD = 48.0


class TooFewElectrodesError(ValueError):
    """The recording has too few electrodes for guarded click removal."""


@dataclass(frozen=True, slots=True)
class ClickRemoval:
    """What counts as a click, and how it is repaired.

    The detection and repair defaults are wavetracker's ``ClickConfig``,
    measured there on field recordings of weakly electric fish and on lab
    recordings with known identities. The electrode-coincidence guard
    (``coincidence``, ``isolated_threshold``) is chirpdetector's and cannot
    be switched off; it needs `min_electrodes` electrodes. The configuration
    is frozen, validated on construction and round-trips through JSON
    (`to_dict` / `from_dict`), so it can be stored in model bundles.

    Parameters
    ----------
    threshold
        A click reaches this many robust standard deviations (median
        absolute value / 0.6745) of its channel's high band, which Gaussian
        noise never does.
    extent_threshold
        A click lasts for as long as it stays above this many.
    high_band
        Clicks are searched above this fraction of the sampling rate ...
    high_band_edge
        ... reached over this further fraction, with a raised-sine edge.
    max_duration_s
        Events longer than this are signals, not clicks.
    guard_s
        Also repaired on both sides of a click: its decay below the
        threshold.
    resolution_s
        Time resolution of the search (bin of the envelope).
    edge_s
        Not searched at the very ends of a block: the join of the periodic
        extension shows in the high band there.
    line_margin_db
        What stands this far above the spectral floor is tonal and is never
        repaired.
    line_width_hz
        Frequency resolution of the spectral floor.
    floor_width_hz
        Range the floor is the ``floor_quantile`` of.
    floor_quantile
        Low, because a loud tone with its harmonics can fill most of that
        range.
    local_window_s
        Window over which the tonal part in a hole is scaled to the samples
        around it.
    fade_s
        Fade that makes a block periodic for the spectral split.
    sample_stride
        Every n-th sample estimates the noise of the high band.
    block_s
        Length of the blocks fixed in absolute time.
    margin_s
        Data on each side of a block it is computed from; at least
        ``max_duration_s``, so that a long event touching a block is seen
        whole enough to be told from a click.
    coincidence
        Guard for chirps, not in audian, and mandatory: the fraction of the
        electrodes a click must reach (at ``threshold``, within one event)
        to be repaired on all of them, in [0.5, 1]. An event on fewer
        electrodes is repaired only where it reaches ``isolated_threshold``.
        The harmonics of a loud chirp reach the high band only on the
        electrodes next to the fish; static from the amplifier or the water
        reaches most of them. Higher values protect chirps on smaller grids
        (see `min_electrodes`) but leave more clicks.
    isolated_threshold
        What a click on fewer than ``coincidence`` of the electrodes must
        reach, in robust standard deviations of the high band; at least
        ``threshold`` and `MIN_ISOLATED_THRESHOLD` (48).
    """

    threshold: float = 12.0
    extent_threshold: float = 8.0
    high_band: float = 0.2
    high_band_edge: float = 0.05
    max_duration_s: float = 0.5
    guard_s: float = 0.002
    resolution_s: float = 0.001
    edge_s: float = 0.005
    line_margin_db: float = 15.0
    line_width_hz: float = 10.0
    floor_width_hz: float = 1000.0
    floor_quantile: float = 0.2
    local_window_s: float = 0.05
    fade_s: float = 0.5
    sample_stride: int = 16
    block_s: float = 2.0
    margin_s: float = 0.5
    coincidence: float = 0.5
    isolated_threshold: float = 48.0

    def __post_init__(self) -> None:
        _canonicalize(self)
        _validate(self)

    @property
    def min_electrodes(self) -> int:
        """Fewest electrodes the guard protects chirps on.

        An event must reach ``ceil(coincidence * n)`` electrodes to be
        repaired everywhere; this exceeds `MAX_CHIRP_ELECTRODES` from
        ``floor(MAX_CHIRP_ELECTRODES / coincidence) + 1`` electrodes on
        (9 for ``coincidence=0.5``, 5 for 1.0).
        """
        return math.floor(MAX_CHIRP_ELECTRODES / self.coincidence + 1e-9) + 1

    def block_samples(self, samplerate_hz: float) -> int:
        """Length of a block in samples (at least 64)."""
        return max(_MIN_SAMPLES, round(self.block_s * samplerate_hz))

    def margin_samples(self, samplerate_hz: float) -> int:
        """Samples on each side a block is computed from."""
        return max(0, round(self.margin_s * samplerate_hz))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of all fields."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClickRemoval:
        """Build a config from `to_dict` output (or parsed JSON).

        Missing keys take their defaults; unknown keys raise, so that a
        bundle written by a newer version is not silently misread.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            msg = f"unknown ClickRemoval keys: {sorted(unknown)}"
            raise ValueError(msg)
        return cls(**data)


def _is_real(value: object) -> TypeGuard[int | float | np.integer | np.floating]:
    return (
        isinstance(value, int | float | np.integer | np.floating)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _canonicalize(cfg: ClickRemoval) -> None:
    """Coerce numeric fields to plain ``float`` / ``int``."""
    for f in fields(cfg):
        value = getattr(cfg, f.name)
        if f.name == "sample_stride":
            if isinstance(value, int | np.integer) and not isinstance(value, bool):
                object.__setattr__(cfg, f.name, int(value))
        elif _is_real(value):
            object.__setattr__(cfg, f.name, float(value))


def _validate(cfg: ClickRemoval) -> None:
    for f in fields(cfg):
        if f.name != "sample_stride" and not _is_real(getattr(cfg, f.name)):
            msg = f"{f.name} must be a finite number, got {getattr(cfg, f.name)!r}"
            raise ValueError(msg)
    positive = (
        "threshold",
        "extent_threshold",
        "high_band",
        "high_band_edge",
        "max_duration_s",
        "resolution_s",
        "line_width_hz",
        "floor_width_hz",
        "local_window_s",
        "block_s",
    )
    for name in positive:
        if getattr(cfg, name) <= 0:
            msg = f"{name} must be positive, got {getattr(cfg, name)!r}"
            raise ValueError(msg)
    for name in ("guard_s", "edge_s", "line_margin_db", "fade_s", "margin_s"):
        if getattr(cfg, name) < 0:
            msg = f"{name} must be >= 0, got {getattr(cfg, name)!r}"
            raise ValueError(msg)
    if cfg.extent_threshold > cfg.threshold:
        msg = (
            f"extent_threshold ({cfg.extent_threshold}) must not exceed "
            f"threshold ({cfg.threshold})"
        )
        raise ValueError(msg)
    if cfg.high_band + cfg.high_band_edge > 0.5:
        msg = (
            "high_band + high_band_edge must be at most 0.5 (Nyquist), got "
            f"{cfg.high_band} + {cfg.high_band_edge}"
        )
        raise ValueError(msg)
    if not 0 < cfg.floor_quantile <= 1:
        msg = f"floor_quantile must lie in (0, 1], got {cfg.floor_quantile!r}"
        raise ValueError(msg)
    stride: object = cfg.sample_stride
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        msg = f"sample_stride must be a positive integer, got {stride!r}"
        raise ValueError(msg)
    if not 0.5 <= cfg.coincidence <= 1:
        msg = (
            f"coincidence must lie in [0.5, 1], got {cfg.coincidence!r}: the "
            "electrode-coincidence guard is mandatory, since without it loud "
            "chirps are repaired as clicks"
        )
        raise ValueError(msg)
    least_isolated = max(cfg.threshold, MIN_ISOLATED_THRESHOLD)
    if cfg.isolated_threshold < least_isolated:
        msg = (
            f"isolated_threshold ({cfg.isolated_threshold}) must be at least "
            f"{least_isolated} (threshold, and {MIN_ISOLATED_THRESHOLD}): a "
            "lower bar repairs loud chirps on the electrodes next to the fish"
        )
        raise ValueError(msg)
    if cfg.margin_s < cfg.max_duration_s:
        msg = (
            f"margin_s ({cfg.margin_s}) must be at least max_duration_s "
            f"({cfg.max_duration_s}), or a long event at a block edge can be "
            "mistaken for a click"
        )
        raise ValueError(msg)


# ------------------------------------------------------------- one channel


def spectral_floor(
    power: NDArray[np.float64], df_hz: float, config: ClickRemoval
) -> NDArray[np.float64]:
    """Smooth floor under a power spectrum ``power`` (bins,).

    A low quantile over a wide range, of medians over narrow ranges: a tone
    can fill most of the spectrum around it with its harmonics and its
    modulations, and the floor must stay below that however loud it is.
    """
    n_bins = len(power)
    chunk = min(max(8, round(config.line_width_hz / df_hz)), n_bins)
    n_chunks = n_bins // chunk
    medians = np.median(power[: n_chunks * chunk].reshape(n_chunks, chunk), axis=-1)
    width = max(3, round(config.floor_width_hz / (chunk * df_hz))) | 1
    rank = max(1, math.ceil(config.floor_quantile * width))
    # `nearest` is the replicate padding of the original
    floor = rank_filter(medians, rank - 1, size=width, mode="nearest")
    floor = np.repeat(floor, chunk)
    return np.pad(floor, (0, n_bins - len(floor)), mode="edge")


def _periodic(x: NDArray[np.float64], n_fft: int) -> NDArray[np.float64]:
    """``x`` extended to ``n_fft`` samples that lead smoothly back to its start.

    A sine that is not periodic in the transform spreads into skirts below
    any margin, which would ring from both ends of the block in the part
    that is repaired.
    """
    n = len(x)
    m = n_fft - n
    if m <= 0:
        return x
    # point reflections continue the value and the slope at both ends
    after = 2 * x[-1] - x[-(m + 1) : -1][::-1]
    before = 2 * x[0] - x[1 : m + 1][::-1]
    k = (np.arange(m) + 0.5) / m
    fade = np.cos(0.5 * np.pi * k) ** 2
    return np.concatenate([x, after * fade + before * (1 - fade)])


def _long_runs(hot: BoolArray, length: int) -> BoolArray:
    """True within runs of at least ``length`` consecutive True."""
    n = len(hot)
    if length > n:
        return np.zeros_like(hot)
    total = np.concatenate([[0], np.cumsum(hot, dtype=np.int64)])
    full = (total[length:] - total[:-length]) == length
    starts = np.concatenate([[0], np.cumsum(full, dtype=np.int64)])
    i = np.arange(n)
    lo = np.clip(i - length + 1, 0, None)
    hi = np.clip(i + 1, None, len(full))
    return (starts[hi] - starts[lo]) > 0


def _runs_holding(runs: BoolArray, seeds: BoolArray) -> BoolArray:
    """The runs of True in ``runs`` that hold a True of ``seeds``."""
    # every sample of a run has the same count of False before it
    labels = np.cumsum(~runs, dtype=np.int64)
    held = np.bincount(labels, weights=seeds & runs, minlength=len(runs) + 1)
    return runs & (held[labels] > 0)


def _hot_bins(
    high: NDArray[np.float64], samplerate_hz: float, config: ClickRemoval
) -> tuple[BoolArray, BoolArray, int]:
    """Where one channel's high band holds clicks, per bin of the search.

    Returns
    -------
    tuple
        ``(hot, strong, size)``: the runs of bins (``resolution_s`` each,
        ``size`` samples) that reach ``threshold`` and last while above
        ``extent_threshold``; those of them that reach
        ``isolated_threshold``; and the bin size.
    """
    n = len(high)
    size = max(1, round(config.resolution_s * samplerate_hz))
    n_bins = n // size
    magnitude = np.abs(high)
    # median over what is not digital silence: half a silent block has
    # noise too
    sample = magnitude[:: config.sample_stride]
    sample = sample[sample > 0]
    if n_bins == 0 or len(sample) == 0:
        return np.zeros(n_bins, dtype=bool), np.zeros(n_bins, dtype=bool), size
    sigma = np.median(sample) / 0.6745
    # the largest value per bin: fast, and bridges zero crossings
    envelope = magnitude[: n_bins * size].reshape(n_bins, size).max(axis=-1)
    longest = max(1, round(config.max_duration_s / config.resolution_s))
    # hysteresis: a click reaches the threshold, and then lasts for as long
    # as it stays above the lower one. A long loud event holds no click,
    # and neither does a long event around one.
    trigger = envelope > config.threshold * sigma
    trigger &= ~_long_runs(trigger, longest)
    hot = _runs_holding(envelope > config.extent_threshold * sigma, trigger)
    hot &= ~_long_runs(hot, longest)
    edge = min(round(config.edge_s / config.resolution_s), n_bins // 2)
    if edge:
        hot[:edge] = False
        hot[-edge:] = False
    strong = _runs_holding(
        hot, trigger & (envelope > config.isolated_threshold * sigma)
    )
    return hot, strong, size


def _bins_to_mask(hot: BoolArray, n: int, size: int, config: ClickRemoval) -> BoolArray:
    """Samples of the runs ``hot``, widened by ``guard_s`` on both sides."""
    guard = round(config.guard_s / config.resolution_s)
    wide = hot
    if guard and hot.any():
        wide = maximum_filter1d(hot, 2 * guard + 1, mode="constant", cval=False)
    mask = np.zeros(n, dtype=bool)
    mask[: len(hot) * size] = np.repeat(wide, size)
    return mask


def find_clicks(
    high: NDArray[np.float64], samplerate_hz: float, config: ClickRemoval
) -> BoolArray:
    """Mask of the clicks in ``high``, one channel's high band (samples,)."""
    hot, _, size = _hot_bins(high, samplerate_hz, config)
    return _bins_to_mask(hot, len(high), size, config)


def _coincident(
    hot: BoolArray, strong: BoolArray, fraction: float, guard_bins: int
) -> BoolArray:
    """The runs of ``hot`` (electrodes, bins) to repair under the guard.

    An event is a run of bins where any electrode is hot, joined across
    gaps up to the guard. Its runs are repaired on every electrode when it
    reaches at least ``fraction`` of them; otherwise only the runs that
    are ``strong`` (reach ``isolated_threshold``) are.
    """
    n_electrodes = hot.shape[0]
    need = max(1, math.ceil(fraction * n_electrodes - 1e-9))
    union = hot.any(axis=0)
    if guard_bins:
        union = maximum_filter1d(union, 2 * guard_bins + 1, mode="constant", cval=False)
    starts = union & ~np.concatenate([[False], union[:-1]])
    label: NDArray[np.int64] = np.where(union, np.cumsum(starts, dtype=np.int64), 0)
    hits = np.zeros((n_electrodes, int(label.max(initial=0)) + 1), dtype=bool)
    for c in range(n_electrodes):
        hits[c, label[hot[c]]] = True
    wide = np.concatenate([[False], hits[:, 1:].sum(axis=0) >= need])
    return (hot & wide[label][None, :]) | strong


def _cumsum0(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.concatenate([[0.0], np.cumsum(x)])


def _repair(
    x: NDArray[np.float64],
    tonal: NDArray[np.float64],
    mask: BoolArray,
    width: int,
) -> NDArray[np.float64]:
    """``x`` with its masked samples bridged under the tonal part ``tonal``.

    In a hole, the tonal part is scaled to what the good samples around it
    hold of it -- the least-squares gain over ``width`` samples, within 0
    and 1: 1 where a tone is there and 0 where it is not yet. What is left,
    the noise and the click, is replaced by a straight line between the
    nearest good samples on either side, and the scaled tonal part is added
    back. Computed at the masked samples and the ends of their holes only.
    """
    n = len(x)
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    # the nearest good samples; none on one side: hold the other
    lo = starts - 1
    hi = ends
    lo = np.where(lo < 0, hi, lo)
    hi = np.where(hi >= n, lo, hi)
    lo = np.clip(lo, 0, n - 1)
    hi = np.clip(hi, 0, n - 1)
    good = ~mask
    num = _cumsum0(x * tonal * good)
    den = _cumsum0(tonal * tonal * good)
    tiny = np.finfo(np.float64).tiny

    def scaled(i: NDArray[np.intp]) -> NDArray[np.float64]:
        a = np.clip(i - width // 2, 0, None)
        b = np.clip(i + width // 2 + 1, None, n)
        gain = (num[b] - num[a]) / np.maximum(den[b] - den[a], tiny)
        return tonal[i] * np.clip(gain, 0.0, 1.0)

    idx = np.flatnonzero(mask)
    run = np.repeat(np.arange(len(starts)), ends - starts)
    left = x[lo] - scaled(lo)
    right = x[hi] - scaled(hi)
    weight = (idx - lo[run]) / np.maximum(hi[run] - lo[run], 1)
    out = x.copy()
    out[idx] = scaled(idx) + left[run] * (1 - weight) + right[run] * weight
    return out


@dataclass(frozen=True, slots=True)
class _Search:
    """One channel's block, split by its spectrum, and its click runs."""

    x: NDArray[np.float64]
    spectrum: NDArray[np.complexfloating[Any]]
    narrow: BoolArray
    n_fft: int
    hot: BoolArray
    strong: BoolArray
    size: int


def _search(
    x: NDArray[np.floating[Any]], samplerate_hz: float, cfg: ClickRemoval
) -> _Search | None:
    """Split ``x`` by its spectrum and find its clicks; None if too short."""
    x64 = np.asarray(x, dtype=np.float64)
    n = len(x64)
    if n < _MIN_SAMPLES:
        return None
    rate = float(samplerate_hz)
    target = n + min(round(cfg.fade_s * rate), n // 2)
    fast = scipy.fft.next_fast_len(target, real=True)
    n_fft = target if fast is None else int(fast)
    spectrum = np.asarray(scipy.fft.rfft(_periodic(x64, n_fft)))
    power = np.real(spectrum) ** 2 + np.imag(spectrum) ** 2
    df_hz = rate / n_fft
    narrow = power > spectral_floor(power, df_hz, cfg) * 10 ** (cfg.line_margin_db / 10)
    freqs = np.arange(len(power)) * df_hz
    # a smooth edge: a sharp one would ring around the clicks
    position = np.clip((freqs / rate - cfg.high_band) / cfg.high_band_edge, 0, 1)
    gain = np.sin(0.5 * np.pi * position) ** 2
    high = np.asarray(scipy.fft.irfft(spectrum * np.where(narrow, 0.0, gain), n=n_fft))
    # round-off of the transform is not noise: digital silence stays silent
    hot, strong, size = _hot_bins(np.where(x64 == 0, 0.0, high[:n]), rate, cfg)
    return _Search(x64, spectrum, narrow, n_fft, hot, strong, size)


def _repaired(
    search: _Search, hot: BoolArray, samplerate_hz: float, cfg: ClickRemoval
) -> tuple[NDArray[np.float64], BoolArray]:
    """``search.x`` with the runs ``hot`` repaired, and the repaired mask."""
    n = len(search.x)
    mask = _bins_to_mask(hot, n, search.size, cfg)
    if not mask.any():
        return search.x, mask
    tonal = np.asarray(
        scipy.fft.irfft(np.where(search.narrow, search.spectrum, 0.0), n=search.n_fft)
    )
    width = max(1, round(cfg.local_window_s * samplerate_hz))
    return _repair(search.x, tonal[:n], mask, width), mask


def _clean_channel(
    x: NDArray[np.floating[Any]],
    samplerate_hz: float,
    config: ClickRemoval | None = None,
) -> tuple[NDArray[np.float64], BoolArray]:
    """Repair the clicks in one channel's block ``x`` (samples,).

    The unguarded algorithm on one block as it is given (audian's, kept for
    testing the port); the result depends on where the block starts and
    ends. Private on purpose: without the electrode-coincidence guard it
    repairs loud chirps. Use `remove_clicks`, `clean_span` or `read_clean`.

    Parameters
    ----------
    x
        One channel's samples.
    samplerate_hz
        Sampling rate.
    config
        Click removal settings; defaults to ``ClickRemoval()``.

    Returns
    -------
    tuple
        ``(out, mask)``: the repaired samples (float64) and which of them
        were repaired. ``out`` equals ``x`` wherever ``mask`` is False, and
        everywhere when nothing was repaired.
    """
    cfg = ClickRemoval() if config is None else config
    search = _search(x, samplerate_hz, cfg)
    if search is None:
        return np.asarray(x, dtype=np.float64), np.zeros(len(x), dtype=bool)
    return _repaired(search, search.hot, samplerate_hz, cfg)


# ------------------------------------------------------- blocks and spans


@dataclass(frozen=True, slots=True)
class CleanPlan:
    """What to read to clean one span of a recording.

    Attributes
    ----------
    start, stop
        Recording samples ``[start, stop)`` to clean.
    read_start, read_stop
        Recording samples to read for that: the blocks the span touches and
        their margins, clipped to the recording.
    n_samples_total
        Length of the whole recording.
    """

    start: int
    stop: int
    read_start: int
    read_stop: int
    n_samples_total: int

    @property
    def n_samples(self) -> int:
        """Number of samples cleaned."""
        return self.stop - self.start

    @property
    def n_read(self) -> int:
        """Number of samples to read."""
        return self.read_stop - self.read_start

    @property
    def read_slice(self) -> slice:
        """``slice(read_start, read_stop)`` for indexing the recording."""
        return slice(self.read_start, self.read_stop)


def plan_clean(
    start_sample: int,
    stop_sample: int,
    n_samples_total: int,
    samplerate_hz: float,
    config: ClickRemoval | None = None,
) -> CleanPlan:
    """Plan the cleaning of recording samples ``[start, stop)``.

    Parameters
    ----------
    start_sample, stop_sample
        Span to clean, ``0 <= start_sample < stop_sample <= n_samples_total``.
    n_samples_total
        Length of the whole recording.
    samplerate_hz
        Sampling rate.
    config
        Click removal settings; defaults to ``ClickRemoval()``.

    Returns
    -------
    CleanPlan
        The span and the samples to read for it. With those samples,
        `clean_span` gives the same values as cleaning the whole recording
        at once with `remove_clicks`.
    """
    cfg = ClickRemoval() if config is None else config
    start, stop, total = int(start_sample), int(stop_sample), int(n_samples_total)
    if not 0 <= start < stop <= total:
        msg = (
            "need 0 <= start_sample < stop_sample <= n_samples_total, "
            f"got {start}, {stop}, {total}"
        )
        raise ValueError(msg)
    size = cfg.block_samples(samplerate_hz)
    margin = cfg.margin_samples(samplerate_hz)
    first_block = start // size
    last_block = (stop - 1) // size
    return CleanPlan(
        start=start,
        stop=stop,
        read_start=max(0, first_block * size - margin),
        read_stop=min(total, (last_block + 1) * size + margin),
        n_samples_total=total,
    )


def _blocks(
    first: int, lo: int, hi: int, n: int, size: int, margin: int
) -> Iterator[tuple[int, int, int, int]]:
    """Blocks of absolute samples ``[first + lo, first + hi)``.

    Yields ``(c0, c1, w0, w1)``: the part of the block to keep and the
    window it is computed from, both in absolute samples, the window
    clipped to ``[first, first + n)``.
    """
    a, b, stop = first + lo, first + hi, first + n
    k = a // size
    while k * size < b:
        yield (
            max(k * size, a),
            min((k + 1) * size, b),
            max(k * size - margin, first),
            min((k + 1) * size + margin, stop),
        )
        k += 1


def _runs_to_repair(found: list[_Search | None], cfg: ClickRemoval) -> list[BoolArray]:
    """Per electrode, the click runs of one block to repair."""
    if any(s is None for s in found):
        # A block too short to search: None on every electrode alike.
        return [np.zeros(0, dtype=bool) for _ in found]
    searches = [s for s in found if s is not None]
    hot = np.stack([s.hot for s in searches])
    strong = np.stack([s.strong for s in searches])
    guard_bins = round(cfg.guard_s / cfg.resolution_s)
    return list(_coincident(hot, strong, cfg.coincidence, guard_bins))


def _clean_range(
    samples: NDArray[np.floating[Any]],
    first: int,
    lo: int,
    hi: int,
    samplerate_hz: float,
    cfg: ClickRemoval,
    workers: int,
) -> tuple[NDArray[np.floating[Any]], BoolArray]:
    """Clean ``samples[lo:hi]``; ``samples[0]`` is recording sample ``first``."""
    n, n_channels = samples.shape
    out = np.array(samples[lo:hi], copy=True)
    mask = np.zeros(out.shape, dtype=bool)
    size = cfg.block_samples(samplerate_hz)
    margin = cfg.margin_samples(samplerate_hz)
    pool = ThreadPoolExecutor(workers) if workers > 1 and n_channels > 1 else None
    try:
        for c0, c1, w0, w1 in _blocks(first, lo, hi, n, size, margin):
            # channel-major, so every channel's samples are contiguous
            window = np.ascontiguousarray(
                samples[w0 - first : w1 - first].T, dtype=np.float64
            )

            def search(c: int, window: NDArray[np.float64] = window) -> _Search | None:
                return _search(window[c], samplerate_hz, cfg)

            channels = range(n_channels)
            found = list(pool.map(search, channels) if pool else map(search, channels))
            runs = _runs_to_repair(found, cfg)

            def repair(
                c: int,
                found: list[_Search | None] = found,
                runs: list[BoolArray] = runs,
            ) -> tuple[NDArray[np.float64], BoolArray] | None:
                s = found[c]
                if s is None or not runs[c].any():
                    return None
                return _repaired(s, runs[c], samplerate_hz, cfg)

            results = list(
                pool.map(repair, channels) if pool else map(repair, channels)
            )
            keep = slice(c0 - w0, c1 - w0)
            rows = slice(c0 - first - lo, c1 - first - lo)
            for c, result in enumerate(results):
                if result is None:
                    continue
                repaired, hit = result
                inside = hit[keep]
                if inside.any():
                    # `out[rows, c]` is a view; only repaired samples change
                    out[rows, c][inside] = repaired[keep][inside]
                    mask[rows, c] = inside
    finally:
        if pool is not None:
            pool.shutdown()
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            "cleaned samples [%d, %d): %.3f%% repaired",
            first + lo,
            first + hi,
            100 * mask.mean() if mask.size else 0.0,
        )
    return out, mask


def can_remove_clicks(n_electrodes: int, config: ClickRemoval | None = None) -> bool:
    """Whether a recording with ``n_electrodes`` can be cleaned safely.

    Pipelines should skip click removal (with a warning) when this is False
    rather than call `remove_clicks`, which raises `TooFewElectrodesError`.
    """
    cfg = ClickRemoval() if config is None else config
    return int(n_electrodes) >= cfg.min_electrodes


def _check_electrodes(n_electrodes: int, cfg: ClickRemoval) -> None:
    if n_electrodes >= cfg.min_electrodes:
        return
    if n_electrodes == 1:
        msg = (
            "click removal needs a multi-electrode recording: on a single "
            "channel a click and a loud chirp's harmonics cannot be told apart, "
            "and the chirp would be repaired as a click. Skip click removal "
            "for single-channel data."
        )
    else:
        msg = (
            f"click removal needs at least {cfg.min_electrodes} electrodes for "
            f"coincidence={cfg.coincidence} (got {n_electrodes}): on fewer, a "
            f"loud chirp (seen on up to {MAX_CHIRP_ELECTRODES} electrodes) can "
            "pass the coincidence guard and be repaired as a click. Skip click "
            "removal for this recording, or raise `coincidence` (fewer clicks "
            "removed) if the grid is only slightly too small."
        )
    raise TooFewElectrodesError(msg)


def _check_samples(samples: NDArray[Any], samplerate_hz: float) -> None:
    if samples.ndim != 2:
        msg = f"samples must be 2-D (n_samples, n_electrodes), got {samples.shape}"
        raise ValueError(msg)
    if not np.issubdtype(samples.dtype, np.floating):
        msg = f"samples must be floating point, got {samples.dtype}"
        raise TypeError(msg)
    if not (math.isfinite(samplerate_hz) and samplerate_hz > 0):
        msg = f"samplerate_hz must be positive, got {samplerate_hz!r}"
        raise ValueError(msg)


def remove_clicks[T: np.floating[Any]](
    samples: NDArray[T],
    samplerate_hz: float,
    config: ClickRemoval | None = None,
    *,
    start_sample: int = 0,
    workers: int = 1,
) -> tuple[NDArray[T], BoolArray]:
    """Remove the clicks from a multi-electrode recording (or a chunk of one).

    The samples are cleaned in blocks fixed in absolute time, each computed
    from itself and its margins, clipped to what ``samples`` holds. With the
    whole recording (``start_sample=0``) this is the reference result that
    `clean_span` reproduces chunk by chunk.

    Parameters
    ----------
    samples
        Shape ``(n_samples, n_electrodes)``, floating point.
    samplerate_hz
        Sampling rate.
    config
        Click removal settings; defaults to ``ClickRemoval()``.
    start_sample
        Recording sample that ``samples[0]`` is; places the block grid.
    workers
        Threads repairing the channels of a block side by side; the result
        does not depend on it.

    Returns
    -------
    tuple
        ``(cleaned, mask)``: the samples with their clicks repaired, in the
        input's dtype, and a boolean ``(n_samples, n_electrodes)`` mask of
        the repaired samples. ``cleaned`` equals ``samples`` exactly wherever
        ``mask`` is False.

    Raises
    ------
    TooFewElectrodesError
        If ``samples`` has fewer than ``config.min_electrodes`` electrodes
        (always for single-channel data); see `can_remove_clicks`.
    """
    _check_samples(samples, samplerate_hz)
    cfg = ClickRemoval() if config is None else config
    _check_electrodes(samples.shape[1], cfg)
    if start_sample < 0:
        msg = f"start_sample must be >= 0, got {start_sample}"
        raise ValueError(msg)
    n = samples.shape[0]
    if n == 0:
        return samples.copy(), np.zeros(samples.shape, dtype=bool)
    out, mask = _clean_range(samples, start_sample, 0, n, samplerate_hz, cfg, workers)
    return out.astype(samples.dtype, copy=False), mask


def clean_span[T: np.floating[Any]](
    samples: NDArray[T],
    plan: CleanPlan,
    samplerate_hz: float,
    config: ClickRemoval | None = None,
    *,
    workers: int = 1,
) -> tuple[NDArray[T], BoolArray]:
    """Clean one span of a recording, as the whole recording would be.

    Parameters
    ----------
    samples
        ``recording[plan.read_start:plan.read_stop]``, shape ``(plan.n_read,
        n_electrodes)``, floating point.
    plan
        Output of `plan_clean` for this recording and config.
    samplerate_hz
        Sampling rate.
    config
        Click removal settings (the ones ``plan`` was made with).
    workers
        Threads repairing the channels of a block side by side.

    Returns
    -------
    tuple
        ``(cleaned, mask)`` for recording samples ``[plan.start,
        plan.stop)``, equal to the same samples of `remove_clicks` on the
        whole recording.

    Raises
    ------
    TooFewElectrodesError
        If there are fewer than ``config.min_electrodes`` electrodes.
    """
    _check_samples(samples, samplerate_hz)
    cfg = ClickRemoval() if config is None else config
    _check_electrodes(samples.shape[1], cfg)
    if samples.shape[0] != plan.n_read:
        msg = (
            f"expected {plan.n_read} samples for clean plan "
            f"[{plan.read_start}, {plan.read_stop}), got {samples.shape[0]}"
        )
        raise ValueError(msg)
    lo = plan.start - plan.read_start
    hi = plan.stop - plan.read_start
    out, mask = _clean_range(
        samples, plan.read_start, lo, hi, samplerate_hz, cfg, workers
    )
    return out.astype(samples.dtype, copy=False), mask


class SampleSource(Protocol):
    """What `read_clean` needs of a recording (e.g. `gridtools.io.Recording`)."""

    @property
    def samplerate_hz(self) -> float:
        """Sampling rate."""
        ...

    @property
    def n_samples(self) -> int:
        """Number of samples per channel."""
        ...

    def read(self, start: int, stop: int) -> NDArray[np.float32]:
        """Samples ``[start, stop)``, shape ``(stop - start, n_channels)``."""
        ...


def read_clean(
    recording: SampleSource,
    start_sample: int,
    stop_sample: int,
    config: ClickRemoval | None = None,
    *,
    workers: int = 1,
) -> tuple[NDArray[np.float32], BoolArray]:
    """Read recording samples ``[start, stop)`` with their clicks removed.

    Reads the margins `plan_clean` asks for, so the result does not depend
    on how a recording is chunked. To feed a spectrogram span, pass
    ``plan.read_start`` and ``plan.read_stop`` of its `SpanPlan`.

    Returns
    -------
    tuple
        ``(cleaned, mask)``, as `clean_span`.

    Raises
    ------
    TooFewElectrodesError
        If the recording has fewer than ``config.min_electrodes``
        electrodes (always for single-channel data); see
        `can_remove_clicks`.
    """
    rate = float(recording.samplerate_hz)
    plan = plan_clean(start_sample, stop_sample, recording.n_samples, rate, config)
    samples = recording.read(plan.read_start, plan.read_stop)
    return clean_span(samples, plan, rate, config, workers=workers)


def click_spans(mask: BoolArray, offset: int = 0) -> NDArray[np.int64]:
    """Runs of repaired samples as ``[start, stop)`` spans.

    Parameters
    ----------
    mask
        ``(n_samples,)`` or ``(n_samples, n_electrodes)``; for the latter a
        sample counts when any electrode was repaired there.
    offset
        Added to every index (e.g. the recording sample of ``mask[0]``).

    Returns
    -------
    NDArray[np.int64]
        Shape ``(n_spans, 2)``, sorted by start.
    """
    hit = np.asarray(mask, dtype=bool)
    if hit.ndim == 2:
        hit = hit.any(axis=1)
    elif hit.ndim != 1:
        msg = f"mask must be 1-D or 2-D, got {hit.shape}"
        raise ValueError(msg)
    edges = np.diff(np.concatenate([[0], hit.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return np.stack([starts, stops], axis=1).astype(np.int64) + int(offset)
