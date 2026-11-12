"""Synthetic signals for the click removal tests.

Two families: audian's own test signals (a 700 Hz tone, Gaussian noise of
1e-3, impulses of 0.2 ms decay), kept so that the ported tests assert what
they asserted there, here laid out on a grid of 9 or 15 electrodes; and a
grid recording with two chirping fish, built with gridtools, for what
matters here: that chirps survive. Click removal refuses grids smaller
than ``ClickRemoval().min_electrodes`` (9), so every multi-electrode
signal has at least 9.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from gridtools.sim import ColoredNoiseParams, Harmonics, colored_noise, synthesize_eod
from numpy.typing import NDArray

RATE = 20_000.0
NOISE = 1e-3
TONE = 0.2
TONE_HZ = 700.0
N = int(6 * RATE)
CLICKS = (int(0.9 * RATE), int(2.3 * RATE), int(4.7 * RATE))

#: Relative harmonic amplitudes of a real EOD next to an electrode
#: (2022-05-12-10_00, electrode 10, 642 Hz): much richer than gridtools'
#: A. leptorhynchus model, which is what makes loud chirps reach the band
#: clicks are searched in.
MEASURED_HARMONICS = Harmonics(
    amplitudes=(1.0, 0.685, 0.333, 0.180, 0.078, 0.050, 0.026, 0.0055),
    phases_rad=(0.0,) * 8,
)


def click(n: int, at: int, amplitude: float = 1.0, tau: float = 0.0002) -> NDArray:
    """An impulse with a 0.2 ms decay: a thousand times the noise."""
    k = np.arange(n) - at
    return np.where(
        k >= 0, amplitude * np.exp(-np.clip(k, 0, None) / (tau * RATE)), 0.0
    )


def noise(n: int, seed: int = 3) -> NDArray:
    return np.random.default_rng(seed).normal(0.0, NOISE, n)


def tone(n: int, start: float = 0.0) -> NDArray:
    t = np.arange(n) / RATE
    return np.where(t >= start, TONE * np.sin(2 * np.pi * TONE_HZ * t), 0.0)


def holes(n: int, at: tuple[int, ...] | list[int]) -> NDArray[np.bool_]:
    """The few milliseconds around each click: where a repair can go wrong."""
    hole = np.zeros(n, dtype=bool)
    for a in at:
        hole[a - int(0.003 * RATE) : a + int(0.006 * RATE)] = True
    return hole


#: Weak clicks (0.05, 22.8 robust sd of the high band: above `threshold`,
#: below `isolated_threshold`) on electrodes 3 and up, i.e. on 6 of 9 or 12
#: of 15 electrodes: repaired by coincidence.
WEAK_MANY = (int(1.6 * RATE), int(3.9 * RATE))
#: The same weak click on electrode 3 alone: left by the guard.
WEAK_ONE = int(5.2 * RATE)
WEAK = 0.05


def channels(n: int = N, n_electrodes: int = 9) -> NDArray[np.float64]:
    """audian's three channels on a grid of ``n_electrodes`` (at least 4).

    Electrode 0: strong clicks (1.0) on a tone; 1: a strong click before a
    tone starts; 2: plain noise; 3 and up: noise with the weak clicks
    `WEAK_MANY` on all of them and `WEAK_ONE` on electrode 3 only. ``n``
    samples, with the clicks of the first `N` repeated every `N`.
    """
    at = [a + k for k in range(0, n, N) for a in CLICKS if a + k < n]
    late = [int(2.5 * RATE) + k for k in range(0, n, N)]
    many = [a + k for k in range(0, n, N) for a in WEAK_MANY if a + k < n]
    one = [WEAK_ONE + k for k in range(0, n, N) if WEAK_ONE + k < n]
    columns = [
        noise(n, 1) + tone(n) + sum(click(n, a) for a in at),
        noise(n, 2) + tone(n, start=3.0) + sum(click(n, a) for a in late),
        noise(n, 4),
    ]
    for e in range(3, n_electrodes):
        x = noise(n, 10 + e) + sum(click(n, a, WEAK) for a in many)
        if e == 3:
            x = x + sum(click(n, a, WEAK) for a in one)
        columns.append(x)
    return np.stack(columns, axis=1)


@pytest.fixture(scope="session")
def sig() -> SimpleNamespace:
    """audian's test signals and their constants, for the ported tests."""
    return SimpleNamespace(
        RATE=RATE,
        NOISE=NOISE,
        TONE_HZ=TONE_HZ,
        N=N,
        CLICKS=CLICKS,
        click=click,
        noise=noise,
        tone=tone,
        holes=holes,
        WEAK=WEAK,
        WEAK_MANY=WEAK_MANY,
        WEAK_ONE=WEAK_ONE,
    )


@pytest.fixture(scope="session")
def grid() -> NDArray[np.float64]:
    """`channels` on 9 electrodes over twice `N` (12 s)."""
    return channels(2 * N, 9)


@pytest.fixture(scope="session")
def grid15() -> NDArray[np.float64]:
    """`channels` on 15 electrodes over `N` (6 s)."""
    return channels(N, 15)


@dataclass(frozen=True)
class ChirpScene:
    """Two chirping fish on a grid, with and without clicks.

    Attributes
    ----------
    truth
        Fish and noise, float32 ``(n, n_electrodes)``.
    signal
        ``truth`` plus the clicks.
    boxes
        ``(t0, t1, f0, f1)`` of every chirp.
    chirp_times_s
        Peak time of every chirp, in the order of ``boxes``.
    clicks
        ``(sample, electrodes)`` of every click.
    at_chirp
        Index into ``boxes`` of the chirp with a click at its peak.
    """

    truth: NDArray[np.float32]
    signal: NDArray[np.float32]
    boxes: list[tuple[float, float, float, float]]
    chirp_times_s: list[float]
    clicks: list[tuple[int, tuple[int, ...]]]
    at_chirp: int


def chirp_scene(
    loudness: float,
    harmonics: Harmonics = MEASURED_HARMONICS,
    seed: int = 7,
    n_electrodes: int = 9,
    dead: tuple[int, ...] = (),
) -> ChirpScene:
    """8 s, fish at 900 and 700 Hz, each chirping small, big, small, big.

    On ``n_electrodes`` (at least 4; those beyond the fourth pick up both
    fish at 2% of their peak). Fish 0 is loudest on electrode 0,
    ``loudness`` times the robust sd of the high band (above 0.2 fs) of
    the pink noise (sd 1); fish 1 is 4x weaker, loudest on electrode 3.
    Small chirps: +60 Hz, 10 ms FWHM; big: +300 Hz, 30 ms. Clicks (0.5 ms
    decay) sit at least 0.3 s from every chirp, except one at the peak of
    fish 0's second big chirp: strong ones (300-3000x the noise sd) on one
    to four electrodes, and a weak one (25x, about 25 robust sd of the
    high band: below `isolated_threshold`) on every live electrode at 6.3 s.
    Electrodes in ``dead`` are digital silence throughout.
    """
    rng = np.random.default_rng(seed)
    n = int(8 * RATE)
    t = np.arange(n) / RATE
    noise_ = colored_noise(
        rng, n, n_electrodes, RATE, ColoredNoiseParams(sd=1.0)
    ).astype(float)
    high_sigma = 0.29  # measured on this noise: robust sd above 4 kHz
    far = [0.02] * (n_electrodes - 4)
    gains = (
        np.array([1.0, 0.3, 0.1, 0.05, *far]) * loudness * high_sigma,
        np.array([0.05, 0.1, 0.3, 1.0, *far]) * loudness * high_sigma / 4,
    )
    truth = noise_.copy()
    boxes: list[tuple[float, float, float, float]] = []
    peaks: list[float] = []
    for fish, (f0, g) in enumerate(zip((900.0, 700.0), gains, strict=True)):
        freq = np.full(n, f0)
        env = np.ones(n)
        for k, tc in enumerate(np.arange(1.0 + 0.6 * fish, 7.0, 1.4)):
            df, fwhm, dip = (60.0, 0.010, 0.15) if k % 2 == 0 else (300.0, 0.030, 0.55)
            bump = np.exp(-0.5 * ((t - tc) / (fwhm / 2.3548)) ** 2)
            freq += df * bump
            env -= dip * bump
            boxes.append(
                (tc - 1.5 * fwhm - 0.01, tc + 1.5 * fwhm + 0.01, f0 - 15, f0 + df + 25)
            )
            peaks.append(float(tc))
        eod = synthesize_eod(freq, env / harmonics.amplitudes[0], RATE, harmonics)
        truth += eod[:, None] * g[None, :]
    signal = truth.copy()
    clicks: list[tuple[int, tuple[int, ...]]] = []
    at_chirp = 3  # fish 0's second big chirp, at 5.2 s
    times = [0.4, 2.0, 3.3, 4.6, 7.6, peaks[at_chirp], 6.3]
    electrodes = [(0,), (1, 2), (3,), (0, 1, 2, 3), (2,), (0, 1)]
    live = tuple(e for e in range(n_electrodes) if e not in dead)
    electrodes.append(live)
    for k, (tc, els) in enumerate(zip(times, electrodes, strict=True)):
        at = int(tc * RATE)
        amplitude = 25.0 if k == len(times) - 1 else 300.0 * 10 ** (k % 2)
        shape = click(n, at, amplitude, tau=0.0005)
        for el in els:
            signal[:, el] += shape
        clicks.append((at, els))
    truth[:, list(dead)] = 0.0
    signal[:, list(dead)] = 0.0
    return ChirpScene(
        truth.astype(np.float32),
        signal.astype(np.float32),
        boxes,
        peaks,
        clicks,
        at_chirp,
    )


@pytest.fixture(scope="session")
def make_chirp_scene() -> Callable[..., ChirpScene]:
    """Factory of `ChirpScene` (cached per loudness)."""
    cache: dict[tuple[float, int, int, tuple[int, ...]], ChirpScene] = {}

    def make(
        loudness: float,
        harmonics: Harmonics = MEASURED_HARMONICS,
        n_electrodes: int = 9,
        dead: tuple[int, ...] = (),
    ) -> ChirpScene:
        key = (loudness, id(harmonics), n_electrodes, dead)
        if key not in cache:
            cache[key] = chirp_scene(
                loudness, harmonics, n_electrodes=n_electrodes, dead=dead
            )
        return cache[key]

    return make


@pytest.fixture(scope="session")
def grid_cleaned(
    grid: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """`grid` cleaned whole: the reference for chunked cleaning."""
    from chirpdetector.preprocessing import remove_clicks

    return remove_clicks(grid, RATE)


@pytest.fixture(scope="session")
def grid15_cleaned(
    grid15: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """`grid15` cleaned whole."""
    from chirpdetector.preprocessing import remove_clicks

    return remove_clicks(grid15, RATE)
