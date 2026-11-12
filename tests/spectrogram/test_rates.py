"""The same physical signal gives the same spectrogram at different rates."""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import erf

from chirpdetector.spectrogram import (
    Spectrogram,
    SpectrogramConfig,
    compute_spectrogram,
    normalize,
)

CFG = SpectrogramConfig()
REFERENCE_FS = 20_000.0
OTHER_RATES = (44_100.0, 48_000.0)
FISH_HZ = (720.0, 941.3)
# A frame centre at all three rates (440 x 100 / 20 kHz = 441 x 220 / 44.1 kHz
# = 440 x 240 / 48 kHz): near it, the frame grids coincide to < 0.35 ms even
# though the 44.1 kHz hop is 220 samples = 4.9887 ms.
CHIRP_T_S = 2.2
CHIRP_HZ = 150.0
CHIRP_SIGMA_S = 0.0085  # 20 ms FWHM


def scene(fs: float, duration_s: float = 2.6) -> NDArray[np.float32]:
    """Two fish on three electrodes; fish 0 chirps (Gaussian excursion).

    The phase is integrated analytically, so every rate samples the same
    continuous-time signal.
    """
    t = np.arange(round(duration_s * fs)) / fs
    scale = CHIRP_SIGMA_S * np.sqrt(2)
    excursion_phase = (
        CHIRP_HZ
        * CHIRP_SIGMA_S
        * np.sqrt(np.pi / 2)
        * (erf((t - CHIRP_T_S) / scale) + erf(CHIRP_T_S / scale))
    )
    fish_a = np.sin(2 * np.pi * (FISH_HZ[0] * t + excursion_phase))
    fish_b = 0.5 * np.sin(2 * np.pi * FISH_HZ[1] * t + 1.0)
    gains = np.array([1.0, 0.4, 0.1])
    return (fish_a[:, None] * gains + fish_b[:, None] * gains[::-1]).astype(np.float32)


SPECS = {
    fs: compute_spectrogram(scene(fs), fs, CFG) for fs in (REFERENCE_FS, *OTHER_RATES)
}


def _nearest_frames(spec: Spectrogram, times_s: NDArray[np.float64]) -> NDArray:
    index = np.round((times_s - spec.times_s[0]) / spec.layout.hop_s).astype(int)
    return np.clip(index, 0, spec.n_times - 1)


@pytest.mark.parametrize("fs", OTHER_RATES)
def test_frequency_grids_agree(fs: float) -> None:
    ref, spec = SPECS[REFERENCE_FS], SPECS[fs]
    assert spec.n_freqs == ref.n_freqs
    assert np.abs(spec.freqs_hz - ref.freqs_hz).max() < 0.07


@pytest.mark.parametrize("fs", OTHER_RATES)
def test_fish_peak_in_the_same_bins(fs: float) -> None:
    peaks = []
    for spec in (SPECS[REFERENCE_FS], SPECS[fs]):
        steady = spec.complete & (np.abs(spec.times_s - CHIRP_T_S) > 0.3)
        mean_power = spec.power.numpy()[:, :, steady].mean(axis=2)
        low = spec.freqs_hz < 0.5 * sum(FISH_HZ)  # fish 0's half of the band
        peaks.append(mean_power[:, low].argmax(axis=1))
        expected = np.argmin(np.abs(spec.freqs_hz - FISH_HZ[0]))
        assert np.all(peaks[-1] == expected)
    np.testing.assert_array_equal(peaks[0], peaks[1])


@pytest.mark.parametrize("fs", [REFERENCE_FS, *OTHER_RATES])
def test_chirp_appears_at_the_same_time(fs: float) -> None:
    spec = SPECS[fs]
    band = (spec.freqs_hz > 790.0) & (spec.freqs_hz < 900.0)
    near = np.abs(spec.times_s - CHIRP_T_S) < 0.1
    energy = spec.power.numpy()[-1][band][:, near].sum(axis=0)  # shortest window
    centroid_s = (energy * spec.times_s[near]).sum() / energy.sum()
    assert abs(centroid_s - CHIRP_T_S) <= spec.layout.hop_s / 2


@pytest.mark.parametrize("fs", OTHER_RATES)
def test_normalized_images_agree_where_the_frame_grids_coincide(fs: float) -> None:
    ref, spec = SPECS[REFERENCE_FS], SPECS[fs]
    selected = ref.complete & (np.abs(ref.times_s - CHIRP_T_S) <= 0.15)
    frames = _nearest_frames(spec, ref.times_s[selected])
    assert np.abs(spec.times_s[frames] - ref.times_s[selected]).max() < 3.5e-4
    image_ref = normalize(ref, CFG).numpy()[:, :, selected]
    image = normalize(spec, CFG).numpy()[:, :, frames]
    diff = np.abs(image - image_ref).reshape(CFG.n_windows, -1)
    assert np.all(diff.max(axis=1) < 0.04)  # < 2.4 dB of the 60 dB range
    assert np.all(diff.mean(axis=1) < 2e-3)


def test_normalized_images_agree_everywhere_at_48_khz() -> None:
    """48 kHz has an exact 5 ms hop: the whole image shares the time grid."""
    ref, spec = SPECS[REFERENCE_FS], SPECS[48_000.0]
    np.testing.assert_array_equal(spec.times_s, ref.times_s)
    both = ref.complete & spec.complete
    image_ref = normalize(ref, CFG).numpy()[:, :, both]
    image = normalize(spec, CFG).numpy()[:, :, both]
    diff = np.abs(image - image_ref).reshape(CFG.n_windows, -1)
    assert np.all(diff.max(axis=1) < 0.04)
    assert np.all(diff.mean(axis=1) < 2e-3)
