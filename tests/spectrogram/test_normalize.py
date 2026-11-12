import dataclasses

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from chirpdetector.spectrogram import (
    SpectrogramConfig,
    compute_spectrogram,
    normalize,
    reference_level_db,
)

# Explicit band: these tests index bins and must not depend on the default band.
CFG = SpectrogramConfig(f_min_hz=400.0, f_max_hz=2000.0)
FS = CFG.samplerate_hz


def scene(n_samples: int = 20_000, seed: int = 0) -> NDArray[np.float32]:
    """Two fish on three electrodes plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / FS
    fish_a = np.sin(2 * np.pi * CFG.freqs_hz[80] * t)
    fish_b = 0.2 * np.sin(2 * np.pi * 933.0 * t + 1.0)
    gains = np.array([[1.0, 0.5, 0.2]])
    noise = 1e-3 * rng.standard_normal((n_samples, 3))
    return (fish_a[:, None] * gains + fish_b[:, None] * gains[:, ::-1] + noise).astype(
        np.float32
    )


def test_output_is_float32_in_unit_interval() -> None:
    spec = compute_spectrogram(scene(), CFG)
    out = normalize(spec, CFG)
    assert out.dtype == torch.float32
    assert out.shape == spec.power.shape
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
    assert float(out.min()) == 0.0  # deep sidelobes clip to the bottom


def test_invariant_to_global_gain() -> None:
    x = scene()
    base = normalize(compute_spectrogram(x, CFG), CFG)
    louder = normalize(compute_spectrogram(10.0 * x, CFG), CFG)
    torch.testing.assert_close(louder, base, rtol=0, atol=1e-5)


def test_reference_is_the_typical_peak_of_the_strongest_fish() -> None:
    x = scene()
    spec = compute_spectrogram(x, CFG)
    ref = reference_level_db(spec)
    assert ref is not None
    # Fish a: amplitude 1 on three electrodes with gains 1, 0.5, 0.2, on a bin.
    expected = 10 * np.log10(0.5 * (1.0 + 0.25 + 0.04))
    assert ref == pytest.approx(expected, abs=0.05)
    out = normalize(spec, CFG)
    level = (CFG.dynamic_range_db - CFG.headroom_db) / CFG.dynamic_range_db
    np.testing.assert_allclose(out.numpy()[0, 80, spec.complete], level, atol=1e-3)


def test_same_reference_for_all_channels() -> None:
    """A stationary on-bin tone maps to the same value in every channel."""
    spec = compute_spectrogram(scene(), CFG)
    out = normalize(spec, CFG).numpy()[:, 80, spec.complete]
    assert np.abs(out - out[0]).max() < 2e-3


def test_reference_ignores_a_loud_transient() -> None:
    x = scene()
    spec = compute_spectrogram(x, CFG)
    x_click = x.copy()
    x_click[10_000] += 1_000.0
    spec_click = compute_spectrogram(x_click, CFG)
    ref, ref_click = reference_level_db(spec), reference_level_db(spec_click)
    assert ref is not None
    assert ref_click is not None
    assert ref_click == pytest.approx(ref, abs=0.1)
    # Away from the click, the image is unchanged (min-max scaling would not be).
    far = torch.from_numpy(np.abs(spec.times_s - 0.5) > 0.15)
    torch.testing.assert_close(
        normalize(spec_click, CFG)[:, :, far],
        normalize(spec, CFG)[:, :, far],
        rtol=0,
        atol=1e-2,
    )


def test_explicit_reference_and_range() -> None:
    spec = compute_spectrogram(scene(), CFG)
    ref = reference_level_db(spec)
    assert ref is not None
    torch.testing.assert_close(
        normalize(spec, CFG, reference_db=ref), normalize(spec, CFG)
    )
    shifted = normalize(spec, CFG, reference_db=ref + 6.0)
    assert float(shifted.mean()) < float(normalize(spec, CFG).mean())
    narrow = dataclasses.replace(CFG, dynamic_range_db=30.0, headroom_db=5.0)
    out = normalize(spec, narrow)
    assert float(out.max()) <= 1.0


def test_config_with_other_stft_parameters_is_rejected() -> None:
    spec = compute_spectrogram(scene(5_000), CFG)
    with pytest.raises(ValueError, match="STFT parameters"):
        normalize(spec, dataclasses.replace(CFG, hop_samples=50))


def test_silence_maps_to_zero() -> None:
    spec = compute_spectrogram(np.zeros((5_000, 2), np.float32), CFG)
    assert reference_level_db(spec) is None
    out = normalize(spec, CFG)
    assert torch.equal(out, torch.zeros_like(out))
