"""Ground-truth definitions and rectified crops of the chirp-parameter study."""

import numpy as np
import pytest
import torch

from chirpdetector.params.crops import (
    RectifiedConfig,
    detector_view,
    rectified_crop,
    shift_frequency,
)
from chirpdetector.params.truth import measure_period_params, per_period_trace

FS = 20_000.0


def _phase(freq_hz: np.ndarray) -> np.ndarray:
    cycles = np.zeros(freq_hz.size)
    cycles[1:] = np.cumsum(0.5 * (freq_hz[:-1] + freq_hz[1:])) / FS
    return 2 * np.pi * cycles


def test_constant_frequency_has_zero_excursion() -> None:
    base = np.full(4000, 800.0)
    tau, delta = per_period_trace(_phase(base), base, FS)
    assert tau.size == pytest.approx(4000 / FS * 800, abs=1)
    np.testing.assert_allclose(delta, 0.0, atol=1e-6)


def test_drifting_baseline_is_removed_locally() -> None:
    t = np.arange(8000) / FS
    base = 700.0 + 50.0 * t  # 50 Hz/s drift
    tau, delta = per_period_trace(_phase(base), base, FS)
    np.testing.assert_allclose(delta, 0.0, atol=1e-6)


def test_gaussian_chirp_parameters() -> None:
    t = np.arange(int(0.2 * FS)) / FS
    sigma = 0.01
    excursion = 300.0 * np.exp(-0.5 * ((t - 0.1) / sigma) ** 2)
    trough = -10.0 * np.exp(-0.5 * ((t - 0.15) / 0.01) ** 2)
    base = np.full(t.size, 900.0)
    tau, delta = per_period_trace(_phase(base + excursion + trough), base, FS)
    p = measure_period_params(tau, delta)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    assert p.height_hz == pytest.approx(300.0, rel=0.01)  # period averaging
    assert p.fwhm_s == pytest.approx(fwhm, rel=0.02)
    assert p.width10_s == pytest.approx(2 * np.sqrt(2 * np.log(10)) * sigma, rel=0.02)
    assert p.undershoot_hz == pytest.approx(10.0, rel=0.05)
    assert not p.edge_clipped


def test_rectified_crop_places_tone_at_delta_f() -> None:
    cfg = RectifiedConfig()
    t = np.arange(int(0.6 * FS)) / FS
    baseline, offset = 812.3, 100.0
    signal = np.sin(2 * np.pi * (baseline + offset) * t)
    seg = torch.as_tensor(signal[None], dtype=torch.float32)
    (crop,) = rectified_crop(seg, 0.0, FS, baseline, (0.2, 0.4), cfg)
    assert crop.shape == (len(cfg.windows_s), cfg.n_freqs, cfg.n_times)
    peak = crop[0].mean(dim=1).argmax().item()
    assert cfg.delta_f_hz[peak] == pytest.approx(offset, abs=cfg.df_step_hz)


def test_detector_view_and_shift_shapes() -> None:
    image = torch.rand(3, 50, 200)
    freqs = np.linspace(500, 745, 50)
    times = np.linspace(0, 0.995, 200)
    out = detector_view(image, freqs, times, (0.2, 0.5, 600, 700), size=(32, 48))
    assert out.shape == (3, 32, 48)
    x = torch.zeros(2, 1, 10, 4)
    x[:, :, 5] = 1.0
    y = shift_frequency(x, torch.tensor([1.0, -0.5]))
    assert y[0, 0, 6, 0] == pytest.approx(1.0)
    assert y[1, 0, 4, 0] == pytest.approx(0.5)
