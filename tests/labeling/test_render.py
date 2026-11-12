"""Cutout rendering: band zoom, geometry, and that the chirp shows up."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from gridtools.io import load_tracks, open_recording
from scipy.io import wavfile

from chirpdetector.labeling.queue import QueueConfig, build_queue
from chirpdetector.labeling.render import (
    CutoutConfig,
    _sampling,
    render_cutout,
    zoom_baseband,
)


def test_zoom_baseband_keeps_amplitude_and_frequency() -> None:
    fs = 20_000.0
    t = np.arange(20_000) / fs
    x = np.stack(
        [0.7 * np.cos(2 * np.pi * 731.0 * t), 0.2 * np.cos(2 * np.pi * 731.0 * t)], 1
    )
    x[:, 0] += 0.5 * np.cos(2 * np.pi * 3000.0 * t)  # outside the band
    y, rate, fc = zoom_baseband(x, fs, 600.0, 900.0, 16)
    assert rate == fs / 16
    assert y.shape == (len(t) // 16, 2)
    middle = slice(len(y) // 4, 3 * len(y) // 4)
    amplitude = np.abs(y[middle])
    np.testing.assert_allclose(amplitude[:, 0], 0.7, rtol=0.01)
    np.testing.assert_allclose(amplitude[:, 1], 0.2, rtol=0.01)
    phase = np.unwrap(np.angle(y[middle, 0]))
    f_measured = np.polyfit(np.arange(len(phase)) / rate, phase, 1)[0] / (2 * np.pi)
    assert f_measured + fc == pytest.approx(731.0, abs=0.01)


def test_zoom_baseband_rejects_too_much_decimation() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        zoom_baseband(np.zeros((1000, 1)), 20_000.0, 0.0, 5000.0, 16)


@pytest.fixture(scope="module")
def chirp_cutout(synthetic_trial: Path):
    queue = build_queue([synthetic_trial], QueueConfig(per_bin=50))
    (item,) = [i for i in queue.items if abs(i.t_center_s - 4.0) < 1e-6]
    rec, tracks = open_recording(synthetic_trial), load_tracks(synthetic_trial)
    return render_cutout(rec, tracks, item, CutoutConfig())


def test_extent_matches_requested_window(chirp_cutout) -> None:
    cut = chirp_cutout
    t0, t1, f0, f1 = cut.window
    assert (t0, t1) == pytest.approx((3.93 - 0.15, 4.07 + 0.15))
    assert (f0, f1) == pytest.approx((680.0 - 40.0, 820.0 + 40.0))
    n_freq, n_time = cut.db.shape
    dt = (cut.extent[1] - cut.extent[0]) / n_time
    df = (cut.extent[3] - cut.extent[2]) / n_freq
    assert dt == pytest.approx(32 / 20_000.0)
    np.testing.assert_allclose(cut.extent[:2], cut.window[:2], atol=dt)
    np.testing.assert_allclose(cut.extent[2:], cut.window[2:], atol=df)


def test_baselines_are_live_tracks_by_frequency(chirp_cutout) -> None:
    assert chirp_cutout.baselines == ((0, 700.0), (1, 760.0))


def test_baseline_rows_hold_the_fish(chirp_cutout) -> None:
    cut = chirp_cutout
    n_freq, _ = cut.db.shape
    df = (cut.extent[3] - cut.extent[2]) / n_freq
    freqs = cut.extent[2] + (np.arange(n_freq) + 0.5) * df
    before = cut.db[:, 0]  # first column, ~0.2 s before the chirp
    for f in (700.0, 760.0):
        near = np.flatnonzero(np.abs(freqs - f) < 10.0)
        peak = near[np.argmax(before[near])]
        assert abs(freqs[peak] - f) <= df


def test_chirp_raises_power_above_the_emitter(chirp_cutout) -> None:
    cut = chirp_cutout
    n_freq, n_time = cut.db.shape
    freqs = (
        cut.extent[2]
        + (np.arange(n_freq) + 0.5) * (cut.extent[3] - cut.extent[2]) / n_freq
    )
    times = (
        cut.extent[0]
        + (np.arange(n_time) + 0.5) * (cut.extent[1] - cut.extent[0]) / n_time
    )
    band = (freqs > 780) & (freqs < 820)
    at_chirp = cut.db[band][:, np.argmin(np.abs(times - 4.0))].mean()
    before = cut.db[band][:, 0].mean()
    assert at_chirp > before + 10.0
    assert cut.clim[1] == pytest.approx(float(cut.db.max()))


def test_cutout_config_is_in_seconds() -> None:
    config = CutoutConfig()
    assert (config.window_s, config.hop_s) == (0.2048, 0.0016)
    with pytest.raises(TypeError):
        CutoutConfig(nfft=4096)  # pyright: ignore[reportCallIssue]
    with pytest.raises(ValueError, match="sample count"):
        CutoutConfig(window_s=4096)
    with pytest.raises(ValueError, match="seconds"):
        CutoutConfig(hop_s=0.0)


def test_sampling_at_20_khz_is_the_former_sample_counts() -> None:
    # 4096-sample window, 32-sample hop, decimated by 32 for a narrow band.
    assert _sampling(0.2048, 0.0016, 20_000.0, 300.0) == (32, 128, 1)
    q, n_window, n_hop = _sampling(0.2048, 0.0016, 20_000.0, 900.0)
    assert (n_window * q, n_hop * q) == (4096, 32)


@pytest.mark.parametrize("fs", [22_050.0, 44_100.0, 48_000.0, 96_000.0])
def test_sampling_at_other_rates_decimates_and_keeps_the_durations(fs: float) -> None:
    q, n_window, n_hop = _sampling(0.2048, 0.0016, fs, 300.0)
    assert fs / q >= 1.25 * 300.0  # the band still fits the decimated rate
    assert q >= 16  # no fallback to full-rate spectra
    assert abs(n_window * q / fs - 0.2048) <= 0.5 * q / fs
    assert abs(n_hop * q / fs - 0.0016) <= 0.5 * q / fs


# The scene of the synthetic trial (tests/conftest.py), recorded at 48 kHz.
FISH_HZ = (700.0, 760.0)
GAINS = (np.array([1.0, 0.5, 0.2, 0.05]), np.array([0.05, 0.2, 0.5, 1.0]))


def _write_48_khz_recording(trial: Path) -> None:
    fs = 48_000.0
    t = np.arange(int(8.0 * fs)) / fs
    excursion = 100.0 * np.exp(-0.5 * ((t - 4.0) / 0.0085) ** 2)
    signal = np.zeros((len(t), 4))
    for k, (f, gains) in enumerate(zip(FISH_HZ, GAINS, strict=True)):
        freq = f + (excursion if k == 0 else np.zeros_like(t))
        signal += np.sin(2 * np.pi * np.cumsum(freq) / fs)[:, None] * gains[None, :]
    signal += 0.01 * np.random.default_rng(1).standard_normal(signal.shape)
    wavfile.write(trial / "recording.wav", int(fs), signal.astype(np.float32))


def test_cutout_at_48_khz_matches_the_one_at_20_khz(
    make_synthetic_trial: Callable[[Path], Path], tmp_path: Path, chirp_cutout
) -> None:
    trial = make_synthetic_trial(tmp_path / "2000-01-01-10_00")
    _write_48_khz_recording(trial)
    rec = open_recording(trial)
    assert rec.samplerate_hz == 48_000.0
    queue = build_queue([trial], QueueConfig(per_bin=50))
    (item,) = [i for i in queue.items if abs(i.t_center_s - 4.0) < 1e-6]
    cut = render_cutout(rec, load_tracks(trial), item, CutoutConfig())
    ref = chirp_cutout  # the same detection at 20 kHz
    n_freq, n_time = cut.db.shape
    dt = (cut.extent[1] - cut.extent[0]) / n_time
    df = (cut.extent[3] - cut.extent[2]) / n_freq
    assert dt == pytest.approx(0.0016, rel=0.01)
    assert df == pytest.approx(
        (ref.extent[3] - ref.extent[2]) / ref.db.shape[0], rel=0.01
    )
    np.testing.assert_allclose(cut.extent, ref.extent, atol=max(dt, df))
    freqs = cut.extent[2] + (np.arange(n_freq) + 0.5) * df
    times = cut.extent[0] + (np.arange(n_time) + 0.5) * dt
    for f in FISH_HZ:
        near = np.flatnonzero(np.abs(freqs - f) < 10.0)
        assert abs(freqs[near[np.argmax(cut.db[near, 0])]] - f) <= df
    band = (freqs > 780) & (freqs < 820)
    at_chirp = cut.db[band][:, np.argmin(np.abs(times - 4.0))].mean()
    assert at_chirp > cut.db[band][:, 0].mean() + 10.0
