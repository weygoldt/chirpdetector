"""Cutout rendering: band zoom, geometry, and that the chirp shows up."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gridtools.io import load_tracks, open_recording

from chirpdetector.labeling.queue import QueueConfig, build_queue
from chirpdetector.labeling.render import CutoutConfig, render_cutout, zoom_baseband


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
