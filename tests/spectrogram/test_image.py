import math

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from chirpdetector.spectrogram import (
    ImageGeometry,
    SpectrogramConfig,
    compute_spectrogram,
    normalize,
    pixel_box_to_timefreq,
    pixel_to_timefreq,
    timefreq_box_to_pixel,
    timefreq_to_pixel,
    to_image,
    to_three_channels,
)

CFG = SpectrogramConfig()
FS = CFG.samplerate_hz


def tone(freq_hz: float, n_samples: int) -> NDArray[np.float32]:
    return np.sin(2 * np.pi * freq_hz * np.arange(n_samples) / FS).astype(np.float32)


def test_image_layout_and_orientation() -> None:
    n = 20_000
    x = np.zeros((n, 1), np.float32)
    x[: n // 2, 0] = tone(600.0, n // 2)  # low tone first
    x[n // 2 :, 0] = tone(1500.0, n - n // 2)  # high tone second
    spec = compute_spectrogram(x, CFG)
    normalized = normalize(spec, CFG)
    image = to_image(normalized)
    assert image.dtype == np.uint8
    assert image.shape == (spec.n_freqs, spec.n_times, spec.n_windows)
    for channel in range(spec.n_windows):
        # Row 0 = highest frequency, column 0 = first frame.
        expected = np.rint(normalized[channel].numpy()[::-1, :] * 255)
        np.testing.assert_array_equal(image[:, :, channel], expected)
    early, late = 50, spec.n_times - 50
    row_low = image[:, early, 0].argmax()
    row_high = image[:, late, 0].argmax()
    assert row_high < row_low
    assert row_low == spec.n_freqs - 1 - np.argmin(np.abs(spec.freqs_hz - 600.0))


def test_to_image_quantisation() -> None:
    values = torch.tensor([0.0, 1.0, 0.5, -0.2, 1.3, 0.25]).reshape(1, 6, 1)
    pixels = to_image(values)[:, 0, 0][::-1]  # undo the frequency flip
    np.testing.assert_array_equal(pixels, [0, 255, 128, 0, 255, 64])
    with pytest.raises(ValueError, match="NaN"):
        to_image(torch.full((1, 2, 2), float("nan")))


def _geometry() -> ImageGeometry:
    spec = compute_spectrogram(np.zeros((12_345, 1), np.float32), CFG, t0_s=7200.0)
    return ImageGeometry.from_spectrogram(spec)


GEOMETRY = _geometry()


def test_pixel_centres_are_frame_times_and_bin_frequencies() -> None:
    spec = compute_spectrogram(np.zeros((12_345, 1), np.float32), CFG, t0_s=7200.0)
    geom = ImageGeometry.from_spectrogram(spec)
    rows = np.arange(spec.n_freqs)
    x, _ = timefreq_to_pixel(spec.times_s, 1000.0, geom)
    np.testing.assert_allclose(x, np.arange(spec.n_times) + 0.5, atol=1e-6)
    _, y = timefreq_to_pixel(spec.times_s[0], spec.freqs_hz, geom)
    np.testing.assert_allclose(y, (spec.n_freqs - 1 - rows) + 0.5, atol=1e-9)
    t_left, t_right = geom.time_extent_s
    f_bottom, f_top = geom.freq_extent_hz
    assert t_left == pytest.approx(spec.times_s[0] - CFG.hop_s / 2)
    assert t_right == pytest.approx(spec.times_s[-1] + CFG.hop_s / 2)
    assert f_bottom == pytest.approx(spec.freqs_hz[0] - CFG.df_hz / 2)
    assert f_top == pytest.approx(spec.freqs_hz[-1] + CFG.df_hz / 2)
    x, y = timefreq_to_pixel([t_left, t_right], [f_top, f_bottom], geom)
    np.testing.assert_allclose(x, [0.0, geom.width_px], atol=1e-6)
    np.testing.assert_allclose(y, [0.0, geom.height_px], atol=1e-9)


@given(
    t_s=st.floats(7100.0, 7300.0),
    f_hz=st.floats(0.0, 10_000.0),
)
def test_timefreq_round_trip(t_s: float, f_hz: float) -> None:
    x, y = timefreq_to_pixel(t_s, f_hz, GEOMETRY)
    t_back, f_back = pixel_to_timefreq(x, y, GEOMETRY)
    assert math.isclose(float(t_back), t_s, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(float(f_back), f_hz, rel_tol=0, abs_tol=1e-9)


@given(x=st.floats(-1e4, 1e4), y=st.floats(-1e3, 1e3))
def test_pixel_round_trip(x: float, y: float) -> None:
    t_s, f_hz = pixel_to_timefreq(x, y, GEOMETRY)
    x_back, y_back = timefreq_to_pixel(t_s, f_hz, GEOMETRY)
    assert math.isclose(float(x_back), x, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(float(y_back), y, rel_tol=0, abs_tol=1e-9)


def test_box_round_trip_and_flip() -> None:
    box = (7200.3, 7200.4, 700.0, 900.0)
    x_min, y_min, x_max, y_max = timefreq_box_to_pixel(*box, GEOMETRY)
    assert x_min < x_max
    assert y_min < y_max  # the high-frequency edge is the top edge
    back = pixel_box_to_timefreq(x_min, y_min, x_max, y_max, GEOMETRY)
    np.testing.assert_allclose(np.array(back, dtype=np.float64), box, atol=1e-9)


@pytest.mark.parametrize("channel", [0, 1, 2])
def test_box_around_a_tone_burst_contains_its_energy(channel: int) -> None:
    n = 30_000
    t0_s = 50.0
    f0_hz, burst_start_s, burst_stop_s, ramp_s = 1013.0, 0.6, 0.75, 0.025
    t = np.arange(n) / FS
    # Cosine on/off ramps keep the burst's own bandwidth to ~2 / duration.
    envelope = np.clip(
        np.minimum(t - burst_start_s, burst_stop_s - t) / ramp_s, 0.0, 1.0
    )
    envelope = np.sin(0.5 * np.pi * envelope) ** 2
    x = (envelope * np.sin(2 * np.pi * f0_hz * t)).astype(np.float32)[:, None]
    spec = compute_spectrogram(x, CFG, t0_s=t0_s)
    image = to_image(normalize(spec, CFG))[:, :, channel]
    power = spec.power[channel].numpy()[::-1, :].astype(np.float64)  # image layout
    geom = ImageGeometry.from_spectrogram(spec)

    # Box = burst extent, widened by this channel's smear: half a window in
    # time; the Hann main lobe (2 native bins) plus the burst bandwidth in
    # frequency.
    window = CFG.window_lengths_samples[channel]
    half_window_s = window / 2 / FS
    half_band_hz = 2 * FS / window + 2 / (burst_stop_s - burst_start_s)
    x_min, y_min, x_max, y_max = timefreq_box_to_pixel(
        t0_s + burst_start_s - half_window_s,
        t0_s + burst_stop_s + half_window_s,
        f0_hz - half_band_hz,
        f0_hz + half_band_hz,
        geom,
    )
    rows = slice(max(0, math.floor(y_min)), math.ceil(y_max))
    cols = slice(max(0, math.floor(x_min)), math.ceil(x_max))
    assert power[rows, cols].sum() >= 0.99 * power.sum()
    peak_row, peak_col = np.unravel_index(image.argmax(), image.shape)
    assert rows.start <= peak_row < rows.stop
    assert cols.start <= peak_col < cols.stop

    # The energy centroid (pixel centres at index + 0.5) is where the burst's
    # centre time and frequency map to.
    centre_x, centre_y = timefreq_to_pixel(
        t0_s + 0.5 * (burst_start_s + burst_stop_s), f0_hz, geom
    )
    col_centroid = (power.sum(axis=0) * (np.arange(power.shape[1]) + 0.5)).sum()
    row_centroid = (power.sum(axis=1) * (np.arange(power.shape[0]) + 0.5)).sum()
    assert col_centroid / power.sum() == pytest.approx(float(centre_x), abs=0.5)
    assert row_centroid / power.sum() == pytest.approx(float(centre_y), abs=0.5)


def test_three_channel_adapter() -> None:
    rng = np.random.default_rng(0)
    one = rng.integers(0, 256, (4, 5, 1), dtype=np.uint8)
    two = rng.integers(0, 256, (4, 5, 2), dtype=np.uint8)
    three = rng.integers(0, 256, (4, 5, 3), dtype=np.uint8)

    zeros = to_three_channels(one, fill="zeros")
    assert zeros.shape == (4, 5, 3)
    np.testing.assert_array_equal(zeros[:, :, 0], one[:, :, 0])
    assert not zeros[:, :, 1:].any()

    repeated = to_three_channels(two, fill="reference")
    np.testing.assert_array_equal(repeated[:, :, :2], two)
    np.testing.assert_array_equal(repeated[:, :, 2], two[:, :, 0])

    assert to_three_channels(three, fill="zeros") is three
    with pytest.raises(ValueError, match="dropping"):
        to_three_channels(np.zeros((4, 5, 4), np.uint8), fill="zeros")
