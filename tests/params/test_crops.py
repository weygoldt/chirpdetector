"""Crop geometry: one definition of the decoder input, wherever it is made."""

from __future__ import annotations

import numpy as np
import pytest

from chirpdetector.data.tiles import slice_spectrogram
from chirpdetector.params.crops import (
    CropConfig,
    canvas_range,
    crop_window,
    decoder_inputs,
)
from chirpdetector.spectrogram import (
    SpectrogramConfig,
    compute_spectrogram,
    compute_spectrogram_span,
    normalize,
    plan_span,
    to_image,
)

FS = 20_000.0
CONFIG = SpectrogramConfig()


def _signal(duration_s: float, seed: int = 0) -> np.ndarray:
    """Two electrodes: a fish at 700 Hz with a 40-ms chirp at 2 s, plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * FS)) / FS
    freq = 700.0 + 150.0 * np.exp(-0.5 * ((t - 2.0) / 0.015) ** 2)
    phase = 2 * np.pi * np.cumsum(freq) / FS
    eod = np.sin(phase)
    noise = 0.01 * rng.normal(size=(t.size, 2))
    return (np.stack([eod, 0.5 * eod], axis=1) + noise).astype(np.float32)


@pytest.fixture(scope="module")
def whole() -> tuple[np.ndarray, object]:
    signal = _signal(8.0)
    return signal, compute_spectrogram(signal, FS, CONFIG)


def test_canvas_range_is_centred_and_shifted_at_edges() -> None:
    assert canvas_range(1000, 500, 640) == (180, 820, False)
    assert canvas_range(1000, 100, 640) == (0, 640, True)
    assert canvas_range(1000, 990, 640) == (360, 1000, True)
    assert canvas_range(300, 100, 640) == (0, 300, True)
    assert crop_window((1.0, 2.0, 100.0, 200.0), 0.1) == (0.9, 2.1, 90.0, 210.0)


def test_crop_from_a_span_equals_crop_from_the_whole_recording(whole) -> None:
    """Inference cuts a span around the box; training the whole scene."""
    signal, spec = whole
    box = np.array([[1.95, 2.06, 680.0, 870.0], [3.0, 3.2, 650.0, 760.0]])
    full = decoder_inputs(spec, box)
    config = CropConfig()
    for t_lo, t_hi in ((0.0, 4.0), (0.3, 5.0)):
        start, stop = int((t_lo) * FS), int(t_hi * FS)
        start = max(0, start - int(config.context_s * FS))
        stop = min(len(signal), stop + int(config.context_s * FS))
        plan = plan_span(start, stop, len(signal), FS, CONFIG)
        part = compute_spectrogram_span(
            signal[plan.read_start : plan.read_stop], plan, CONFIG
        )
        span = decoder_inputs(part, box)
        np.testing.assert_array_equal(span.crops, full.crops)
        np.testing.assert_array_equal(span.scalars, full.scalars)
        assert not span.canvas_shifted.any()


def test_crop_samples_the_normalised_canvas_exactly(whole) -> None:
    """With the crop grid on frames and bins, the crop is the tile image."""
    _, spec = whole
    hop, df = spec.layout.hop_s, spec.layout.df_hz
    k0, k1 = 380, 420  # frames: 40 columns
    b0, b1 = 30, 70  # bins: 40 rows
    box = [
        spec.times_s[k0],
        spec.times_s[k1 - 1],
        spec.freqs_hz[b0],
        spec.freqs_hz[b1 - 1],
    ]
    config = CropConfig(size=(b1 - b0, k1 - k0), margin_rel=0.0)
    inputs = decoder_inputs(spec, [box], config)
    lo, hi, shifted = canvas_range(
        spec.n_times,
        round((0.5 * (box[0] + box[1]) - spec.times_s[0]) / hop),
        config.canvas_frames(hop),
    )
    assert not shifted
    image = to_image(normalize(slice_spectrogram(spec, lo, hi), CONFIG))
    expected = image[::-1][b0:b1, k0 - lo : k1 - lo].transpose(
        2, 0, 1
    )  # row 0 = lowest f
    np.testing.assert_array_equal(inputs.crops[0], expected)
    assert inputs.scalars[0, 0] == pytest.approx(np.log(box[1] - box[0]))
    assert inputs.scalars[0, 1] == pytest.approx(np.log(box[3] - box[2]), rel=1e-6)
    assert df > 0


def test_the_chirp_sits_where_the_box_says(whole) -> None:
    _, spec = whole
    inputs = decoder_inputs(spec, [[1.95, 2.05, 690.0, 860.0]])
    crop = inputs.crops[0, 2].astype(float)  # shortest window
    rows = np.linspace(690 - 25.5, 860 + 25.5, 96)
    peak_row = rows[np.argmax(crop[:, 48])]  # centre column: top of the chirp
    assert peak_row == pytest.approx(850.0, abs=10.0)
    baseline_row = rows[np.argmax(crop[:, 2])]  # first column: the baseline
    assert baseline_row == pytest.approx(700.0, abs=5.0)


def test_edges_of_recording_and_band_are_flagged(whole) -> None:
    _, spec = whole
    inputs = decoder_inputs(
        spec,
        [
            [0.0, 0.2, 650.0, 750.0],  # at the recording's start
            [3.0, 3.2, 1450.0, 1520.0],  # above the band
            [3.0, 3.2, 650.0, 750.0],  # inside
        ],
    )
    assert inputs.outside_time[0] > 0
    assert inputs.canvas_shifted[0]
    assert inputs.outside_freq[1] > 0
    assert inputs.outside_time[1] == 0
    assert inputs.outside_time[2] == 0
    assert inputs.outside_freq[2] == 0
    assert not inputs.canvas_shifted[2]
    # columns outside the recording read 0
    assert (inputs.crops[0][:, :, :5] == 0).all()


def test_bad_boxes_raise_and_empty_is_empty(whole) -> None:
    _, spec = whole
    with pytest.raises(ValueError, match="not finite"):
        decoder_inputs(spec, [[1.0, 0.9, 600.0, 700.0]])
    with pytest.raises(ValueError, match="not finite"):
        decoder_inputs(spec, [[1.0, 1.1, np.nan, 700.0]])
    empty = decoder_inputs(spec, np.zeros((0, 4)))
    assert empty.crops.shape == (0, 3, 96, 96)
    assert empty.scalars.shape == (0, 2)


def test_config_round_trips_and_validates() -> None:
    config = CropConfig(size=(64, 80), margin_rel=0.2)
    assert CropConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="unknown"):
        CropConfig.from_dict({"sizes": [1, 2]})
    with pytest.raises(ValueError, match="canvas_s"):
        CropConfig(canvas_s=0.0)
    assert config.canvas_frames(0.005) == 640
