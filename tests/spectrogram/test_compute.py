import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

import chirpdetector.spectrogram.compute as compute_module
from chirpdetector.spectrogram import (
    SpectrogramConfig,
    compute_spectrogram,
    compute_spectrogram_span,
    n_frames_total,
    plan_span,
)

CFG = SpectrogramConfig()
FS = CFG.samplerate_hz


def tone(
    freq_hz: float, n_samples: int, amplitude: float = 1.0, phase: float = 0.3
) -> NDArray[np.float32]:
    t = np.arange(n_samples) / FS
    return (amplitude * np.sin(2 * np.pi * freq_hz * t + phase)).astype(np.float32)


def as_power(x: torch.Tensor) -> NDArray[np.float32]:
    return x.cpu().numpy()


def test_output_layout() -> None:
    n_samples = 10_037
    spec = compute_spectrogram(np.zeros((n_samples, 4), np.float32), CFG, t0_s=12.5)
    n_frames = n_frames_total(n_samples, CFG)
    assert spec.power.shape == (3, CFG.n_freqs, n_frames)
    assert spec.power.dtype == torch.float32
    np.testing.assert_array_equal(spec.freqs_hz, CFG.freqs_hz)
    np.testing.assert_allclose(
        spec.times_s,
        12.5 + np.arange(n_frames) * CFG.hop_samples / FS,
        rtol=0,
        atol=1e-12,
    )
    assert spec.frame_start == 0
    assert spec.complete.shape == (n_frames,)


@pytest.mark.parametrize("offset_bins", [0.0, 0.2, 0.4, -0.3])
@pytest.mark.parametrize("bin_index", [5, 120, 300])
def test_tone_lands_in_nearest_bin_in_every_channel(
    bin_index: int, offset_bins: float
) -> None:
    freq_hz = CFG.freqs_hz[bin_index] + offset_bins * CFG.df_hz
    spec = compute_spectrogram(tone(freq_hz, 20_000)[:, None], CFG)
    middle = as_power(spec.power)[:, :, spec.complete]
    for channel in range(CFG.n_windows):
        peak_bins = middle[channel].argmax(axis=0)
        assert np.all(peak_bins == bin_index), channel


def test_on_bin_tone_reads_mean_square_power_in_every_channel() -> None:
    amplitude = 0.3
    spec = compute_spectrogram(tone(CFG.freqs_hz[150], 20_000, amplitude)[:, None], CFG)
    levels_db = 10 * np.log10(as_power(spec.power)[:, 150, spec.complete])
    np.testing.assert_allclose(levels_db, 10 * np.log10(amplitude**2 / 2), atol=0.01)


@pytest.mark.parametrize("click_sample", [9_000, 9_049, 9_051, 10_123])
def test_click_peaks_at_its_time_in_every_channel(click_sample: int) -> None:
    x = np.zeros((20_000, 1), np.float32)
    x[click_sample] = 1.0
    t0_s = 100.0
    spec = compute_spectrogram(x, CFG, t0_s=t0_s)
    click_time = t0_s + click_sample / FS
    peak_frames = as_power(spec.power).sum(axis=1).argmax(axis=1)
    for channel, frame in enumerate(peak_frames):
        assert abs(spec.times_s[frame] - click_time) <= CFG.hop_s / 2 + 1e-12, channel
    # Multi-resolution alignment: the same frame in all channels.
    assert len(set(peak_frames.tolist())) == 1


def test_stationary_tone_has_the_same_frequency_index_in_all_channels() -> None:
    freq_hz = 777.7
    spec = compute_spectrogram(tone(freq_hz, 20_000)[:, None], CFG)
    peaks = as_power(spec.power)[:, :, spec.complete].mean(axis=2).argmax(axis=1)
    assert len(set(peaks.tolist())) == 1
    assert peaks[0] == np.argmin(np.abs(CFG.freqs_hz - freq_hz))


def test_short_windows_localise_a_brief_frequency_excursion() -> None:
    """A 20 ms excursion is much narrower in time in the shortest channel."""
    n = 20_000
    t = np.arange(n) / FS
    freq = np.full(n, 800.0)
    chirp = (t >= 0.5) & (t < 0.52)
    freq[chirp] += 200.0
    x = np.sin(2 * np.pi * np.cumsum(freq) / FS).astype(np.float32)[:, None]
    spec = compute_spectrogram(x, CFG)
    row = np.argmin(np.abs(CFG.freqs_hz - 1000.0))
    widths_s = []
    for channel in range(CFG.n_windows):
        trace = as_power(spec.power)[channel, row]
        widths_s.append(np.count_nonzero(trace > trace.max() / 2) * CFG.hop_s)
    long_s, short_s = widths_s[0], widths_s[-1]
    assert short_s <= 0.035
    assert short_s < long_s / 3


def _multi_electrode_recording(n_samples: int, seed: int = 0) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    gains = np.array([1.0, 0.4, 0.1], np.float32)
    fish = tone(720.0, n_samples) + 0.5 * tone(941.3, n_samples)
    noise = 0.05 * rng.standard_normal((n_samples, 3)).astype(np.float32)
    return (fish[:, None] * gains + noise).astype(np.float32)


def test_sum_aggregation_is_the_sum_of_electrode_powers() -> None:
    x = _multi_electrode_recording(12_000)
    total = as_power(compute_spectrogram(x, CFG).power)
    singles = [
        as_power(compute_spectrogram(x[:, [e]], CFG).power) for e in range(x.shape[1])
    ]
    np.testing.assert_allclose(total, np.sum(singles, axis=0), rtol=1e-5)
    # Linear power, not dB: the sum of dB values would be far off.
    db_sum = np.sum([10 * np.log10(p) for p in singles], axis=0)
    assert not np.allclose(10 * np.log10(total), db_sum, atol=1.0)


def test_max_aggregation_is_the_max_of_electrode_powers() -> None:
    x = _multi_electrode_recording(12_000)
    cfg = SpectrogramConfig(aggregation="max")
    loudest = as_power(compute_spectrogram(x, cfg).power)
    singles = [
        as_power(compute_spectrogram(x[:, [e]], cfg).power) for e in range(x.shape[1])
    ]
    np.testing.assert_allclose(loudest, np.max(singles, axis=0), rtol=1e-6)


RECORDING = _multi_electrode_recording(31_337, seed=1)
WHOLE = compute_spectrogram(RECORDING, CFG, t0_s=3600.0)


def _assert_span_matches_whole(start: int, stop: int) -> None:
    n_total = RECORDING.shape[0]
    plan = plan_span(start, stop, n_total, CFG)
    spec = compute_spectrogram_span(RECORDING[plan.read_slice], plan, CFG, t0_s=3600.0)
    expected = WHOLE.power[:, :, plan.frame_start : plan.frame_stop]
    scale = float(WHOLE.power.max())
    torch.testing.assert_close(spec.power, expected, rtol=1e-6, atol=1e-6 * scale)
    np.testing.assert_array_equal(
        spec.times_s, WHOLE.times_s[plan.frame_start : plan.frame_stop]
    )
    np.testing.assert_array_equal(
        spec.complete, WHOLE.complete[plan.frame_start : plan.frame_stop]
    )
    assert spec.frame_start == plan.frame_start


@settings(max_examples=40, deadline=None)
@given(
    start=st.integers(0, RECORDING.shape[0] - 100),
    length=st.integers(100, RECORDING.shape[0]),
)
def test_span_equals_whole_recording(start: int, length: int) -> None:
    stop = min(RECORDING.shape[0], start + length)
    _assert_span_matches_whole(start, stop)


def test_span_at_both_edges_equals_whole_recording() -> None:
    n_total = RECORDING.shape[0]
    _assert_span_matches_whole(0, 150)
    _assert_span_matches_whole(n_total - 150, n_total)


def test_result_does_not_depend_on_block_size(monkeypatch: pytest.MonkeyPatch) -> None:
    bytes_per_frame = RECORDING.shape[1] * (CFG.n_fft // 2 + 1) * 8
    monkeypatch.setattr(compute_module, "_BLOCK_BYTES", 7 * bytes_per_frame)
    blocked = compute_spectrogram(RECORDING, CFG, t0_s=3600.0)
    torch.testing.assert_close(blocked.power, WHOLE.power, rtol=1e-6, atol=0.0)


def test_click_in_the_last_samples_is_not_lost() -> None:
    n_samples = 20_037
    x = np.zeros((n_samples, 1), np.float32)
    x[-3] = 1.0
    spec = compute_spectrogram(x, CFG)
    broadband = as_power(spec.power)[-1].sum(axis=0)
    assert broadband.argmax() == spec.n_times - 1
    assert not spec.complete[-1]


def test_torch_and_float64_inputs_match_float32_numpy() -> None:
    x = RECORDING[:5_000]
    ref = compute_spectrogram(x, CFG).power
    from_torch = compute_spectrogram(torch.from_numpy(x.copy()), CFG).power
    from_f64 = compute_spectrogram(x.astype(np.float64), CFG).power
    torch.testing.assert_close(from_torch, ref, rtol=0, atol=0)
    torch.testing.assert_close(from_f64, ref, rtol=0, atol=0)


def test_bad_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="n_samples, n_electrodes"):
        compute_spectrogram(np.zeros(1_000, np.float32), CFG)
    with pytest.raises(ValueError, match="transposed"):
        compute_spectrogram(np.zeros((15, 1_000), np.float32), CFG)
    with pytest.raises(TypeError, match="floating"):
        compute_spectrogram(np.zeros((1_000, 2), np.int16), CFG)  # pyright: ignore[reportArgumentType]
    plan = plan_span(0, 1_000, 10_000, CFG)
    with pytest.raises(ValueError, match="expected"):
        compute_spectrogram_span(np.zeros((10, 2), np.float32), plan, CFG)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu_and_spans_match_whole() -> None:
    x = RECORDING[:8_000, :2]
    cpu = compute_spectrogram(x, CFG).power
    gpu_spec = compute_spectrogram(x, CFG, device="cuda")
    assert gpu_spec.power.device.type == "cuda"
    scale = float(cpu.max())
    torch.testing.assert_close(gpu_spec.power.cpu(), cpu, rtol=1e-4, atol=1e-5 * scale)
    plan = plan_span(2_000, 5_000, x.shape[0], CFG)
    span = compute_spectrogram_span(x[plan.read_slice], plan, CFG, device="cuda")
    torch.testing.assert_close(
        span.power,
        gpu_spec.power[:, :, plan.frame_start : plan.frame_stop],
        rtol=1e-6,
        atol=1e-6 * scale,
    )
