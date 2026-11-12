import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

import chirpdetector.spectrogram.compute as compute_module
from chirpdetector.spectrogram import (
    Spectrogram,
    SpectrogramConfig,
    compute_spectrogram,
    compute_spectrogram_span,
    n_frames_total,
    plan_span,
)

# Explicit band: these tests index bins and must not depend on the default band.
CFG = SpectrogramConfig(f_min_hz=400.0, f_max_hz=2000.0)
RATES = (20_000.0, 44_100.0, 48_000.0)
# 96 kHz: odd FFT length (19661) with an even reference window (19660).
RATES_WITH_ODD_FFT = (*RATES, 96_000.0)


def tone(
    freq_hz: float,
    duration_s: float,
    fs: float,
    amplitude: float = 1.0,
    phase: float = 0.3,
) -> NDArray[np.float32]:
    t = np.arange(round(duration_s * fs)) / fs
    return (amplitude * np.sin(2 * np.pi * freq_hz * t + phase)).astype(np.float32)


def as_power(x: torch.Tensor) -> NDArray[np.float32]:
    return x.cpu().numpy()


@pytest.mark.parametrize("fs", RATES)
def test_output_layout(fs: float) -> None:
    n_samples = round(0.5 * fs) + 37
    spec = compute_spectrogram(np.zeros((n_samples, 4), np.float32), fs, CFG, t0_s=12.5)
    layout = CFG.layout(fs)
    n_frames = n_frames_total(n_samples, fs, CFG)
    assert spec.power.shape == (3, CFG.n_freqs, n_frames)
    assert spec.power.dtype == torch.float32
    assert spec.layout == layout
    assert spec.samplerate_hz == fs
    np.testing.assert_array_equal(spec.freqs_hz, layout.freqs_hz)
    # Exact frame times on the recording's absolute sample grid.
    np.testing.assert_array_equal(
        spec.times_s, 12.5 + (np.arange(n_frames) * layout.hop_samples) / fs
    )
    assert spec.frame_start == 0
    assert spec.complete.shape == (n_frames,)


@pytest.mark.parametrize("fs", RATES)
@pytest.mark.parametrize("offset_bins", [0.0, 0.2, 0.4, -0.3])
@pytest.mark.parametrize("bin_index", [5, 120, 300])
def test_tone_lands_in_nearest_bin_in_every_channel(
    fs: float, bin_index: int, offset_bins: float
) -> None:
    layout = CFG.layout(fs)
    freq_hz = layout.freqs_hz[bin_index] + offset_bins * layout.df_hz
    spec = compute_spectrogram(tone(freq_hz, 1.0, fs)[:, None], fs, CFG)
    middle = as_power(spec.power)[:, :, spec.complete]
    for channel in range(CFG.n_windows):
        peak_bins = middle[channel].argmax(axis=0)
        assert np.all(peak_bins == bin_index), channel


@pytest.mark.parametrize("fs", RATES_WITH_ODD_FFT)
def test_on_bin_tone_reads_mean_square_power_in_every_channel(fs: float) -> None:
    amplitude = 0.3
    freq_hz = CFG.layout(fs).freqs_hz[150]
    spec = compute_spectrogram(tone(freq_hz, 1.0, fs, amplitude)[:, None], fs, CFG)
    levels_db = 10 * np.log10(as_power(spec.power)[:, 150, spec.complete])
    np.testing.assert_allclose(levels_db, 10 * np.log10(amplitude**2 / 2), atol=0.01)


@pytest.mark.parametrize("fs", RATES_WITH_ODD_FFT)
@pytest.mark.parametrize("click_s", [0.45, 0.45245, 0.45255, 0.50615])
def test_click_peaks_at_its_time_in_every_channel(fs: float, click_s: float) -> None:
    x = np.zeros((round(fs), 1), np.float32)
    click_sample = round(click_s * fs)
    x[click_sample] = 1.0
    t0_s = 100.0
    spec = compute_spectrogram(x, fs, CFG, t0_s=t0_s)
    click_time = t0_s + click_sample / fs
    peak_frames = as_power(spec.power).sum(axis=1).argmax(axis=1)
    for channel, frame in enumerate(peak_frames):
        error_s = abs(spec.times_s[frame] - click_time)
        assert error_s <= spec.layout.hop_s / 2 + 1e-12, channel
    # Multi-resolution alignment: the same frame in all channels.
    assert len(set(peak_frames.tolist())) == 1


@pytest.mark.parametrize("fs", RATES)
def test_stationary_tone_has_the_same_frequency_index_in_all_channels(
    fs: float,
) -> None:
    freq_hz = 777.7
    spec = compute_spectrogram(tone(freq_hz, 1.0, fs)[:, None], fs, CFG)
    peaks = as_power(spec.power)[:, :, spec.complete].mean(axis=2).argmax(axis=1)
    assert len(set(peaks.tolist())) == 1
    assert peaks[0] == np.argmin(np.abs(spec.freqs_hz - freq_hz))


@pytest.mark.parametrize("fs", RATES)
def test_short_windows_localise_a_brief_frequency_excursion(fs: float) -> None:
    """A 20 ms excursion is much narrower in time in the shortest channel."""
    t = np.arange(round(fs)) / fs
    freq = np.full(t.size, 800.0)
    freq[(t >= 0.5) & (t < 0.52)] += 200.0
    x = np.sin(2 * np.pi * np.cumsum(freq) / fs).astype(np.float32)[:, None]
    spec = compute_spectrogram(x, fs, CFG)
    row = np.argmin(np.abs(spec.freqs_hz - 1000.0))
    widths_s = []
    for channel in range(CFG.n_windows):
        trace = as_power(spec.power)[channel, row]
        widths_s.append(np.count_nonzero(trace > trace.max() / 2) * spec.layout.hop_s)
    long_s, short_s = widths_s[0], widths_s[-1]
    assert short_s <= 0.035
    assert short_s < long_s / 3


def _multi_electrode_recording(
    duration_s: float, fs: float, seed: int = 0, n_electrodes: int = 3
) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    gains = np.geomspace(1.0, 0.1, n_electrodes).astype(np.float32)
    fish = tone(720.0, duration_s, fs) + 0.5 * tone(941.3, duration_s, fs)
    noise = 0.05 * rng.standard_normal((fish.size, n_electrodes)).astype(np.float32)
    return (fish[:, None] * gains + noise).astype(np.float32)


@pytest.mark.parametrize("fs", RATES)
def test_sum_aggregation_is_the_sum_of_electrode_powers(fs: float) -> None:
    x = _multi_electrode_recording(0.6, fs)
    total = as_power(compute_spectrogram(x, fs, CFG).power)
    singles = [
        as_power(compute_spectrogram(x[:, [e]], fs, CFG).power)
        for e in range(x.shape[1])
    ]
    np.testing.assert_allclose(total, np.sum(singles, axis=0), rtol=1e-5)
    # Linear power, not dB: the sum of dB values would be far off.
    db_sum = np.sum([10 * np.log10(p) for p in singles], axis=0)
    assert not np.allclose(10 * np.log10(total), db_sum, atol=1.0)


def test_max_aggregation_is_the_max_of_electrode_powers() -> None:
    fs = 20_000.0
    x = _multi_electrode_recording(0.6, fs)
    cfg = SpectrogramConfig(aggregation="max")
    loudest = as_power(compute_spectrogram(x, fs, cfg).power)
    singles = [
        as_power(compute_spectrogram(x[:, [e]], fs, cfg).power)
        for e in range(x.shape[1])
    ]
    np.testing.assert_allclose(loudest, np.max(singles, axis=0), rtol=1e-6)


T0_S = 3600.0
# 9 and 15 electrodes: torch.sum over electrodes reorders its reduction with
# the block width there (3 electrodes did not show it).
ELECTRODES = (3, 9, 15)
RECORDINGS = {
    (fs, n): _multi_electrode_recording(1.037, fs, seed=1, n_electrodes=n)
    for fs in RATES
    for n in ELECTRODES
}
WHOLES = {
    key: compute_spectrogram(x, key[0], CFG, t0_s=T0_S) for key, x in RECORDINGS.items()
}


def _assert_span_matches_whole(
    fs: float, start: int, stop: int, n_electrodes: int = 3
) -> None:
    recording, whole = RECORDINGS[fs, n_electrodes], WHOLES[fs, n_electrodes]
    plan = plan_span(start, stop, recording.shape[0], fs, CFG)
    assert plan.samplerate_hz == fs
    spec = compute_spectrogram_span(recording[plan.read_slice], plan, CFG, t0_s=T0_S)
    _assert_frames_equal(spec, whole, plan.frame_start, plan.frame_stop)


def _assert_frames_equal(
    spec: Spectrogram, whole: Spectrogram, frame_start: int, frame_stop: int
) -> None:
    # Bit-identical on CPU, not merely close.
    assert torch.equal(spec.power, whole.power[:, :, frame_start:frame_stop])
    np.testing.assert_array_equal(spec.times_s, whole.times_s[frame_start:frame_stop])
    np.testing.assert_array_equal(spec.complete, whole.complete[frame_start:frame_stop])
    assert spec.frame_start == frame_start
    assert spec.layout == whole.layout


@settings(max_examples=45, deadline=None)
@given(
    fs=st.sampled_from(RATES),
    n_electrodes=st.sampled_from(ELECTRODES),
    start_frac=st.floats(0.0, 1.0),
    length_frac=st.floats(0.0, 1.0),
)
def test_span_equals_whole_recording(
    fs: float, n_electrodes: int, start_frac: float, length_frac: float
) -> None:
    n_total = RECORDINGS[fs, n_electrodes].shape[0]
    hop = CFG.layout(fs).hop_samples
    start = round(start_frac * (n_total - hop))
    stop = min(n_total, start + hop + round(length_frac * n_total))
    _assert_span_matches_whole(fs, start, stop, n_electrodes)


@pytest.mark.parametrize("n_electrodes", ELECTRODES)
@pytest.mark.parametrize("fs", RATES)
def test_span_at_both_edges_equals_whole_recording(
    fs: float, n_electrodes: int
) -> None:
    n_total = RECORDINGS[fs, n_electrodes].shape[0]
    hop = CFG.layout(fs).hop_samples
    _assert_span_matches_whole(fs, 0, hop + 50, n_electrodes)
    _assert_span_matches_whole(fs, n_total - hop - 50, n_total, n_electrodes)


@pytest.mark.parametrize("n_electrodes", ELECTRODES)
@pytest.mark.parametrize("fs", RATES)
def test_result_does_not_depend_on_block_size(
    fs: float, n_electrodes: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = RECORDINGS[fs, n_electrodes]
    bytes_per_frame = recording.shape[1] * (CFG.layout(fs).n_fft // 2 + 1) * 8
    monkeypatch.setattr(compute_module, "_BLOCK_BYTES", 7 * bytes_per_frame)
    blocked = compute_spectrogram(recording, fs, CFG, t0_s=T0_S)
    assert torch.equal(blocked.power, WHOLES[fs, n_electrodes].power)


@pytest.mark.parametrize("fs", RATES_WITH_ODD_FFT)
def test_click_in_the_last_samples_is_not_lost(fs: float) -> None:
    n_samples = round(fs) + 37
    x = np.zeros((n_samples, 1), np.float32)
    x[-3] = 1.0
    spec = compute_spectrogram(x, fs, CFG)
    broadband = as_power(spec.power)[-1].sum(axis=0)
    assert broadband.argmax() == spec.n_times - 1
    assert not spec.complete[-1]


def test_torch_and_float64_inputs_match_float32_numpy() -> None:
    fs = 20_000.0
    x = RECORDINGS[fs, 3][:5_000]
    ref = compute_spectrogram(x, fs, CFG).power
    from_torch = compute_spectrogram(torch.from_numpy(x.copy()), fs, CFG).power
    from_f64 = compute_spectrogram(x.astype(np.float64), fs, CFG).power
    torch.testing.assert_close(from_torch, ref, rtol=0, atol=0)
    torch.testing.assert_close(from_f64, ref, rtol=0, atol=0)


def test_bad_inputs_are_rejected() -> None:
    fs = 20_000.0
    with pytest.raises(ValueError, match="n_samples, n_electrodes"):
        compute_spectrogram(np.zeros(5_000, np.float32), fs, CFG)
    with pytest.raises(ValueError, match="transposed"):
        compute_spectrogram(np.zeros((15, 5_000), np.float32), fs, CFG)
    with pytest.raises(TypeError, match="floating"):
        compute_spectrogram(np.zeros((5_000, 2), np.int16), fs, CFG)  # pyright: ignore[reportArgumentType]
    plan = plan_span(0, 1_000, 10_000, fs, CFG)
    with pytest.raises(ValueError, match="expected"):
        compute_spectrogram_span(np.zeros((10, 2), np.float32), plan, CFG)


@pytest.mark.parametrize("fs", RATES)
def test_input_shorter_than_the_reference_window_is_rejected(fs: float) -> None:
    n_window = CFG.layout(fs).window_lengths_samples[0]
    with pytest.raises(ValueError, match="shorter than the reference window"):
        compute_spectrogram(np.zeros((n_window - 1, 2), np.float32), fs, CFG)
    compute_spectrogram(np.zeros((n_window, 2), np.float32), fs, CFG)


def test_band_above_nyquist_is_rejected_per_call() -> None:
    x = np.zeros((10_000, 1), np.float32)
    with pytest.raises(ValueError, match="Nyquist"):
        compute_spectrogram(x, 3_000.0, CFG)  # band reaches 2000 Hz
    compute_spectrogram(x, 4_000.0, CFG)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("fs", [20_000.0, 48_000.0])
def test_cuda_matches_cpu_and_spans_match_whole(fs: float) -> None:
    x = RECORDINGS[fs, 9][: round(0.4 * fs)]
    cpu = compute_spectrogram(x, fs, CFG).power
    gpu_spec = compute_spectrogram(x, fs, CFG, device="cuda")
    assert gpu_spec.power.device.type == "cuda"
    scale = float(cpu.max())
    torch.testing.assert_close(gpu_spec.power.cpu(), cpu, rtol=1e-4, atol=1e-5 * scale)
    plan = plan_span(round(0.1 * fs), round(0.25 * fs), x.shape[0], fs, CFG)
    span = compute_spectrogram_span(x[plan.read_slice], plan, CFG, device="cuda")
    torch.testing.assert_close(
        span.power,
        gpu_spec.power[:, :, plan.frame_start : plan.frame_stop],
        rtol=1e-6,
        atol=1e-6 * scale,
    )
