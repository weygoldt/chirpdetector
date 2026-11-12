import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from chirpdetector.spectrogram import (
    SpectrogramConfig,
    complete_frames,
    context_samples,
    frame_times_s,
    n_frames_total,
    plan_span,
)

CFG = SpectrogramConfig()
RATES = (20_000.0, 44_100.0, 48_000.0)
# 96 kHz has an odd FFT length (19661): frames cover [c - 9830, c + 9831).
RATES_WITH_ODD_FFT = (*RATES, 96_000.0)


@pytest.mark.parametrize("fs", RATES)
@pytest.mark.parametrize("n_samples", [1, 99, 100, 101, 4096, 20_000, 20_037])
def test_frames_cover_the_tail(fs: float, n_samples: int) -> None:
    hop = CFG.layout(fs).hop_samples
    n_frames = n_frames_total(n_samples, fs, CFG)
    last_centre = (n_frames - 1) * hop
    assert last_centre <= n_samples - 1
    assert last_centre > n_samples - 1 - hop


def test_context_is_half_the_fft_length() -> None:
    assert context_samples(20_000.0, CFG) == 2048
    assert context_samples(48_000.0, CFG) == 4915
    assert context_samples(96_000.0, CFG) == 19661 // 2
    single = SpectrogramConfig(reference_window_s=0.0512, extra_windows_s=())
    assert context_samples(20_000.0, single) == 512


@pytest.mark.parametrize("fs", RATES)
def test_frame_times_are_exact_sample_times(fs: float) -> None:
    frames = np.array([0, 1, 7, 123_456_789], dtype=np.int64)
    hop = CFG.layout(fs).hop_samples
    times = frame_times_s(frames, fs, CFG, t0_s=10.0)
    np.testing.assert_array_equal(times, 10.0 + (frames * hop) / fs)


@given(
    fs=st.sampled_from(RATES_WITH_ODD_FFT),
    n_seconds=st.floats(0.25, 10.0),
    cuts=st.lists(st.floats(0.0, 1.0), max_size=8),
)
def test_spans_partition_the_frames(
    fs: float, n_seconds: float, cuts: list[float]
) -> None:
    n_samples = round(n_seconds * fs)
    hop = CFG.layout(fs).hop_samples
    edges = sorted({0, n_samples, *(int(c * n_samples) for c in cuts)})
    frames: list[int] = []
    for start, stop in itertools.pairwise(edges):
        if -(-start // hop) == -(-stop // hop):  # span holds no frame centre
            with pytest.raises(ValueError, match="no frame centre"):
                plan_span(start, stop, n_samples, fs, CFG)
            continue
        plan = plan_span(start, stop, n_samples, fs, CFG)
        frames.extend(range(plan.frame_start, plan.frame_stop))
    assert frames == list(range(n_frames_total(n_samples, fs, CFG)))


@given(
    fs=st.sampled_from(RATES_WITH_ODD_FFT),
    n_seconds=st.floats(0.25, 10.0),
    start_frac=st.floats(0.0, 1.0),
    length_s=st.floats(0.0, 3.0),
)
def test_plan_reads_exactly_the_needed_samples(
    fs: float, n_seconds: float, start_frac: float, length_s: float
) -> None:
    layout = CFG.layout(fs)
    hop, n_fft = layout.hop_samples, layout.n_fft
    n_samples = round(n_seconds * fs)
    start = min(int(start_frac * n_samples), n_samples - 1)
    stop = min(n_samples, start + 1 + round(length_s * fs))
    if -(-start // hop) == -(-stop // hop):
        return
    plan = plan_span(start, stop, n_samples, fs, CFG)
    assert plan.samplerate_hz == fs
    half = context_samples(fs, CFG)
    first = plan.frame_start * hop - half
    last = (plan.frame_stop - 1) * hop - half + n_fft
    assert plan.read_start - plan.pad_before == first
    assert plan.read_stop + plan.pad_after == last
    assert 0 <= plan.read_start < plan.read_stop <= n_samples
    assert plan.pad_before == max(0, -first)
    assert plan.pad_after == max(0, last - n_samples)
    assert plan.n_read + plan.pad_before + plan.pad_after == (
        (plan.n_frames - 1) * hop + n_fft
    )
    assert plan.read_slice == slice(plan.read_start, plan.read_stop)


@pytest.mark.parametrize(
    ("start", "stop", "total"),
    [
        (0, 0, 10_000),
        (10, 5, 10_000),
        (-1, 50, 10_000),
        (0, 10_001, 10_000),
        (1, 50, 10_000),  # no frame centre (multiples of 100) inside
    ],
)
def test_invalid_spans_are_rejected(start: int, stop: int, total: int) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - message varies by case
        plan_span(start, stop, total, 20_000.0, CFG)


@pytest.mark.parametrize("fs", RATES)
def test_recording_shorter_than_the_reference_window_is_rejected(fs: float) -> None:
    n_window = CFG.layout(fs).window_lengths_samples[0]
    with pytest.raises(ValueError, match="shorter than the reference window"):
        plan_span(0, n_window - 1, n_window - 1, fs, CFG)
    plan_span(0, n_window, n_window, fs, CFG)


def test_band_above_nyquist_is_rejected_when_planning() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        plan_span(0, 10_000, 10_000, 2_000.0, CFG)


@pytest.mark.parametrize("fs", RATES_WITH_ODD_FFT)
def test_complete_frames_flag_zero_padded_edges(fs: float) -> None:
    layout = CFG.layout(fs)
    n_samples = round(1.0 * fs)
    flags = complete_frames(0, n_frames_total(n_samples, fs, CFG), n_samples, fs, CFG)
    centres = np.arange(flags.size) * layout.hop_samples
    first = centres - layout.n_fft // 2
    expected = (first >= 0) & (first + layout.n_fft <= n_samples)
    np.testing.assert_array_equal(flags, expected)
    assert flags.any()
    assert not flags[0]
    assert not flags[-1]
