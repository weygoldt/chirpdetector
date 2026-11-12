import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from chirpdetector.spectrogram import (
    SpectrogramConfig,
    complete_frames,
    context_samples,
    n_frames_total,
    plan_span,
)

CFG = SpectrogramConfig()


@pytest.mark.parametrize("n_samples", [1, 99, 100, 101, 4096, 20_000, 20_037])
def test_frames_cover_the_tail(n_samples: int) -> None:
    hop = CFG.hop_samples
    n_frames = n_frames_total(n_samples, CFG)
    last_centre = (n_frames - 1) * hop
    assert last_centre <= n_samples - 1
    assert last_centre > n_samples - 1 - hop


def test_context_is_half_the_reference_window() -> None:
    assert context_samples(CFG) == 2048
    assert context_samples(SpectrogramConfig(window_lengths_samples=(1024,))) == 512


@given(
    n_samples=st.integers(1_000, 200_000),
    cuts=st.lists(st.floats(0.0, 1.0), max_size=8),
)
def test_spans_partition_the_frames(n_samples: int, cuts: list[float]) -> None:
    hop = CFG.hop_samples
    edges = sorted({0, n_samples, *(int(c * n_samples) for c in cuts)})
    frames: list[int] = []
    for start, stop in itertools.pairwise(edges):
        if -(-start // hop) == -(-stop // hop):  # span holds no frame centre
            with pytest.raises(ValueError, match="no frame centre"):
                plan_span(start, stop, n_samples, CFG)
            continue
        plan = plan_span(start, stop, n_samples, CFG)
        frames.extend(range(plan.frame_start, plan.frame_stop))
    assert frames == list(range(n_frames_total(n_samples, CFG)))


@given(
    n_samples=st.integers(1_000, 200_000),
    start_frac=st.floats(0.0, 1.0),
    length=st.integers(100, 50_000),
)
def test_plan_reads_exactly_the_needed_samples(
    n_samples: int, start_frac: float, length: int
) -> None:
    start = min(int(start_frac * n_samples), n_samples - 1)
    stop = min(n_samples, start + length)
    if -(-start // CFG.hop_samples) == -(-stop // CFG.hop_samples):
        return
    plan = plan_span(start, stop, n_samples, CFG)
    half = context_samples(CFG)
    first = plan.frame_start * CFG.hop_samples - half
    last = (plan.frame_stop - 1) * CFG.hop_samples + half
    assert plan.read_start - plan.pad_before == first
    assert plan.read_stop + plan.pad_after == last
    assert 0 <= plan.read_start < plan.read_stop <= n_samples
    assert plan.pad_before == max(0, -first)
    assert plan.pad_after == max(0, last - n_samples)
    assert plan.n_read + plan.pad_before + plan.pad_after == (
        (plan.n_frames - 1) * CFG.hop_samples + CFG.n_fft
    )
    assert plan.read_slice == slice(plan.read_start, plan.read_stop)


@pytest.mark.parametrize(
    ("start", "stop", "total"),
    [(0, 0, 1000), (10, 5, 1000), (-1, 50, 1000), (0, 1001, 1000), (1, 50, 1000)],
)
def test_invalid_spans_are_rejected(start: int, stop: int, total: int) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - message varies by case
        plan_span(start, stop, total, CFG)


def test_complete_frames_flag_zero_padded_edges() -> None:
    n_samples = 20_000
    flags = complete_frames(0, n_frames_total(n_samples, CFG), n_samples, CFG)
    centres = np.arange(flags.size) * CFG.hop_samples
    expected = (centres >= 2048) & (centres + 2048 <= n_samples)
    np.testing.assert_array_equal(flags, expected)
    assert flags.any()
    assert not flags[0]
    assert not flags[-1]
