"""Click removal: what it repairs, what it leaves, and that it is stable.

`TestRepair` and part of `TestSpans` are ported from audian's
``tests/test_clicks.py`` (commit 6fa5274), where each number asserted
against was measured first and the margins are a factor of three or more
beyond it. The port itself is bit-identical to audian's on those signals.
The rest pins what chirpdetector needs: chirps survive, and a recording
cleaned in spans gives the same samples as cleaned whole.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from gridtools.io import open_recording
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt

from chirpdetector.preprocessing import (
    ClickRemoval,
    clean_channel,
    clean_span,
    click_spans,
    plan_clean,
    read_clean,
    remove_clicks,
)
from chirpdetector.spectrogram import (
    SpectrogramConfig,
    compute_spectrogram,
    compute_spectrogram_span,
    plan_span,
)

RATE = 20_000.0

# ------------------------------------------------------------------ config


class TestConfig:
    def test_defaults_are_wavetrackers(self) -> None:
        cfg = ClickRemoval()
        assert (cfg.threshold, cfg.extent_threshold) == (12.0, 8.0)
        assert (cfg.high_band, cfg.high_band_edge) == (0.2, 0.05)
        assert (cfg.max_duration_s, cfg.guard_s, cfg.resolution_s) == (
            0.5,
            0.002,
            0.001,
        )
        assert (cfg.line_margin_db, cfg.line_width_hz, cfg.floor_width_hz) == (
            15.0,
            10.0,
            1000.0,
        )
        assert (cfg.block_s, cfg.margin_s) == (2.0, 0.5)

    @pytest.mark.parametrize(
        "cfg",
        [
            ClickRemoval(),
            ClickRemoval(threshold=48.0, extent_threshold=32.0, high_band=0.4),
        ],
    )
    def test_json_round_trip(self, cfg: ClickRemoval) -> None:
        back = ClickRemoval.from_dict(json.loads(json.dumps(cfg.to_dict())))
        assert back == cfg
        assert hash(back) == hash(cfg)

    def test_missing_keys_take_defaults(self) -> None:
        assert ClickRemoval.from_dict({"threshold": 20}) == ClickRemoval(threshold=20.0)

    def test_unknown_keys_raise(self) -> None:
        with pytest.raises(ValueError, match="unknown ClickRemoval keys"):
            ClickRemoval.from_dict({"treshold": 20})

    def test_numpy_scalars_are_plain_numbers(self) -> None:
        cfg = ClickRemoval(threshold=np.float32(12.0), sample_stride=np.int64(16))  # pyright: ignore[reportArgumentType]
        assert cfg == ClickRemoval()
        assert type(cfg.threshold) is float
        assert type(cfg.sample_stride) is int
        json.dumps(cfg.to_dict())

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"threshold": 0.0}, "threshold must be positive"),
            ({"extent_threshold": 13.0}, "must not exceed"),
            ({"high_band": 0.48}, "Nyquist"),
            ({"high_band_edge": 0.0}, "high_band_edge must be positive"),
            ({"margin_s": 0.4}, "at least max_duration_s"),
            ({"sample_stride": 0}, "sample_stride"),
            ({"floor_quantile": 0.0}, "floor_quantile"),
            ({"guard_s": -1.0}, "guard_s must be >= 0"),
            ({"block_s": float("nan")}, "finite"),
        ],
    )
    def test_invalid_settings_raise(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ClickRemoval(**kwargs)

    def test_blocks_and_margins_in_samples(self) -> None:
        cfg = ClickRemoval()
        assert cfg.block_samples(RATE) == 40_000
        assert cfg.margin_samples(RATE) == 10_000
        assert ClickRemoval(block_s=0.001).block_samples(RATE) == 64


# ------------------------------------------------- one channel (from audian)


class TestRepair:
    """Ported from audian, on its own signals (fixture ``sig``)."""

    def test_every_click_is_found(self, sig: SimpleNamespace) -> None:
        n = sig.N
        x = sig.noise(n) + sig.tone(n) + sum(sig.click(n, a) for a in sig.CLICKS)
        _, mask = clean_channel(x, RATE)
        assert all(mask[a] for a in sig.CLICKS)

    def test_a_click_on_a_loud_tone_is_repaired_and_the_tone_kept(
        self, sig: SimpleNamespace
    ) -> None:
        """Measured in audian: the error in the holes drops from 1.0 to
        0.012, and in the tone's own band from 0.035 to 0.0032."""
        n = sig.N
        truth = sig.noise(n) + sig.tone(n)
        x = truth + sum(sig.click(n, a) for a in sig.CLICKS)
        out, _ = clean_channel(x, RATE)
        hole = sig.holes(n, sig.CLICKS)
        assert np.abs(x - truth)[hole].max() > 0.9
        assert np.abs(out - truth)[hole].max() < 0.05
        band_hz = [sig.TONE_HZ - 50, sig.TONE_HZ + 50]
        sos = butter(4, band_hz, "bandpass", fs=RATE, output="sos")
        band = np.abs(sosfiltfilt(sos, out) - sosfiltfilt(sos, truth))[hole]
        assert band.max() < 0.01

    def test_a_tone_that_starts_later_is_not_written_into_the_hole(
        self, sig: SimpleNamespace
    ) -> None:
        n = sig.N
        at = int(2.5 * RATE)
        x = sig.noise(n) + sig.tone(n, start=3.0) + sig.click(n, at)
        out, mask = clean_channel(x, RATE)
        assert mask[at]
        hole = mask & (np.arange(n) < int(2.6 * RATE))
        assert np.abs(out[hole]).max() < 5 * sig.NOISE

    def test_a_long_loud_event_is_a_signal_and_stays(
        self, sig: SimpleNamespace
    ) -> None:
        n = sig.N
        t = np.arange(n) / RATE
        burst = np.where((t > 1.0) & (t < 2.0), 0.05, 0.0)
        x = sig.noise(n) + burst * np.random.default_rng(5).normal(0.0, 1.0, n)
        out, mask = clean_channel(x, RATE)
        assert not mask.any()
        assert np.array_equal(out, x)

    def test_nothing_outside_a_click_changes(self, sig: SimpleNamespace) -> None:
        n = sig.N
        x = sig.noise(n) + sig.tone(n) + sum(sig.click(n, a) for a in sig.CLICKS)
        out, mask = clean_channel(x, RATE)
        assert 0 < mask.sum() < 0.01 * n
        assert np.array_equal(out[~mask], x[~mask])

    def test_a_channel_without_clicks_is_returned_as_it_is(
        self, sig: SimpleNamespace
    ) -> None:
        x = sig.noise(sig.N) + sig.tone(sig.N)
        out, mask = clean_channel(x, RATE)
        assert not mask.any()
        assert np.array_equal(out, x)

    def test_digital_silence_stays_silent(self, sig: SimpleNamespace) -> None:
        n = sig.N
        x = sig.noise(n)
        x[: n // 2] = 0.0
        x += sig.click(n, int(4 * RATE))
        out, mask = clean_channel(x, RATE)
        assert mask[int(4 * RATE)]
        assert np.all(out[: n // 2] == 0.0)

    def test_a_higher_threshold_leaves_a_weaker_click(
        self, sig: SimpleNamespace
    ) -> None:
        """The 0.05 click reaches 22.8 robust sd of its high band."""
        n = sig.N
        x = sig.noise(n) + sig.click(n, int(3 * RATE), amplitude=0.05)
        _, low = clean_channel(x, RATE, ClickRemoval(threshold=12, extent_threshold=8))
        strict = ClickRemoval(threshold=50, extent_threshold=50 * 8 / 12)
        _, high = clean_channel(x, RATE, strict)
        assert low[int(3 * RATE)]
        assert not high.any()

    def test_too_short_to_tell(self) -> None:
        x = np.ones(40)
        out, mask = clean_channel(x, RATE)
        assert np.array_equal(out, x)
        assert not mask.any()


# ------------------------------------------------------------- recordings


class TestRecording:
    def test_float32_in_float32_out_and_untouched_outside_the_mask(
        self, three_channels: NDArray[np.float64]
    ) -> None:
        x = three_channels.astype(np.float32)
        out, mask = remove_clicks(x, RATE)
        assert out.dtype == np.float32
        assert mask.shape == x.shape
        assert mask[:, 0].any()
        assert mask[:, 1].any()
        assert np.array_equal(out[~mask], x[~mask])
        assert not np.array_equal(out, x)

    def test_a_channel_without_clicks_comes_out_bit_for_bit(
        self, three_channels: NDArray[np.float64]
    ) -> None:
        out, mask = remove_clicks(three_channels, RATE)
        assert out.dtype == np.float64
        assert not mask[:, 2].any()
        assert np.array_equal(out[:, 2], three_channels[:, 2])

    def test_injected_clicks_are_removed(
        self, make_chirp_scene: Callable[..., Any]
    ) -> None:
        """Every click (300 or 3000x the noise sd) is found on every
        electrode it is on, and what is left around it is the noise the
        bridge replaced: RMS error over the 12 ms around a click 0.4 to
        2.7 (noise sd 1), the 2.7 under the peak of a big chirp."""
        scene = make_chirp_scene(30.0)
        out, mask = remove_clicks(scene.signal, RATE)
        for at, electrodes in scene.clicks:
            for el in electrodes:
                assert mask[at, el]
                hole = slice(at - 40, at + 200)
                excess = np.abs(scene.signal[hole, el] - scene.truth[hole, el]).max()
                error = out[hole, el].astype(np.float64) - scene.truth[hole, el]
                assert excess >= 300
                assert np.sqrt(np.mean(error**2)) < 5.0

    def test_a_clean_recording_is_returned_bit_for_bit(
        self, make_chirp_scene: Callable[..., Any]
    ) -> None:
        """Two chirping fish with real harmonics, 30x the high-band noise
        on their loudest electrode, no clicks: nothing is repaired."""
        scene = make_chirp_scene(30.0)
        out, mask = remove_clicks(scene.truth, RATE)
        assert not mask.any()
        assert np.array_equal(out, scene.truth)

    def test_threads_give_the_same_samples(
        self, three_channels: NDArray[np.float64]
    ) -> None:
        one, m1 = remove_clicks(three_channels, RATE)
        three, m3 = remove_clicks(three_channels, RATE, workers=3)
        assert np.array_equal(one, three)
        assert np.array_equal(m1, m3)

    def test_bad_input_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            remove_clicks(np.zeros(100), RATE)
        with pytest.raises(TypeError, match="floating"):
            remove_clicks(np.zeros((100, 2), dtype=np.int16), RATE)  # pyright: ignore[reportArgumentType]
        with pytest.raises(ValueError, match="start_sample"):
            remove_clicks(np.zeros((100, 2)), RATE, start_sample=-1)


# ---------------------------------------------------------------- chirps


def box_power(
    samples: NDArray[np.floating[Any]], box: tuple[float, ...], nfft: int
) -> float:
    """Power in a chirp box of the electrode-summed spectrogram."""
    config = SpectrogramConfig(
        window_lengths_samples=(nfft,), hop_samples=50, f_min_hz=500.0, f_max_hz=1500.0
    )
    spec = compute_spectrogram(torch.from_numpy(np.ascontiguousarray(samples)), config)
    t0, t1, f0, f1 = box
    ti = (spec.times_s >= t0) & (spec.times_s <= t1)
    fi = (spec.freqs_hz >= f0) & (spec.freqs_hz <= f1)
    return float(spec.power[0].numpy()[np.ix_(fi, ti)].sum())


class TestChirps:
    @pytest.mark.parametrize("nfft", [4096, 1024])
    def test_chirps_away_from_clicks_keep_their_power(
        self, make_chirp_scene: Callable[..., Any], nfft: int
    ) -> None:
        """Clicks at least 0.3 s from a chirp: its box sees no repaired
        sample, and its power is exactly that of the click-free truth."""
        scene = make_chirp_scene(30.0)
        out, _ = remove_clicks(scene.signal, RATE)
        for k, box in enumerate(scene.boxes):
            if k == scene.at_chirp:
                continue
            assert box_power(out, box, nfft) == box_power(scene.truth, box, nfft)

    @pytest.mark.parametrize("nfft", [4096, 1024])
    def test_a_chirp_under_a_click_is_restored(
        self, make_chirp_scene: Callable[..., Any], nfft: int
    ) -> None:
        """A click at the peak of a big chirp lifts its box by 18.9 dB
        (nfft 4096; 19.5 dB at 1024). Repaired, the box is 0.10 dB (0.11
        dB) below the truth: what the bridge cannot know of the chirp
        under the click, 18-19 dB below the chirp's own power."""
        scene = make_chirp_scene(30.0)
        out, _ = remove_clicks(scene.signal, RATE)
        box = scene.boxes[scene.at_chirp]
        truth = box_power(scene.truth, box, nfft)
        assert 10 * np.log10(box_power(scene.signal, box, nfft) / truth) > 15
        assert abs(10 * np.log10(box_power(out, box, nfft) / truth)) < 0.3

    def test_loud_chirps_with_real_harmonics_reach_the_click_band(
        self, make_chirp_scene: Callable[..., Any]
    ) -> None:
        """The limitation the validation report quantifies: a fish at 300x
        the high-band noise with the harmonics of a real EOD sweeps its
        4th-6th harmonics through 4-5 kHz in a big chirp, which the default
        threshold takes for a click. A threshold of 48 leaves it."""
        scene = make_chirp_scene(300.0)
        _, default = remove_clicks(scene.truth, RATE)
        _, strict = remove_clicks(
            scene.truth, RATE, ClickRemoval(threshold=48.0, extent_threshold=32.0)
        )
        big = [scene.chirp_times_s[k] for k in (1, 3)]
        assert any(
            default[int(t * RATE) - 300 : int(t * RATE) + 300].any() for t in big
        )
        assert not strict.any()


class TestCoincidenceGuard:
    """``coincidence``: clicks on few electrodes must reach a higher bar."""

    def test_off_by_default(self) -> None:
        assert ClickRemoval().coincidence == 0.0
        assert ClickRemoval().isolated_threshold == 48.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"coincidence": 1.5}, "coincidence"),
            ({"coincidence": -0.1}, "coincidence"),
            ({"coincidence": 0.5, "isolated_threshold": 10.0}, "isolated_threshold"),
        ],
    )
    def test_invalid_settings_raise(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ClickRemoval(**kwargs)

    def test_a_weak_click_counts_on_many_electrodes_only(
        self, sig: SimpleNamespace
    ) -> None:
        """The 0.05 click of audian's tests reaches 22.8 robust sd: below
        the isolated threshold of 48 on one of four electrodes, a click on
        three of them (half or more)."""
        n = sig.N
        rng = np.random.default_rng(1)
        x = rng.normal(0.0, sig.NOISE, (n, 4))
        single, multi, loud = int(1.5 * RATE), int(3.0 * RATE), int(4.5 * RATE)
        x[:, 0] += sig.click(n, single, amplitude=0.05)
        x[:, :3] += sig.click(n, multi, amplitude=0.05)[:, None]
        x[:, 2] += sig.click(n, loud, amplitude=1.0)
        _, plain = remove_clicks(x, RATE)
        _, guarded = remove_clicks(x, RATE, ClickRemoval(coincidence=0.5))
        assert plain[single, 0]
        assert not guarded[single].any()
        assert guarded[multi, :3].all()
        assert guarded[loud, 2]
        assert not (guarded & ~plain).any()

    def test_loud_chirps_are_left_and_clicks_still_found(
        self, make_chirp_scene: Callable[..., Any]
    ) -> None:
        """Nine electrodes, the fish at 300x the high-band noise with real
        harmonics: its big chirps reach the click band on two electrodes
        (8260 samples repaired without the guard, none with it), and every
        click, single or on several electrodes, is still repaired."""
        scene = make_chirp_scene(300.0, n_electrodes=9)
        cfg = ClickRemoval(coincidence=0.5)
        _, plain = remove_clicks(scene.truth, RATE)
        out, mask = remove_clicks(scene.truth, RATE, cfg)
        assert plain.any()
        assert not mask.any()
        assert np.array_equal(out, scene.truth)
        _, mask = remove_clicks(scene.signal, RATE, cfg)
        for at, electrodes in scene.clicks:
            assert mask[at, list(electrodes)].all()

    @settings(max_examples=8, deadline=None)
    @given(cuts=st.lists(st.integers(1, 239_999), max_size=4, unique=True))
    def test_spans_give_the_same_samples_as_the_whole(
        self,
        three_channels: NDArray[np.float64],
        cuts: list[int],
    ) -> None:
        cfg = ClickRemoval(coincidence=0.5)
        whole, whole_mask = _guarded_whole(three_channels, cfg)
        edges = [0, *sorted(cuts), len(three_channels)]
        pieces, masks = [], []
        for a, b in itertools.pairwise(edges):
            plan = plan_clean(a, b, len(three_channels), RATE, cfg)
            out, mask = clean_span(three_channels[plan.read_slice], plan, RATE, cfg)
            pieces.append(out)
            masks.append(mask)
        assert np.array_equal(np.concatenate(pieces), whole)
        assert np.array_equal(np.concatenate(masks), whole_mask)


_GUARDED: dict[int, tuple[NDArray[np.float64], NDArray[np.bool_]]] = {}


def _guarded_whole(
    x: NDArray[np.float64], cfg: ClickRemoval
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """`remove_clicks` of ``x``, computed once per array."""
    if id(x) not in _GUARDED:
        _GUARDED[id(x)] = remove_clicks(x, RATE, cfg)
    return _GUARDED[id(x)]


# ------------------------------------------------------------------ spans


class TestSpans:
    def test_plan_reads_the_blocks_and_their_margins(self) -> None:
        cfg = ClickRemoval()
        total = 1_000_000
        plan = plan_clean(50_000, 90_000, total, RATE, cfg)
        # blocks 1 and 2 ([40000, 120000)) and 10000 on either side
        assert (plan.read_start, plan.read_stop) == (30_000, 130_000)
        assert plan.n_samples == 40_000
        assert plan.read_slice == slice(30_000, 130_000)
        edge = plan_clean(0, 10, total, RATE, cfg)
        assert (edge.read_start, edge.read_stop) == (0, 50_000)
        tail = plan_clean(total - 10, total, total, RATE, cfg)
        assert tail.read_stop == total

    @pytest.mark.parametrize(("start", "stop"), [(-1, 10), (10, 10), (0, 101)])
    def test_invalid_spans_raise(self, start: int, stop: int) -> None:
        with pytest.raises(ValueError, match="need 0 <= start_sample"):
            plan_clean(start, stop, 100, RATE)

    def test_a_slice_gives_the_same_samples_as_the_whole(
        self, three_channels: NDArray[np.float64]
    ) -> None:
        """From audian: blocks are fixed in absolute time, so a frame comes
        out the same in a buffer that starts anywhere, as long as the
        buffer reaches a block and a margin beyond it."""
        x = three_channels
        cfg = ClickRemoval()
        whole, _ = remove_clicks(x, RATE, cfg)
        reserve = cfg.block_samples(RATE) + cfg.margin_samples(RATE)
        for a, b in ((13_579, 201_234), (40_000, 240_000), (1, 177_777)):
            part, _ = remove_clicks(x[a:b], RATE, cfg, start_sample=a)
            inner = slice(reserve, (b - a) - reserve)
            assert (whole[a:b][inner] != x[a:b][inner]).sum() > 100
            assert np.array_equal(part[inner], whole[a:b][inner]), (a, b)

    @settings(max_examples=15, deadline=None)
    @given(cuts=st.lists(st.integers(1, 239_999), max_size=5, unique=True))
    def test_spans_give_the_same_samples_as_the_whole(
        self,
        three_channels: NDArray[np.float64],
        three_channels_cleaned: tuple[NDArray[np.float64], NDArray[np.bool_]],
        cuts: list[int],
    ) -> None:
        x = three_channels
        whole, whole_mask = three_channels_cleaned
        edges = [0, *sorted(cuts), len(x)]
        pieces, masks = [], []
        for a, b in itertools.pairwise(edges):
            plan = plan_clean(a, b, len(x), RATE)
            out, mask = clean_span(x[plan.read_slice], plan, RATE)
            assert out.shape == (b - a, x.shape[1])
            pieces.append(out)
            masks.append(mask)
        assert np.array_equal(np.concatenate(pieces), whole)
        assert np.array_equal(np.concatenate(masks), whole_mask)

    def test_clean_span_checks_its_samples(self) -> None:
        plan = plan_clean(0, 1000, 100_000, RATE)
        with pytest.raises(ValueError, match="expected"):
            clean_span(np.zeros((10, 2)), plan, RATE)

    def test_read_clean_feeds_a_spectrogram_span(
        self, three_channels: NDArray[np.float64]
    ) -> None:
        """The documented use: clean exactly what a spectrogram span reads.
        The frames equal those of the whole recording, cleaned whole."""
        x = three_channels.astype(np.float32)
        recording = _Recording(x)
        config = SpectrogramConfig()
        whole = compute_spectrogram(remove_clicks(x, RATE)[0], config)
        start, stop = 70_001, 150_000
        plan = plan_span(start, stop, len(x), config)
        samples, _ = read_clean(recording, plan.read_start, plan.read_stop)
        part = compute_spectrogram_span(samples, plan, config)
        frames = slice(plan.frame_start, plan.frame_stop)
        assert torch.equal(part.power, whole.power[:, :, frames])
        dirty = compute_spectrogram_span(x[plan.read_slice], plan, config)
        assert not torch.equal(dirty.power, part.power)

    def test_click_spans(self) -> None:
        mask = np.zeros((10, 2), dtype=bool)
        mask[2:4, 0] = True
        mask[3:6, 1] = True
        mask[8, 1] = True
        assert click_spans(mask).tolist() == [[2, 6], [8, 9]]
        assert click_spans(mask[:, 0], offset=100).tolist() == [[102, 104]]
        assert click_spans(np.zeros(5, dtype=bool)).shape == (0, 2)


class _Recording:
    """The surface of `gridtools.io.Recording` that `read_clean` uses."""

    def __init__(self, data: NDArray[np.float32]) -> None:
        self.data = data
        self.samplerate_hz = RATE
        self.n_samples = len(data)

    def read(self, start: int, stop: int) -> NDArray[np.float32]:
        return self.data[start:stop].copy()


# ------------------------------------------------------------- real data

REAL = Path("/mnt/data2/2019_tube_competition/2020-06-03-10_10")


@pytest.mark.data
def test_a_real_recording_cleaned_in_spans_equals_one_excerpt() -> None:
    """20 s of a 15-electrode trial with multi-electrode clicks (about 15
    a minute), cleaned in three spans, equals the same 20 s cleaned as
    part of one larger excerpt that holds every block's margins."""
    recording = open_recording(REAL)
    cfg = ClickRemoval()
    a = recording.time_to_sample(10_790.0)
    b = a + round(20 * RATE)
    reserve = cfg.block_samples(RATE) + cfg.margin_samples(RATE)
    excerpt = recording.read(a - reserve, b + reserve)
    whole, whole_mask = remove_clicks(excerpt, RATE, cfg, start_sample=a - reserve)
    cuts = [a, a + 123_457, a + 250_000, b]
    pieces = [read_clean(recording, s, e, cfg) for s, e in itertools.pairwise(cuts)]
    cleaned = np.concatenate([p[0] for p in pieces])
    mask = np.concatenate([p[1] for p in pieces])
    inner = slice(reserve, reserve + (b - a))
    assert np.array_equal(cleaned, whole[inner])
    assert np.array_equal(mask, whole_mask[inner])
    raw = recording.read(a, b)
    assert len(click_spans(mask)) > 0
    assert np.array_equal(cleaned[~mask], raw[~mask])
