"""Click removal: what it repairs, what it leaves, and that it is stable.

`TestRepair` and part of `TestSpans` are ported from audian's
``tests/test_clicks.py`` (commit 6fa5274), where each number asserted
against was measured first and the margins are a factor of three or more
beyond it. `TestRepair` runs the unguarded per-channel algorithm
(`_clean_channel`), which is bit-identical to audian's. The rest pins what
chirpdetector needs, always under the mandatory electrode-coincidence
guard and on grids of 9 or 15 electrodes: chirps survive, clicks go,
recordings with too few electrodes are refused, and a recording cleaned in
spans gives the same samples as cleaned whole.
"""

from __future__ import annotations

import itertools
import json
import math
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
from scipy.signal import butter, resample_poly, sosfiltfilt

from chirpdetector.preprocessing import (
    MAX_CHIRP_ELECTRODES,
    MIN_ISOLATED_THRESHOLD,
    ClickRemoval,
    TooFewElectrodesError,
    can_remove_clicks,
    clean_span,
    click_spans,
    plan_clean,
    read_clean,
    remove_clicks,
)
from chirpdetector.preprocessing.clicks import _clean_channel as clean_channel
from chirpdetector.preprocessing.clicks import _high_band_gain
from chirpdetector.spectrogram import (
    SpectrogramConfig,
    compute_spectrogram,
    compute_spectrogram_span,
    plan_span,
)

RATE = 20_000.0

# ------------------------------------------------------------------ config


class TestConfig:
    def test_detection_defaults_are_wavetrackers(self) -> None:
        cfg = ClickRemoval()
        assert (cfg.threshold, cfg.extent_threshold) == (12.0, 8.0)
        # audian's 0.2 and 0.05 of the sampling rate, at 20 kHz
        assert (cfg.high_band_hz, cfg.high_band_edge_hz) == (4000.0, 1000.0)
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
            ClickRemoval(threshold=48.0, extent_threshold=32.0, high_band_hz=8000.0),
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

    @pytest.mark.parametrize(
        ("key", "hint"),
        [
            ("high_band", "high_band_hz=4000"),
            ("high_band_edge", "high_band_edge_hz=1000"),
        ],
    )
    def test_outdated_keys_say_what_replaced_them(self, key: str, hint: str) -> None:
        """Bundles written before the band was in Hz are refused, with the
        value that reproduces them at 20 kHz."""
        with pytest.raises(ValueError, match=f"outdated ClickRemoval keys.*{hint}"):
            ClickRemoval.from_dict({key: 0.2})

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
            ({"high_band_hz": 0.0}, "high_band_hz must be positive"),
            ({"high_band_edge_hz": 0.0}, "high_band_edge_hz must be positive"),
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
        strict = ClickRemoval(
            threshold=50, extent_threshold=50 * 8 / 12, isolated_threshold=200
        )
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
    """Recordings, on audian's signals laid out on 9 or 15 electrodes."""

    def test_float32_in_float32_out_and_untouched_outside_the_mask(
        self, grid: NDArray[np.float64]
    ) -> None:
        x = grid.astype(np.float32)
        out, mask = remove_clicks(x, RATE)
        assert out.dtype == np.float32
        assert mask.shape == x.shape
        assert mask[:, 0].any()
        assert mask[:, 1].any()
        assert np.array_equal(out[~mask], x[~mask])
        assert not np.array_equal(out, x)

    def test_a_channel_without_clicks_comes_out_bit_for_bit(
        self, grid: NDArray[np.float64]
    ) -> None:
        out, mask = remove_clicks(grid, RATE)
        assert out.dtype == np.float64
        assert not mask[:, 2].any()
        assert np.array_equal(out[:, 2], grid[:, 2])

    @pytest.mark.parametrize("n_electrodes", [9, 15])
    def test_injected_clicks_are_removed(
        self, make_chirp_scene: Callable[..., Any], n_electrodes: int
    ) -> None:
        """Every click is found on every electrode it is on: strong ones
        (300 or 3000x the noise sd) on one to four electrodes, and a weak
        one (25x) on all of them. What is left around a strong click is the
        noise the bridge replaced: RMS error over the 12 ms around it 0.4
        to 2.7 (noise sd 1), the 2.7 under the peak of a big chirp."""
        scene = make_chirp_scene(30.0, n_electrodes=n_electrodes)
        out, mask = remove_clicks(scene.signal, RATE)
        for at, electrodes in scene.clicks:
            assert mask[at, list(electrodes)].all()
            if len(electrodes) == n_electrodes:
                continue
            for el in electrodes:
                hole = slice(at - 40, at + 200)
                excess = np.abs(scene.signal[hole, el] - scene.truth[hole, el]).max()
                error = out[hole, el].astype(np.float64) - scene.truth[hole, el]
                assert excess >= 300
                assert np.sqrt(np.mean(error**2)) < 5.0

    @pytest.mark.parametrize("n_electrodes", [9, 15])
    def test_a_clean_recording_is_returned_bit_for_bit(
        self, make_chirp_scene: Callable[..., Any], n_electrodes: int
    ) -> None:
        """Two chirping fish with real harmonics, 30x the high-band noise
        on their loudest electrode, no clicks: nothing is repaired."""
        scene = make_chirp_scene(30.0, n_electrodes=n_electrodes)
        out, mask = remove_clicks(scene.truth, RATE)
        assert not mask.any()
        assert np.array_equal(out, scene.truth)

    def test_threads_give_the_same_samples(self, grid: NDArray[np.float64]) -> None:
        one, m1 = remove_clicks(grid, RATE)
        three, m3 = remove_clicks(grid, RATE, workers=3)
        assert np.array_equal(one, three)
        assert np.array_equal(m1, m3)

    def test_bad_input_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            remove_clicks(np.zeros(100), RATE)
        with pytest.raises(TypeError, match="floating"):
            remove_clicks(np.zeros((100, 9), dtype=np.int16), RATE)  # pyright: ignore[reportArgumentType]
        with pytest.raises(ValueError, match="start_sample"):
            remove_clicks(np.zeros((100, 9)), RATE, start_sample=-1)


# ---------------------------------------------------------------- chirps


def box_power(
    samples: NDArray[np.floating[Any]], box: tuple[float, ...], nfft: int
) -> float:
    """Power in a chirp box of the electrode-summed spectrogram."""
    config = SpectrogramConfig(
        reference_window_s=nfft / RATE,
        extra_windows_s=(),
        hop_s=50 / RATE,
        f_min_hz=500.0,
        f_max_hz=1500.0,
    )
    spec = compute_spectrogram(
        torch.from_numpy(np.ascontiguousarray(samples)), RATE, config
    )
    t0, t1, f0, f1 = box
    ti = (spec.times_s >= t0) & (spec.times_s <= t1)
    fi = (spec.freqs_hz >= f0) & (spec.freqs_hz <= f1)
    return float(spec.power[0].numpy()[np.ix_(fi, ti)].sum())


def big_chirps_touched(
    scene: Any, mask: NDArray[np.bool_], with_clicks: bool = False
) -> bool:
    """Whether any repair falls within 15 ms of a big chirp of fish 0.

    With clicks, the big chirp that has a click at its peak (`at_chirp`)
    is left out: repairing that click is the point.
    """
    big = [k for k in (1, 3) if not (with_clicks and k == scene.at_chirp)]
    times = [scene.chirp_times_s[k] for k in big]
    return any(mask[int(t * RATE) - 300 : int(t * RATE) + 300].any() for t in times)


class TestChirps:
    @pytest.mark.parametrize("n_electrodes", [9, 15])
    @pytest.mark.parametrize("nfft", [4096, 1024])
    def test_chirps_away_from_clicks_keep_their_power(
        self, make_chirp_scene: Callable[..., Any], nfft: int, n_electrodes: int
    ) -> None:
        """Clicks at least 0.3 s from a chirp: its box sees no repaired
        sample, and its power is exactly that of the click-free truth."""
        scene = make_chirp_scene(30.0, n_electrodes=n_electrodes)
        out, _ = remove_clicks(scene.signal, RATE)
        for k, box in enumerate(scene.boxes):
            if k == scene.at_chirp:
                continue
            assert box_power(out, box, nfft) == box_power(scene.truth, box, nfft)

    @pytest.mark.parametrize("nfft", [4096, 1024])
    def test_a_chirp_under_a_click_is_restored(
        self, make_chirp_scene: Callable[..., Any], nfft: int
    ) -> None:
        """A click at the peak of a big chirp lifts its box by 15 dB or
        more; repaired, the box is within 0.3 dB of the truth: what the
        bridge cannot know of the chirp under the click."""
        scene = make_chirp_scene(30.0)
        out, _ = remove_clicks(scene.signal, RATE)
        box = scene.boxes[scene.at_chirp]
        truth = box_power(scene.truth, box, nfft)
        assert 10 * np.log10(box_power(scene.signal, box, nfft) / truth) > 15
        assert abs(10 * np.log10(box_power(out, box, nfft) / truth)) < 0.3

    def test_loud_chirps_reach_the_click_band_of_their_electrode(
        self, make_chirp_scene: Callable[..., Any]
    ) -> None:
        """Why the guard is mandatory: a fish at 300x the high-band noise
        with the harmonics of a real EOD sweeps its 4th-6th harmonics
        through 4-5 kHz in a big chirp, and the per-channel algorithm
        (audian's, unguarded) repairs it as a click."""
        scene = make_chirp_scene(300.0)
        _, mask = clean_channel(scene.truth[:, 0], RATE)
        assert big_chirps_touched(scene, mask[:, None])

    @pytest.mark.parametrize("n_electrodes", [9, 15])
    def test_the_default_leaves_loud_chirps_and_finds_the_clicks(
        self, make_chirp_scene: Callable[..., Any], n_electrodes: int
    ) -> None:
        """The same loud chirps under the default guard: nothing repaired
        without clicks, every click still repaired with them."""
        scene = make_chirp_scene(300.0, n_electrodes=n_electrodes)
        out, mask = remove_clicks(scene.truth, RATE)
        assert not mask.any()
        assert np.array_equal(out, scene.truth)
        _, mask = remove_clicks(scene.signal, RATE)
        for at, electrodes in scene.clicks:
            assert mask[at, list(electrodes)].all()
        assert not big_chirps_touched(scene, mask, with_clicks=True)


# ------------------------------------------------------------------ guard


class TestCoincidenceGuard:
    """Clicks on few electrodes must reach `isolated_threshold`."""

    def test_on_by_default(self) -> None:
        cfg = ClickRemoval()
        assert cfg.coincidence == 0.5
        assert cfg.isolated_threshold == MIN_ISOLATED_THRESHOLD == 48.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"coincidence": 0.0},
            {"coincidence": 0.3},
            {"coincidence": 0.49},
            {"coincidence": 1.5},
            {"isolated_threshold": 12.0},
            {"isolated_threshold": 47.9},
            {"threshold": 60.0, "extent_threshold": 40.0},
        ],
    )
    def test_unguarded_configs_are_rejected(self, kwargs: dict[str, Any]) -> None:
        """Without the guard, or with an isolated bar it never applies,
        loud chirps are repaired: such configs cannot be built, nor read
        from a bundle."""
        with pytest.raises(ValueError, match=r"coincidence|isolated_threshold"):
            ClickRemoval(**kwargs)
        with pytest.raises(ValueError, match=r"coincidence|isolated_threshold"):
            ClickRemoval.from_dict(kwargs)

    def test_a_weak_click_counts_on_many_electrodes_only(
        self, grid: NDArray[np.float64], sig: SimpleNamespace
    ) -> None:
        """The 0.05 click of audian's tests reaches 22.8 robust sd: above
        `threshold`, below `isolated_threshold`. On 6 of 9 electrodes it is
        repaired on all six; alone on one it is left. The strong clicks
        (1.0) on single electrodes are repaired."""
        _, mask = remove_clicks(grid, RATE)
        for at in sig.WEAK_MANY:
            assert mask[at, 3:].all()
        assert not mask[sig.WEAK_ONE].any()
        assert all(mask[at, 0] for at in sig.CLICKS)
        assert mask[int(2.5 * RATE), 1]

    def test_a_weak_click_on_too_few_electrodes_is_left(
        self, sig: SimpleNamespace
    ) -> None:
        """On 4 of 9 electrodes (below half) the weak click is left, on 5
        (half rounded up) it is repaired."""
        n = sig.N
        x = np.random.default_rng(1).normal(0.0, sig.NOISE, (n, 9))
        four, five = int(2.0 * RATE), int(4.0 * RATE)
        x[:, :4] += sig.click(n, four, sig.WEAK)[:, None]
        x[:, 4:] += sig.click(n, five, sig.WEAK)[:, None]
        _, mask = remove_clicks(x, RATE)
        assert not mask[four].any()
        assert mask[five, 4:].all()

    @pytest.mark.parametrize("loudness", [30.0, 300.0])
    def test_dead_electrodes_still_count_and_chirps_stay_safe(
        self, make_chirp_scene: Callable[..., Any], loudness: float
    ) -> None:
        """9 electrodes, 2 of them dead (digital silence): the weak click on
        the 7 live ones still reaches half of the 9 and is repaired, the
        strong single-electrode clicks too, the dead ones stay silent, and
        loud chirps are left."""
        dead = (5, 7)
        scene = make_chirp_scene(loudness, n_electrodes=9, dead=dead)
        out, mask = remove_clicks(scene.truth, RATE)
        assert not mask.any()
        out, mask = remove_clicks(scene.signal, RATE)
        for at, electrodes in scene.clicks:
            assert mask[at, list(electrodes)].all()
        assert not mask[:, list(dead)].any()
        assert np.all(out[:, list(dead)] == 0.0)
        assert not big_chirps_touched(scene, mask, with_clicks=True)

    @settings(max_examples=8, deadline=None)
    @given(cuts=st.lists(st.integers(1, 119_999), max_size=4, unique=True))
    def test_spans_give_the_same_samples_as_the_whole_on_15(
        self,
        grid15: NDArray[np.float64],
        grid15_cleaned: tuple[NDArray[np.float64], NDArray[np.bool_]],
        cuts: list[int],
    ) -> None:
        whole, whole_mask = grid15_cleaned
        edges = [0, *sorted(cuts), len(grid15)]
        pieces, masks = [], []
        for a, b in itertools.pairwise(edges):
            plan = plan_clean(a, b, len(grid15), RATE)
            out, mask = clean_span(grid15[plan.read_slice], plan, RATE)
            pieces.append(out)
            masks.append(mask)
        assert np.array_equal(np.concatenate(pieces), whole)
        assert np.array_equal(np.concatenate(masks), whole_mask)


# ------------------------------------------------------------ sample rate


class TestSampleRate:
    """Everything in seconds or Hz: any sampling rate is cleaned alike."""

    def test_the_band_at_20khz_is_audians_bit_for_bit(self) -> None:
        """At 20 kHz the gain of the search band equals audian's, written in
        fractions of the sampling rate (0.2, 0.05), to the last bit, and so
        does every result built on it."""
        freqs = np.fft.rfftfreq(70_000, 1.0 / RATE)
        audian = np.sin(0.5 * np.pi * np.clip((freqs / RATE - 0.2) / 0.05, 0, 1)) ** 2
        assert np.array_equal(_high_band_gain(freqs, RATE, ClickRemoval()), audian)

    def test_the_band_stays_put_at_other_rates(self) -> None:
        cfg = ClickRemoval()
        for rate in (20_000.0, 44_100.0, 48_000.0):
            freqs = np.fft.rfftfreq(round(3 * rate), 1.0 / rate)
            gain = _high_band_gain(freqs, rate, cfg)
            assert gain[freqs <= 4000.0].max() == 0.0
            assert gain[freqs >= 5000.0].min() == 1.0

    @pytest.mark.parametrize("loudness", [30.0, 300.0])
    def test_the_same_clicks_are_found_at_20_and_48_khz(
        self, make_chirp_scene: Callable[..., Any], loudness: float
    ) -> None:
        """The 9-electrode scene, resampled to 48 kHz: the same click
        events at the same times (within 1 ms; a repair under a loud
        chirp may end a few ms later), every click on every electrode,
        and nothing repaired without clicks, loud chirps included."""
        scene = make_chirp_scene(loudness, n_electrodes=9)
        fast = 48_000.0
        signal = resample_poly(scene.signal.astype(np.float64), 12, 5, axis=0)
        truth = resample_poly(scene.truth.astype(np.float64), 12, 5, axis=0)
        _, slow_mask = remove_clicks(scene.signal, RATE)
        _, fast_mask = remove_clicks(signal, fast)
        slow = click_spans(slow_mask) / RATE
        quick = click_spans(fast_mask) / fast
        assert slow.shape == quick.shape
        assert np.abs(slow[:, 0] - quick[:, 0]).max() < 0.001
        for at, electrodes in scene.clicks:
            assert fast_mask[round(at * fast / RATE), list(electrodes)].all()
        _, mask = remove_clicks(truth, fast)
        assert not mask.any()

    def test_too_low_a_rate_is_refused(self) -> None:
        """The band reaches 5 kHz, so the rate must be at least 10 kHz."""
        cfg = ClickRemoval()
        assert cfg.min_samplerate_hz == 10_000.0
        low = 8_000.0
        x = np.random.default_rng(0).normal(0.0, 1e-3, (int(low), 9))
        with pytest.raises(ValueError, match="needs at least 10000 Hz"):
            remove_clicks(x, low)
        plan = plan_clean(0, len(x), len(x), low)
        with pytest.raises(ValueError, match="needs at least 10000 Hz"):
            clean_span(x, plan, low)
        with pytest.raises(ValueError, match="needs at least 10000 Hz"):
            clean_channel(x[:, 0], low)
        out, mask = remove_clicks(x, 10_000.0)
        assert not mask.any()
        assert np.array_equal(out, x)
        assert not can_remove_clicks(9, cfg, samplerate_hz=low)
        assert can_remove_clicks(9, cfg, samplerate_hz=10_000.0)
        assert not can_remove_clicks(8, cfg, samplerate_hz=48_000.0)


# -------------------------------------------------------------- electrodes


class TestElectrodes:
    """The guard needs a grid: fewer electrodes than `min_electrodes` are refused."""

    @pytest.mark.parametrize(
        ("coincidence", "expected"), [(0.5, 9), (0.75, 6), (1.0, 5)]
    )
    def test_min_electrodes(self, coincidence: float, expected: int) -> None:
        cfg = ClickRemoval(coincidence=coincidence)
        assert cfg.min_electrodes == expected
        # half rounded up of `min_electrodes` exceeds a chirp's reach, one
        # electrode fewer does not
        assert math.ceil(coincidence * expected) > MAX_CHIRP_ELECTRODES
        assert math.ceil(coincidence * (expected - 1) - 1e-9) <= MAX_CHIRP_ELECTRODES

    @pytest.mark.parametrize("n_electrodes", [1, 2, 4, 8])
    def test_too_few_electrodes_are_refused(self, n_electrodes: int) -> None:
        x = np.random.default_rng(0).normal(0.0, 1e-3, (int(RATE), n_electrodes))
        match = "single channel" if n_electrodes == 1 else "at least 9 electrodes"
        with pytest.raises(TooFewElectrodesError, match=match):
            remove_clicks(x, RATE)
        plan = plan_clean(0, len(x), len(x), RATE)
        with pytest.raises(TooFewElectrodesError, match=match):
            clean_span(x, plan, RATE)
        with pytest.raises(TooFewElectrodesError, match=match):
            read_clean(_Recording(x.astype(np.float32)), 0, len(x))
        assert issubclass(TooFewElectrodesError, ValueError)

    def test_a_higher_coincidence_accepts_smaller_grids(self) -> None:
        x = np.random.default_rng(0).normal(0.0, 1e-3, (int(RATE), 5))
        with pytest.raises(TooFewElectrodesError):
            remove_clicks(x, RATE)
        out, mask = remove_clicks(x, RATE, ClickRemoval(coincidence=1.0))
        assert not mask.any()
        assert np.array_equal(out, x)

    @pytest.mark.parametrize("n_electrodes", [9, 15])
    def test_enough_electrodes_are_accepted(self, n_electrodes: int) -> None:
        x = np.random.default_rng(0).normal(0.0, 1e-3, (int(RATE), n_electrodes))
        out, mask = remove_clicks(x, RATE)
        assert mask.shape == x.shape
        assert np.array_equal(out, x)

    @pytest.mark.parametrize(
        ("n_electrodes", "coincidence", "expected"),
        [
            (1, 0.5, False),
            (1, 1.0, False),
            (4, 1.0, False),
            (5, 1.0, True),
            (8, 0.5, False),
            (9, 0.5, True),
            (15, 0.5, True),
            (8, 0.75, True),
        ],
    )
    def test_can_remove_clicks(
        self, n_electrodes: int, coincidence: float, expected: bool
    ) -> None:
        cfg = ClickRemoval(coincidence=coincidence)
        assert can_remove_clicks(n_electrodes, cfg) is expected
        if coincidence == 0.5:
            assert can_remove_clicks(n_electrodes) is expected


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
        self, grid: NDArray[np.float64], grid_cleaned: tuple[NDArray, NDArray]
    ) -> None:
        """From audian: blocks are fixed in absolute time, so a frame comes
        out the same in a buffer that starts anywhere, as long as the
        buffer reaches a block and a margin beyond it."""
        x = grid
        cfg = ClickRemoval()
        whole, _ = grid_cleaned
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
        grid: NDArray[np.float64],
        grid_cleaned: tuple[NDArray[np.float64], NDArray[np.bool_]],
        cuts: list[int],
    ) -> None:
        x = grid
        whole, whole_mask = grid_cleaned
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
            clean_span(np.zeros((10, 9)), plan, RATE)

    def test_read_clean_feeds_a_spectrogram_span(
        self, grid: NDArray[np.float64]
    ) -> None:
        """The documented use: clean exactly what a spectrogram span reads.
        The samples, and so the span's frames, equal those of the whole
        recording cleaned whole. (Frames of a span and of the whole
        recording can differ in the last float32 bits on 9 electrodes, raw
        or cleaned, through the spectrogram's own blocking; that is
        compared within 1e-6 here.)"""
        x = grid.astype(np.float32)
        recording = _Recording(x)
        config = SpectrogramConfig()
        cleaned_whole, _ = remove_clicks(x, RATE)
        start, stop = 70_001, 150_000
        plan = plan_span(start, stop, len(x), RATE, config)
        samples, _ = read_clean(recording, plan.read_start, plan.read_stop)
        assert np.array_equal(samples, cleaned_whole[plan.read_slice])
        part = compute_spectrogram_span(samples, plan, config)
        reference = compute_spectrogram_span(
            cleaned_whole[plan.read_slice], plan, config
        )
        assert torch.equal(part.power, reference.power)
        whole = compute_spectrogram(cleaned_whole, RATE, config)
        frames = slice(plan.frame_start, plan.frame_stop)
        torch.testing.assert_close(
            part.power, whole.power[:, :, frames], rtol=1e-6, atol=0.0
        )
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
