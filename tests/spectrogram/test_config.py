import json
from typing import Any

import numpy as np
import pytest

from chirpdetector.spectrogram import SpectrogramConfig

COMMON_RATES = (20_000.0, 22_050.0, 40_000.0, 44_100.0, 48_000.0, 96_000.0)


def test_defaults_are_physical_and_keep_the_long_reference_window() -> None:
    cfg = SpectrogramConfig()
    assert cfg.reference_window_s == 0.2048
    assert cfg.extra_windows_s == (0.0512, 0.0128)
    assert cfg.window_durations_s == (0.2048, 0.0512, 0.0128)
    assert cfg.hop_s == 0.005
    assert cfg.df_hz == pytest.approx(20_000 / 4096)
    assert cfg.n_windows == 3


def test_default_band_is_the_v1_production_band() -> None:
    # Author's decision (2026-11-12): 500-1500 Hz, as in the v1 production run.
    cfg = SpectrogramConfig()
    assert (cfg.f_min_hz, cfg.f_max_hz) == (500.0, 1500.0)
    assert cfg.nominal_freqs_hz[0] >= 500.0
    assert cfg.nominal_freqs_hz[-1] <= 1500.0


def test_layout_at_20_khz_is_the_previous_sample_count_geometry() -> None:
    layout = SpectrogramConfig().layout(20_000.0)
    assert layout.window_lengths_samples == (4096, 1024, 256)
    assert layout.n_fft == 4096
    assert layout.hop_samples == 100
    assert layout.df_hz == 20_000 / 4096
    np.testing.assert_array_equal(layout.freqs_hz, SpectrogramConfig().nominal_freqs_hz)


@pytest.mark.parametrize("fs", COMMON_RATES)
def test_layout_rounding_rules(fs: float) -> None:
    cfg = SpectrogramConfig()
    layout = cfg.layout(fs)
    assert layout.samplerate_hz == fs
    assert layout.n_fft == round(fs * cfg.reference_window_s)
    assert layout.hop_samples == round(fs * cfg.hop_s)
    for n, duration in zip(
        layout.window_lengths_samples, cfg.window_durations_s, strict=True
    ):
        assert n % 2 == 0
        assert n <= layout.n_fft
        assert abs(n - fs * duration) <= 1.0
    assert layout.context_samples == layout.n_fft // 2


@pytest.mark.parametrize("fs", COMMON_RATES)
def test_frequency_grid_matches_the_target_within_rounding(fs: float) -> None:
    cfg = SpectrogramConfig()
    layout = cfg.layout(fs)
    # Same bins, same image height, at every rate.
    assert (layout.freq_bin_start, layout.freq_bin_stop) == (
        cfg.freq_bin_start,
        cfg.freq_bin_stop,
    )
    deviation = np.abs(layout.freqs_hz - cfg.nominal_freqs_hz)
    # |fs / n_fft - df| <= df * 0.5 / n_fft, times the bin index.
    bins = np.arange(cfg.freq_bin_start, cfg.freq_bin_stop)
    bound = bins * cfg.df_hz * 0.5 / layout.n_fft
    assert np.all(deviation <= bound * (1 + 1e-9))
    assert deviation.max() < 0.07  # Hz, at the 1500 Hz edge


def test_frequency_grid_is_reference_bins_inside_crop() -> None:
    cfg = SpectrogramConfig(f_min_hz=400.0, f_max_hz=2000.0)
    freqs = cfg.nominal_freqs_hz
    assert freqs.shape == (cfg.n_freqs,)
    assert freqs[0] >= 400.0
    assert freqs[0] - cfg.df_hz < 400.0
    assert freqs[-1] <= 2000.0
    assert freqs[-1] + cfg.df_hz > 2000.0
    np.testing.assert_allclose(np.diff(freqs), cfg.df_hz)


def test_crop_edges_on_a_bin_are_inclusive() -> None:
    df = 1 / 0.2048
    cfg = SpectrogramConfig(f_min_hz=100 * df, f_max_hz=200 * df)
    assert cfg.freq_bin_start == 100
    assert cfg.freq_bin_stop == 201


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"extra_windows_s": ()},
        {"reference_window_s": 0.1024, "extra_windows_s": (0.0256,), "hop_s": 0.0025},
        {"aggregation": "max", "window": "blackman"},
        {"f_min_hz": 0.0, "f_max_hz": 22_050.0},
        {"dynamic_range_db": 80.0, "headroom_db": 0.0},
    ],
)
def test_json_round_trip(kwargs: dict[str, Any]) -> None:
    cfg = SpectrogramConfig(**kwargs)
    restored = SpectrogramConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert restored == cfg
    assert restored.to_dict() == cfg.to_dict()


def test_numeric_types_are_canonical() -> None:
    cfg = SpectrogramConfig(
        hop_s=np.float64(0.005),  # pyright: ignore[reportArgumentType]
        extra_windows_s=[0.0512, 0.0128],  # pyright: ignore[reportArgumentType]
        f_min_hz=500,
        f_max_hz=np.int64(1500),  # pyright: ignore[reportArgumentType]
    )
    assert cfg == SpectrogramConfig()
    data = cfg.to_dict()
    assert type(data["hop_s"]) is float
    assert type(data["f_max_hz"]) is float
    assert data["extra_windows_s"] == [0.0512, 0.0128]
    assert set(data) == {
        "reference_window_s",
        "extra_windows_s",
        "window",
        "hop_s",
        "f_min_hz",
        "f_max_hz",
        "aggregation",
        "dynamic_range_db",
        "headroom_db",
    }
    json.dumps(data)


def test_from_dict_fills_defaults_and_rejects_unknown_keys() -> None:
    assert SpectrogramConfig.from_dict({"hop_s": 0.0025}) == SpectrogramConfig(
        hop_s=0.0025
    )
    with pytest.raises(ValueError, match="unknown"):
        SpectrogramConfig.from_dict({"nfft": 4096})


@pytest.mark.parametrize(
    ("key", "value", "hint"),
    [
        ("window_lengths_samples", [4096, 1024, 256], "reference_window_s=0.2048"),
        ("hop_samples", 100, "hop_s"),
        ("samplerate_hz", 20_000.0, "comes from the recording"),
    ],
)
def test_sample_count_keys_are_rejected_with_the_conversion(
    key: str, value: Any, hint: str
) -> None:
    with pytest.raises(ValueError, match="seconds and Hz") as info:
        SpectrogramConfig.from_dict({key: value})
    assert hint in str(info.value)


def test_a_whole_sample_count_bundle_config_is_rejected() -> None:
    old = {
        "samplerate_hz": 20_000.0,
        "window_lengths_samples": [4096, 1024, 256],
        "window": "hann",
        "hop_samples": 100,
        "f_min_hz": 500.0,
        "f_max_hz": 1500.0,
        "aggregation": "sum",
        "dynamic_range_db": 60.0,
        "headroom_db": 10.0,
    }
    with pytest.raises(ValueError, match="sample-count keys") as info:
        SpectrogramConfig.from_dict(old)
    message = str(info.value)
    for key in ("hop_samples", "samplerate_hz", "window_lengths_samples"):
        assert key in message


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_window_s": 0.0},
        {"reference_window_s": float("nan")},
        {"reference_window_s": 4096},  # samples, not seconds: no longer valid
        {"extra_windows_s": (0.3,)},  # longer than the reference window
        {"extra_windows_s": (0.0512, 0.0512)},
        {"extra_windows_s": (-0.01,)},
        {"extra_windows_s": "0.0512"},
        {"hop_s": 0.0},
        {"hop_s": 0.0065},  # more than half the shortest window (12.8 ms)
        {"hop_s": True},
        {"f_min_hz": 2000.0, "f_max_hz": 2000.0},
        {"f_min_hz": -1.0},
        {"f_min_hz": 401.0, "f_max_hz": 402.0},  # no bin inside
        {"f_max_hz": float("nan")},
        {"window": "kaiser"},
        {"aggregation": "db_sum"},
        {"dynamic_range_db": 0.0},
        {"headroom_db": 60.0},
        {"headroom_db": -1.0},
    ],
)
def test_invalid_configs_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - message varies by field
        SpectrogramConfig(**kwargs)


def test_sample_count_arguments_no_longer_exist() -> None:
    with pytest.raises(TypeError):
        SpectrogramConfig(window_lengths_samples=(4096,))  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize(
    ("fs", "kwargs", "match"),
    [
        (2_000.0, {}, "Nyquist"),  # 1500 Hz band edge above 1000 Hz
        (0.0, {}, "positive"),
        (float("nan"), {}, "positive"),
        (50.0, {"f_min_hz": 1.0, "f_max_hz": 10.0}, "fewer than 2 samples"),
        (100.0, {"f_min_hz": 1.0, "f_max_hz": 10.0}, "hop"),  # 0.5 -> 0 samples
        (
            100.0,
            {"f_min_hz": 1.0, "f_max_hz": 10.0, "extra_windows_s": (0.0128, 0.012)},
            "coincide",
        ),
    ],
)
def test_configs_that_do_not_fit_a_rate_are_rejected_per_rate(
    fs: float, kwargs: dict[str, Any], match: str
) -> None:
    cfg = SpectrogramConfig(**kwargs)
    with pytest.raises(ValueError, match=match):
        cfg.layout(fs)
