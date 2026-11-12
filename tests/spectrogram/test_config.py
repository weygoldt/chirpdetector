import json
from typing import Any

import numpy as np
import pytest

from chirpdetector.spectrogram import SpectrogramConfig


def test_defaults_keep_the_long_reference_window() -> None:
    cfg = SpectrogramConfig()
    assert cfg.window_lengths_samples[0] == 4096
    assert cfg.n_fft == 4096
    assert cfg.df_hz == pytest.approx(20_000 / 4096)
    assert cfg.hop_s == pytest.approx(0.005)
    assert cfg.n_windows == 3


def test_default_band_is_the_v1_production_band() -> None:
    # Author's decision (2026-11-12): 500-1500 Hz, as in the v1 production run.
    cfg = SpectrogramConfig()
    assert (cfg.f_min_hz, cfg.f_max_hz) == (500.0, 1500.0)
    assert cfg.freqs_hz[0] >= 500.0
    assert cfg.freqs_hz[-1] <= 1500.0


def test_frequency_grid_is_reference_bins_inside_crop() -> None:
    cfg = SpectrogramConfig(f_min_hz=400.0, f_max_hz=2000.0)
    freqs = cfg.freqs_hz
    assert freqs.shape == (cfg.n_freqs,)
    assert freqs[0] >= 400.0
    assert freqs[0] - cfg.df_hz < 400.0
    assert freqs[-1] <= 2000.0
    assert freqs[-1] + cfg.df_hz > 2000.0
    np.testing.assert_allclose(np.diff(freqs), cfg.df_hz)
    np.testing.assert_allclose(freqs / cfg.df_hz, np.round(freqs / cfg.df_hz))


def test_crop_edges_on_a_bin_are_inclusive() -> None:
    df = 20_000 / 4096
    cfg = SpectrogramConfig(f_min_hz=100 * df, f_max_hz=200 * df)
    assert cfg.freq_bin_start == 100
    assert cfg.freq_bin_stop == 201


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"window_lengths_samples": (4096,)},
        {"window_lengths_samples": (2048, 512), "hop_samples": 50},
        {"aggregation": "max", "window": "blackman"},
        {"samplerate_hz": 44_100.0, "f_min_hz": 0.0, "f_max_hz": 22_050.0},
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
        samplerate_hz=20_000,
        hop_samples=np.int64(100),  # pyright: ignore[reportArgumentType]
        window_lengths_samples=[4096, 1024, 256],  # pyright: ignore[reportArgumentType]
    )
    assert cfg == SpectrogramConfig()
    data = cfg.to_dict()
    assert type(data["samplerate_hz"]) is float
    assert type(data["hop_samples"]) is int
    assert data["window_lengths_samples"] == [4096, 1024, 256]
    json.dumps(data)


def test_from_dict_fills_defaults_and_rejects_unknown_keys() -> None:
    assert SpectrogramConfig.from_dict({"hop_samples": 64}) == SpectrogramConfig(
        hop_samples=64
    )
    with pytest.raises(ValueError, match="unknown"):
        SpectrogramConfig.from_dict({"nfft": 4096})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hop_samples": 300},  # hop > shortest window
        {"hop_samples": 129},  # hop > half the shortest window
        {"hop_samples": 0},
        {"hop_samples": True},
        {"f_max_hz": 10_001.0},  # above Nyquist
        {"f_min_hz": 2000.0, "f_max_hz": 2000.0},
        {"f_min_hz": -1.0},
        {"f_min_hz": 401.0, "f_max_hz": 402.0},  # no bin inside
        {"f_max_hz": float("nan")},
        {"window_lengths_samples": ()},
        {"window_lengths_samples": (1024, 4096)},  # reference not longest
        {"window_lengths_samples": (4096, 1023)},  # odd
        {"window_lengths_samples": (4096, 1024, 1024)},
        {"window_lengths_samples": (4096.0, 1024)},
        {"window": "kaiser"},
        {"aggregation": "db_sum"},
        {"samplerate_hz": 0.0},
        {"samplerate_hz": float("inf")},
        {"dynamic_range_db": 0.0},
        {"headroom_db": 60.0},
        {"headroom_db": -1.0},
    ],
)
def test_invalid_configs_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - message varies by field
        SpectrogramConfig(**kwargs)
