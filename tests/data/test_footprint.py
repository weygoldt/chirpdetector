"""chirpdetector's front end as the spectrogram of gridtools' footprints."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gridtools.io import open_recording
from gridtools.sim.footprint import chirp_footprint

from chirpdetector.data.footprint import FrontEndSpectrogram, footprints_match
from chirpdetector.spectrogram import SpectrogramConfig, compute_spectrogram

CONFIG = SpectrogramConfig()


def _recording(scene_dir: Path) -> tuple[np.ndarray, float]:
    rec = open_recording(scene_dir / "recording.wav")
    return rec.read(0, rec.n_samples), rec.samplerate_hz


def test_frames_lie_on_the_recordings_grid(scene_dirs: dict[str, Path]) -> None:
    x, fs = _recording(scene_dirs["grid9"])
    whole = compute_spectrogram(x, fs, CONFIG)
    fn = FrontEndSpectrogram(CONFIG)
    start, stop = 12_345, 12_345 + 30_000  # not on the 100-sample hop grid
    power, freqs, times = fn(x[start:stop], fs, start / fs)
    hop = CONFIG.layout(fs).hop_samples
    frames = np.rint(times * fs / hop).astype(int)
    np.testing.assert_allclose(times, frames * hop / fs, rtol=0, atol=1e-12)
    # complete frames only: the whole reference window inside the span
    n_fft = CONFIG.layout(fs).n_fft
    assert frames[0] * hop - n_fft // 2 >= start
    assert frames[-1] * hop + n_fft // 2 <= stop
    assert frames[0] * hop - n_fft // 2 - hop < start
    np.testing.assert_array_equal(freqs, whole.freqs_hz)
    reference = whole.power[0].double().numpy()[:, frames]
    np.testing.assert_allclose(power, reference, rtol=1e-5, atol=0)


def test_other_channels_and_short_spans() -> None:
    with pytest.raises(ValueError, match="channel 3"):
        FrontEndSpectrogram(CONFIG, channel=3)
    fn = FrontEndSpectrogram(CONFIG, channel=2)
    assert fn.window_s == pytest.approx(0.2048)
    power, _freqs, times = fn(np.zeros((1000, 2), np.float32), 20_000.0, 0.0)
    assert power.shape == (CONFIG.n_freqs, 0)
    assert times.size == 0
    with pytest.raises(ValueError, match="sample grid"):
        fn(np.zeros((5000, 2), np.float32), 20_000.0, 0.00001)


def test_footprints_agree_with_the_scenes_hann_boxes(
    scene_dirs: dict[str, Path],
) -> None:
    """Same cells up to the frame grid: frequency edges equal, time within a hop."""
    scene = scene_dirs["grid9"]
    chirps = pd.read_csv(scene / "chirps.csv")
    x, _ = _recording(scene)
    fn = FrontEndSpectrogram(CONFIG)
    inner = chirps.loc[~chirps["truncated"]].head(8)
    assert len(inner) >= 5
    for row in inner.itertuples():
        fp = chirp_footprint(scene, int(row.chirp_id), fn, recording=x)
        stored = (row.box_t0_s, row.box_t1_s, row.box_f0_hz, row.box_f1_hz)
        assert fp.box[2] == pytest.approx(stored[2], abs=1e-6)
        assert fp.box[3] == pytest.approx(stored[3], abs=1e-6)
        assert fp.box[0] == pytest.approx(stored[0], abs=CONFIG.hop_s + 1e-9)
        assert fp.box[1] == pytest.approx(stored[1], abs=CONFIG.hop_s + 1e-9)
        # box edges on the dataset's pixel edges (frame centre +- hop / 2)
        for edge in fp.box[:2]:
            k = (edge - CONFIG.hop_s / 2) / CONFIG.hop_s
            assert k == pytest.approx(round(k), abs=1e-6)
        if abs(max(row.snr_baseline_db, row.snr_excursion_db)) > 1.0:
            assert fp.visible == row.visible


def test_scene_footprints_match_the_default_front_end(
    scene_dirs: dict[str, Path],
) -> None:
    info = json.loads((scene_dirs["grid9"] / "scene.json").read_text())
    fp = info["footprint"]
    assert footprints_match(fp, CONFIG)
    assert not footprints_match({**fp, "hop_s": 0.0025}, CONFIG)
    assert not footprints_match({**fp, "enabled": False}, CONFIG)
    assert not footprints_match({}, CONFIG)
    assert not footprints_match(fp, dataclasses.replace(CONFIG, f_max_hz=1400.0))
    assert not footprints_match(fp, dataclasses.replace(CONFIG, window="hamming"))
    assert not footprints_match(fp, dataclasses.replace(CONFIG, aggregation="max"))
