"""Chunked inference equals `predict_span` on the whole recording."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gridtools.io import open_recording

from chirpdetector.bundle import Bundle
from chirpdetector.inference import merge_tiles, predict_span
from chirpdetector.pipeline.stream import (
    RecordingSlice,
    detect_recording,
    merge_global,
    tile_grid,
)
from chirpdetector.prediction import load_detector

COLUMNS = [
    "tile",
    "t_start_s",
    "t_end_s",
    "f_low_hz",
    "f_high_hz",
    "score",
    "emitter_hz",
]


@pytest.fixture
def recording(tmp_path: Path, make_synthetic_trial: Callable[[Path], Path]):
    trial = make_synthetic_trial(tmp_path / "trial")
    return open_recording(trial / "recording.wav")


def _reference(bundle: Bundle, rec, threshold: float = 0.05) -> pd.DataFrame:
    return predict_span(
        bundle,
        rec,
        0.0,
        rec.n_samples / rec.samplerate_hz,
        name="r",
        score_threshold=threshold,
        detector=load_detector(bundle),
        **_no_postprocess(),
    )


def _no_postprocess() -> dict:
    import inspect

    return (
        {"postprocess": False}
        if "postprocess" in inspect.signature(predict_span).parameters
        else {}
    )


@pytest.mark.parametrize(("chunk_tiles", "prefetch"), [(16, 1), (16, 3), (32, 2)])
def test_chunks_equal_one_pass(
    stub_bundle: Bundle, recording, chunk_tiles: int, prefetch: int
) -> None:
    ref = _reference(stub_bundle, recording)
    grid = tile_grid(stub_bundle, recording.samplerate_hz, recording.n_samples)
    assert grid.n_tiles > chunk_tiles or chunk_tiles == 32  # several chunks
    out = detect_recording(
        stub_bundle,
        recording,
        name="r",
        detector=load_detector(stub_bundle),
        store_threshold=0.05,
        chunk_tiles=chunk_tiles,
        prefetch=prefetch,
        click_workers=1,
    )
    assert len(ref) > 5
    pd.testing.assert_frame_equal(
        out.detections.loc[:, COLUMNS].reset_index(drop=True),
        ref.loc[:, COLUMNS].reset_index(drop=True),
        check_exact=True,
    )
    # chunk table covers the tiles once
    assert int(out.chunks["n_tiles"].to_numpy().sum()) == grid.n_tiles
    assert (
        out.detections["span_id"].to_numpy()
        == out.detections["tile"].to_numpy() // chunk_tiles
    ).all()


def test_store_threshold_is_a_superset(stub_bundle: Bundle, recording) -> None:
    """Thresholding the stored table later equals running at that threshold."""
    low = _reference(stub_bundle, recording, 0.05)
    high = _reference(stub_bundle, recording, 0.3)
    again = low.loc[low["score"] >= 0.3, COLUMNS].reset_index(drop=True)
    pd.testing.assert_frame_equal(again, high.loc[:, COLUMNS].reset_index(drop=True))


def test_time_range_matches_the_full_run_inside(stub_bundle: Bundle, recording) -> None:
    det = load_detector(stub_bundle)
    full = detect_recording(
        stub_bundle, recording, name="r", detector=det, chunk_tiles=16, click_workers=1
    ).detections
    part = detect_recording(
        stub_bundle,
        recording,
        name="r",
        detector=det,
        chunk_tiles=16,
        click_workers=1,
        t_range_s=(6.0, 7.0),
    ).detections
    assert len(part)
    # the range is processed in whole chunks of the full run's grid
    tiles = sorted(set(part["tile"]))
    assert tiles[0] % 16 == 0
    inner = full.loc[full["tile"].isin(tiles[1:-1]), COLUMNS].reset_index(drop=True)
    got = part.loc[part["tile"].isin(tiles[1:-1]), COLUMNS].reset_index(drop=True)
    pd.testing.assert_frame_equal(got, inner)


def test_slice_is_a_recording(recording) -> None:
    part = RecordingSlice(recording, 1000, 5000)
    assert part.n_samples == 4000
    np.testing.assert_array_equal(part.read(10, 20), recording.read(1010, 1020))
    with pytest.raises(IndexError):
        part.read(0, 4001)


def _rows(rng: np.random.Generator, n: int, n_tiles: int) -> pd.DataFrame:
    """Detections clustered at tile boundaries, each in a tile that owns it."""
    boundary = rng.integers(1, n_tiles - 1, n)
    centre = boundary + 0.02 + rng.normal(0, 0.02, n)
    tile = np.where(centre < boundary + 0.02, boundary - 1, boundary)
    f0 = rng.choice([700.0, 705.0, 900.0], n)
    return pd.DataFrame(
        {
            "recording": "r",
            "image_index": tile,
            "t_start_s": centre - 0.01,
            "t_end_s": centre + 0.01,
            "f_low_hz": f0,
            "f_high_hz": f0 + 80,
            "score": rng.uniform(0, 1, n),
        }
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_grouped_merge_equals_one_merge(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_tiles = 30
    owned = [
        (-np.inf, 1.02),
        *[(k + 0.02, k + 1.02) for k in range(1, n_tiles - 1)],
        (n_tiles - 0.98, np.inf),
    ]
    df = _rows(rng, 300, n_tiles)
    expected = merge_tiles(df, owned)
    got = merge_global(df, owned, group_rows=7)
    assert len(expected) < len(df)  # some twins were merged
    pd.testing.assert_frame_equal(got, expected)


@pytest.fixture(scope="module")
def clicky_recording(tmp_path_factory: pytest.TempPathFactory):
    """9 electrodes, two fish and clicks on most electrodes every 0.7 s."""
    from scipy.io import wavfile

    fs = 20_000
    rng = np.random.default_rng(3)
    t = np.arange(6 * fs) / fs
    gains = rng.uniform(0.2, 1.0, (2, 9))
    signal = np.zeros((len(t), 9))
    for k, f in enumerate((700.0, 760.0)):
        signal += np.sin(2 * np.pi * f * t)[:, None] * gains[k]
    signal += 0.01 * rng.standard_normal(signal.shape)
    for c in np.arange(0.3, 6.0, 0.7):
        i = int(c * fs)
        signal[i : i + 20, :7] += 5.0 * rng.standard_normal((20, 7))
    path = tmp_path_factory.mktemp("clicky") / "recording.wav"
    wavfile.write(path, fs, signal.astype(np.float32))
    return open_recording(path)


@pytest.fixture(scope="module")
def click_bundle(
    tmp_path_factory: pytest.TempPathFactory, stub_bundle: Bundle
) -> Bundle:
    from chirpdetector.bundle import load_bundle, write_bundle
    from chirpdetector.preprocessing import ClickRemoval

    directory = tmp_path_factory.mktemp("click_bundle")
    weights = directory / "w.bin"
    weights.write_bytes(b"click stub")
    write_bundle(
        directory / "bundle",
        weights=weights,
        model_type=stub_bundle.model_type,
        model=dict(stub_bundle.model),
        spectrogram=stub_bundle.spectrogram,
        tiling=stub_bundle.tiling,
        preprocessing={"click_removal": ClickRemoval().to_dict()},
        score_threshold=0.5,
        training={},
    )
    return load_bundle(directory / "bundle")


def test_click_removal_in_processes_equals_one_pass(
    click_bundle: Bundle, clicky_recording
) -> None:
    ref = _reference(click_bundle, clicky_recording)
    det = load_detector(click_bundle)
    threads = detect_recording(
        click_bundle,
        clicky_recording,
        name="r",
        detector=det,
        chunk_tiles=16,
        click_workers=2,
    )
    procs = detect_recording(
        click_bundle,
        clicky_recording,
        name="r",
        detector=det,
        chunk_tiles=16,
        click_workers=1,
        clean_processes=2,
        prefetch=3,
    )
    assert threads.click_removal
    assert procs.click_removal
    for out in (threads, procs):
        pd.testing.assert_frame_equal(
            out.detections.loc[:, COLUMNS].reset_index(drop=True),
            ref.loc[:, COLUMNS].reset_index(drop=True),
            check_exact=True,
        )
    # clicks were found and repaired, and counted per chunk
    assert (threads.chunks["n_click_runs"].to_numpy() > 0).all()
    pd.testing.assert_frame_equal(
        threads.chunks.loc[:, ["click_fraction", "n_click_runs"]],
        procs.chunks.loc[:, ["click_fraction", "n_click_runs"]],
    )
    np.testing.assert_array_equal(
        threads.detections["click_fraction_box"], procs.detections["click_fraction_box"]
    )
