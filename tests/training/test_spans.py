"""Span prediction: tiling, ownership of overlaps, cross-tile duplicates."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gridtools.io import load_tracks, open_recording

from chirpdetector.bundle import Bundle, load_bundle, write_bundle
from chirpdetector.data import Tiling
from chirpdetector.inference import (
    merge_tiles,
    predict_span,
    render_span,
    render_tiles,
    tile_starts,
)
from chirpdetector.postprocess import COLUMNS, PostprocessConfig
from chirpdetector.spectrogram import (
    SpectrogramConfig,
    compute_spectrogram,
    normalize,
    timefreq_to_pixel,
    to_image,
)

from .conftest import STUB, StubBackend

CONFIG = SpectrogramConfig()
TILING = Tiling(
    tile_duration_s=0.64, overlap_s=0.2, width_px=128, height_px=CONFIG.n_freqs
)


@pytest.mark.parametrize(
    ("n", "width", "overlap"), [(128, 128, 40), (500, 128, 40), (1000, 128, 0)]
)
def test_tiles_cover_every_frame_without_overhang(
    n: int, width: int, overlap: int
) -> None:
    starts = tile_starts(n, width, overlap)
    assert starts[0] == 0
    assert starts[-1] == n - width
    covered = np.zeros(n, bool)
    for s in starts:
        covered[s : s + width] = True
    assert covered.all()
    assert all(b - a <= width - overlap for a, b in itertools.pairwise(starts))
    with pytest.raises(ValueError, match="fewer than one tile"):
        tile_starts(width - 1, width, overlap)


def test_tiles_are_normalised_on_their_own_frames() -> None:
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(20_000 * 2, 2)).astype(np.float32)
    spec = compute_spectrogram(signal, 20_000.0, CONFIG)
    tiles = render_tiles(spec, TILING)
    for image, start, geometry in zip(
        tiles.images, tiles.starts, tiles.geometries, strict=True
    ):
        assert image.shape == (CONFIG.n_freqs, 128, 3)
        assert geometry.t_first_s == pytest.approx(spec.times_s[start])
    # owned intervals partition the time axis
    lo = [o[0] for o in tiles.owned_s]
    hi = [o[1] for o in tiles.owned_s]
    assert lo[0] == -np.inf
    assert hi[-1] == np.inf
    assert lo[1:] == hi[:-1]
    from chirpdetector.inference.spans import _slice

    first = _slice(spec, 0, 128)
    np.testing.assert_array_equal(tiles.images[0], to_image(normalize(first, CONFIG)))


def _row(
    tile: int, centre: float, score: float, f: tuple[float, float] = (700, 800)
) -> dict:
    return {
        "recording": "r",
        "image_index": tile,
        "t_start_s": centre - 0.01,
        "t_end_s": centre + 0.01,
        "f_low_hz": f[0],
        "f_high_hz": f[1],
        "score": score,
    }


def test_merge_keeps_owned_detections_and_drops_cross_tile_twins() -> None:
    owned = [(-np.inf, 1.0), (1.0, 2.0), (2.0, np.inf)]
    rows = [
        _row(0, 0.5, 0.9),  # owned by tile 0
        _row(1, 0.5, 0.95),  # same chirp seen by tile 1, outside its part
        _row(0, 0.99, 0.6),  # at the boundary, tile 0's side
        _row(1, 1.01, 0.7),  # the same chirp, tile 1's side: duplicate
        _row(1, 1.5, 0.8),  # two fish chirping at once in one tile:
        _row(1, 1.5, 0.7, (705, 790)),  # both kept (no NMS within a tile)
        _row(2, 2.03, 0.5, (900, 950)),  # near the boundary, other band: kept
        _row(1, 1.99, 0.6),
    ]
    out = merge_tiles(pd.DataFrame(rows), owned)
    kept = list(
        zip(
            out["tile"], np.round(out["t_start_s"] + 0.01, 3), out["score"], strict=True
        )
    )
    assert kept == [
        (0, 0.5, 0.9),
        (1, 1.01, 0.7),
        (1, 1.5, 0.8),
        (1, 1.5, 0.7),
        (1, 1.99, 0.6),
        (2, 2.03, 0.5),
    ]


@pytest.fixture
def span_bundle(tmp_path: Path, stub_backend: StubBackend) -> Bundle:
    weights = tmp_path / "w.bin"
    weights.write_bytes(b"stub")
    write_bundle(
        tmp_path / "bundle",
        weights=weights,
        model_type=STUB,
        model={},
        spectrogram=CONFIG,
        tiling=TILING,
        preprocessing={"click_removal": None},
        score_threshold=0.5,
        training={},
    )
    return load_bundle(tmp_path / "bundle")


def test_predict_span_maps_each_tile(
    span_bundle: Bundle,
    stub_backend: StubBackend,
    make_synthetic_trial: Callable[[Path], Path],
    tmp_path: Path,
) -> None:
    trial = make_synthetic_trial(tmp_path / "trial")
    recording = open_recording(trial / "recording.wav")
    stub_backend.output = pd.DataFrame(
        {
            "x_min": [60.0, 0.0, 118.0],
            "y_min": [10.0, 10.0, 10.0],
            "x_max": [70.0, 10.0, 128.0],
            "y_max": [40.0, 40.0, 40.0],
            "score": [0.9, 0.8, 0.7],
        }
    )
    out = predict_span(span_bundle, recording, 2.0, 5.0, name="trial")
    n_tiles = out["tile"].max() + 1
    # the middle box of every tile, the left one of the first, the right one of the last
    assert len(out) == n_tiles + 2
    assert set(out["recording"]) == {"trial"}
    centres = 0.5 * (out["t_start_s"].to_numpy() + out["t_end_s"].to_numpy())
    assert centres.min() >= 2.0 - 0.01
    assert centres.max() <= 5.0 + 0.01
    assert (np.diff(centres) >= 0).all()
    # a short span is widened to one tile
    short = predict_span(span_bundle, recording, 4.0, 4.1, name="trial")
    assert set(short["tile"]) == {0}


def test_predict_span_assigns_then_drops_copies_and_duplicates(
    span_bundle: Bundle,
    stub_backend: StubBackend,
    make_synthetic_trial: Callable[[Path], Path],
    tmp_path: Path,
) -> None:
    """Per tile: a chirp of fish 0 (700 Hz), its second harmonic and a second,
    narrower box on the chirp; the emitter head predicts their baselines."""
    trial = make_synthetic_trial(tmp_path / "trial")
    recording = open_recording(trial / "recording.wav")
    tracks = load_tracks(trial)
    geometry = render_span(span_bundle, recording, 2.0, 5.0).geometries[0]

    def rows(*freqs: float) -> list[float]:
        return [float(timefreq_to_pixel(0.0, f, geometry)[1]) for f in freqs]

    lo, hi = rows(670.0, 1370.0, 670.0), rows(800.0, 1500.0, 790.0)
    stub_backend.output = pd.DataFrame(
        {
            "x_min": [60.0, 60.0, 61.0],
            "y_min": np.minimum(lo, hi),
            "x_max": [70.0, 70.0, 70.0],
            "y_max": np.maximum(lo, hi),
            "score": [0.9, 0.8, 0.7],
            "emitter_y": rows(700.0, 1400.0, 701.0),
        }
    )
    kwargs = {"name": "trial", "tracks": tracks}
    off = predict_span(span_bundle, recording, 2.0, 5.0, postprocess=False, **kwargs)
    n_tiles = len(off) // 3
    assert len(off) == 3 * n_tiles
    assert not set(COLUMNS) & set(off.columns)
    assert off.attrs["provenance"]["postprocess"] is None
    # the chirp and its narrower twin are assigned to fish 0, the copy to none
    assert off["emitter_id"].tolist() == [0, pd.NA, 0] * n_tiles

    flagged = predict_span(
        span_bundle, recording, 2.0, 5.0, keep_dropped=True, **kwargs
    )
    pd.testing.assert_frame_equal(flagged.drop(columns=list(COLUMNS)), off)
    assert flagged["drop_reason"].tolist() == ["", "harmonic", "duplicate"] * n_tiles
    assert flagged["harmonic_of_id"].dropna().unique().tolist() == [0]
    provenance = flagged.attrs["provenance"]
    assert provenance["tracks"] is True
    assert provenance["emitter_tolerance_hz"] == 50.0
    assert provenance["postprocess"]["counts"]["harmonic"] == n_tiles
    assert provenance["postprocess"]["harmonic_tolerance_hz"] == 15.0

    kept = predict_span(span_bundle, recording, 2.0, 5.0, **kwargs)
    assert len(kept) == n_tiles
    assert (kept["score"].to_numpy() == 0.9).all()
    assert (kept["emitter_id"].to_numpy() == 0).all()

    # without tracks: the copy is only flagged (pair rule), the twin dropped
    trackless = predict_span(span_bundle, recording, 2.0, 5.0, name="trial")
    assert "emitter_id" not in trackless.columns
    assert trackless["score"].tolist() == [0.9, 0.8] * n_tiles
    assert trackless["harmonic_pair_of"].tolist() == [
        x for k in range(n_tiles) for x in (pd.NA, 2 * k)
    ]
    dropping = predict_span(
        span_bundle,
        recording,
        2.0,
        5.0,
        name="trial",
        postprocess=PostprocessConfig(trackless="drop"),
    )
    assert len(dropping) == n_tiles


def test_tiling_equals_the_dataset_builders() -> None:
    """One definition of tiles: inference cuts exactly the dataset's tiles.

    Runs once the dataset builder (``chirpdetector.data.tiles``) is merged;
    then inference should import those functions instead of its own.
    """
    tiles_module = pytest.importorskip("chirpdetector.data.tiles")
    layout = tiles_module.TileLayout(tile_duration_s=3.2, overlap_s=1.0)
    rng = np.random.default_rng(1)
    signal = rng.normal(size=(20_000 * 9, 3)).astype(np.float32)
    spec = compute_spectrogram(signal, 20_000.0, CONFIG)
    tiling = layout.tiling(CONFIG)
    ours = render_tiles(spec, tiling)
    assert ours.starts == layout.starts(spec.n_times, CONFIG)
    for start, image, geometry in zip(
        ours.starts, ours.images, ours.geometries, strict=True
    ):
        theirs, their_geometry = tiles_module.render_tile(spec, start, tiling)
        np.testing.assert_array_equal(image, theirs)
        assert geometry == their_geometry
