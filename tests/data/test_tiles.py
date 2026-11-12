"""Tiling of spectrograms and the assignment of boxes to tiles."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from chirpdetector.data.tiles import (
    TileLayout,
    assign_to_tile,
    concat_boxes,
    render_tile,
    slice_spectrogram,
)
from chirpdetector.preprocessing import ClickRemoval, remove_clicks
from chirpdetector.spectrogram import (
    ImageGeometry,
    SpectrogramConfig,
    compute_spectrogram,
    normalize,
    to_image,
)

CONFIG = SpectrogramConfig()


def test_default_layout_is_640_frames_with_200_overlap() -> None:
    layout = TileLayout()
    assert layout.width_frames(CONFIG) == 640
    assert layout.overlap_frames(CONFIG) == 200
    tiling = layout.tiling(CONFIG)
    assert (tiling.width_px, tiling.height_px) == (640, 205)
    assert tiling.problems(CONFIG) == []


@pytest.mark.parametrize(
    ("n_frames", "expected"),
    [
        (640, [0]),
        (641, [0, 1]),
        (1080, [0, 440]),
        (1600, [0, 440, 880, 960]),
        (12000, [*range(0, 11360, 440), 11360]),
    ],
)
def test_tiles_cover_every_frame_and_end_at_the_last(
    n_frames: int, expected: list[int]
) -> None:
    starts = TileLayout().starts(n_frames, CONFIG)
    assert starts == expected
    covered = np.zeros(n_frames, dtype=bool)
    for s in starts:
        covered[s : s + 640] = True
    assert covered.all()
    assert starts[-1] + 640 == n_frames
    assert all(b - a <= 440 for a, b in itertools.pairwise(starts))


def test_layout_rejects_short_input_and_odd_widths() -> None:
    with pytest.raises(ValueError, match="fewer than one tile"):
        TileLayout().starts(639, CONFIG)
    with pytest.raises(ValueError, match="multiple of 32"):
        TileLayout(tile_duration_s=3.0).width_frames(CONFIG)
    with pytest.raises(ValueError, match="overlap"):
        TileLayout(overlap_s=3.2).overlap_frames(CONFIG)


GEOMETRY = ImageGeometry(
    width_px=640,
    height_px=205,
    t_first_s=10.0,
    dt_s=0.005,
    f_lowest_hz=502.9,
    df_hz=4.88,
)
T0, T1 = GEOMETRY.time_extent_s  # 9.9975, 13.1975


def _box(t0: float, t1: float, tc: float, label: str = "chirp") -> dict[str, object]:
    return {
        "t_start_s": t0,
        "t_end_s": t1,
        "f_low_hz": 700.0,
        "f_high_hz": 800.0,
        "t_center_s": tc,
        "label": label,
        "emitter_hz": 700.0,
        "chirp_id": 0,
    }


def test_edge_rule_keeps_boxes_mostly_inside() -> None:
    boxes = pd.DataFrame(
        [
            _box(11.0, 11.25, 11.1),  # inside
            _box(T0 - 0.07, T0 + 0.18, T0 + 0.05),  # 72% inside, centre in
            _box(T0 - 0.10, T0 + 0.15, T0 + 0.05),  # 60% inside
            _box(T1 - 0.20, T1 + 0.05, T1 + 0.01),  # 80% inside, centre out
            _box(12.0, 12.3, 12.1, "invisible"),
            _box(12.5, 12.8, 12.6, "harmonic"),
            _box(T1 - 0.001, T1 + 0.2, T1 + 0.1),  # a 0.2 px sliver
            _box(20.0, 20.3, 20.1),  # elsewhere
        ]
    ).assign(chirp_id=range(8))
    chirps, ignore = assign_to_tile(boxes, GEOMETRY)
    assert chirps["chirp_id"].tolist() == [0, 1]
    assert chirps.loc[1, "t_start_s"] == pytest.approx(T0)
    assert chirps.loc[1, "t_center_s"] == pytest.approx(T0 + 0.05)
    assert dict(zip(ignore["chirp_id"], ignore["reason"], strict=True)) == {
        2: "edge",
        3: "edge",
        4: "invisible",
        5: "harmonic",
    }
    assert bool((ignore["t_start_s"] >= T0).all())
    assert bool((ignore["t_end_s"] <= T1).all())
    mid = 0.5 * (ignore["t_start_s"] + ignore["t_end_s"])
    np.testing.assert_allclose(ignore["t_center_s"], mid)
    assert "label" not in chirps.columns
    assert "label" not in ignore.columns


def test_edge_fraction_is_configurable() -> None:
    boxes = pd.DataFrame([_box(T0 - 0.10, T0 + 0.15, T0 + 0.05)])
    chirps, ignore = assign_to_tile(boxes, GEOMETRY, edge_fraction=0.5)
    assert len(chirps) == 1
    assert len(ignore) == 0


def test_no_boxes_gives_empty_tables() -> None:
    boxes = pd.DataFrame([_box(11.0, 11.2, 11.1)]).iloc[:0]
    chirps, ignore = assign_to_tile(boxes, GEOMETRY)
    assert len(chirps) == 0
    assert len(ignore) == 0
    assert "reason" in ignore.columns


def test_concat_boxes_keeps_missing_live_tracks_as_none() -> None:
    a = pd.DataFrame({"t_start_s": [0.0], "live_tracks_hz": [[700.0, 750.0]]})
    b = pd.DataFrame({"t_start_s": [1.0]})
    out = concat_boxes([a, b])
    assert out["live_tracks_hz"].tolist() == [[700.0, 750.0], None]


def _signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(4.0 * 20_000)) / 20_000
    tone = np.sin(2 * np.pi * 700 * t)[:, None] * np.linspace(0.2, 1.0, 9)[None, :]
    return (tone + 0.01 * rng.standard_normal((t.size, 9))).astype(np.float32)


def test_tiles_are_normalized_on_their_own_frames() -> None:
    spec = compute_spectrogram(_signal(), 20_000.0, CONFIG)
    image, geometry = render_tile(spec, 40, TileLayout().tiling(CONFIG))
    tile = slice_spectrogram(spec, 40, 680)
    np.testing.assert_array_equal(image, to_image(normalize(tile, CONFIG)))
    assert geometry.t_first_s == pytest.approx(spec.times_s[40])
    assert geometry.width_px == 640
    assert geometry.height_px == CONFIG.n_freqs


def test_images_do_not_depend_on_the_recording_unit() -> None:
    """Volts, millivolts or simulation units give the same images."""
    x = _signal()
    images = []
    for gain in (1.0, 1e-3, 1e3):
        cleaned, mask = remove_clicks(x * np.float32(gain), 20_000.0, ClickRemoval())
        spec = compute_spectrogram(cleaned, 20_000.0, CONFIG)
        images.append((render_tile(spec, 0, TileLayout().tiling(CONFIG))[0], mask))
    for image, mask in images[1:]:
        np.testing.assert_array_equal(mask, images[0][1])
        diff = np.abs(image.astype(int) - images[0][0].astype(int))
        assert diff.max() <= 1
        assert (diff > 0).mean() < 1e-3
