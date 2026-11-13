"""The single-candidate view: tracks, zoom, snapping, pixel <-> data maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gridtools.io import Tracks, load_tracks, open_recording
from hypothesis import given
from hypothesis import strategies as st

from chirpdetector.labeling.queue import QueueItem
from chirpdetector.labeling.view import (
    TrackView,
    ViewConfig,
    colour_lut,
    colourize,
    data_to_pixel,
    pixel_to_data,
    prediction_of,
    render_view,
    snap,
    track_views,
    view_windows,
)

from .conftest import CHIRP


def _item(box: tuple[float, float, float, float], candidate=()) -> QueueItem:
    return QueueItem(
        item_id="r#x",
        trial="r",
        det_index=0,
        t_start_s=box[0],
        t_end_s=box[1],
        f_low_hz=box[2],
        f_high_hz=box[3],
        n_candidates=2,
        delta_f_hz=4.0,
        df_bin="0-10",
        candidate=candidate,
    )


def _tracks(*lines: tuple[float, float], starts: tuple[float, ...] = ()) -> Tracks:
    """Tracks sampled every 0.3277 s (as wavetracker): f = a + b t."""
    times = np.round(np.arange(0.0, 20.0, 0.3277), 6)
    freqs, idents, idx = [], [], []
    for k, (a, b) in enumerate(lines):
        t = times[times >= (starts[k] if k < len(starts) else 0.0)]
        freqs.append(a + b * t)
        idents.append(np.full(len(t), float(k)))
        idx.append(np.searchsorted(times, t))
    return Tracks(
        times_s=times,
        freqs_hz=np.concatenate(freqs),
        idents=np.concatenate(idents),
        time_indices=np.concatenate(idx),
    )


def _track(key: int, ident: int, t: list[float], f: list[float]) -> TrackView:
    return TrackView(key, ident, f[0], np.asarray(t, float), np.asarray(f, float))


# pixel <-> data ------------------------------------------------------------


def test_pixel_edges_are_the_extent() -> None:
    extent = (10.0, 10.6, 690.0, 714.0)
    assert pixel_to_data(0, 0, 1200, 480, extent) == (10.0, 714.0)
    assert pixel_to_data(1200, 480, 1200, 480, extent) == pytest.approx((10.6, 690.0))
    assert pixel_to_data(600, 240, 1200, 480, extent) == pytest.approx((10.3, 702.0))
    # 480 px for 24 Hz: one pixel is 0.05 Hz
    _, f = pixel_to_data(600, 241, 1200, 480, extent)
    assert 702.0 - f == pytest.approx(0.05)


@given(
    x=st.floats(0, 1, allow_nan=False),
    y=st.floats(0, 1, allow_nan=False),
    w=st.floats(10, 4000),
    h=st.floats(10, 3000),
    t0=st.floats(0, 20000),
    dt=st.floats(0.05, 5),
    f0=st.floats(0, 2000),
    df=st.floats(1, 1000),
)
def test_pixel_to_data_round_trips(x, y, w, h, t0, dt, f0, df) -> None:
    extent = (t0, t0 + dt, f0, f0 + df)
    t, f = pixel_to_data(x * w, y * h, w, h, extent)
    assert t0 <= t <= t0 + dt + 1e-9
    assert f0 - 1e-9 <= f <= f0 + df + 1e-9
    xb, yb = data_to_pixel(t, f, w, h, extent)
    assert xb == pytest.approx(x * w, abs=1e-6 * max(w, t0 / dt))
    assert yb == pytest.approx(y * h, abs=1e-6 * max(h, f0 / df))


def test_points_outside_the_panel_are_refused() -> None:
    with pytest.raises(ValueError, match="outside"):
        pixel_to_data(-1, 5, 100, 100, (0, 1, 0, 1))
    with pytest.raises(ValueError, match="empty"):
        pixel_to_data(0, 0, 0, 100, (0, 1, 0, 1))


# snapping ------------------------------------------------------------------


def test_snap_takes_the_nearest_track_at_the_click_time() -> None:
    flat = _track(1, 5, [0.0, 1.0], [700.0, 700.0])
    rising = _track(2, 9, [0.0, 1.0], [690.0, 710.0])  # crosses 700 at 0.5 s
    tracks = (flat, rising)
    early = snap(tracks, 0.1, 693.0, 15.0)
    assert early is not None
    assert early.track.track_id == 9
    assert early.track_hz == pytest.approx(692.0)
    assert early.offset_hz == pytest.approx(1.0)
    late = snap(tracks, 0.9, 701.0, 15.0)
    assert late is not None
    assert late.track.track_id == 5  # rising is at 708 Hz by then
    assert late.offset_hz == pytest.approx(1.0)


def test_snap_rejects_clicks_beyond_the_tolerance() -> None:
    tracks = (_track(1, 5, [0.0, 1.0], [700.0, 700.0]),)
    assert snap(tracks, 0.5, 714.9, 15.0) is not None
    assert snap(tracks, 0.5, 715.1, 15.0) is None
    assert snap((), 0.5, 700.0, 15.0) is None


def test_snap_ties_go_to_the_lower_key() -> None:
    tracks = (
        _track(1, 3, [0.0, 1.0], [700.0, 700.0]),
        _track(2, 1, [0.0, 1.0], [704.0, 704.0]),
    )
    found = snap(tracks, 0.5, 702.0, 15.0)
    assert found is not None
    assert found.track.key == 1


def test_track_frequency_is_held_beyond_the_polyline() -> None:
    track = _track(1, 0, [1.0, 2.0], [700.0, 710.0])
    assert track.frequency_at(0.0) == 700.0
    assert track.frequency_at(1.5) == pytest.approx(705.0)
    assert track.frequency_at(5.0) == 710.0


# tracks and zoom -----------------------------------------------------------


def test_tracks_are_numbered_by_baseline_at_the_box_start() -> None:
    # track 0 falls from 712 Hz, track 1 is flat at 704 Hz: at the box start
    # (4.0 s) track 0 is at 708 Hz, so the flat track is number 1
    tracks = _tracks((712.0, -1.0), (704.0, 0.0))
    views = track_views(tracks, _item((4.0, 4.1, 690.0, 800.0)), ViewConfig())
    assert [(v.key, v.track_id) for v in views] == [(1, 1), (2, 0)]
    assert views[1].baseline_hz == pytest.approx(708.0)
    assert views[0].baseline_hz == pytest.approx(704.0)
    t0, t1 = 4.05 - 1.0, 4.05 + 1.0  # span_s (2 s) centred on the box
    assert views[1].times_s[0] == pytest.approx(t0)
    assert views[1].times_s[-1] == pytest.approx(t1)
    np.testing.assert_allclose(views[1].freqs_hz, 712.0 - views[1].times_s)


def test_tracks_across_gaps_are_shown_with_their_gap() -> None:
    """wavetracker drops detections for seconds; the view bridges them."""
    times = np.round(np.arange(0.0, 20.0, 0.3277), 6)
    keep = (times < 2.6) | (times > 5.6)  # a 3 s gap around the box at 4 s
    t = times[keep]
    tracks = Tracks(
        times_s=times,
        freqs_hz=np.concatenate([np.full(len(times), 700.0), np.full(len(t), 704.0)]),
        idents=np.concatenate([np.zeros(len(times)), np.ones(len(t))]),
        time_indices=np.concatenate([np.arange(len(times)), np.flatnonzero(keep)]),
    )
    item = _item((4.0, 4.1, 690.0, 800.0))
    views = track_views(tracks, item, ViewConfig())
    assert [(v.track_id, v.baseline_hz) for v in views] == [(0, 700.0), (1, 704.0)]
    assert views[0].gap_s < 0.33
    assert views[1].gap_s == pytest.approx(t[t > 4.05].min() - t[t < 4.05].max())
    assert views[1].gap_s > 3.0
    strict = track_views(tracks, item, ViewConfig(max_gap_s=1.0))
    assert [v.track_id for v in strict] == [0]


def test_tracks_live_only_at_the_centre_count() -> None:
    late = _tracks((700.0, 0.0), (760.0, 0.0), starts=(0.0, 5.0))
    views = track_views(late, _item((0.43, 0.57, 690.0, 800.0)), ViewConfig())
    assert [v.track_id for v in views] == [0]


def test_zoom_spans_the_near_baselines_only() -> None:
    config = ViewConfig()
    item = _item((4.0, 4.1, 695.0, 820.0))
    # 700 and 704 Hz are near the box bottom; 900 Hz is above the box
    views = track_views(_tracks((700.0, 0), (704.0, 0), (900.0, 0)), item, config)
    windows = view_windows(item, views, config, 20_000.0)
    t0, t1, lo, hi = windows["zoom"]
    assert (lo, hi) == pytest.approx((690.0, 714.0))  # 4 Hz + 2 x 8 Hz < 24 Hz
    # overview: 500 Hz centred on box and near baselines (655-860 Hz needed)
    assert windows["overview"][2:] == pytest.approx((507.5, 1007.5))
    assert (t0, t1) == pytest.approx((3.05, 5.05))  # 2 s centred on the box
    # a 60 Hz pair: the zoom covers both baselines plus the padding
    views = track_views(_tracks((700.0, 0), (760.0, 0)), item, config)
    assert view_windows(item, views, config, 20_000.0)["zoom"][2:] == pytest.approx(
        (692.0, 768.0)
    )


def test_spans_grow_for_long_chirps_and_wide_baselines() -> None:
    config = ViewConfig()
    item = _item((4.0, 6.5, 700.0, 1350.0))  # a 2.5 s, 650 Hz box
    views = track_views(
        _tracks(
            (690.0, 0),
        ),
        item,
        config,
    )
    t0, t1, lo, hi = view_windows(item, views, config, 20_000.0)["overview"]
    assert (t0, t1) == pytest.approx((3.75, 6.75))  # box + pad_s
    assert (lo, hi) == pytest.approx((650.0, 1390.0))  # baseline/box + pad_hz


def test_zoom_falls_back_to_all_tracks_and_to_the_box() -> None:
    config = ViewConfig()
    item = _item((4.0, 4.1, 1400.0, 1600.0))  # e.g. a harmonic
    views = track_views(_tracks((700.0, 0), (760.0, 0)), item, config)
    assert view_windows(item, views, config, 20_000.0)["zoom"][2:] == pytest.approx(
        (692.0, 768.0)
    )
    lo, hi = view_windows(item, (), config, 20_000.0)["zoom"][2:]
    assert (lo, hi) == pytest.approx((1388.0, 1412.0))


# rendering -----------------------------------------------------------------


@pytest.fixture(scope="module")
def rendered(synthetic_trial: Path):
    item = _item(CHIRP)
    rec, tracks = open_recording(synthetic_trial), load_tracks(synthetic_trial)
    return render_view(rec, tracks, item, ViewConfig())


def test_panels_match_their_windows(rendered) -> None:
    view = rendered
    windows = view_windows(view.item, view.tracks, ViewConfig(), 20_000.0)
    for name in ("overview", "zoom"):
        panel = view.panel(name)
        n_freq, n_time = panel.db.shape
        dt = (panel.extent[1] - panel.extent[0]) / n_time
        df = (panel.extent[3] - panel.extent[2]) / n_freq
        np.testing.assert_allclose(panel.extent[:2], windows[name][:2], atol=dt)
        np.testing.assert_allclose(panel.extent[2:], windows[name][2:], atol=df)
    # zoom bins are at most zoom_bin_hz wide (zero padding)
    zoom = view.zoom
    assert (zoom.extent[3] - zoom.extent[2]) / zoom.db.shape[0] <= 0.25 + 1e-9
    for panel in (view.overview, zoom):  # noise floor to peak
        assert panel.clim == pytest.approx(
            (float(np.percentile(panel.db, 5.0)), float(panel.db.max()))
        )


def test_zoom_peaks_sit_on_the_tracks(rendered) -> None:
    """The brightest zoom row before the chirp is the track's frequency."""
    zoom = rendered.zoom
    n_freq, _ = zoom.db.shape
    df = (zoom.extent[3] - zoom.extent[2]) / n_freq
    freqs = zoom.extent[2] + (np.arange(n_freq) + 0.5) * df
    before = zoom.db[:, 0]
    for f in (700.0, 760.0):
        near = np.flatnonzero(np.abs(freqs - f) < 5.0)
        assert abs(freqs[near[np.argmax(before[near])]] - f) <= df


def test_short_window_changes_the_overview_only(
    synthetic_trial: Path, rendered
) -> None:
    item = _item(CHIRP)
    rec, tracks = open_recording(synthetic_trial), load_tracks(synthetic_trial)
    view = render_view(rec, tracks, item, ViewConfig(), window_s=0.0512)
    assert view.window_s == 0.0512
    # the zoom keeps the long window: close baselines stay separable
    np.testing.assert_array_equal(view.zoom.db, rendered.zoom.db)
    # 4x shorter window: 4x coarser native resolution, 4x finer in time
    n_short, n_long = view.overview.db.shape[0], rendered.overview.db.shape[0]
    assert n_short == pytest.approx(n_long / 4, rel=0.1)


def test_colourize_orientation(rendered) -> None:
    lut = colour_lut()
    up = colourize(rendered.zoom, lut)
    down = colourize(rendered.zoom, lut, top_down=True)
    assert up.shape == (*rendered.zoom.db.shape, 3)
    np.testing.assert_array_equal(down, up[::-1])
    peak = np.unravel_index(np.argmax(rendered.zoom.db), rendered.zoom.db.shape)
    np.testing.assert_array_equal(up[peak], lut[-1])


def test_prediction_comes_from_the_candidate_columns() -> None:
    assert prediction_of(_item(CHIRP)) is None
    item = _item(
        CHIRP,
        (
            ("candidate_id", "a"),
            ("candidate_score", 0.8),
            ("candidate_emitter_hz", 701.5),
            ("candidate_track_id", None),
            ("candidate_source", "m"),
        ),
    )
    p = prediction_of(item)
    assert p is not None
    assert (p.emitter_hz, p.track_id, p.score, p.source) == (701.5, None, 0.8, "m")
