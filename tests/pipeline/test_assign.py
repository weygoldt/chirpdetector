"""Assignment to wavetracker tracks: liveness, gaps, tolerance, ids, confidence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from gridtools.io import Tracks

from chirpdetector.pipeline.assign import (
    ASSIGN_COLUMNS,
    SIM_V4_DUALHEAD,
    assign_detections,
    track_baselines,
)
from chirpdetector.pipeline.config import AssignmentConfig

DT = 0.1
TIMES = np.round(np.arange(0.0, 100.0, DT), 6)


def _tracks(spec: dict[float, tuple[np.ndarray, np.ndarray]], noise: int = 0) -> Tracks:
    """Tracks from ``{ident: (times, freqs)}``, plus unassigned detections."""
    fund, ident, idx = [], [], []
    for i, (t, f) in spec.items():
        fund.append(np.asarray(f, float))
        ident.append(np.full(len(t), float(i)))
        idx.append(np.searchsorted(TIMES, np.round(t, 6)))
    if noise:
        rng = np.random.default_rng(0)
        fund.append(rng.uniform(600, 900, noise))
        ident.append(np.full(noise, np.nan))
        idx.append(rng.integers(0, len(TIMES), noise))
    return Tracks(
        times_s=TIMES,
        freqs_hz=np.concatenate(fund),
        idents=np.concatenate(ident),
        time_indices=np.concatenate(idx),
    )


def _flat(t0: float, t1: float, f: float) -> tuple[np.ndarray, np.ndarray]:
    t = TIMES[(t0 <= TIMES) & (t1 >= TIMES)]
    return t, np.full(len(t), f)


def _dets(onsets, emitters) -> pd.DataFrame:
    onsets = np.asarray(onsets, float)
    return pd.DataFrame(
        {
            "t_start_s": onsets,
            "t_end_s": onsets + 0.1,
            "emitter_hz": np.asarray(emitters, float),
        }
    )


CFG = AssignmentConfig(tolerance_hz=15.0, max_gap_s=1.0, baseline_window_s=1.0)


def test_nearest_live_track_keeps_the_wavetracker_id() -> None:
    tracks = _tracks(
        {1044.0: _flat(0, 100, 700.0), 3.0: _flat(0, 100, 760.0)}, noise=500
    )
    out = assign_detections(_dets([10, 20, 30], [701.0, 758.0, 729.5]), tracks, CFG)
    assert list(out.columns) == list(ASSIGN_COLUMNS)
    assert out["track_id"].dtype == np.float64
    np.testing.assert_array_equal(out["track_id"].to_numpy()[:2], [1044.0, 3.0])
    assert list(out["assign_status"]) == ["assigned", "assigned", "beyond_tolerance"]
    assert np.isnan(out["track_id"].iloc[2])
    # nearest/second are kept even when unassigned
    assert out["nearest_track_id"].iloc[2] == 1044.0
    assert out["second_track_id"].iloc[2] == 3.0
    np.testing.assert_allclose(out["nearest_df_hz"], [1.0, 2.0, 29.5])
    np.testing.assert_allclose(out["assign_margin_hz"], [58.0, 56.0, 1.0])
    np.testing.assert_allclose(out["track_df_other_hz"], [60.0, 60.0, 60.0])
    assert (out["n_live_tracks"].to_numpy() == 2).all()
    np.testing.assert_allclose(out["track_hz"].to_numpy()[:2], [700.0, 760.0])


def test_tracks_in_a_long_gap_are_not_live() -> None:
    t1, f1 = _flat(0, 40, 700.0)
    t2, f2 = _flat(45, 100, 702.0)  # 5-s gap
    tracks = _tracks({7.0: (np.r_[t1, t2], np.r_[f1, f2]), 8.0: _flat(0, 100, 900.0)})
    out = assign_detections(_dets([42.0, 50.0], [701.0, 702.0]), tracks, CFG)
    assert out["assign_status"].tolist() == ["beyond_tolerance", "assigned"]
    assert out["nearest_track_id"].iloc[0] == 8.0
    assert out["n_live_tracks"].tolist() == [1, 2]
    # a longer max_gap bridges it, with the baseline interpolated at the onset
    wide = AssignmentConfig(tolerance_hz=15.0, max_gap_s=10.0)
    bridged = assign_detections(_dets([42.0], [701.0]), tracks, wide)
    assert bridged["track_id"].iloc[0] == 7.0
    assert bridged["track_hz"].iloc[0] == pytest.approx(
        700.0 + 2.0 * (42.0 - 40.0) / 5.0
    )


def test_no_live_track_no_tracks_and_no_emitter() -> None:
    tracks = _tracks({1.0: _flat(50, 100, 700.0)})
    out = assign_detections(
        _dets([10.0, 60.0, 70.0], [700.0, 700.0, np.nan]), tracks, CFG
    )
    assert out["assign_status"].tolist() == ["no_live_track", "assigned", "no_emitter"]
    assert out["n_live_tracks"].tolist() == [0, 1, 1]
    none = assign_detections(_dets([10.0], [700.0]), None, CFG)
    assert none["assign_status"].tolist() == ["no_tracks"]
    box_only = assign_detections(
        _dets([60.0], [700.0]).drop(columns="emitter_hz"), tracks, CFG
    )
    assert box_only["assign_status"].tolist() == ["no_emitter"]
    assert np.isnan(box_only["track_id"]).all()
    assert len(assign_detections(_dets([], []), tracks, CFG)) == 0


def test_baseline_is_taken_before_the_onset() -> None:
    """A track that follows the chirp's excursion is read before the onset."""
    t, f = _flat(0, 100, 700.0)
    f = f.copy()
    f[(t >= 30.0) & (t <= 30.3)] = 760.0  # the tracker follows a big chirp
    tracks = _tracks({5.0: (t, f), 6.0: _flat(0, 100, 745.0)})
    out = assign_detections(_dets([30.0], [700.5]), tracks, CFG)
    assert out["track_id"].iloc[0] == 5.0
    assert out["track_hz"].iloc[0] == pytest.approx(700.0)
    base_ids, base = track_baselines(tracks, [30.0], CFG)
    assert base_ids.tolist() == [5.0, 6.0]
    assert base[0, 0] == pytest.approx(700.0)


def test_close_tracks_lower_the_confidence() -> None:
    tracks = _tracks({1.0: _flat(0, 100, 700.0), 2.0: _flat(0, 100, 703.0)})
    far = _tracks({1.0: _flat(0, 100, 700.0), 2.0: _flat(0, 100, 760.0)})
    close = assign_detections(_dets([10.0, 10.0], [700.5, 701.5]), tracks, CFG)
    apart = assign_detections(_dets([10.0], [700.5]), far, CFG)
    assert close["track_id"].tolist() == [1.0, 1.0]
    assert close["assign_confidence"].iloc[0] < apart["assign_confidence"].iloc[0]
    assert close["assign_confidence"].iloc[1] < close["assign_confidence"].iloc[0]
    assert 0.5 < close["assign_confidence"].iloc[0] < 0.9
    assert apart["assign_confidence"].iloc[0] > 0.95
    np.testing.assert_allclose(close["track_df_other_hz"], 3.0)


def test_posterior_sums_to_one_with_the_untracked_option() -> None:
    err = np.array([[0.5, 3.0, np.nan], [80.0, np.nan, np.nan]])
    post = SIM_V4_DUALHEAD.posterior(err)
    total = np.nansum(post, axis=1)
    assert np.all(total < 1)
    assert total[0] > 0.95  # a close track explains it
    assert total[1] < 0.5  # 80 Hz off: probably untracked
    assert np.isnan(post[0, 2])


def test_blocks_give_the_same_result() -> None:
    tracks = _tracks({1.0: _flat(0, 100, 700.0), 2.0: _flat(20, 100, 740.0)}, noise=50)
    rng = np.random.default_rng(1)
    dets = _dets(rng.uniform(0, 99, 300), rng.uniform(690, 760, 300))
    pd.testing.assert_frame_equal(
        assign_detections(dets, tracks, CFG, block=17),
        assign_detections(dets, tracks, CFG),
    )
