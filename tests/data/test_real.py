"""The real test set from hand labels (needs the recordings on /mnt/data2)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gridtools.io import read_labels, write_labels

from chirpdetector.data import validate_dataset
from chirpdetector.data.real import LABEL_FILE, build_real_dataset, hand_boxes

TRACKED = Path("/mnt/data2/2019_tube_competition/2019-12-09-10_00")
UNTRACKED = Path("/mnt/data2/2022_tube_competition/raw/2022-05-12-10_00")
LINKED = ("traces-grid1.raw", "fishgrid.cfg", "times.npy", "fund_v.npy")
LINKED += ("ident_v.npy", "idx_v.npy", "sign_v.npy")


def _short_copy(recording: Path, root: Path, duration_s: float) -> Path:
    """The recording (linked) with its first labelled region cut to ``duration_s``."""
    out = root / recording.name
    out.mkdir(parents=True)
    for name in LINKED:
        if (recording / name).exists():
            os.symlink(recording / name, out / name)
    labels = read_labels(recording / LABEL_FILE)
    region = labels.loc[labels["category"] == "labeled_region"].iloc[0].copy()
    region["t_end_s"] = region["t_start_s"] + duration_s
    mid = 0.5 * (labels["t_start_s"] + labels["t_end_s"])
    chirps = labels.loc[
        (labels["category"] == "chirp")
        & (mid >= region["t_start_s"])
        & (mid <= region["t_end_s"])
    ]
    write_labels(pd.concat([region.to_frame().T, chirps]), out / LABEL_FILE)
    return out


def test_hand_boxes_use_tracks_or_the_lower_edge() -> None:
    labels = pd.DataFrame(
        {
            "t_start_s": [1.0, 2.0],
            "t_end_s": [1.2, 2.2],
            "f_low_hz": [705.0, 600.0],
            "f_high_hz": [800.0, 700.0],
        }
    )

    class FakeTrack:
        def __init__(self, ident: int, f: float, stop: float) -> None:
            self.ident, self.f, self.stop = ident, f, stop

        def frequency_at(self, t: float) -> np.ndarray:
            return np.array([self.f if t <= self.stop else np.nan])

    tracks = [FakeTrack(0, 700.0, 10.0), FakeTrack(1, 760.0, 1.5)]
    boxes = hand_boxes(labels, tracks)
    assert boxes["emitter_hz"].tolist() == [700.0, 700.0]
    assert boxes["emitter_id"].tolist() == [0, 0]
    assert boxes["live_tracks_hz"].tolist() == [[700.0, 760.0], [700.0]]
    assert set(boxes["emitter_hz_source"]) == {"nearest_track_to_f_low"}
    proxy = hand_boxes(labels, None)
    assert proxy["emitter_hz"].tolist() == [705.0, 600.0]
    assert proxy["live_tracks_hz"].tolist() == [[705.0], [600.0]]
    assert bool(proxy["emitter_id"].isna().all())
    assert set(proxy["emitter_hz_source"]) == {"f_low_proxy"}
    assert set(proxy["label_source"]) == {"hand"}
    np.testing.assert_allclose(proxy["t_center_s"], [1.1, 2.1])


@pytest.mark.data
@pytest.mark.slow
def test_real_dataset_from_one_short_region(tmp_path: Path) -> None:
    recordings = [
        _short_copy(TRACKED, tmp_path / "rec", 12.0),
        _short_copy(UNTRACKED, tmp_path / "rec", 12.0),
    ]
    dataset = build_real_dataset(tmp_path / "dataset", recordings)
    manifest = validate_dataset(dataset.directory)
    assert set(manifest.splits.values()) == {"test"}
    assert set(manifest.sources) == {TRACKED.name, UNTRACKED.name}
    assert manifest.sources[TRACKED.name].tracks is not None
    assert manifest.sources[UNTRACKED.name].tracks is None
    for name in manifest.sources:
        extra = manifest.sources[name].extra
        assert extra["click_removal_applied"] is True  # 9 and 11 electrodes
        assert extra["n_electrodes"] >= 9
    boxes = dataset.boxes
    chirps = boxes.loc[(boxes["kind"] == "chirp").to_numpy()]
    assert len(chirps) > 0
    assert (chirps["label_source"] == "hand").all()
    of = chirps["image_id"].str.split("_r").str[0]
    sources = dict(zip(of, chirps["emitter_hz_source"], strict=True))
    assert sources[TRACKED.name] == "nearest_track_to_f_low"
    assert sources[UNTRACKED.name] == "f_low_proxy"
    # every hand label lies in some tile as a chirp or an edge ignore region
    for recording in recordings:
        labels = read_labels(recording / LABEL_FILE)
        rows = labels.loc[labels["category"] == "chirp"].index
        mine = boxes.loc[
            boxes["image_id"].str.startswith(recording.name).to_numpy()
            & boxes["label_row"].notna().to_numpy()
        ]
        assert set(mine["label_row"].astype(int)) == set(rows)
        region = labels.loc[labels["category"] == "labeled_region"].iloc[0]
        records = [r for r in manifest.images if r.source == recording.name]
        assert records[0].extent[0] == pytest.approx(region["t_start_s"], abs=0.005)
        assert records[-1].extent[1] == pytest.approx(region["t_end_s"], abs=0.005)
        assert records[0].extra["labelled_f_low_hz"] == pytest.approx(
            region["f_low_hz"]
        )
    unlabelled = boxes.loc[(boxes["reason"] == "unlabelled").to_numpy()]
    assert len(unlabelled) >= len(manifest.images)
