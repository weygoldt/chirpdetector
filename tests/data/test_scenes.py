"""Detector datasets from simulated scenes: tiling, labels, splits, geometry."""

from __future__ import annotations

import dataclasses
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gridtools.io import load_tracks

from chirpdetector.data import Dataset, load_dataset, read_image, validate_dataset
from chirpdetector.data.scenes import (
    DEFAULT_SPLIT,
    SceneDatasetConfig,
    SplitRule,
    build_scene_dataset,
    find_scenes,
    scene_boxes,
    tile_scene,
)
from chirpdetector.preprocessing import ClickRemoval
from chirpdetector.spectrogram import SpectrogramConfig, timefreq_to_pixel

SPLIT = SplitRule(ranges={"train": (0, 4), "val": (4, 5), "test": (5, 6)})
CONFIG = SceneDatasetConfig(split=SPLIT, workers=1)


@pytest.fixture(scope="module")
def dataset(
    scene_dirs: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> Dataset:
    out = tmp_path_factory.mktemp("sim") / "dataset"
    return build_scene_dataset(list(scene_dirs.values()), out, CONFIG)


def _scene_chirps(dataset: Dataset, source: str) -> pd.DataFrame:
    scene_dir = Path(dataset.manifest.sources[source].extra["scene_dir"])
    return pd.read_csv(scene_dir / "chirps.csv")


def _rows(dataset: Dataset, source: str) -> pd.DataFrame:
    ids = [r.id for r in dataset.manifest.images if r.source == source]
    return dataset.boxes.loc[dataset.boxes["image_id"].isin(ids).to_numpy()]


# ------------------------------------------------------------------ splits


def test_default_split_by_seed() -> None:
    splits = DEFAULT_SPLIT.assign([0, 1599, 1600, 1799, 1800, 1999])
    assert list(splits.values()) == ["train", "train", "val", "val", "test", "test"]
    with pytest.raises(ValueError, match="no split range"):
        DEFAULT_SPLIT.assign([2000])


def test_split_by_fractions_is_seeded_and_complete() -> None:
    rule = SplitRule.parse("train=0.8,val=0.1,test=0.1", seed=3)
    seeds = range(100)
    first, again = rule.assign(seeds), rule.assign(reversed(seeds))
    assert first == again
    counts = pd.Series(first).value_counts().to_dict()
    assert counts == {"train": 80, "val": 10, "test": 10}
    other = SplitRule.parse("train=0.8,val=0.1,test=0.1", seed=4).assign(seeds)
    assert other != first
    small = rule.assign(range(3))
    assert sorted(small.values()) == ["test", "train", "val"]


def test_split_rule_parsing_and_errors() -> None:
    rule = SplitRule.parse("train=0:1600, val=1600:1800, test=1800:2000")
    assert rule == DEFAULT_SPLIT
    with pytest.raises(ValueError, match="overlap"):
        SplitRule.parse("train=0:10,val=5:20")
    with pytest.raises(ValueError, match="sum to 1"):
        SplitRule.parse("train=0.8,val=0.1")
    with pytest.raises(ValueError, match="unknown splits"):
        SplitRule.parse("training=0:10")
    with pytest.raises(ValueError, match="mix"):
        SplitRule.parse("train=0:10,val=0.5")


def test_find_scenes_refuses_incomplete_directories(
    scene_dirs: dict[str, Path], tmp_path: Path
) -> None:
    root = scene_dirs["grid9"].parent
    assert find_scenes(root) == sorted(scene_dirs.values())
    assert find_scenes(str(root / "00000[34]")) == [
        scene_dirs["grid9"],
        scene_dirs["grid4"],
    ]
    (tmp_path / "000001").mkdir()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        find_scenes(tmp_path)
    with pytest.raises(FileNotFoundError, match="no scene"):
        find_scenes(tmp_path / "nothing*")


# ---------------------------------------------------------------- dataset


def test_dataset_is_valid_with_one_group_per_scene(dataset: Dataset) -> None:
    manifest = validate_dataset(dataset.directory)
    assert set(manifest.sources) == {"test-000003", "test-000004", "test-000005"}
    assert manifest.splits == {
        "test-000003": "train",
        "test-000004": "val",
        "test-000005": "test",
    }
    for source in manifest.sources.values():
        assert source.group == source.id
        assert source.kind == "simulation"
        assert source.samplerate_hz == 20_000.0
        assert Path(str(source.tracks)).is_dir()
    assert manifest.tiling.width_px == 640
    assert manifest.tiling.height_px == 205
    assert manifest.spectrogram == SpectrogramConfig()
    assert manifest.provenance["builder"].endswith("build_scene_dataset")
    assert (dataset.directory / "diagnostics" / "contact_sheet.png").is_file()
    counts = json.loads((dataset.directory / "diagnostics" / "counts.json").read_text())
    assert set(counts) == {"train", "val", "test"}


def test_tiles_cover_each_scene_end_to_end(dataset: Dataset) -> None:
    hop = 0.005
    for source in dataset.manifest.sources:
        records = sorted(
            (r for r in dataset.manifest.images if r.source == source),
            key=lambda r: r.geometry.t_first_s,
        )
        assert [r.extra["frame_start"] for r in records] == [0, 440, 880, 960]
        assert records[0].geometry.t_first_s == pytest.approx(0.0)
        last = records[-1].geometry
        # 8 s at 5 ms: frames 0..1599, the last centred 5 ms before the end
        assert last.t_first_s + (last.width_px - 1) * last.dt_s == pytest.approx(
            8.0 - hop
        )
        for a, b in itertools.pairwise(records):
            assert a.extent[1] - b.extent[0] >= 1.0 - 1e-9


def test_click_removal_follows_the_electrode_count(dataset: Dataset) -> None:
    sources = dataset.manifest.sources
    assert sources["test-000003"].extra["n_electrodes"] == 9
    assert sources["test-000003"].extra["click_removal_applied"] is True
    assert sources["test-000004"].extra["n_electrodes"] == 4
    assert sources["test-000004"].extra["click_removal_applied"] is False
    assert dataset.manifest.preprocessing == {"click_removal": ClickRemoval().to_dict()}


def test_labels_follow_visibility_and_the_edge_rule(dataset: Dataset) -> None:
    """Every box is what the footprint table and the 70 % rule say it is."""
    checked = {"chirp": 0, "edge": 0, "invisible": 0, "harmonic": 0}
    for source in dataset.manifest.sources:
        chirps = _scene_chirps(dataset, source).set_index("chirp_id")
        rows = _rows(dataset, source)
        for record in (r for r in dataset.manifest.images if r.source == source):
            t0, t1, _, _ = record.extent
            here = rows.loc[(rows["image_id"] == record.id).to_numpy()]
            got = {
                (int(r.chirp_id), "chirp" if r.kind == "chirp" else str(r.reason))
                for r in here.itertuples()
            }
            expected = set()
            for c in chirps.reset_index().to_dict(orient="records"):
                cid = int(c["chirp_id"])
                for label, edges in (
                    ("fundamental", ("box_t0_s", "box_t1_s")),
                    ("harmonic", ("harmonic_box_t0_s", "harmonic_box_t1_s")),
                ):
                    b0, b1 = float(c[edges[0]]), float(c[edges[1]])
                    if not (np.isfinite(b0) and b1 > t0 and b0 < t1):
                        continue
                    if min(b1, t1) - max(b0, t0) < 0.5 * record.geometry.dt_s:
                        continue  # sliver
                    if label == "harmonic":
                        expected.add((cid, "harmonic"))
                    elif not c["visible"]:
                        expected.add((cid, "invisible"))
                    elif (min(b1, t1) - max(b0, t0)) >= 0.7 * (b1 - b0) - 1e-12 and (
                        t0 <= c["t_center_s"] <= t1
                    ):
                        expected.add((cid, "chirp"))
                    else:
                        expected.add((cid, "edge"))
            assert got == expected, record.id
            for _, kind in got:
                checked[kind] += 1
    assert all(n > 0 for n in checked.values()), checked


def test_every_visible_chirp_is_a_positive_somewhere(dataset: Dataset) -> None:
    for source in dataset.manifest.sources:
        chirps = _scene_chirps(dataset, source)
        rows = _rows(dataset, source)
        positives = set(rows.loc[rows["kind"] == "chirp", "chirp_id"].astype(int))
        visible = set(chirps.loc[chirps["visible"], "chirp_id"].astype(int))
        assert visible == positives
        invisible = set(chirps.loc[~chirps["visible"], "chirp_id"].astype(int))
        assert not invisible & positives


def test_positive_boxes_are_the_clipped_footprints(dataset: Dataset) -> None:
    source = "test-000003"
    chirps = _scene_chirps(dataset, source).set_index("chirp_id")
    rows = _rows(dataset, source)
    positives = rows.loc[(rows["kind"] == "chirp").to_numpy()]
    for row in positives.itertuples():
        c = chirps.loc[int(row.chirp_id)]
        t0, t1, f0, f1 = dataset.manifest.image(str(row.image_id)).extent
        assert row.t_start_s == pytest.approx(max(c["box_t0_s"], t0))
        assert row.t_end_s == pytest.approx(min(c["box_t1_s"], t1))
        assert row.f_low_hz == pytest.approx(max(c["box_f0_hz"], f0))
        assert row.f_high_hz == pytest.approx(min(c["box_f1_hz"], f1))
        assert row.t_center_s == pytest.approx(c["t_center_s"])
        assert row.emitter_hz == pytest.approx(c["baseline_hz"])
        assert row.emitter_id == c["emitter_id"]
        assert row.family == c["family"]
        assert row.snr_db == pytest.approx(c["snr_db"])
        assert row.template_id == c["template_id"]


def test_live_tracks_hold_every_fish_and_the_emitter(dataset: Dataset) -> None:
    for source in ("test-000003", "test-000005"):
        n_fish = 1 if source == "test-000005" else 3
        rows = _rows(dataset, source)
        chirps = rows.loc[(rows["kind"] == "chirp").to_numpy()]
        assert len(chirps)
        for row in chirps.itertuples():
            live = np.asarray(row.live_tracks_hz)
            assert live.size == n_fish
            assert np.min(np.abs(live - row.emitter_hz)) < 0.5


def test_boxes_land_on_the_bright_excursion(dataset: Dataset) -> None:
    """Clean single fish: the excursion is bright inside its box, dark outside."""
    source = "test-000005"
    chirps = _scene_chirps(dataset, source).set_index("chirp_id")
    rows = _rows(dataset, source)
    positives = rows.loc[(rows["kind"] == "chirp").to_numpy()]
    assert len(positives) >= 5
    checked = 0
    for row in positives.itertuples():
        record = dataset.manifest.image(str(row.image_id))
        image = read_image(dataset.image_path(record.id))
        c = chirps.loc[int(row.chirp_id)]
        g = record.geometry
        # the excursion's peak lies inside the box
        x, y = timefreq_to_pixel(
            c["t_peak_s"], c["baseline_hz"] + 0.6 * c["excursion_hz"], g
        )
        x0, y_top = timefreq_to_pixel(row.t_start_s, row.f_high_hz, g)
        x1, y_bottom = timefreq_to_pixel(row.t_end_s, row.f_low_hz, g)
        if not (0 <= x < g.width_px):
            continue
        assert x0 <= x <= x1
        assert y_top <= y <= y_bottom
        # rows of the excursion (above the baseline band) in the reference
        # window: bright in the box
        _, y_base = timefreq_to_pixel(0.0, c["baseline_hz"] + 30.0, g)
        top, bottom = int(np.ceil(y_top)), int(np.floor(y_base))
        if bottom - top < 2:
            continue
        inside = image[top:bottom, int(np.ceil(x0)) : int(np.floor(x1)), 0]
        assert inside.max() >= 64
        # same rows outside every box of this image are dark (frames whose
        # window reaches past the scene's edges show the onset's leakage)
        times = g.t_first_s + np.arange(g.width_px) * g.dt_s
        mask = (times > 0.11) & (times < 8.0 - 0.11)
        for other in rows.loc[(rows["image_id"] == record.id).to_numpy()].itertuples():
            a, _ = timefreq_to_pixel(other.t_start_s, 0.0, g)
            b, _ = timefreq_to_pixel(other.t_end_s, 0.0, g)
            mask[max(0, int(a) - 2) : int(np.ceil(b)) + 2] = False
        if mask.sum() > 20:
            outside = image[top:bottom, mask, 0]
            assert outside.max() < inside.max() / 2
            checked += 1
    assert checked >= 3


def test_scene_boxes_without_chirps() -> None:
    columns = ["chirp_id", "t_center_s", "baseline_hz", "emitter_id", "visible"]
    empty = pd.DataFrame(columns=columns)
    boxes = scene_boxes(empty, [])
    assert len(boxes) == 0
    assert "live_tracks_hz" in boxes.columns


def test_other_front_ends_recompute_the_footprints(
    scene_dirs: dict[str, Path],
) -> None:
    spectrogram = dataclasses.replace(SpectrogramConfig(), f_max_hz=1200.0)
    config = dataclasses.replace(CONFIG, spectrogram=spectrogram)
    tiles = tile_scene(scene_dirs["clean"], config)
    assert tiles.stats["footprints"] == "recomputed"
    assert tiles.images[0][1].shape == (spectrogram.n_freqs, 640, 3)
    positives = pd.concat(tiles.chirps, ignore_index=True)
    assert len(positives)
    assert bool((positives["f_high_hz"] <= 1200.0 + 2.5).all())
    default = tile_scene(scene_dirs["clean"], CONFIG)
    assert default.stats["footprints"] == "scene"
    same = pd.concat(default.chirps, ignore_index=True)
    # same chirps; boxes equal up to the frame grid (one hop) below 1200 Hz
    assert sorted(same["chirp_id"]) == sorted(positives["chirp_id"])
    merged = same.merge(positives, on=["chirp_id", "t_center_s"], suffixes=("", "_r"))
    np.testing.assert_allclose(merged["t_start_s"], merged["t_start_s_r"], atol=0.0051)
    np.testing.assert_allclose(merged["f_low_hz"], merged["f_low_hz_r"], atol=1e-6)


def test_tracks_are_ground_truth_baselines(scene_dirs: dict[str, Path]) -> None:
    tracks = load_tracks(scene_dirs["grid9"])
    assert len(list(tracks)) == 3


@pytest.mark.slow
def test_worker_processes_give_the_same_dataset(
    scene_dirs: dict[str, Path], dataset: Dataset, tmp_path: Path
) -> None:
    config = dataclasses.replace(CONFIG, workers=2, threads_per_worker=1)
    parallel = build_scene_dataset(
        list(scene_dirs.values()), tmp_path / "parallel", config, diagnostics=False
    )
    assert parallel.manifest.content_sha256 == dataset.manifest.content_sha256
    assert load_dataset(tmp_path / "parallel").manifest.images == (
        dataset.manifest.images
    )
