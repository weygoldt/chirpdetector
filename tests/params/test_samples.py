"""Sample sets: scene discovery and splits, box matching, extraction, scene truth."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chirpdetector.bundle import Bundle
from chirpdetector.params.samples import (
    SAMPLE_COLUMNS,
    BoxSource,
    detector_overlap,
    discover_scenes,
    extract_samples,
    load_samples,
    match_predictions,
    parse_split,
)
from chirpdetector.params.truth import scene_truth

from .conftest import footprint_detections


def _preds(rows: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["t_start_s", "t_end_s", "f_low_hz", "f_high_hz", "score"]
    )


def test_matching_is_one_to_one_by_time_and_frequency() -> None:
    chirps = pd.DataFrame(
        {
            "t_center_s": [1.0, 1.05, 3.0, 5.0],
            "f_low_hz": [700.0, 900.0, 700.0, 700.0],
            "f_high_hz": [800.0, 1000.0, 800.0, 800.0],
        }
    )
    preds = _preds(
        [
            (0.95, 1.07, 690.0, 790.0, 0.9),  # chirp 0
            (1.0, 1.1, 905.0, 990.0, 0.8),  # chirp 1 (other band)
            (2.9, 3.1, 710.0, 790.0, 0.5),  # chirp 2 ...
            (2.95, 3.05, 705.0, 795.0, 0.95),  # ... closer and higher score
            (4.75, 4.85, 700.0, 800.0, 0.9),  # 200 ms off: no match
            (5.0, 5.1, 820.0, 900.0, 0.9),  # no frequency overlap
        ]
    )
    index, dt = match_predictions(preds, chirps, 0.1)
    np.testing.assert_array_equal(index, [0, 1, -1, 2, -1, -1])
    assert dt[0] == pytest.approx(0.01)
    assert np.isnan(dt[2])
    empty, _ = match_predictions(preds, chirps.iloc[:0])
    assert (empty == -1).all()


def test_scenes_are_split_by_directory_and_filtered_by_background(
    params_scenes: Path, tmp_path: Path
) -> None:
    refs = discover_scenes([params_scenes])
    assert {(r.split, r.seed) for r in refs} == {
        ("train", 0),
        ("train", 1),
        ("train", 2),
        ("val", 1600),
        ("test", 1800),
        ("test", 1801),
    }
    assert {r.kind for r in refs} == {"simulation"}
    assert discover_scenes([params_scenes], background="hybrid") == []
    assert len(discover_scenes([params_scenes], background="synthetic")) == len(refs)
    with pytest.raises(ValueError, match="background"):
        discover_scenes([params_scenes], background="real")
    # a scene in a 'val' directory whose seed (0) says 'train' under the old
    # seed ranges: by default the directory decides, a seed rule must agree
    moved = tmp_path / "val" / "000000"
    shutil.copytree(params_scenes / "train" / "000000", moved)
    assert [r.split for r in discover_scenes([tmp_path])] == ["val"]
    with pytest.raises(ValueError, match="'val' directory"):
        discover_scenes([tmp_path], parse_split("train=0:1600,val=1600:1800"))
    # scenes outside split directories need a seed rule
    loose = tmp_path / "loose"
    shutil.copytree(params_scenes / "test" / "001800", loose / "001800")
    with pytest.raises(ValueError, match="no train/val/test directory"):
        discover_scenes([loose])
    rule = parse_split("train=0:1600,val=1600:1800,test=1800:2000")
    assert [r.split for r in discover_scenes([loose], rule)] == ["test"]
    assert parse_split("directory") is None


def test_seeds_outside_the_old_ranges_are_split_by_directory(
    params_scenes: Path, tmp_path: Path
) -> None:
    """train_v3 uses seeds 2000-4999, outside the detector's old 0-1999 ranges."""
    for split, old, new in (
        ("train", 0, 2000),
        ("val", 1600, 4400),
        ("test", 1800, 4700),
    ):
        target = tmp_path / split / f"{new:06d}"
        shutil.copytree(params_scenes / split / f"{old:06d}", target)
        info = json.loads((target / "scene.json").read_text())
        info["seed"] = new
        (target / "scene.json").write_text(json.dumps(info))
    refs = discover_scenes([tmp_path])
    assert {(r.split, r.seed) for r in refs} == {
        ("train", 2000),
        ("val", 4400),
        ("test", 4700),
    }
    with pytest.raises(ValueError, match="no split range"):
        discover_scenes(
            [tmp_path], parse_split("train=0:1600,val=1600:1800,test=1800:2000")
        )


def test_scene_truth_recovers_the_templates(clean_scene: Path) -> None:
    """A noise-free fish at 800 Hz: per-period truth = the Gaussian templates."""
    chirps = pd.read_csv(clean_scene / "chirps.csv")
    info = json.loads((clean_scene / "scene.json").read_text())
    library = pd.read_parquet(Path(info["templates"]["path"]) / "templates.parquet")
    truth = scene_truth(clean_scene, chirps).add_prefix("true_")
    table = chirps.join(truth).merge(
        library[["template_id", "duration_50_s"]], on="template_id"
    )
    t = table.loc[~table["truncated"].to_numpy(bool)]
    assert len(t) >= 3
    np.testing.assert_array_equal(t["true_chirp_id"], t["chirp_id"])
    # period averaging lowers the peak by ~1 % for the 8-ms templates
    np.testing.assert_allclose(t["true_height_hz"], t["excursion_hz"], rtol=0.03)
    np.testing.assert_allclose(t["true_fwhm_s"], t["duration_50_s"], rtol=0.05)
    assert (t["true_undershoot_hz"] < 0.05 * t["true_height_hz"]).all()


def test_footprint_extraction(params_scenes: Path, tmp_path: Path) -> None:
    refs = discover_scenes([params_scenes])
    boxes = BoxSource(mode="footprint", jitter_t_s=0.01, jitter_f_hz=5.0)
    samples = extract_samples(
        refs, tmp_path / "s", boxes=boxes, workers=0, device="cpu"
    )
    table = samples.table
    assert set(SAMPLE_COLUMNS) <= set(table.columns)
    assert samples.crops.shape == (len(table), 3, 96, 96)
    assert table["matched"].to_numpy(bool).all()
    assert set(table["split"]) == {"train", "val", "test"}
    assert table["usable"].to_numpy(bool).mean() > 0.8
    # jitter only in train: other splits keep the footprint boxes exactly
    test = table[table["split"] == "test"]
    np.testing.assert_allclose(test["t_start_s"], test["box_t0_s"])
    train = table[table["split"] == "train"]
    assert not np.allclose(train["t_start_s"], train["box_t0_s"])
    # reused when asked again with the same settings, refused with others
    again = extract_samples(refs, tmp_path / "s", boxes=boxes, workers=0, device="cpu")
    assert again.manifest["files"] == samples.manifest["files"]
    with pytest.raises(ValueError, match="other"):
        extract_samples(
            refs,
            tmp_path / "s",
            boxes=BoxSource(mode="footprint", jitter_t_s=0.02),
            workers=0,
        )
    loaded = load_samples(tmp_path / "s")
    np.testing.assert_array_equal(loaded.crops[:3], samples.crops[:3])


def test_detector_extraction_matches_predictions(
    params_scenes: Path,
    stub_detector: Bundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import chirpdetector.inference as inference

    refs = discover_scenes([params_scenes])
    monkeypatch.setattr(
        inference,
        "detect_in_spectrogram",
        footprint_detections({r.source_id: r.directory for r in refs}),
    )
    samples = extract_samples(
        refs,
        tmp_path / "d",
        boxes=BoxSource(mode="detector"),
        detector=stub_detector,
        workers=1,
        device="cpu",
    )
    table = samples.table
    assert (
        samples.manifest["detector"]["weights_sha256"] == stub_detector.weights_sha256
    )
    # the bogus box per scene is kept (unmatched) in val and test only
    is_matched = table["matched"].to_numpy(bool)
    unmatched = table.loc[~is_matched]
    assert set(unmatched["split"]) == {"val", "test"}
    assert (unmatched["chirp_id"].to_numpy() == -1).all()
    assert np.isnan(unmatched["true_height_hz"].to_numpy(float)).all()
    assert not unmatched["visible"].to_numpy(bool).any()
    matched = table.loc[is_matched]
    assert (np.abs(matched["dt_match_s"].to_numpy(float)) <= 0.1).all()
    # every matched box is its chirp's footprint
    np.testing.assert_allclose(matched["t_start_s"], matched["box_t0_s"])
    assert samples.manifest["detector_overlap"] == {"train": 0, "val": 0, "test": 0}


def test_scenes_the_detector_saw_are_not_val_or_test(
    params_scenes: Path, stub_detector: Bundle, tmp_path: Path
) -> None:
    """A decoder must not be calibrated or scored on the detector's own scenes."""
    from dataclasses import replace

    refs = discover_scenes([params_scenes])
    test_id = next(r.source_id for r in refs if r.split == "test")
    train_id = next(r.source_id for r in refs if r.split == "train")
    seen = replace(
        stub_detector,
        training={"run_id": "stub", "train_groups": [test_id], "val_groups": []},
    )
    assert detector_overlap(refs, seen)["test"] == [test_id]
    with pytest.raises(ValueError, match="val/test scenes of the decoder"):
        extract_samples(refs, tmp_path / "x", detector=seen, workers=0, device="cpu")
    assert not (tmp_path / "x").exists()
    # the detector's training scenes may be the decoder's training scenes
    fine = replace(
        stub_detector,
        training={"run_id": "stub", "train_groups": [], "val_groups": [train_id]},
    )
    assert detector_overlap(refs, fine) == {"train": [train_id], "val": [], "test": []}


def test_natural_prior_weights(params_scenes: Path) -> None:
    from chirpdetector.params.evaluation import natural_prior_weights

    scenes = [r.directory for r in discover_scenes([params_scenes])]
    ids = np.concatenate(
        [pd.read_csv(d / "chirps.csv")["template_id"].to_numpy() for d in scenes]
    )
    for level in ("bin", "template"):
        weights, info = natural_prior_weights(ids, scenes, level=level)
        assert weights.shape == ids.shape
        assert weights.mean() == pytest.approx(1.0)
        assert (weights > 0).all()
        assert info["pool_size"] == 12
        assert 0 < info["ess"] <= ids.size
    with pytest.raises(ValueError, match="level"):
        natural_prior_weights(ids, scenes, level="individual")


def test_a_chirp_truth_ignores_the_emitters_other_chirps(clean_scene: Path) -> None:
    """A window reaching into the next chirp of the same fish must not change
    the truth: the peak and undershoot of a neighbour are not this chirp's."""
    from gridtools.sim.scene import load_scene_model

    from chirpdetector.params.truth import chirp_truth

    chirps = pd.read_csv(clean_scene / "chirps.csv").sort_values("t_onset_s")
    chirps = chirps.loc[~chirps["truncated"].to_numpy(bool)]
    model = load_scene_model(clean_scene)
    pairs = [
        (a, b)
        for (_, a), (_, b) in zip(
            chirps.iloc[:-1].iterrows(), chirps.iloc[1:].iterrows(), strict=True
        )
        if b["excursion_hz"] > 1.2 * a["excursion_hz"]
    ]
    assert pairs, "the clean scene needs a chirp followed by a larger one"
    a, b = pairs[0]
    support = (float(a["t_support_start_s"]), float(a["t_support_stop_s"]))
    reach = float(b["t_support_stop_s"]) - support[1] + 0.01  # covers chirp b
    own = chirp_truth(model, int(a["emitter_id"]), support, chirp_id=int(a["chirp_id"]))
    wide = chirp_truth(
        model, int(a["emitter_id"]), support, pad_s=reach, chirp_id=int(a["chirp_id"])
    )
    for name in ("height_hz", "fwhm_s", "width10_s", "undershoot_hz"):
        assert getattr(wide, name) == pytest.approx(getattr(own, name), abs=1e-6)
    # with every chirp in the phase, the window would report chirp b's peak
    mixed = chirp_truth(model, int(a["emitter_id"]), support, pad_s=reach)
    assert mixed.height_hz > 1.1 * own.height_hz
    with pytest.raises(ValueError, match="not a chirp of fish"):
        chirp_truth(model, int(a["emitter_id"]), support, chirp_id=10_000)
