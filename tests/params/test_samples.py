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
    discover_scenes,
    extract_samples,
    load_samples,
    match_predictions,
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


def test_scenes_are_split_by_seed_and_filtered_by_background(
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
    # a scene in a 'val' directory whose seed says 'train'
    wrong = tmp_path / "val" / "000000"
    shutil.copytree(params_scenes / "train" / "000000", wrong)
    with pytest.raises(ValueError, match="'val' directory"):
        discover_scenes([tmp_path])


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
