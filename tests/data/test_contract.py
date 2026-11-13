"""The detector dataset contract: writer, validator, geometry, ground truth."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from chirpdetector.data import (
    DatasetError,
    DatasetWriter,
    Manifest,
    Tiling,
    dataset_ground_truth,
    load_dataset,
    read_image,
    validate_dataset,
    write_image,
)
from chirpdetector.data.contract import (
    labels_to_pixel_boxes,
    pixel_boxes_to_timefreq,
    read_yolo_labels,
)
from chirpdetector.data.synthetic import SMALL_SPECTROGRAM, make_synthetic_dataset
from chirpdetector.evaluation import EvaluationConfig, evaluate
from chirpdetector.evaluation.schema import floats, select, strings
from chirpdetector.spectrogram import ImageGeometry, SpectrogramConfig


def _problems(directory: Path) -> str:
    with pytest.raises(DatasetError) as info:
        validate_dataset(directory)
    return str(info.value)


# ------------------------------------------------------------------ valid data


def test_synthetic_dataset_is_valid_and_complete(synthetic_dir: Path) -> None:
    dataset = load_dataset(synthetic_dir)
    manifest = dataset.manifest
    assert manifest.classes == ("chirp",)
    assert len(manifest.images) == 12
    assert manifest.groups("train") == tuple(f"scene-00{k}" for k in range(4))
    assert manifest.groups("val") == ("scene-004",)
    assert manifest.groups("test") == ("scene-005",)
    kinds = np.asarray(strings(dataset.boxes["kind"]))
    chirps = select(dataset.boxes, kinds == "chirp")
    assert (floats(chirps["emitter_hz"]) > 0).all()
    assert (kinds == "ignore").sum() == 12
    assert "snr_db" in dataset.boxes.columns  # extra columns survive


def test_manifest_round_trip_and_stable_hash(
    synthetic_dir: Path, tmp_path: Path
) -> None:
    manifest = Manifest.read(synthetic_dir)
    again = Manifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
    assert again == manifest
    assert again.content_sha256 == manifest.content_sha256
    rebuilt = make_synthetic_dataset(tmp_path / "again", seed=3)
    assert rebuilt.manifest.content_sha256 == manifest.content_sha256


def test_split_lookups_never_mix_groups(synthetic_dir: Path) -> None:
    manifest = Manifest.read(synthetic_dir)
    for split in ("train", "val", "test"):
        groups = set(manifest.groups(split))
        for image_id in manifest.image_ids(split):
            assert manifest.image(image_id).group in groups


def test_png_channels_are_window_order(tmp_path: Path) -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[..., 0], image[..., 1], image[..., 2] = 10, 20, 30
    path = tmp_path / "x.png"
    write_image(path, image)
    np.testing.assert_array_equal(read_image(path), image)
    rgb = np.asarray(Image.open(path).convert("RGB"))
    assert tuple(rgb[0, 0]) == (10, 20, 30)  # red = reference window


def test_labels_agree_with_boxes(synthetic_dir: Path) -> None:
    dataset = load_dataset(synthetic_dir)
    for record in dataset.manifest.images:
        labels = read_yolo_labels(dataset.label_path(record.id))
        assert (labels[:, 0] == 0).all()
        g = record.geometry
        timefreq = pixel_boxes_to_timefreq(
            labels_to_pixel_boxes(labels, g.width_px, g.height_px), g
        )
        expected = dataset.boxes_of(record.id, "chirp")[
            ["t_start_s", "t_end_s", "f_low_hz", "f_high_hz"]
        ].to_numpy()
        np.testing.assert_allclose(timefreq, expected, rtol=0, atol=1e-9)


def test_ground_truth_scores_perfect_predictions(synthetic_dir: Path) -> None:
    dataset = load_dataset(synthetic_dir)
    truth = dataset_ground_truth(dataset)
    assert set(truth.labels["label"]) == {"chirp", "ignore"}
    assert "snr_db" in truth.labels.columns  # box extras, for strata
    kinds = np.asarray(strings(dataset.boxes["kind"]))
    chirps = select(dataset.boxes, kinds == "chirp")
    predictions = chirps.rename(columns={"image_id": "recording"})[
        ["recording", "t_start_s", "t_end_s", "f_low_hz", "f_high_hz"]
    ].assign(score=0.9)
    result = evaluate(predictions, truth, config=EvaluationConfig(n_bootstrap=0))
    pooled = result.metrics()["detection"]["all"]
    assert pooled["precision"] == 1.0
    assert pooled["recall"] == 1.0
    assert pooled["labels"] == len(chirps)


def test_ground_truth_centres_are_box_midpoints(synthetic_dir: Path) -> None:
    """Labels are matched at their box midpoint (as predictions are); the
    chirp's own centre time stays available as ``chirp_t_center_s``."""
    dataset = load_dataset(synthetic_dir)
    labels = dataset_ground_truth(dataset).labels
    mid = 0.5 * (labels["t_start_s"] + labels["t_end_s"])
    np.testing.assert_allclose(labels["t_center_s"], mid, rtol=0, atol=1e-12)
    boxes = dataset.boxes
    np.testing.assert_allclose(
        np.sort(labels["chirp_t_center_s"].to_numpy(dtype=np.float64)),
        np.sort(boxes["t_center_s"].to_numpy(dtype=np.float64)),
        rtol=0,
        atol=1e-12,
    )


# --------------------------------------------------------------- the validator


def test_a_split_per_image_is_refused(broken: Callable[..., Path]) -> None:
    def per_image_split(data: dict[str, Any]) -> None:
        data["images"][0]["split"] = "val"

    assert "unknown keys ['split']" in _problems(broken(manifest=per_image_split))


def test_groups_need_a_valid_split(broken: Callable[..., Path]) -> None:
    def bad_split(data: dict[str, Any]) -> None:
        data["groups"]["scene-004"] = "validation"

    assert "'validation' is not one of" in _problems(broken(manifest=bad_split))

    def missing_group(data: dict[str, Any]) -> None:
        del data["groups"]["scene-004"]

    assert "group 'scene-004' has no split" in _problems(broken(manifest=missing_group))


def test_image_group_must_be_its_sources(broken: Callable[..., Path]) -> None:
    def regroup(data: dict[str, Any]) -> None:
        data["images"][0]["group"] = "scene-005"

    assert "differs from its source's group" in _problems(broken(manifest=regroup))


def test_identical_images_in_two_splits_are_refused(
    broken: Callable[..., Path],
) -> None:
    def leak(data: dict[str, Any]) -> None:
        test_image = next(r for r in data["images"] if r["group"] == "scene-005")
        test_image["image_sha256"] = data["images"][0]["image_sha256"]

    assert "identical but in different splits" in _problems(broken(manifest=leak))


@pytest.mark.parametrize(
    "classes", [["chirp", "background"], ["background", "chirp"], ["item"]]
)
def test_class_map_is_only_chirp(
    broken: Callable[..., Path], classes: list[str]
) -> None:
    def remap(data: dict[str, Any]) -> None:
        data["classes"] = classes

    assert "classes must be ['chirp']" in _problems(broken(manifest=remap))


def test_class_one_labels_are_refused(broken: Callable[..., Path]) -> None:
    directory = broken()
    label = next((directory / "labels").glob("*.txt"))
    lines = label.read_text().splitlines()
    label.write_text("".join("1" + line[1:] + "\n" for line in lines))
    problems = _problems(directory)
    assert "class indices [1.0]" in problems
    assert "single_cls" in problems


def test_ignore_regions_must_lie_inside_their_image(
    broken: Callable[..., Path],
) -> None:
    def shift(table: pd.DataFrame) -> pd.DataFrame:
        ignore = int(np.flatnonzero(table["kind"].to_numpy() == "ignore")[0])
        table.loc[ignore, ["t_start_s", "t_end_s"]] += 100.0
        table.loc[ignore, "t_center_s"] += 100.0
        return table

    assert "ignore box(es) outside the image extent" in _problems(broken(boxes=shift))


def test_chirps_need_an_emitter_frequency(broken: Callable[..., Path]) -> None:
    def drop(table: pd.DataFrame) -> pd.DataFrame:
        table.loc[table["kind"] == "chirp", "emitter_hz"] = np.nan
        return table

    assert "chirps without a positive, finite emitter_hz" in _problems(
        broken(boxes=drop)
    )


def test_chirps_need_live_tracks_or_source_tracks(
    broken: Callable[..., Path],
) -> None:
    def drop(table: pd.DataFrame) -> pd.DataFrame:
        table["live_tracks_hz"] = None
        return table

    assert "neither live_tracks_hz nor tracks" in _problems(broken(boxes=drop))

    def with_tracks(data: dict[str, Any]) -> None:
        for source in data["sources"].values():
            source["tracks"] = "/somewhere/tracks"

    validate_dataset(broken(boxes=drop, manifest=with_tracks))


def test_edits_without_rehashing_are_detected(
    synthetic_dir: Path, tmp_path: Path
) -> None:
    directory = tmp_path / "edited"
    shutil.copytree(synthetic_dir, directory)
    data = json.loads((directory / "dataset.json").read_text())
    data["images"][0]["extra"] = {"note": "edited"}
    (directory / "dataset.json").write_text(json.dumps(data))
    assert "content_sha256 does not match" in _problems(directory)

    directory = tmp_path / "relabelled"
    shutil.copytree(synthetic_dir, directory)
    label = next((directory / "labels").glob("*.txt"))
    label.write_text(label.read_text() + "0 0.5 0.5 0.1 0.1\n")
    problems = _problems(directory)
    assert "sha256 differs from the manifest" in problems
    assert "lines, the manifest says" in problems


def test_stray_npy_caches_are_refused(broken: Callable[..., Path]) -> None:
    directory = broken()
    np.save(directory / "images" / "scene-000_t000.npy", np.zeros(3))
    assert ".npy files" in _problems(directory)


def test_tiling_must_match_the_front_end(broken: Callable[..., Path]) -> None:
    def taller(data: dict[str, Any]) -> None:
        data["tiling"]["height_px"] += 1

    assert "tiling.height_px" in _problems(broken(manifest=taller))


# ------------------------------------------------------------------ the writer


def _writer(directory: Path, width: int = 64) -> tuple[DatasetWriter, ImageGeometry]:
    config = SMALL_SPECTROGRAM
    layout = config.layout(20_000.0)
    tiling = Tiling(
        tile_duration_s=width * config.hop_s,
        overlap_s=0.0,
        width_px=width,
        height_px=config.n_freqs,
    )
    writer = DatasetWriter(directory, spectrogram=config, tiling=tiling)
    geometry = ImageGeometry(
        width_px=width,
        height_px=config.n_freqs,
        t_first_s=10.0,
        dt_s=layout.hop_s,
        f_lowest_hz=float(layout.freqs_hz[0]),
        df_hz=layout.df_hz,
    )
    writer.add_source("rec", group="rec", kind="real", tracks="tracks")
    writer.assign_split("rec", "test")
    return writer, geometry


def test_writer_refuses_boxes_outside_the_image(tmp_path: Path) -> None:
    writer, geometry = _writer(tmp_path / "ds")
    image = np.zeros((geometry.height_px, geometry.width_px, 3), np.uint8)
    t0, t1 = geometry.time_extent_s
    f0, f1 = geometry.freq_extent_hz
    box = {"f_low_hz": f0, "f_high_hz": f0 + 50, "emitter_hz": f0 + 1}
    late = pd.DataFrame([box | {"t_start_s": t1 - 0.05, "t_end_s": t1 + 0.05}])
    with pytest.raises(ValueError, match="extend beyond the image"):
        writer.add_image("a", image, geometry, source="rec", chirps=late)
    # rounding-level overshoot is clipped
    exact = pd.DataFrame([box | {"t_start_s": t0 - 1e-9, "t_end_s": t0 + 0.05}])
    writer.add_image("b", image, geometry, source="rec", chirps=exact)
    dataset = writer.close()
    assert dataset.boxes_of("b")["t_start_s"].iloc[0] == t0
    assert f1 > f0


def test_writer_refuses_inconsistent_input(tmp_path: Path) -> None:
    writer, geometry = _writer(tmp_path / "ds")
    image = np.zeros((geometry.height_px, geometry.width_px, 3), np.uint8)
    with pytest.raises(ValueError, match="unknown source"):
        writer.add_image("a", image, geometry, source="nope")
    with pytest.raises(ValueError, match="shape"):
        writer.add_image("a", image[:, :-1], geometry, source="rec")
    with pytest.raises(ValueError, match="must match"):
        writer.add_image("../a", image, geometry, source="rec")
    with pytest.raises(ValueError, match="already in split"):
        writer.assign_split("rec", "train")
    chirp = pd.DataFrame(
        [{"t_start_s": 10.1, "t_end_s": 10.12, "f_low_hz": 600.0, "f_high_hz": 650.0}]
    )
    with pytest.raises(ValueError, match="emitter_hz"):
        writer.add_image("a", image, geometry, source="rec", chirps=chirp)
    writer.add_image("a", image, geometry, source="rec")
    with pytest.raises(ValueError, match="written twice"):
        writer.add_image("a", image, geometry, source="rec")
    dataset = writer.close()
    assert dataset.manifest.image("a").n_chirps == 0


def test_tiling_checks_the_channel_layout() -> None:
    one_window = SpectrogramConfig(extra_windows_s=(), f_min_hz=500, f_max_hz=810)
    tiling = Tiling(
        tile_duration_s=0.64, overlap_s=0.0, width_px=128, height_px=one_window.n_freqs
    )
    assert any("channel_fill" in p for p in tiling.problems(one_window))
    filled = Tiling(
        tile_duration_s=0.64,
        overlap_s=0.0,
        width_px=128,
        height_px=one_window.n_freqs,
        channel_fill="zeros",
    )
    assert filled.problems(one_window) == []
