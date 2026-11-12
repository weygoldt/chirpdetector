"""Bundles (hash verification, metrics guard) and pixel -> time/frequency."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from chirpdetector.bundle import BundleError, load_bundle, write_bundle
from chirpdetector.cli import app
from chirpdetector.data import Dataset, dataset_ground_truth, read_image
from chirpdetector.data.contract import labels_to_pixel_boxes, read_yolo_labels
from chirpdetector.evaluation import (
    EvaluationConfig,
    assign_by_frequency,
    attach_to_bundle,
    evaluate,
)
from chirpdetector.evaluation.schema import floats, select, strings
from chirpdetector.prediction import predict_dataset, predict_images
from chirpdetector.spectrogram import ImageGeometry, pixel_box_to_timefreq

from .conftest import STUB, StubBackend


def _bundle(directory: Path, dataset: Dataset, weights: bytes = b"weights") -> Path:
    source = directory.parent / f"{directory.name}.bin"
    source.write_bytes(weights)
    write_bundle(
        directory,
        weights=source,
        model_type=STUB,
        model={"kind": "stub"},
        spectrogram=dataset.manifest.spectrogram,
        tiling=dataset.manifest.tiling,
        preprocessing=dataset.manifest.preprocessing,
        score_threshold=0.3,
        training={"run_id": "r", "fit": "final", "dataset_sha256": "d"},
    )
    return directory


# ------------------------------------------------------------------- bundle


def test_bundle_round_trip(tmp_path: Path, dataset: Dataset) -> None:
    bundle = load_bundle(_bundle(tmp_path / "b", dataset))
    assert bundle.model_type == STUB
    assert bundle.classes == ("chirp",)
    assert bundle.spectrogram == dataset.manifest.spectrogram
    assert bundle.tiling == dataset.manifest.tiling
    assert bundle.score_threshold == 0.3
    assert bundle.metrics is None
    record = json.loads((bundle.path / "bundle.json").read_text())
    assert record["weights_sha256"] == bundle.weights_sha256


def test_tampered_weights_are_refused(tmp_path: Path, dataset: Dataset) -> None:
    directory = _bundle(tmp_path / "b", dataset)
    (directory / "weights.pt").write_bytes(b"other weights")
    with pytest.raises(BundleError, match="replaced after the bundle was made"):
        load_bundle(directory)


def test_bundle_must_be_chirp_only(tmp_path: Path, dataset: Dataset) -> None:
    directory = _bundle(tmp_path / "b", dataset)
    record = json.loads((directory / "bundle.json").read_text())
    record["classes"] = ["background", "chirp"]
    (directory / "bundle.json").write_text(json.dumps(record))
    with pytest.raises(BundleError, match="classes"):
        load_bundle(directory)


def _evaluation(dataset: Dataset, model_sha256: str | None):
    chirps = select(dataset.boxes, dataset.boxes["kind"].to_numpy() == "chirp")
    predictions = pd.DataFrame(
        {
            "recording": strings(chirps["image_id"]),
            **{
                c: floats(chirps[c])
                for c in ("t_start_s", "t_end_s", "f_low_hz", "f_high_hz")
            },
            "score": 0.9,
        }
    )
    if model_sha256 is not None:
        predictions["model_sha256"] = model_sha256
    return evaluate(
        predictions,
        dataset_ground_truth(dataset),
        config=EvaluationConfig(n_bootstrap=0),
    )


def test_metrics_attach_only_to_their_own_weights(
    tmp_path: Path, dataset: Dataset
) -> None:
    directory = _bundle(tmp_path / "b", dataset)
    bundle = load_bundle(directory)
    with pytest.raises(ValueError, match="cannot be attached"):
        attach_to_bundle(_evaluation(dataset, None), directory, bundle.weights_path)
    with pytest.raises(ValueError, match="cannot be attached"):
        attach_to_bundle(_evaluation(dataset, "0" * 64), directory, bundle.weights_path)
    path = attach_to_bundle(
        _evaluation(dataset, bundle.weights_sha256), directory, bundle.weights_path
    )
    assert json.loads(path.read_text())["weights_sha256"] == bundle.weights_sha256
    assert load_bundle(directory).metrics is not None


def test_metrics_of_other_weights_are_refused_on_load(
    tmp_path: Path, dataset: Dataset
) -> None:
    directory = _bundle(tmp_path / "b", dataset)
    (directory / "metrics.json").write_text(json.dumps({"weights_sha256": "abc"}))
    with pytest.raises(BundleError, match="computed for weights"):
        load_bundle(directory)


def test_bundle_inspect_cli(tmp_path: Path, dataset: Dataset) -> None:
    directory = _bundle(tmp_path / "b", dataset)
    result = CliRunner().invoke(app, ["bundle", "inspect", str(directory)])
    assert result.exit_code == 0, result.output
    assert "(verified)" in result.output
    assert "metrics     none attached" in result.output
    (directory / "weights.pt").write_bytes(b"x")
    result = CliRunner().invoke(app, ["bundle", "inspect", str(directory)])
    assert result.exit_code == 1


# --------------------------------------------------------------- prediction


def test_pixel_boxes_map_exactly_to_time_and_frequency(
    tmp_path: Path, dataset: Dataset, stub_backend: StubBackend
) -> None:
    pixel = pd.DataFrame(
        {
            "x_min": [0.0, 10.25, 100.0],
            "y_min": [0.0, 5.5, 40.0],
            "x_max": [3.0, 30.75, 128.0],
            "y_max": [2.0, 20.0, 63.0],
            "score": [0.9, 0.5, 0.1],
            "emitter_y": [2.0, 20.0, 63.0],
        }
    )
    stub_backend.output = pixel
    bundle = load_bundle(_bundle(tmp_path / "b", dataset))
    geometry = ImageGeometry(
        width_px=bundle.tiling.width_px,
        height_px=bundle.tiling.height_px,
        t_first_s=123.4,
        dt_s=bundle.spectrogram.hop_s,
        f_lowest_hz=503.5,
        df_hz=bundle.spectrogram.df_hz,
    )
    image = np.zeros((geometry.height_px, geometry.width_px, 3), np.uint8)
    out = predict_images(bundle, [image, image], [geometry, geometry])
    kept = pixel.iloc[:2]  # the bundle's threshold is 0.3
    assert len(out) == 4
    t0, t1, f0, f1 = pixel_box_to_timefreq(
        kept["x_min"], kept["y_min"], kept["x_max"], kept["y_max"], geometry
    )
    first = out[out["image_index"] == 0]
    np.testing.assert_array_equal(first["t_start_s"], t0)
    np.testing.assert_array_equal(first["t_end_s"], t1)
    np.testing.assert_array_equal(first["f_low_hz"], f0)
    np.testing.assert_array_equal(first["f_high_hz"], f1)
    # the emitter's row is mapped like a box edge: row y -> frequency
    np.testing.assert_allclose(first["emitter_hz"], f0)
    assert set(out["model_sha256"]) == {bundle.weights_sha256}
    assert set(out["recording"]) == {"image-0", "image-1"}
    assert len(predict_images(bundle, [image], [geometry], score_threshold=0.0)) == 3


def test_predictions_of_the_true_pixels_score_perfectly(
    tmp_path: Path, dataset: Dataset, stub_backend: StubBackend
) -> None:
    """Labels -> pixels -> predict_images -> evaluation recovers every chirp."""
    record = dataset.manifest.images[0]
    g = record.geometry
    boxes = labels_to_pixel_boxes(
        read_yolo_labels(dataset.label_path(record.id)), g.width_px, g.height_px
    )
    stub_backend.output = pd.DataFrame(
        boxes, columns=["x_min", "y_min", "x_max", "y_max"]
    ).assign(score=0.8)
    bundle = load_bundle(_bundle(tmp_path / "b", dataset))
    out = predict_dataset(bundle, dataset, [record.id])
    expected = dataset.boxes_of(record.id, "chirp")
    for column in ("t_start_s", "t_end_s", "f_low_hz", "f_high_hz"):
        np.testing.assert_allclose(out[column], expected[column], rtol=0, atol=1e-9)
    result = evaluate(
        out,
        dataset_ground_truth(dataset, [record.id]),
        config=EvaluationConfig(n_bootstrap=0),
    )
    pooled = result.metrics()["detection"]["all"]
    assert (pooled["precision"], pooled["recall"]) == (1.0, 1.0)
    path = attach_to_bundle(result, bundle.path, bundle.weights_path)
    assert json.loads(path.read_text())["weights_sha256"] == bundle.weights_sha256


def test_images_must_match_the_bundle(tmp_path: Path, dataset: Dataset) -> None:
    bundle = load_bundle(_bundle(tmp_path / "b", dataset))
    record = dataset.manifest.images[0]
    image = read_image(dataset.image_path(record.id))
    with pytest.raises(ValueError, match="bundle's tiling"):
        predict_images(bundle, [image[:, :-1]], [record.geometry])
    other = ImageGeometry(
        width_px=record.geometry.width_px,
        height_px=record.geometry.height_px,
        t_first_s=0.0,
        dt_s=0.02,
        f_lowest_hz=500.0,
        df_hz=bundle.spectrogram.df_hz,
    )
    with pytest.raises(ValueError, match="another front end"):
        predict_images(bundle, [image], [other])


# ------------------------------------------------------ assignment by frequency


def test_assign_by_frequency_picks_the_nearest_live_track() -> None:
    from gridtools.io import Tracks

    times = np.arange(0.0, 10.0, 0.1)
    late = times >= 5.0
    tracks = Tracks(
        times_s=times,
        freqs_hz=np.concatenate([np.full(100, 700.0), np.full(late.sum(), 712.0)]),
        idents=np.concatenate([np.zeros(100), np.full(late.sum(), 3.0)]),
        time_indices=np.concatenate([np.arange(100), np.flatnonzero(late)]),
    )
    predictions = pd.DataFrame(
        {
            "recording": ["r", "r", "r", "r", "other"],
            "t_start_s": [1.0, 6.0, 6.0, 6.0, 1.0],
            "t_end_s": [1.1, 6.1, 6.1, 6.1, 1.1],
            "emitter_hz": [709.0, 709.0, 730.0, np.nan, 700.0],
        }
    )
    out = assign_by_frequency(predictions, {"r": tracks}, tolerance_hz=5.0)
    # 1 s: only track 0 is live, 9 Hz away -> none; 6 s: track 3 is 3 Hz away
    assert out["emitter_id"].tolist()[:2] == [pd.NA, 3]
    assert out["emitter_id"].isna().tolist() == [True, False, True, True, True]
    assert out["emitter_df_hz"].iloc[1] == pytest.approx(3.0)
    wide = assign_by_frequency(predictions, {"r": tracks}, tolerance_hz=10.0)
    assert wide["emitter_id"].iloc[0] == 0
