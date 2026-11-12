"""Dual-head detector: emitter target, loss, geometry, bundles, GPU cap, smoke run."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch
from gridtools.io import open_recording
from hypothesis import given
from hypothesis import strategies as st

pytest.importorskip("transformers")

from transformers.loss.loss_rt_detr import RTDetrLoss

from chirpdetector.bundle import Bundle, load_bundle
from chirpdetector.data import Dataset, Tiling, dataset_ground_truth, read_image
from chirpdetector.data.contract import sha256_file
from chirpdetector.evaluation import EvaluationConfig, attach_to_bundle, evaluate
from chirpdetector.inference import predict_span
from chirpdetector.models import _dualhead_engine as engine
from chirpdetector.models import get_backend
from chirpdetector.models.dualhead import (
    DualHeadModelConfig,
    decode_emitter,
    encode_emitter,
    input_shape,
)
from chirpdetector.prediction import load_detector, predict_dataset, predict_images
from chirpdetector.spectrogram import SpectrogramConfig, timefreq_to_pixel
from chirpdetector.training import DetectorTrainingConfig, train_detector

from .conftest import TINY

# ------------------------------------------------------------------- config


def test_model_config_defaults_and_validation() -> None:
    backend = get_backend("dualhead")
    resolved = backend.resolve_model_config({})
    assert resolved["architecture"] == "rtdetr_v2_r18vd"
    assert resolved["pretrained"] is True
    with pytest.raises(ValueError, match="unknown"):
        backend.resolve_model_config({"archtecture": "tiny"})
    with pytest.raises(ValueError, match="no pretrained weights"):
        DualHeadModelConfig(architecture="tiny", pretrained=True)
    with pytest.raises(ValueError, match="max_det"):
        DualHeadModelConfig(architecture="tiny", pretrained=False, max_det=500)
    config = DetectorTrainingConfig.from_dict({"model": {"type": "dualhead", **TINY}})
    assert config.to_dict()["model"]["type"] == "dualhead"


def test_inputs_are_padded_to_the_stride_not_resized() -> None:
    assert input_shape(205, 640) == (224, 640)
    image = np.full((205, 640, 3), 7, np.uint8)
    x = engine.to_model_input([image], (224, 640))
    assert x.shape == (1, 3, 224, 640)
    assert (x[0, :, :205] == 7).all()
    assert (x[0, :, 205:] == engine.PAD_VALUE).all()


# ----------------------------------------------------------- emitter target


@given(
    emitter=st.floats(-300, 500),
    y_max=st.floats(0, 205),
    height=st.floats(10, 1000),
)
def test_emitter_encoding_round_trips(
    emitter: float, y_max: float, height: float
) -> None:
    offset = encode_emitter(emitter, y_max, height)
    assert decode_emitter(offset, y_max, height) == pytest.approx(emitter, abs=1e-9)


def test_emitter_offset_is_frequency_relative_to_the_box(
    dualhead_dataset: Dataset,
) -> None:
    """Offset = (emitter_hz - f_low_hz) / image span: no absolute frequency."""
    image_id = next(
        i
        for i in dualhead_dataset.manifest.image_ids()
        if len(dualhead_dataset.boxes_of(i))
    )
    g = dualhead_dataset.manifest.image(image_id).geometry
    rows = dualhead_dataset.boxes_of(image_id)
    _, y_low = timefreq_to_pixel(0.0, rows["f_low_hz"].to_numpy(), g)
    _, y_emitter = timefreq_to_pixel(0.0, rows["emitter_hz"].to_numpy(), g)
    span = g.freq_extent_hz[1] - g.freq_extent_hz[0]
    np.testing.assert_allclose(
        encode_emitter(y_emitter, y_low, g.height_px),
        (rows["emitter_hz"] - rows["f_low_hz"]).to_numpy() / span,
        atol=1e-12,
    )
    # moving box and emitter up by 37 Hz leaves the offset unchanged
    _, y_low2 = timefreq_to_pixel(0.0, rows["f_low_hz"].to_numpy() + 37, g)
    _, y_emitter2 = timefreq_to_pixel(0.0, rows["emitter_hz"].to_numpy() + 37, g)
    np.testing.assert_allclose(
        encode_emitter(y_emitter2, y_low2, g.height_px),
        encode_emitter(y_emitter, y_low, g.height_px),
        atol=1e-12,
    )


def test_model_decodes_the_emitter_from_the_predicted_lower_edge() -> None:
    model, _ = engine.build_model(DualHeadModelConfig(**TINY), emitter_unit=63 / 64)
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 64, 128))
    boxes = out["boxes"].numpy()
    lower = boxes[..., 1] + boxes[..., 3] / 2
    # an untrained head outputs its initial offset everywhere
    expected = decode_emitter(engine.EMITTER_INIT, lower * 64, 63) / 64
    np.testing.assert_allclose(out["emitter"].numpy(), expected, atol=1e-5)


# --------------------------------------------------------------------- loss


def _criterion(unit: float = 1.0, weight: float = 10.0) -> engine.DualHeadLoss:
    config = engine.hf_config(DualHeadModelConfig(**TINY))
    return engine.DualHeadLoss(config, emitter_weight=weight, emitter_unit=unit)


def _outputs(logits: list[float], boxes: list[list[float]]) -> dict[str, torch.Tensor]:
    return {
        "logits": torch.tensor(logits).reshape(1, -1, 1).requires_grad_(),
        "pred_boxes": torch.tensor(boxes).reshape(1, -1, 4),
    }


def _target(ignore: list[list[float]]) -> dict[str, torch.Tensor]:
    return {
        "class_labels": torch.zeros(1, dtype=torch.int64),
        "boxes": torch.tensor([[0.2, 0.5, 0.1, 0.2]]),
        "emitter_y": torch.tensor([0.58]),
        "ignore_boxes": torch.tensor(ignore).reshape(-1, 4),
    }


BOXES = [[0.2, 0.5, 0.1, 0.2], [0.7, 0.5, 0.1, 0.2], [0.45, 0.2, 0.1, 0.1]]
"""Query 0 on the chirp, query 1 on an ignore region, query 2 on background."""
REGION = [[0.6, 0.35, 0.8, 0.65]]


def test_ignore_regions_remove_the_classification_loss_of_unmatched_queries() -> None:
    criterion = _criterion()
    matched = [(torch.tensor([0]), torch.tensor([0]))]
    grads = {}
    for name, regions in (("plain", []), ("ignore", REGION)):
        outputs = _outputs([2.0, 3.0, 1.0], BOXES)
        loss = criterion.loss_labels_vfl(outputs, [_target(regions)], matched, 1.0)
        loss["loss_vfl"].backward()
        grad = outputs["logits"].grad
        assert grad is not None
        grads[name] = grad.reshape(-1)
    assert grads["plain"][1] != 0
    assert grads["ignore"][1] == 0  # the query on the ignore region is free
    assert grads["ignore"][0] == pytest.approx(float(grads["plain"][0]))
    assert grads["ignore"][2] == pytest.approx(float(grads["plain"][2]))


def test_a_matched_query_inside_an_ignore_region_is_still_trained() -> None:
    criterion = _criterion()
    outputs = _outputs([2.0, 3.0, 1.0], BOXES)
    regions = [[0.1, 0.3, 0.3, 0.7]]  # covers the chirp itself
    matched = [(torch.tensor([0]), torch.tensor([0]))]
    loss = criterion.loss_labels_vfl(outputs, [_target(regions)], matched, 1.0)
    loss["loss_vfl"].backward()
    grad = outputs["logits"].grad
    assert grad is not None
    assert grad.reshape(-1)[0] != 0


def test_without_ignore_regions_the_loss_is_transformers() -> None:
    """Pins the copied varifocal loss to transformers' implementation."""
    criterion = _criterion()
    reference = RTDetrLoss(engine.hf_config(DualHeadModelConfig(**TINY)))
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(2, 7, 1)).astype(np.float32)
    boxes = rng.uniform(0.2, 0.8, size=(2, 7, 4)).astype(np.float32)
    boxes[..., 2:] *= 0.3
    targets = [
        {
            "class_labels": torch.zeros(2, dtype=torch.int64),
            "boxes": torch.tensor([[0.3, 0.3, 0.1, 0.2], [0.6, 0.5, 0.2, 0.1]]),
            "ignore_boxes": torch.zeros(0, 4),
        },
        {
            "class_labels": torch.zeros(1, dtype=torch.int64),
            "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
            "ignore_boxes": torch.zeros(0, 4),
        },
    ]
    outputs = {
        "logits": torch.from_numpy(logits),
        "pred_boxes": torch.from_numpy(boxes),
    }
    indices = reference.matcher(outputs, targets)
    ours = criterion.loss_labels_vfl(outputs, targets, indices, 3.0)["loss_vfl"]
    theirs = reference.loss_labels_vfl(outputs, targets, indices, 3.0)["loss_vfl"]
    assert float(ours) == pytest.approx(float(theirs), rel=1e-6)


def test_emitter_loss_is_l1_on_matched_queries_in_image_span_units() -> None:
    unit = 205 / 224
    criterion = _criterion(unit=unit)
    pred = torch.tensor([[0.50, 0.70, 0.90]], requires_grad=True)
    outputs = {"pred_emitter": pred}
    target = _target([])
    loss = criterion.loss_emitter(
        outputs, [target], [(torch.tensor([1]), torch.tensor([0]))], 1.0
    )["loss_emitter"]
    assert float(loss.detach()) == pytest.approx(abs(0.70 - 0.58) / unit)
    loss.backward()
    assert pred.grad is not None
    np.testing.assert_array_equal(pred.grad.numpy()[0] != 0, [False, True, False])
    # layers without emitter predictions (encoder proposals) add nothing
    assert criterion.loss_emitter({}, [target], [], 1.0) == {}


def test_full_loss_has_every_term_and_reaches_the_emitter_head() -> None:
    model, _ = engine.build_model(DualHeadModelConfig(**TINY), emitter_unit=63 / 64)
    model.train()
    criterion = engine.DualHeadLoss(model.config, 10.0, 63 / 64)
    target = _target(REGION)
    empty = {
        "class_labels": torch.zeros(0, dtype=torch.int64),
        "boxes": torch.zeros(0, 4),
        "emitter_y": torch.zeros(0),
        "ignore_boxes": torch.zeros(0, 4),
    }
    targets = [target, empty]
    outputs = model(torch.rand(2, 3, 64, 128), labels=targets)
    assert outputs["denoising_meta_values"] is not None
    loss, terms = engine.compute_loss(criterion, outputs, targets)
    assert torch.isfinite(loss)
    assert {"loss_vfl", "loss_bbox", "loss_giou", "loss_emitter"} <= set(terms)
    assert any(k.startswith("loss_emitter_dn") for k in terms)
    assert any(k.startswith("loss_emitter_aux") for k in terms)
    loss.backward()
    last = cast("Any", model.emitter_embed)[-1].layers[-1]
    assert last.weight.grad is not None
    assert last.weight.grad.abs().sum() > 0


# --------------------------------------------------------------------- data


def test_time_shift_moves_content_and_turns_cut_chirps_into_ignore_regions() -> None:
    image = np.zeros((20, 100, 3), np.uint8)
    image[:, 10:20] = 200
    boxes = np.array([[10, 2, 20, 8], [90, 2, 100, 8]], np.float32)
    emitter = np.array([9.0, 7.0], np.float32)
    ignore = np.array([[50, 0, 60, 5]], np.float32)
    out, b, e, ig = engine.time_shift(image, boxes, emitter, ignore, 6)
    assert (out[:, 16:26] == 200).all()
    assert (out[:, :6] == engine.PAD_VALUE).all()
    np.testing.assert_array_equal(b, [[16, 2, 26, 8]])  # rows unchanged
    np.testing.assert_array_equal(e, [9.0])
    # the right chirp keeps 4 of 10 columns: ignore region, clipped
    np.testing.assert_array_equal(ig, [[56, 0, 66, 5], [96, 2, 100, 8]])
    out, b, _, _ = engine.time_shift(image, boxes, emitter, ignore, -3)
    np.testing.assert_array_equal(b, [[7, 2, 17, 8], [87, 2, 97, 8]])
    assert (out[:, 7:17] == 200).all()


def test_targets_are_normalised_by_the_padded_input(dualhead_dataset: Dataset) -> None:
    ids = [
        i
        for i in dualhead_dataset.manifest.image_ids()
        if len(dualhead_dataset.boxes_of(i))
    ]
    item = engine.make_items(dualhead_dataset, ids[:1])[0]
    target = engine.make_target(item.boxes, item.emitter_y, item.ignore, (64, 128))
    cx, cy, w, h = target["boxes"].numpy().T
    np.testing.assert_allclose(cx * 128 - w * 64, item.boxes[:, 0], atol=1e-4)
    np.testing.assert_allclose(cy * 64 + h * 32, item.boxes[:, 3], atol=1e-4)
    np.testing.assert_allclose(
        target["emitter_y"].numpy() * 64, item.emitter_y, atol=1e-4
    )
    assert target["ignore_boxes"].shape == (len(item.ignore), 4)


# ------------------------------------------------------- geometry round trip


class _Oracle(torch.nn.Module):
    """Returns given last-layer boxes and emitter rows (normalised) with score 1."""

    def __init__(self, boxes: torch.Tensor, emitter: torch.Tensor) -> None:
        super().__init__()
        self.boxes, self.emitter = boxes, emitter

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        n = x.shape[0]
        q = self.boxes.shape[0]
        return {
            "logits": torch.full((n, 1, q, 1), 10.0),
            "boxes": self.boxes.expand(n, 1, q, 4).clone(),
            "emitter": self.emitter.expand(n, 1, q).clone(),
        }


def test_predictions_map_back_to_the_labels_in_seconds_and_hz(
    dualhead_dataset: Dataset,
    make_tiny_bundle: Callable[[SpectrogramConfig, Tiling], Bundle],
) -> None:
    """Dataset boxes -> training targets -> model output -> predict_images -> Hz."""
    manifest = dualhead_dataset.manifest
    bundle = make_tiny_bundle(manifest.spectrogram, manifest.tiling)
    image_id = max(
        manifest.image_ids(), key=lambda i: len(dualhead_dataset.boxes_of(i))
    )
    rows = dualhead_dataset.boxes_of(image_id)
    item = engine.make_items(dualhead_dataset, [image_id])[0]
    shape = tuple(bundle.model["input_shape"])
    target = engine.make_target(item.boxes, item.emitter_y, item.ignore, shape)
    detector = load_detector(bundle, device="cpu")
    detector.model = _Oracle(target["boxes"], target["emitter_y"])  # type: ignore[attr-defined]
    out = predict_images(
        bundle,
        [read_image(dualhead_dataset.image_path(image_id))],
        [manifest.image(image_id).geometry],
        detector=detector,
    )
    out = out.sort_values("t_start_s").reset_index(drop=True)
    rows = rows.sort_values("t_start_s").reset_index(drop=True)
    assert len(out) == len(rows)
    for column in ("t_start_s", "t_end_s", "f_low_hz", "f_high_hz", "emitter_hz"):
        np.testing.assert_allclose(out[column], rows[column], atol=1e-6, err_msg=column)


def test_postprocess_keeps_the_best_queries_above_threshold() -> None:
    boxes = torch.tensor(
        [[[[0.5, 0.5, 0.2, 0.2], [0.5, 0.95, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]]]
    )
    outputs = {
        "logits": torch.tensor([[[[3.0], [2.0], [-3.0]]]]),
        "boxes": boxes,
        "emitter": torch.tensor([[[0.55, 2.0, 0.1]]]),
    }
    (table,) = engine.postprocess(
        outputs, input_hw=(64, 128), image_hw=(63, 128), score_threshold=0.5, max_det=10
    )
    assert len(table) == 2
    assert table["y_max"].to_numpy().max() <= 63  # clipped to the image, not the input
    assert table["emitter_y"].to_numpy().max() == 63  # the emitter lies in its band
    np.testing.assert_allclose(
        table.loc[0, ["x_min", "x_max"]], [51.2, 76.8], atol=1e-4
    )


# ----------------------------------------------------------------- GPU cap


class _Stop(Exception):
    pass


def test_training_and_inference_honour_the_vram_fraction(
    monkeypatch: pytest.MonkeyPatch,
    dualhead_dataset: Dataset,
    make_tiny_bundle: Callable[[SpectrogramConfig, Tiling], Bundle],
    tmp_path: Path,
) -> None:
    calls: list[tuple[float, int]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device=0: calls.append((fraction, device)),
    )
    monkeypatch.setenv("CHIRP_MAX_VRAM_FRACTION", "0.3")

    def stop(*_: object, **__: object) -> None:
        assert calls, "the cap must be applied before the model is built"
        raise _Stop

    monkeypatch.setattr(engine, "build_model", stop)
    monkeypatch.setattr(engine, "load_weights", stop)
    from chirpdetector.models import FitRequest

    manifest = dualhead_dataset.manifest
    for fraction, expected in ((None, 0.3), (0.2, 0.2)):
        config = DetectorTrainingConfig.from_dict(
            {
                "device": "cuda",
                "max_vram_fraction": fraction,
                "model": {"type": "dualhead", **TINY},
                "cv": {"n_folds": 0},
            }
        )
        request = FitRequest(
            dualhead_dataset,
            manifest.image_ids("train"),
            manifest.image_ids("val"),
            config,
            tmp_path / f"fit-{fraction}",
            "final",
        )
        calls.clear()
        with pytest.raises(_Stop):
            engine.fit(request)
        assert calls == [(expected, 0)]
    bundle = make_tiny_bundle(manifest.spectrogram, manifest.tiling)
    calls.clear()
    with pytest.raises(_Stop):
        engine.DualHeadDetector(bundle, device="cuda:0")
    assert calls == [(0.3, 0)]


# --------------------------------------------------------------- smoke runs


def _smoke(tmp_path: Path, dataset: Dataset, dataset_dir: Path, device: str) -> None:
    config = DetectorTrainingConfig.from_dict(
        {
            "device": device,
            "model": {"type": "dualhead", **TINY, "time_shift_px": 8},
            "optim": {"epochs": 2, "batch": 4, "workers": 0, "warmup_epochs": 0.0},
            "cv": {"n_folds": 0},
        }
    )
    result = train_detector(dataset_dir, config, tmp_path / "run")
    assert result.bundle is not None
    bundle = load_bundle(result.bundle)
    assert bundle.model_type == "dualhead"
    assert bundle.model["input_shape"] == [64, 128]
    fit = result.fits[0]
    assert fit.curves is not None
    curves = pd.read_csv(fit.curves)
    assert len(curves) == 2
    assert {"train/emitter", "val/ap", "val/assign_accuracy"} <= set(curves.columns)
    assert fit.details["start_weights"]["source"] == "random"
    # the weights load without network access or chirpdetector's trainer
    data = torch.load(bundle.weights_path, map_location="cpu", weights_only=True)
    assert data["format"] == engine.WEIGHTS_FORMAT
    test_ids = list(dataset.manifest.image_ids("test"))
    predictions = predict_dataset(
        bundle, dataset, test_ids, score_threshold=0.0, device=device
    )
    assert "emitter_hz" in predictions.columns
    assert np.isfinite(predictions["emitter_hz"]).all()
    assert set(predictions["model_sha256"]) == {bundle.weights_sha256}
    result_eval = evaluate(
        predictions,
        dataset_ground_truth(dataset, test_ids),
        config=EvaluationConfig(n_bootstrap=0),
    )
    path = attach_to_bundle(result_eval, bundle.path, bundle.weights_path)
    assert json.loads(path.read_text())["weights_sha256"] == bundle.weights_sha256
    assert load_bundle(bundle.path).metrics is not None


@pytest.mark.slow
def test_smoke_training_on_cpu(
    tmp_path: Path,
    dualhead_dataset: Dataset,
    dualhead_dataset_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHIRP_MAX_VRAM_FRACTION", raising=False)
    _smoke(tmp_path, dualhead_dataset, dualhead_dataset_dir, "cpu")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_smoke_training_on_gpu(
    tmp_path: Path,
    dualhead_dataset: Dataset,
    dualhead_dataset_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHIRP_MAX_VRAM_FRACTION", "0.05")
    _smoke(tmp_path, dualhead_dataset, dualhead_dataset_dir, "cuda")


def test_predict_span_returns_the_emitter_frequency(
    make_tiny_bundle: Callable[[SpectrogramConfig, Tiling], Bundle],
    make_synthetic_trial: Callable[[Path], Path],
    tmp_path: Path,
) -> None:
    config = SpectrogramConfig()
    tiling = Tiling(
        tile_duration_s=0.64, overlap_s=0.2, width_px=128, height_px=config.n_freqs
    )
    bundle = make_tiny_bundle(config, tiling)
    trial = make_synthetic_trial(tmp_path / "trial")
    recording = open_recording(trial / "recording.wav")
    out = predict_span(
        bundle, recording, 2.0, 5.0, name="trial", score_threshold=0.0, device="cpu"
    )
    assert len(out)
    assert "emitter_hz" in out.columns
    low, high = config.f_min_hz - 5, config.f_max_hz + 5
    assert out["emitter_hz"].between(low, high).all()


def test_training_can_continue_from_a_weights_file(tmp_path: Path) -> None:
    first, start = engine.build_model(DualHeadModelConfig(**TINY), emitter_unit=63 / 64)
    assert start["source"] == "random"
    with torch.no_grad():
        for p in first.emitter_embed.parameters():
            p.add_(0.5)  # differs from a fresh head
    path = tmp_path / "start.pt"
    engine.export_weights(first, path)
    config = DualHeadModelConfig(**{**TINY, "weights": str(path), "num_queries": 30})
    assert config.repository is None
    model, start = engine.build_model(config, emitter_unit=63 / 64)
    assert start["source"] == "file"
    assert start["sha256"] == sha256_file(path)
    assert model.config.num_queries == 30
    for name, value in first.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, msg=name)
    with pytest.raises(ValueError, match="emitter unit"):
        engine.build_model(config, emitter_unit=200 / 224)
    with pytest.raises(FileNotFoundError):
        engine.build_model(
            DualHeadModelConfig(**{**TINY, "weights": str(tmp_path / "no.pt")}), 1.0
        )
    with pytest.raises(ValueError, match=r"\.pt"):
        DualHeadModelConfig(**{**TINY, "weights": "start.bin"})
