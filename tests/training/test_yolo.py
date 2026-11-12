"""The YOLO26 backend: input pipeline, ignore-aware loss and validation, smoke run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from chirpdetector.bundle import load_bundle
from chirpdetector.data import Dataset, dataset_ground_truth, read_image
from chirpdetector.evaluation import EvaluationConfig, attach_to_bundle, evaluate
from chirpdetector.models import _yolo_engine as engine
from chirpdetector.models.yolo import input_shape
from chirpdetector.prediction import predict_dataset
from chirpdetector.training import DetectorTrainingConfig, train_detector

from .conftest import cuda_available


def _dataset(
    dataset: Dataset, ids: list[str], *, augment: bool, time_flip: float = 0.0
):
    from ultralytics.cfg import get_cfg

    tiling = dataset.manifest.tiling
    return engine.SpectrogramDataset(
        engine._items(dataset, ids),
        shape=input_shape(tiling.height_px, tiling.width_px),
        time_flip=time_flip,
        augment=augment,
        hyp=get_cfg(),
        batch_size=2,
    )


def test_training_input_equals_inference_input(dataset: Dataset) -> None:
    """Channel order, padding and scale are identical in training and inference."""
    ids = list(dataset.manifest.image_ids("train"))[:2]
    data = _dataset(dataset, ids, augment=False)
    tiling = dataset.manifest.tiling
    shape = input_shape(tiling.height_px, tiling.width_px)
    assert shape == (64, 128)
    for k, path in enumerate(data.im_files):
        sample = data[k]
        expected = engine.to_model_input([read_image(path)], shape)[0]
        assert torch.equal(sample["img"], expected)
        assert sample["img"].shape == (3, *shape)
        # channel 0 is the reference window (red in the PNG)
        assert torch.equal(
            sample["img"][0, : tiling.height_px, : tiling.width_px],
            torch.from_numpy(read_image(path)[:, :, 0].copy()),
        )
        assert int(sample["img"][:, tiling.height_px :, :].max()) == engine.PAD_VALUE


def test_labels_are_normalised_by_the_padded_input(dataset: Dataset) -> None:
    image_id = dataset.manifest.image_ids("train")[0]
    data = _dataset(dataset, [image_id], augment=False)
    sample = data[0]
    h_in, w_in = sample["img"].shape[1:]
    xywh = sample["bboxes"].numpy() * np.array([w_in, h_in, w_in, h_in])
    xyxy = np.concatenate(
        [xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1
    )
    np.testing.assert_allclose(
        xyxy, engine._items(dataset, [image_id])[0].boxes, atol=1e-3
    )
    assert (sample["cls"] == 0).all()
    assert len(sample["ignore_bboxes"]) == dataset.manifest.image(image_id).n_ignore


def test_pipeline_has_no_other_augmentation(dataset: Dataset) -> None:
    data = _dataset(dataset, list(dataset.manifest.image_ids("train")), augment=True)
    transforms = data.transforms.transforms
    assert len(transforms) == 1
    assert isinstance(transforms[0], engine.SpectrogramFormat)
    assert transforms[0].time_flip == 0.0
    first, again = data[0]["img"], data[0]["img"]
    assert torch.equal(first, again)  # deterministic: nothing random is applied


def test_time_flip_mirrors_image_and_boxes(dataset: Dataset) -> None:
    image_id = dataset.manifest.image_ids("train")[0]
    plain = _dataset(dataset, [image_id], augment=False)[0]
    flipped = _dataset(dataset, [image_id], augment=True, time_flip=1.0)[0]
    width = dataset.manifest.tiling.width_px
    w_in = plain["img"].shape[2]
    assert torch.equal(flipped["img"][:, :, :width], plain["img"][:, :, :width].flip(2))
    xc_plain = plain["bboxes"][:, 0] * w_in
    xc_flip = flipped["bboxes"][:, 0] * w_in
    torch.testing.assert_close(xc_flip, width - xc_plain)


def test_collate_keeps_ignore_regions_per_image(dataset: Dataset) -> None:
    ids = list(dataset.manifest.image_ids("train"))[:3]
    data = _dataset(dataset, ids, augment=False)
    batch = engine.SpectrogramDataset.collate_fn([data[k] for k in range(3)])
    counts = torch.bincount(batch["ignore_batch_idx"].long(), minlength=3).tolist()
    assert sum(counts) == len(batch["ignore_bboxes"])
    assert counts == [len(data[k]["ignore_bboxes"]) for k in range(3)]


# ------------------------------------------------------------------- loss


def _loss_setup():
    from ultralytics.cfg import get_cfg

    model = engine.ChirpDetectionModel("yolo26n.yaml", nc=1, ch=3, verbose=False)
    setattr(model, "args", get_cfg())  # noqa: B010 (nn.Module typing)
    criterion = engine.IgnoreAwareDetectionLoss(model, tal_topk=10)
    torch.manual_seed(0)
    x = torch.rand(2, 3, 64, 128)
    preds = model(x)["one2many"]
    return model, criterion, preds


def test_ignore_regions_remove_background_loss() -> None:
    _, criterion, preds = _loss_setup()
    scores = preds["scores"].detach().clone().requires_grad_(True)
    preds = {**preds, "scores": scores}
    batch = {
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.2, 0.5, 0.1, 0.3]]),  # a chirp in image 0
        "ignore_bboxes": torch.tensor([[0.7, 0.5, 0.3, 0.6]]),  # ignore in image 1
        "ignore_batch_idx": torch.tensor([1.0]),
    }
    loss, _ = criterion.loss(preds, batch)
    loss.sum().backward()
    grad = scores.grad
    assert grad is not None
    anchors, strides = engine.make_anchors(preds["feats"], criterion.stride, 0.5)
    points = anchors * strides
    inside = (
        (points[:, 0] >= 0.55 * 128)
        & (points[:, 0] <= 0.85 * 128)
        & (points[:, 1] >= 0.2 * 64)
        & (points[:, 1] <= 0.8 * 64)
    )
    assert inside.any()
    # image 1: no gradient inside the ignore region, gradient outside it
    assert grad[1, 0, inside].abs().max() == 0
    assert grad[1, 0, ~inside].abs().min() > 0
    # image 0 has no ignore region: gradient everywhere
    assert grad[0, 0, inside].abs().min() > 0


def test_without_ignore_regions_the_loss_is_ultralytics() -> None:
    from ultralytics.utils.loss import v8DetectionLoss

    model, criterion, preds = _loss_setup()
    batch = {
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.2, 0.5, 0.1, 0.3], [0.6, 0.4, 0.2, 0.2]]),
        "ignore_bboxes": torch.zeros((0, 4)),
        "ignore_batch_idx": torch.zeros(0),
    }
    ours, _ = criterion.loss(preds, batch)
    theirs, _ = v8DetectionLoss(model, tal_topk=10).loss(preds, batch)
    torch.testing.assert_close(ours, theirs)


def test_validation_drops_predictions_in_ignore_regions() -> None:
    preds = [
        {
            "bboxes": torch.tensor(
                [[10.0, 10.0, 20.0, 20.0], [80.0, 20.0, 100.0, 40.0]]
            ),
            "conf": torch.tensor([0.9, 0.8]),
            "cls": torch.zeros(2),
            "extra": torch.zeros((2, 0)),
        },
        {
            "bboxes": torch.tensor([[80.0, 20.0, 100.0, 40.0]]),
            "conf": torch.tensor([0.7]),
            "cls": torch.zeros(1),
            "extra": torch.zeros((1, 0)),
        },
    ]
    batch = {
        "img": torch.zeros(2, 3, 64, 128),
        "ignore_bboxes": torch.tensor([[90 / 128, 30 / 64, 30 / 128, 30 / 64]]),
        "ignore_batch_idx": torch.tensor([0.0]),
    }
    out = engine.drop_ignored(preds, batch)
    assert out[0]["bboxes"].tolist() == [[10.0, 10.0, 20.0, 20.0]]
    assert len(out[1]["bboxes"]) == 1  # the region belongs to image 0 only


# --------------------------------------------------------------- smoke run


def _smoke_config(device: str) -> DetectorTrainingConfig:
    return DetectorTrainingConfig.from_dict(
        {
            "device": device,
            "model": {"architecture": "yolo26n", "pretrained": False},
            "optim": {"epochs": 1, "batch": 4, "workers": 0, "warmup_epochs": 0.0},
            "cv": {"n_folds": 0},
        }
    )


def _smoke(tmp_path: Path, dataset: Dataset, dataset_dir: Path, device: str) -> None:
    result = train_detector(dataset_dir, _smoke_config(device), tmp_path / "run")
    assert result.bundle is not None
    bundle = load_bundle(result.bundle)
    assert bundle.model_type == "ultralytics-yolo"
    assert bundle.model["input_shape"] == [64, 128]
    assert bundle.model["head"] == "one2one"
    fit = result.fits[0]
    assert fit.curves is not None
    assert fit.curves.is_file()
    curves = pd.read_csv(fit.curves)
    assert len(curves) == 1
    assert fit.details["start_weights"]["source"] == "random"
    # the bundle's weights load with plain ultralytics (no chirpdetector classes)
    ckpt = torch.load(bundle.weights_path, map_location="cpu", weights_only=False)
    assert type(ckpt["model"]).__name__ == "DetectionModel"
    test_ids = list(dataset.manifest.image_ids("test"))
    predictions = predict_dataset(bundle, dataset, test_ids, score_threshold=0.0)
    assert set(predictions["model_sha256"]) <= {bundle.weights_sha256}
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
    tmp_path: Path, dataset: Dataset, dataset_dir: Path, no_vram_env: None
) -> None:
    _smoke(tmp_path, dataset, dataset_dir, "cpu")


@pytest.mark.gpu
@pytest.mark.skipif(not cuda_available(), reason="no CUDA device")
def test_smoke_training_on_gpu(
    tmp_path: Path, dataset: Dataset, dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHIRP_MAX_VRAM_FRACTION", "0.1")
    _smoke(tmp_path, dataset, dataset_dir, "cuda")
