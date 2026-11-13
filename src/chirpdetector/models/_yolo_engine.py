"""ultralytics YOLO26 training and inference on chirp spectrograms.

Imported lazily (ultralytics is slow to import). The ultralytics trainer is
used as is (optimizer, schedule, EMA, AMP, early stopping, checkpoints),
with four replacements:

Dataset
    Images come from the dataset contract, not from a directory scan:
    positives from the YOLO label files, ignore regions from
    ``boxes.parquet``. No label cache is written into the dataset, no
    ``.npy`` image cache is read, and images are never resized.
Augmentation
    The pipeline is ours, not ultralytics' ``v8_transforms``: optional time
    reversal, then zero padding at the bottom and right to the stride,
    nothing else. The physically meaningless ultralytics augmentations
    are not just set to zero, they are not in the pipeline at all:
    vertical flips (they turn up-chirps into down-chirps and swap high and
    low frequencies), HSV, colour and channel-swap jitter (channels are
    window lengths, not colours), rotation, shear and perspective (they
    tilt the frequency axis), scaling (it changes chirp excursions and
    durations in Hz and s, and the width of every baseline), translation
    (it pads with an artificial border and moves fish to EOD frequencies
    the scene does not have; time offsets come from the tiling), mosaic,
    mixup, cutmix and copy-paste (they put spectrogram pieces of different
    frequency bands and times side by side, so rows no longer map to one
    frequency), and Albumentations (blur, CLAHE, grey conversion). Domain
    randomisation happens in signal space, in the simulator.
    *Time reversal* (``time_flip``, default 0): a reversed spectrogram is
    the exact spectrogram of the time-reversed signal, but the signal is
    not a natural one: chirps rise fast and decay slowly, many small chirps
    have an undershoot after the excursion, and rises have a slow decay
    over seconds. Reversal would teach the model to accept these shapes
    backwards and dilutes the asymmetry that separates chirps from other
    transients. With simulation as an unlimited data source, the extra
    variety is not worth that; it stays available for an ablation.
Loss (ignore regions)
    YOLO has no ignore label. The classification loss of an anchor point
    that is background (no chirp assigned to it) and lies inside an ignore
    region of its image is set to zero; everything else is unchanged (box
    and DFL losses only exist for assigned anchors; the normalisation by
    the assigned scores is unchanged). The model is thus free to fire or not
    on an invisible chirp or a harmonic copy. Ignore regions are not
    painted over in the image: masking with "local background" cannot be
    done for real recordings without inventing pixels, and for harmonic
    copies it would erase the fish's harmonic baseline as well, teaching
    the model an artefact. The cost of loss masking is a dependence on
    ultralytics internals (``v8DetectionLoss.bce``), pinned by tests.
    Anchor points are cell centres; an ignore region narrower than a
    stride may contain no anchor point at the coarse levels, so those
    still see it as background (ignore footprints are as large as chirp
    footprints, so this is rare).
Validation
    During training (early stopping, selection of ``best.pt``), predictions
    whose centre lies inside an ignore region are dropped before the
    metrics, and the NMS-free one-to-one head is validated (``nms=False``),
    as it is deployed. The metrics are ultralytics' (precision, recall,
    mAP50, mAP50-95; fitness = 0.1 mAP50 + 0.9 mAP50-95), on the
    validation images of the fit; the matching-rule metrics of
    `chirpdetector.evaluation` are computed afterwards, from the bundle.
"""

from __future__ import annotations

import logging
import math
import os
import random
import shutil
from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest import mock

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # never pip-install at runtime

import cv2
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from ultralytics import __version__ as ULTRALYTICS_VERSION
from ultralytics.data.augment import Compose
from ultralytics.data.dataset import YOLODataset
from ultralytics.engine import validator as ultralytics_validator
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import RANK, nms
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import make_anchors

from chirpdetector.data.contract import (
    CLASSES,
    Dataset,
    labels_to_pixel_boxes,
    read_yolo_labels,
    sha256_file,
)
from chirpdetector.models import PIXEL_COLUMNS, FitOutput, FitRequest
from chirpdetector.models.yolo import MODEL_TYPE, STRIDE, YoloModelConfig, input_shape
from chirpdetector.resources import apply_vram_limit
from chirpdetector.spectrogram import timefreq_box_to_pixel

if TYPE_CHECKING:
    from chirpdetector.bundle import Bundle

log = logging.getLogger(__name__)

PAD_VALUE = 0
"""Value of the padding pixels: the bottom of the normalised dB range."""


def weights_cache() -> Path:
    """COCO weights of ultralytics: ``weights/`` in chirpdetector's cache."""
    from chirpdetector.pretrained import cache_dir

    return cache_dir() / "weights"


_NAMES = dict(enumerate(CLASSES))


def _disable_telemetry() -> None:
    """Switch off ultralytics' anonymous usage events for this process."""
    try:
        from ultralytics.utils.events import events

        events.enabled = False
    except ImportError:  # pragma: no cover - layout of other versions
        pass


_disable_telemetry()


# ---------------------------------------------------------------- inputs


def to_model_input(
    images: Sequence[NDArray[np.uint8]], shape: tuple[int, int]
) -> torch.Tensor:
    """Images in window order -> ``(n, 3, H_in, W_in)`` uint8, zero-padded.

    The same layout the training pipeline produces: the image at the top
    left, `PAD_VALUE` below and to the right, channel ``c`` = window ``c``.
    """
    height, width = shape
    out = np.full((len(images), 3, height, width), PAD_VALUE, dtype=np.uint8)
    for k, image in enumerate(images):
        h, w = image.shape[:2]
        if image.shape[2] != 3 or h > height or w > width:
            msg = f"image of shape {image.shape} does not fit the input {shape}"
            raise ValueError(msg)
        out[k, :, :h, :w] = image.transpose(2, 0, 1)
    return torch.from_numpy(out)


@dataclass(frozen=True)
class _Item:
    """One training or validation image with pixel boxes (x_min, y_min, x_max, y_max)."""

    path: str
    shape: tuple[int, int]
    boxes: NDArray[np.float32]
    ignore: NDArray[np.float32]


def _items(dataset: Dataset, image_ids: Sequence[str]) -> list[_Item]:
    out = []
    for image_id in image_ids:
        record = dataset.manifest.image(image_id)
        g = record.geometry
        labels = read_yolo_labels(dataset.label_path(image_id))
        boxes = labels_to_pixel_boxes(labels, g.width_px, g.height_px)
        rows = dataset.boxes_of(image_id, "ignore")
        if len(rows):
            x0, y0, x1, y1 = timefreq_box_to_pixel(
                rows["t_start_s"].to_numpy(dtype=np.float64),
                rows["t_end_s"].to_numpy(dtype=np.float64),
                rows["f_low_hz"].to_numpy(dtype=np.float64),
                rows["f_high_hz"].to_numpy(dtype=np.float64),
                g,
            )
            ignore = np.stack([x0, y0, x1, y1], axis=1)
        else:
            ignore = np.zeros((0, 4))
        out.append(
            _Item(
                path=str(dataset.image_path(image_id)),
                shape=(g.height_px, g.width_px),
                boxes=boxes.astype(np.float32).reshape(-1, 4),
                ignore=ignore.astype(np.float32).reshape(-1, 4),
            )
        )
    return out


def _flip_time(boxes: NDArray[np.float32], width: int) -> NDArray[np.float32]:
    out = boxes.copy()
    out[:, 0] = width - boxes[:, 2]
    out[:, 2] = width - boxes[:, 0]
    return out


class SpectrogramFormat:
    """Our whole augmentation and formatting pipeline (see the module docstring)."""

    def __init__(self, shape: tuple[int, int], time_flip: float) -> None:
        self.shape = shape
        self.time_flip = float(time_flip)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Time-flip (optionally), pad, and convert one sample to tensors."""
        bgr = labels.pop("img")
        boxes = labels.pop("bboxes")
        ignore = labels.pop("ignore")
        cls = labels.pop("cls")
        for key in ("segments", "keypoints", "normalized", "bbox_format"):
            labels.pop(key, None)
        image = bgr[:, :, ::-1]  # OpenCV's BGR -> window order
        height, width = image.shape[:2]
        if self.time_flip > 0 and random.random() < self.time_flip:
            image = image[:, ::-1]
            boxes, ignore = _flip_time(boxes, width), _flip_time(ignore, width)
        h_in, w_in = self.shape
        scale = np.array([w_in, h_in, w_in, h_in], dtype=np.float32)
        labels["img"] = to_model_input([np.ascontiguousarray(image)], self.shape)[0]
        labels["cls"] = torch.from_numpy(
            np.asarray(cls, dtype=np.float32).reshape(-1, 1)
        )
        labels["bboxes"] = torch.from_numpy(xyxy2xywh(boxes) / scale).float()
        labels["batch_idx"] = torch.zeros(len(boxes))
        labels["ignore_bboxes"] = torch.from_numpy(xyxy2xywh(ignore) / scale).float()
        labels["ratio_pad"] = ((1.0, 1.0), (0.0, 0.0))
        labels["resized_shape"] = (height, width)
        return labels


class SpectrogramDataset(YOLODataset):
    """Contract images for the ultralytics trainer (see the module docstring)."""

    def __init__(
        self,
        items: Sequence[_Item],
        *,
        shape: tuple[int, int],
        time_flip: float,
        augment: bool,
        hyp: Any,
        prefix: str = "",
        batch_size: int = 16,
    ) -> None:
        self._items = {str(Path(item.path)): item for item in items}
        self._shape = shape
        self._time_flip = time_flip
        super().__init__(
            img_path=[item.path for item in items],
            imgsz=max(shape),
            cache=False,
            augment=augment,
            hyp=hyp,
            prefix=prefix,
            rect=False,
            batch_size=batch_size,
            stride=STRIDE,
            pad=0.0,
            single_cls=False,
            classes=None,
            fraction=1.0,
            data={"names": _NAMES, "nc": len(CLASSES), "channels": 3},
            task="detect",
        )

    def get_labels(self) -> list[dict[str, Any]]:
        """Labels from the items, in the order of ``self.im_files``."""
        labels = []
        for f in self.im_files:
            item = self._items[str(Path(f))]
            labels.append(
                {
                    "im_file": f,
                    "shape": item.shape,
                    "cls": np.zeros((len(item.boxes), 1), dtype=np.float32),
                    "bboxes": item.boxes,
                    "ignore": item.ignore,
                    "segments": [],
                    "keypoints": None,
                    "normalized": False,
                    "bbox_format": "xyxy",
                }
            )
        return labels

    def load_image(
        self, i: int, rect_mode: bool = True, resize_short: bool = False
    ) -> tuple[NDArray[np.uint8], tuple[int, int], tuple[int, int]]:
        """Read image ``i`` as stored (BGR, as ultralytics expects); never resized."""
        path = self.im_files[i]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            msg = f"image not found: {path}"
            raise FileNotFoundError(msg)
        expected = self._items[str(Path(path))].shape
        if image.shape[:2] != expected:
            msg = f"{path}: shape {image.shape[:2]}, expected {expected}"
            raise ValueError(msg)
        hw = (int(image.shape[0]), int(image.shape[1]))
        return cast("NDArray[np.uint8]", image), hw, hw

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        """Keep the raw pixel boxes; `SpectrogramFormat` converts them."""
        return label

    def build_transforms(self, hyp: Any = None) -> Compose:
        """`SpectrogramFormat` only (time flip only when augmenting)."""
        flip = self._time_flip if self.augment else 0.0
        return Compose([SpectrogramFormat(self._shape, flip)])

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """ultralytics' collation plus ignore boxes and their image index."""
        ignore = [sample.pop("ignore_bboxes") for sample in batch]
        out = YOLODataset.collate_fn(batch)
        out["ignore_bboxes"] = torch.cat(ignore, 0)
        out["ignore_batch_idx"] = torch.cat(
            [torch.full((len(x),), float(i)) for i, x in enumerate(ignore)]
        )
        return out


# ------------------------------------------------------------------ loss


def ignore_anchor_mask(
    points: torch.Tensor,
    ignore_xywh: torch.Tensor,
    ignore_batch_idx: torch.Tensor,
    batch_size: int,
    imgsz: torch.Tensor,
) -> torch.Tensor:
    """Anchor points inside an ignore region of their image.

    Parameters
    ----------
    points
        Anchor points in input pixels, ``(n_anchors, 2)`` (x, y).
    ignore_xywh
        Ignore regions, normalised ``xc, yc, w, h``, ``(m, 4)``.
    ignore_batch_idx
        Image of each region, ``(m,)``.
    batch_size
        Images in the batch.
    imgsz
        Input size ``(height, width)``.

    Returns
    -------
    torch.Tensor
        Boolean ``(batch_size, n_anchors)``.
    """
    mask = torch.zeros(batch_size, points.shape[0], device=points.device)
    if ignore_xywh.shape[0] == 0:
        return mask.bool()
    scale = imgsz[[1, 0, 1, 0]].to(points.device, points.dtype)
    xyxy = xywh2xyxy(ignore_xywh.to(points.device, points.dtype)) * scale
    x, y = points[None, :, 0], points[None, :, 1]
    inside = (
        (x >= xyxy[:, 0:1])
        & (x <= xyxy[:, 2:3])
        & (y >= xyxy[:, 1:2])
        & (y <= xyxy[:, 3:4])
    )
    mask.index_add_(0, ignore_batch_idx.to(points.device).long(), inside.to(mask.dtype))
    return mask > 0


class _MaskedBCE:
    """``BCEWithLogitsLoss(reduction="none")`` that zeroes ignored background."""

    def __init__(self, bce: Any, ignored: torch.Tensor) -> None:
        self.bce = bce
        self.ignored = ignored

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.bce(pred, target)  # (batch, anchors, classes)
        background = target.sum(-1) <= 0
        keep = ~(self.ignored & background)
        return loss * keep.unsqueeze(-1).to(loss.dtype)


class IgnoreAwareDetectionLoss(v8DetectionLoss):
    """ultralytics' detection loss with ignore regions (module docstring)."""

    def get_assigned_targets_and_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, Any],
        **assign_kwargs: Any,
    ) -> tuple:
        """Mask the classification loss of ignored background anchors."""
        ignore = batch.get("ignore_bboxes")
        if ignore is None or ignore.shape[0] == 0:
            return super().get_assigned_targets_and_loss(preds, batch, **assign_kwargs)
        feats = cast("list[torch.Tensor]", preds["feats"])
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        stride = float(cast("torch.Tensor", self.stride)[0])
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device) * stride
        ignored = ignore_anchor_mask(
            anchor_points * stride_tensor,
            ignore,
            batch["ignore_batch_idx"],
            preds["scores"].shape[0],
            imgsz,
        )
        bce = self.bce
        self.bce = _MaskedBCE(bce, ignored)
        try:
            return super().get_assigned_targets_and_loss(preds, batch, **assign_kwargs)
        finally:
            self.bce = bce


class ChirpDetectionModel(DetectionModel):
    """YOLO26 with `IgnoreAwareDetectionLoss`; exported as a plain ``DetectionModel``."""

    def init_criterion(self) -> Any:
        """One-to-many and one-to-one losses, both ignore-aware."""
        head = cast("Any", self.model)[-1]
        if getattr(head, "one2one_cv2", None) is not None:
            return E2ELoss(self, loss_fn=IgnoreAwareDetectionLoss)
        return IgnoreAwareDetectionLoss(self)


# ------------------------------------------------------------ validation


def drop_ignored(
    preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]
) -> list[dict[str, torch.Tensor]]:
    """Remove predictions whose centre lies inside an ignore region of their image."""
    ignore = batch.get("ignore_bboxes")
    if ignore is None or ignore.shape[0] == 0:
        return preds
    height, width = batch["img"].shape[2:]
    scale = torch.tensor([width, height, width, height], dtype=torch.float32)
    xyxy = xywh2xyxy(ignore.float().cpu()) * scale
    owner = batch["ignore_batch_idx"].cpu()
    out = []
    for si, pred in enumerate(preds):
        own = xyxy[owner == si].to(pred["bboxes"].device)
        if own.shape[0] and pred["bboxes"].shape[0]:
            c = (pred["bboxes"][:, :2] + pred["bboxes"][:, 2:]) / 2
            inside = (
                (c[:, None, 0] >= own[None, :, 0])
                & (c[:, None, 0] <= own[None, :, 2])
                & (c[:, None, 1] >= own[None, :, 1])
                & (c[:, None, 1] <= own[None, :, 3])
            ).any(dim=1)
            pred = {k: v[~inside] for k, v in pred.items()}
        out.append(pred)
    return out


class ChirpValidator(DetectionValidator):
    """Detection validator that leaves ignore regions out of the metrics.

    Outside training (the final validation of ``best.pt``), ultralytics
    looks the dataset up by name; it gets the trainer's dataset dict
    instead.
    """

    chirp_data: dict[str, Any] | None = None

    def __call__(self, trainer: Any = None, model: Any = None, **kwargs: Any) -> Any:
        """Validate; outside training with the trainer's dataset dict."""
        if trainer is not None or self.chirp_data is None:
            return super().__call__(trainer=trainer, model=model, **kwargs)
        with mock.patch.object(
            ultralytics_validator, "check_det_dataset", return_value=self.chirp_data
        ):
            return super().__call__(trainer=trainer, model=model, **kwargs)

    def update_metrics(
        self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]
    ) -> None:
        """Drop ignored predictions, then count as usual."""
        super().update_metrics(drop_ignored(preds, batch), batch)


# --------------------------------------------------------------- trainer


class ChirpTrainer(DetectionTrainer):
    """ultralytics' detection trainer on a chirpdetector dataset."""

    def __init__(
        self,
        *,
        items: Mapping[str, Sequence[_Item]],
        shape: tuple[int, int],
        time_flip: float,
        overrides: dict[str, Any],
    ) -> None:
        self._chirp_items = dict(items)
        self._chirp_shape = shape
        self._chirp_time_flip = time_flip
        super().__init__(overrides=overrides)

    def get_dataset(self) -> dict[str, Any]:
        """The split names stand in for paths; `build_dataset` resolves them."""
        return {
            "train": "train",
            "val": "val",
            "nc": len(CLASSES),
            "names": _NAMES,
            "channels": 3,
        }

    def build_dataset(
        self, img_path: str, mode: str = "train", batch: int | None = None
    ) -> SpectrogramDataset:
        """A `SpectrogramDataset` of the fit's training or validation images."""
        return SpectrogramDataset(
            self._chirp_items[img_path],
            shape=self._chirp_shape,
            time_flip=self._chirp_time_flip,
            augment=mode == "train",
            hyp=self.args,
            prefix=f"{mode}: ",
            batch_size=batch or self.batch_size,
        )

    def get_model(
        self, cfg: Any = None, weights: Any = None, verbose: bool = True
    ) -> ChirpDetectionModel:
        """A fresh `ChirpDetectionModel`, optionally initialised from weights."""
        model = ChirpDetectionModel(
            cfg, nc=len(CLASSES), ch=3, verbose=verbose and RANK == -1
        )
        model = self.set_model_names_for_load(model)
        if weights:
            model.load(weights)
        return model

    def get_validator(self) -> ChirpValidator:
        """`ChirpValidator` on the fit's validation images."""
        validator = ChirpValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
        validator.chirp_data = self.data
        return validator


# ------------------------------------------------------------------- fit


def ultralytics_device(device: str) -> str:
    """Our device string (``auto``, ``cpu``, ``cuda``, ``cuda:N``, ``N``) for ultralytics."""
    device = device.strip().lower()
    if device == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "0"
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    return device


def _starting_weights(config: YoloModelConfig) -> tuple[str, dict[str, Any]]:
    """The ``model`` argument for ultralytics and a record of where it came from."""
    if config.weights is not None:
        path = Path(config.weights).expanduser().resolve()
        if not path.is_file():
            msg = f"starting weights {path} do not exist"
            raise FileNotFoundError(msg)
        return str(path), {
            "source": "file",
            "path": str(path),
            "sha256": sha256_file(path),
        }
    if config.pretrained:
        cache = weights_cache()
        cache.mkdir(parents=True, exist_ok=True)
        path = Path(attempt_download_asset(str(cache / f"{config.architecture}.pt")))
        return str(path), {
            "source": "ultralytics-coco",
            "path": str(path),
            "sha256": sha256_file(path),
        }
    return f"{config.architecture}.yaml", {
        "source": "random",
        "path": None,
        "sha256": None,
    }


def export_weights(checkpoint: Path, target: Path) -> None:
    """Write a self-contained ultralytics checkpoint of the trained model.

    The model class is set back to ultralytics' ``DetectionModel`` (ours only
    changes the loss), optimizer state is dropped, and the one-to-one head
    is selected for inference.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ckpt.get("model") or ckpt.get("ema")
    if model is None:
        msg = f"{checkpoint} holds no model"
        raise ValueError(msg)
    model.__class__ = DetectionModel
    if hasattr(model, "criterion"):
        model.criterion = None
    model.end2end = True
    torch.save(
        {
            "model": model.half(),
            "train_args": ckpt.get("train_args"),
            "train_metrics": ckpt.get("train_metrics"),
            "date": ckpt.get("date"),
            "version": ckpt.get("version"),
        },
        target,
    )


def _jsonable_metrics(metrics: Mapping[str, Any] | None) -> dict[str, float]:
    out = {}
    for key, value in (metrics or {}).items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out[str(key)] = number
    return out


def fit(request: FitRequest) -> FitOutput:
    """Train one fresh YOLO26 model on the request's images.

    The ultralytics run directory is ``out_dir / "ultralytics"``; the
    exported weights are ``out_dir / "weights.pt"`` and the epoch curves
    ``out_dir / "curves.csv"``.
    """
    cfg = request.config
    model_cfg = YoloModelConfig(**cfg.model)
    tiling = request.dataset.manifest.tiling
    shape = input_shape(tiling.height_px, tiling.width_px)
    if not request.train_ids or not request.val_ids:
        msg = f"{request.name}: needs training and validation images"
        raise ValueError(msg)
    items = {
        "train": _items(request.dataset, request.train_ids),
        "val": _items(request.dataset, request.val_ids),
    }
    device = ultralytics_device(cfg.device)
    vram = apply_vram_limit(cfg.max_vram_fraction, device if device != "cpu" else "cpu")
    model_arg, start = _starting_weights(model_cfg)
    optim, out_dir = cfg.optim, request.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    zero = dict.fromkeys(
        (
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "degrees",
            "translate",
            "scale",
            "shear",
            "perspective",
            "flipud",
            "bgr",
            "mosaic",
            "mixup",
            "cutmix",
            "copy_paste",
            "multi_scale",
        ),
        0.0,
    )
    overrides: dict[str, Any] = {
        "task": "detect",
        "mode": "train",
        "model": model_arg,
        "data": "chirpdetector",
        "pretrained": model_arg.endswith(".pt"),
        "epochs": optim.epochs,
        "batch": optim.batch,
        "imgsz": max(shape),
        "device": device,
        "workers": optim.workers,
        "project": str(out_dir),
        "name": "ultralytics",
        "exist_ok": True,
        "optimizer": optim.optimizer,
        "lr0": optim.lr0,
        "lrf": optim.lrf,
        "cos_lr": optim.schedule == "cosine",
        "warmup_epochs": optim.warmup_epochs,
        "weight_decay": optim.weight_decay,
        "momentum": optim.momentum,
        "patience": optim.patience,
        "amp": optim.amp,
        "seed": cfg.seed,
        "deterministic": cfg.deterministic,
        "single_cls": False,
        "rect": False,
        "cache": False,
        "close_mosaic": 0,
        "fliplr": cfg.augment.time_flip,
        **zero,
        "nms": False,
        "max_det": model_cfg.max_det,
        "plots": False,
        "val": True,
        "save_period": -1,
        "cls_pw": 0.0,
        "verbose": True,
    }
    trainer = ChirpTrainer(
        items=items, shape=shape, time_flip=cfg.augment.time_flip, overrides=overrides
    )
    trainer.train()
    selected = trainer.best if trainer.best.exists() else trainer.last
    weights = out_dir / "weights.pt"
    export_weights(selected, weights)
    curves = None
    if trainer.csv.exists():
        curves = out_dir / "curves.csv"
        shutil.copyfile(trainer.csv, curves)
    best_epoch = int(trainer.stopper.best_epoch) + 1 if trainer.best.exists() else None
    model_info = {
        "architecture": model_cfg.architecture,
        "input_shape": list(shape),
        "pad_value": PAD_VALUE,
        "padding": "bottom-right",
        "channels": "window order (0 = reference window)",
        "head": "one2one",
        "max_det": model_cfg.max_det,
        "ultralytics": ULTRALYTICS_VERSION,
    }
    details = {
        "start_weights": start,
        "selected_checkpoint": selected.name,
        "batch_used": int(trainer.batch_size),
        "batch_requested": optim.batch,
        "device": str(trainer.device),
        "amp_used": bool(trainer.amp),
        "max_vram_fraction": vram,
        "ultralytics_dir": str(trainer.save_dir),
        "n_train_images": len(request.train_ids),
        "n_val_images": len(request.val_ids),
    }
    if details["batch_used"] != optim.batch:
        log.warning(
            "%s: ultralytics reduced the batch from %d to %d after running out of memory",
            request.name,
            optim.batch,
            details["batch_used"],
        )
    return FitOutput(
        weights=weights,
        model=model_info,
        curves=curves,
        best_epoch=best_epoch,
        epochs_run=int(trainer.epoch) + 1,
        metrics=_jsonable_metrics(trainer.metrics),
        details=details,
    )


# ------------------------------------------------------------- inference


class YoloDetector:
    """A bundle's YOLO model, ready for inference (`chirpdetector.models.Detector`)."""

    def __init__(
        self, bundle: Bundle, *, device: str | None = None, batch_size: int = 16
    ) -> None:
        if bundle.model_type != MODEL_TYPE:
            msg = f"bundle model type {bundle.model_type!r} is not {MODEL_TYPE!r}"
            raise ValueError(msg)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        apply_vram_limit(None, self.device)
        model, _ = load_checkpoint(str(bundle.weights_path), device="cpu", fuse=False)
        model = model.float().eval().to(self.device)
        model.end2end = bundle.model.get("head", "one2one") == "one2one"
        self.model = model
        height_in, width_in = (int(v) for v in bundle.model["input_shape"])
        self.shape = (height_in, width_in)
        self.image_shape = (bundle.tiling.height_px, bundle.tiling.width_px)
        self.max_det = int(bundle.model.get("max_det", 300))
        self.batch_size = batch_size
        names = getattr(model, "names", None)
        if names is not None and dict(names) != _NAMES:
            msg = f"model class names {names} are not {_NAMES}"
            raise ValueError(msg)

    def predict(
        self, images: Sequence[NDArray[np.uint8]], *, score_threshold: float
    ) -> list[pd.DataFrame]:
        """Pixel boxes per image (`chirpdetector.models.Detector.predict`)."""
        out: list[pd.DataFrame] = []
        height, width = self.image_shape
        for start in range(0, len(images), self.batch_size):
            chunk = list(images[start : start + self.batch_size])
            for image in chunk:
                if image.shape != (height, width, 3) or image.dtype != np.uint8:
                    msg = (
                        f"expected ({height}, {width}, 3) uint8 images, got "
                        f"{image.dtype} {image.shape}"
                    )
                    raise ValueError(msg)
            x = to_model_input(chunk, self.shape).to(self.device).float() / 255
            with torch.inference_mode():
                raw = self.model(x)
            results = nms.non_max_suppression(
                raw,
                score_threshold,
                0.7,
                max_det=self.max_det,
                nc=0,
                end2end=bool(self.model.end2end),
            )
            for result in results:
                det = cast("torch.Tensor", result).detach().float().cpu().numpy()
                boxes = det[:, :4].astype(np.float64)
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
                frame = pd.DataFrame(boxes, columns=list(PIXEL_COLUMNS[:4]))
                frame["score"] = det[:, 4].astype(np.float64)
                keep = (frame["x_max"] > frame["x_min"]) & (
                    frame["y_max"] > frame["y_min"]
                )
                out.append(frame.loc[keep.to_numpy()].reset_index(drop=True))
        return out
