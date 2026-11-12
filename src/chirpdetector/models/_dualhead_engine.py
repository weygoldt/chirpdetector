"""RT-DETRv2 with an emitter head: model, loss, data, training, inference.

Imported lazily (transformers is slow to import). See
`chirpdetector.models.dualhead` for the design; the pieces here:

Model
    `DualHeadRTDetr`: Hugging Face's ``RTDetrV2ForObjectDetection`` plus one
    MLP per decoder layer on the decoder's query states (the same states the
    class and box heads read). It outputs the emitter's row in normalised
    input coordinates, decoded from the offset relative to the query's
    predicted lower box edge (`chirpdetector.models.dualhead.decode_emitter`;
    the edge is detached, so the emitter loss never moves boxes).
Loss
    `DualHeadLoss`: transformers' ``RTDetrLoss`` (Hungarian matching,
    varifocal loss, L1 + GIoU box losses; on the last decoder layer, every
    auxiliary layer, the encoder's top-k proposals and the denoising groups)
    with two changes:

    * an L1 loss between the predicted and the true emitter row of every
      matched query, in units of the image's frequency span (layers with
      emitter predictions only, i.e. not the encoder proposals);
    * ignore regions: the classification loss of a query that is *not*
      matched to a chirp and whose predicted box centre lies inside an
      ignore region of its image is zero. Such a query may fire or not on an
      invisible chirp or a harmonic copy; matched queries are trained as
      usual. Images are not painted over (see `_yolo_engine`).
Data
    Contract images (`read_image`, window order), zero-padded at the bottom
    and right to the stride; boxes from ``boxes.parquet`` via the image
    geometry; optional time shift (`time_shift`).
Training
    AdamW (backbone at a reduced learning rate), linear warm-up, cosine or
    linear decay, gradient clipping, bf16 autocast on CUDA, an exponential
    moving average of the weights. After every epoch the EMA weights are
    validated with `chirpdetector.evaluation` on the fit's validation
    images (the matching rule of the benchmark, ignore regions as
    ``ignore`` labels): precision, recall and F1 at the config's score
    threshold, AP, the emitter error of the true positives and the
    assignment accuracy (nearest live track to the predicted emitter
    frequency). The fitness is AP x assignment accuracy; the best epoch is
    shipped, and training stops after ``optim.patience`` epochs without
    improvement.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import logging
import math
import random
import shutil
import signal
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn
from transformers import RTDetrV2Config, RTDetrV2ForObjectDetection
from transformers import __version__ as TRANSFORMERS_VERSION
from transformers.loss.loss_for_object_detection import box_iou
from transformers.loss.loss_rt_detr import RTDetrLoss
from transformers.models.rt_detr_v2.modeling_rt_detr_v2 import (
    RTDetrV2MLPPredictionHead,
)

from chirpdetector.data.contract import Dataset, dataset_ground_truth, read_image
from chirpdetector.models import PIXEL_COLUMNS, FitOutput, FitRequest
from chirpdetector.models.dualhead import (
    MODEL_TYPE,
    DualHeadModelConfig,
    input_shape,
)
from chirpdetector.resources import apply_vram_limit
from chirpdetector.spectrogram import (
    ImageGeometry,
    pixel_box_to_timefreq,
    pixel_to_timefreq,
    timefreq_box_to_pixel,
    timefreq_to_pixel,
)

if TYPE_CHECKING:
    from chirpdetector.bundle import Bundle

log = logging.getLogger(__name__)

WEIGHTS_FORMAT = "chirpdetector-dualhead-weights"
WEIGHTS_FORMAT_VERSION = 1
STATE_FORMAT = "chirpdetector-dualhead-training-state-1"
PAD_VALUE = 0
"""Value of the padding pixels: the bottom of the normalised dB range."""
EMITTER_SCALE = 0.1
"""Raw head output x this = emitter offset in units of the image's span."""
EMITTER_INIT = 0.04
"""Initial emitter offset (40 Hz above the lower box edge at 1000 Hz span)."""
MIN_VISIBLE = 0.5
"""A time-shifted chirp keeping less of its width becomes an ignore region."""
VAL_SCORE_FLOOR = 0.05
"""Validation keeps detections down to this score (for AP)."""
VAL_MAX_DET = 50
"""Validation keeps at most this many detections per image (sim tiles hold at
most 19 chirps); early in training most queries pass the score floor, and the
evaluation's cost grows with the number of predictions."""


# ------------------------------------------------------------------- model


def tiny_config() -> RTDetrV2Config:
    """A miniature RT-DETRv2 (0.3 M parameters) for tests."""
    return RTDetrV2Config(
        backbone_config={
            "model_type": "rt_detr_resnet",
            "embedding_size": 8,
            "hidden_sizes": [8, 16, 32, 64],
            "depths": [1, 1, 1, 1],
            "layer_type": "basic",
            "out_features": ["stage2", "stage3", "stage4"],
            "out_indices": [2, 3, 4],
        },
        encoder_in_channels=[16, 32, 64],
        encoder_hidden_dim=32,
        encoder_ffn_dim=64,
        encoder_attention_heads=2,
        d_model=32,
        decoder_in_channels=[32, 32, 32],
        decoder_ffn_dim=64,
        decoder_attention_heads=2,
        decoder_layers=2,
    )


def hf_config(model_cfg: DualHeadModelConfig) -> RTDetrV2Config:
    """The transformers config of a model config (one class, native input size)."""
    from chirpdetector.models.dualhead import ARCHITECTURES

    repository = ARCHITECTURES[model_cfg.architecture]
    if repository is None:
        config = tiny_config()
    else:
        config = cast("RTDetrV2Config", RTDetrV2Config.from_pretrained(repository))
    config.num_labels = 1
    config.id2label = {0: "chirp"}
    config.label2id = {"chirp": 0}
    config.num_queries = model_cfg.num_queries
    config.num_denoising = model_cfg.num_denoising
    # anchors and position embeddings from the actual feature-map sizes
    config.anchor_image_size = None
    config.eval_size = None
    return config


class DualHeadRTDetr(RTDetrV2ForObjectDetection):
    """RT-DETRv2 whose queries also predict their emitter's baseline row.

    ``emitter_unit`` is the image height over the input height: the emitter
    offset is in units of the image's frequency span, rows are normalised by
    the (padded) input height.
    """

    def __init__(self, config: RTDetrV2Config) -> None:
        super().__init__(config)
        self.emitter_embed = nn.ModuleList(
            RTDetrV2MLPPredictionHead(config.d_model, config.d_model, 1, num_layers=3)
            for _ in range(config.decoder_layers)
        )
        self.emitter_unit = 1.0
        self.reset_emitter_head()

    def reset_emitter_head(self) -> None:
        """Start every emitter head at the offset `EMITTER_INIT`."""
        for head in self.emitter_embed:
            last = cast("nn.Linear", cast("Any", head).layers[-1])
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, EMITTER_INIT / EMITTER_SCALE)

    def forward(  # type: ignore[override]
        self,
        pixel_values: torch.Tensor,
        labels: list[dict[str, torch.Tensor]] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Run the detector.

        Returns a dict with, per decoder layer (dim 1) and query (dim 2):
        ``logits`` (B, L, Q, 1), ``boxes`` (B, L, Q, 4; cx, cy, w, h
        normalised by the input size), ``emitter`` (B, L, Q; row normalised
        by the input height); the encoder's top-k proposals
        (``enc_topk_logits``, ``enc_topk_bboxes``) and, in training with
        denoising, ``denoising_meta_values`` (denoising queries come first).
        """
        outputs = self.model(pixel_values, labels=labels)
        hidden = outputs.intermediate_hidden_states  # (B, L, Q, d)
        boxes = outputs.intermediate_reference_points  # (B, L, Q, 4)
        raw = torch.stack(
            [head(hidden[:, i]) for i, head in enumerate(self.emitter_embed)], dim=1
        )[..., 0]
        lower = (boxes[..., 1] + 0.5 * boxes[..., 3]).detach()
        emitter = lower - raw * (EMITTER_SCALE * self.emitter_unit)
        return {
            "logits": outputs.intermediate_logits,
            "boxes": boxes,
            "emitter": emitter,
            "enc_topk_logits": outputs.enc_topk_logits,
            "enc_topk_bboxes": outputs.enc_topk_bboxes,
            "denoising_meta_values": outputs.denoising_meta_values
            if self.training
            else None,
        }


def build_model(
    model_cfg: DualHeadModelConfig, emitter_unit: float
) -> tuple[DualHeadRTDetr, dict[str, Any]]:
    """A model to train (COCO weights, random or a weights file) and its start."""
    config = hf_config(model_cfg)
    if model_cfg.weights is not None:
        return _continue_from(Path(model_cfg.weights), config, emitter_unit)
    repository = model_cfg.repository
    if repository is not None:
        model = DualHeadRTDetr.from_pretrained(
            repository, config=config, ignore_mismatched_sizes=True
        )
        start = {"source": "huggingface-coco", "repository": repository}
    else:
        model = DualHeadRTDetr(config)
        start = {"source": "random", "repository": None}
    model = cast("DualHeadRTDetr", model)
    model.reset_emitter_head()
    model.emitter_unit = float(emitter_unit)
    return model, start


def _as_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    return dict(config) if isinstance(config, Mapping) else dict(config.to_dict())


def _continue_from(
    path: Path, config: RTDetrV2Config, emitter_unit: float
) -> tuple[DualHeadRTDetr, dict[str, Any]]:
    """A model initialised from a weights file of the same architecture."""
    from chirpdetector.data.contract import sha256_file

    path = path.expanduser().resolve()
    if not path.is_file():
        msg = f"starting weights {path} do not exist"
        raise FileNotFoundError(msg)
    model, data = load_weights(path)
    stored = RTDetrV2Config.from_dict(dict(data["hf_config"]))
    keys = ("d_model", "decoder_layers", "encoder_in_channels", "encoder_hidden_dim")
    mismatch = [k for k in keys if getattr(stored, k) != getattr(config, k)]
    stored_backbone = _as_dict(stored.backbone_config)
    wanted_backbone = _as_dict(config.backbone_config)
    for k in ("hidden_sizes", "depths", "layer_type"):
        if stored_backbone.get(k) != wanted_backbone.get(k):
            mismatch.append(f"backbone {k}")
    if mismatch:
        msg = f"{path} is not of the configured architecture ({', '.join(mismatch)})"
        raise ValueError(msg)
    if not math.isclose(model.emitter_unit, emitter_unit, rel_tol=1e-9):
        msg = (
            f"{path} was trained for emitter unit {model.emitter_unit:g} (image / "
            f"input height), the dataset needs {emitter_unit:g}"
        )
        raise ValueError(msg)
    # query counts are runtime settings of the config, not weights
    model.config.num_queries = config.num_queries
    model.config.num_denoising = config.num_denoising
    model.model.config.num_queries = config.num_queries
    model.model.config.num_denoising = config.num_denoising
    return model, {"source": "file", "path": str(path), "sha256": sha256_file(path)}


# -------------------------------------------------------------------- loss


def ignored_queries(
    pred_boxes: torch.Tensor,
    targets: Sequence[Mapping[str, torch.Tensor]],
    matched: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Unmatched queries whose predicted centre lies in an ignore region.

    Parameters
    ----------
    pred_boxes
        ``(B, Q, 4)`` cx, cy, w, h (normalised).
    targets
        Per image, ``ignore_boxes`` ``(m, 4)`` x0, y0, x1, y1 (normalised).
    matched
        ``(batch index, query index)`` of the matched queries.

    Returns
    -------
    torch.Tensor
        Boolean ``(B, Q)``.
    """
    batch, queries = pred_boxes.shape[:2]
    mask = torch.zeros(batch, queries, dtype=torch.bool, device=pred_boxes.device)
    centres = pred_boxes[..., :2].detach()
    for b, target in enumerate(targets):
        regions = target.get("ignore_boxes")
        if regions is None or regions.shape[0] == 0:
            continue
        regions = regions.to(centres)
        x, y = centres[b, :, None, 0], centres[b, :, None, 1]
        inside = (
            (x >= regions[None, :, 0])
            & (x <= regions[None, :, 2])
            & (y >= regions[None, :, 1])
            & (y <= regions[None, :, 3])
        )
        mask[b] = inside.any(dim=1)
    mask[matched[0].to(mask.device), matched[1].to(mask.device)] = False
    return mask


class DualHeadLoss(RTDetrLoss):
    """``RTDetrLoss`` plus the emitter loss and ignore regions (module docstring)."""

    def __init__(
        self, config: RTDetrV2Config, emitter_weight: float, emitter_unit: float
    ) -> None:
        super().__init__(config)
        self.weight_dict["loss_emitter"] = float(emitter_weight)
        self.losses = ["vfl", "boxes", "emitter"]
        self.emitter_unit = float(emitter_unit)

    def get_loss(self, loss, outputs, targets, indices, num_boxes):  # type: ignore[override]
        """Dispatch ``emitter`` here, everything else to ``RTDetrLoss``."""
        if loss == "emitter":
            return self.loss_emitter(outputs, targets, indices, num_boxes)
        return super().get_loss(loss, outputs, targets, indices, num_boxes)

    def loss_emitter(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: Sequence[Mapping[str, torch.Tensor]],
        indices: Sequence[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
    ) -> dict[str, torch.Tensor]:
        """L1 between predicted and true emitter rows of matched queries."""
        if "pred_emitter" not in outputs:
            return {}
        idx = self._get_source_permutation_idx(indices)
        pred = outputs["pred_emitter"][idx].float()
        true = torch.cat(
            [t["emitter_y"][j] for t, (_, j) in zip(targets, indices, strict=True)]
        ).to(pred)
        error = (pred - true).abs() / self.emitter_unit
        return {"loss_emitter": error.sum() / num_boxes}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):  # type: ignore[override]
        """``RTDetrLoss.loss_labels_vfl`` with ignored queries masked out.

        The body follows transformers 5.22 line by line; the only change is
        the ``keep`` factor.
        """
        idx = self._get_source_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices, strict=True)], dim=0
        )
        ious, _ = box_iou(_corners(src_boxes.detach()), _corners(target_boxes))
        ious = torch.diag(ious)
        src_logits = outputs["logits"]
        dtype = src_logits.dtype
        target_classes_original = torch.cat(
            [t["class_labels"][i] for t, (_, i) in zip(targets, indices, strict=True)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_original
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        target_score_original = torch.zeros_like(target_classes, dtype=dtype)
        target_score_original[idx] = ious.to(dtype)
        target_score = target_score_original.unsqueeze(-1) * target
        pred_score = F.sigmoid(src_logits.detach())
        weight = (
            self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        ).to(dtype)
        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        keep = ~ignored_queries(outputs["pred_boxes"], targets, idx)
        loss = loss * keep.unsqueeze(-1).to(loss.dtype)
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}


def _corners(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def compute_loss(
    criterion: DualHeadLoss,
    outputs: Mapping[str, Any],
    targets: Sequence[Mapping[str, torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Total loss and its terms, in float32 (call outside autocast).

    Mirrors transformers' ``RTDetrForObjectDetectionLoss``: denoising
    queries are split off and supervised through their known matches, every
    decoder layer and the encoder's top-k proposals get the matching losses.
    """
    logits = outputs["logits"].float()
    boxes = outputs["boxes"].float()
    emitter = outputs["emitter"].float()
    meta = outputs.get("denoising_meta_values")
    denoising: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    if meta is not None:
        split = meta["dn_num_split"]
        dn_logits, logits = torch.split(logits, split, dim=2)
        dn_boxes, boxes = torch.split(boxes, split, dim=2)
        dn_emitter, emitter = torch.split(emitter, split, dim=2)
        denoising = (dn_logits, dn_boxes, dn_emitter)
    n_layers = logits.shape[1]

    def layer(i: int, lg: torch.Tensor, bx: torch.Tensor, em: torch.Tensor) -> dict:
        return {"logits": lg[:, i], "pred_boxes": bx[:, i], "pred_emitter": em[:, i]}

    loss_inputs: dict[str, Any] = layer(n_layers - 1, logits, boxes, emitter)
    auxiliary = [layer(i, logits, boxes, emitter) for i in range(n_layers - 1)]
    auxiliary.append(
        {
            "logits": outputs["enc_topk_logits"].float(),
            "pred_boxes": outputs["enc_topk_bboxes"].float(),
        }
    )
    loss_inputs["auxiliary_outputs"] = auxiliary
    if denoising is not None:
        loss_inputs["dn_auxiliary_outputs"] = [
            layer(i, *denoising) for i in range(n_layers)
        ]
        loss_inputs["denoising_meta_values"] = meta
    terms = criterion(loss_inputs, list(targets))
    return cast("torch.Tensor", sum(terms.values())), terms


def loss_summary(terms: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Loss terms summed over layers: ``vfl``, ``bbox``, ``giou``, ``emitter``."""
    out: dict[str, float] = dict.fromkeys(("vfl", "bbox", "giou", "emitter"), 0.0)
    for key, value in terms.items():
        for name in out:
            if key.startswith(f"loss_{name}"):
                out[name] += float(value.detach())
    return out


# -------------------------------------------------------------------- data


@dataclass(frozen=True)
class Item:
    """One image with pixel boxes ``(x0, y0, x1, y1)`` and emitter rows."""

    path: Path
    boxes: NDArray[np.float32]
    emitter_y: NDArray[np.float32]
    ignore: NDArray[np.float32]


def _pixel_boxes(rows: pd.DataFrame, geometry: ImageGeometry) -> NDArray[np.float64]:
    if len(rows) == 0:
        return np.zeros((0, 4))
    x0, y0, x1, y1 = timefreq_box_to_pixel(
        rows["t_start_s"].to_numpy(np.float64),
        rows["t_end_s"].to_numpy(np.float64),
        rows["f_low_hz"].to_numpy(np.float64),
        rows["f_high_hz"].to_numpy(np.float64),
        geometry,
    )
    return np.stack([x0, y0, x1, y1], axis=1).reshape(-1, 4)


def make_items(dataset: Dataset, image_ids: Sequence[str]) -> list[Item]:
    """Boxes, emitter rows and ignore regions of images, in pixels."""
    out = []
    for image_id in image_ids:
        geometry = dataset.manifest.image(image_id).geometry
        chirps = dataset.boxes_of(image_id, "chirp")
        _, emitter = timefreq_to_pixel(
            chirps["t_center_s"].to_numpy(np.float64),
            chirps["emitter_hz"].to_numpy(np.float64),
            geometry,
        )
        out.append(
            Item(
                path=dataset.image_path(image_id),
                boxes=_pixel_boxes(chirps, geometry).astype(np.float32),
                emitter_y=np.asarray(emitter, np.float32).reshape(-1),
                ignore=_pixel_boxes(
                    dataset.boxes_of(image_id, "ignore"), geometry
                ).astype(np.float32),
            )
        )
    return out


def time_shift(
    image: NDArray[np.uint8],
    boxes: NDArray[np.float32],
    emitter_y: NDArray[np.float32],
    ignore: NDArray[np.float32],
    shift: int,
) -> tuple[
    NDArray[np.uint8], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]
]:
    """Move an image's content by ``shift`` columns (> 0: later).

    The vacated columns are `PAD_VALUE`. Boxes move with the content and
    are clipped to the image; a chirp that keeps less than `MIN_VISIBLE` of
    its width becomes an ignore region (as chirps cut by tile edges are in
    the datasets). Rows, and thus frequencies, are untouched.
    """
    width = image.shape[1]
    shift = int(shift)
    out = np.full_like(image, PAD_VALUE)
    if shift >= 0:
        out[:, shift:] = image[:, : width - shift]
    else:
        out[:, :shift] = image[:, -shift:]

    def move(b: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
        moved = b.copy()
        moved[:, [0, 2]] = np.clip(moved[:, [0, 2]] + shift, 0, width)
        full = np.maximum(b[:, 2] - b[:, 0], 1e-9)
        return moved, (moved[:, 2] - moved[:, 0]) / full

    chirps, visible = move(boxes)
    regions, region_visible = move(ignore)
    keep = visible >= MIN_VISIBLE
    lost = (~keep) & (visible > 0)
    regions = np.concatenate([regions[region_visible > 0], chirps[lost]])
    return (
        out,
        chirps[keep],
        emitter_y[keep],
        regions.astype(np.float32).reshape(-1, 4),
    )


def to_model_input(
    images: Sequence[NDArray[np.uint8]], shape: tuple[int, int]
) -> torch.Tensor:
    """Images in window order -> ``(n, 3, H_in, W_in)`` uint8, zero-padded."""
    height, width = shape
    out = np.full((len(images), 3, height, width), PAD_VALUE, dtype=np.uint8)
    for k, image in enumerate(images):
        h, w = image.shape[:2]
        if image.ndim != 3 or image.shape[2] != 3 or h > height or w > width:
            msg = f"image of shape {image.shape} does not fit the input {shape}"
            raise ValueError(msg)
        out[k, :, :h, :w] = image.transpose(2, 0, 1)
    return torch.from_numpy(out)


def make_target(
    boxes: NDArray[np.float32],
    emitter_y: NDArray[np.float32],
    ignore: NDArray[np.float32],
    shape: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Pixel labels -> the loss's target dict (normalised by the input size)."""
    height, width = shape
    scale = np.array([width, height, width, height], np.float32)
    b = boxes.reshape(-1, 4) / scale
    centre = np.stack(
        [
            (b[:, 0] + b[:, 2]) / 2,
            (b[:, 1] + b[:, 3]) / 2,
            b[:, 2] - b[:, 0],
            b[:, 3] - b[:, 1],
        ],
        axis=1,
    )
    return {
        "class_labels": torch.zeros(len(b), dtype=torch.int64),
        "boxes": torch.from_numpy(centre.astype(np.float32)).reshape(-1, 4),
        "emitter_y": torch.from_numpy(
            (emitter_y.reshape(-1) / height).astype(np.float32)
        ),
        "ignore_boxes": torch.from_numpy(
            (ignore.reshape(-1, 4) / scale).astype(np.float32)
        ),
    }


class TileDataset(torch.utils.data.Dataset):
    """Training or validation images and their targets."""

    def __init__(
        self, items: Sequence[Item], shape: tuple[int, int], time_shift_px: int = 0
    ) -> None:
        self.items = list(items)
        self.shape = shape
        self.time_shift_px = int(time_shift_px)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, k: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        item = self.items[k]
        image = read_image(item.path)
        boxes, emitter, ignore = item.boxes, item.emitter_y, item.ignore
        if self.time_shift_px > 0:
            shift = int(torch.randint(-self.time_shift_px, self.time_shift_px + 1, ()))
            if shift:
                image, boxes, emitter, ignore = time_shift(
                    image, boxes, emitter, ignore, shift
                )
        x = to_model_input([image], self.shape)[0]
        return x, make_target(boxes, emitter, ignore, self.shape)


def collate(
    batch: Sequence[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    """Stack images; keep one target dict per image."""
    return torch.stack([x for x, _ in batch]), [t for _, t in batch]


# --------------------------------------------------------------- inference


def postprocess(
    outputs: Mapping[str, torch.Tensor],
    *,
    input_hw: tuple[int, int],
    image_hw: tuple[int, int],
    score_threshold: float,
    max_det: int,
) -> list[pd.DataFrame]:
    """Last-layer predictions -> pixel tables (`PIXEL_COLUMNS` + ``emitter_y``).

    The highest-scoring ``max_det`` queries with ``score >=
    score_threshold``; boxes and emitter rows clipped to the image (not the
    padded input): the emitter's baseline lies in the image's band.
    """
    height_in, width_in = input_hw
    height, width = image_hw
    scores = outputs["logits"][:, -1, :, 0].float().sigmoid()
    boxes = outputs["boxes"][:, -1].float()
    emitter = outputs["emitter"][:, -1].float()
    k = min(max_det, scores.shape[1])
    top, index = scores.topk(k, dim=1)
    tables = []
    for b in range(scores.shape[0]):
        keep = top[b] >= score_threshold
        q = index[b][keep]
        cx, cy, w, h = boxes[b, q].unbind(-1)
        frame = pd.DataFrame(
            {
                "x_min": ((cx - w / 2) * width_in).clamp(0, width).cpu().numpy(),
                "y_min": ((cy - h / 2) * height_in).clamp(0, height).cpu().numpy(),
                "x_max": ((cx + w / 2) * width_in).clamp(0, width).cpu().numpy(),
                "y_max": ((cy + h / 2) * height_in).clamp(0, height).cpu().numpy(),
                "score": top[b][keep].cpu().numpy(),
                "emitter_y": (emitter[b, q] * height_in).clamp(0, height).cpu().numpy(),
            }
        ).astype(np.float64)
        valid = (frame["x_max"] > frame["x_min"]) & (frame["y_max"] > frame["y_min"])
        tables.append(frame.loc[valid.to_numpy()].reset_index(drop=True))
    return tables


def resolve_device(device: str | None) -> torch.device:
    """``auto``/None, ``cpu``, ``cuda``, ``cuda:N`` or ``N`` -> a torch device."""
    name = (device or "auto").strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.isdigit():
        return torch.device("cuda", int(name))
    return torch.device(name)


def _autocast(device: torch.device, enabled: bool) -> Any:
    use = enabled and device.type == "cuda"
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=use)


def load_weights(path: str | Path) -> tuple[DualHeadRTDetr, dict[str, Any]]:
    """Rebuild a model from an exported weights file (no network access)."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format") != WEIGHTS_FORMAT:
        msg = f"{path} is not a {WEIGHTS_FORMAT} file"
        raise ValueError(msg)
    if data.get("format_version") != WEIGHTS_FORMAT_VERSION:
        msg = f"{path}: weights format version {data.get('format_version')!r}"
        raise ValueError(msg)
    config = RTDetrV2Config.from_dict(dict(data["hf_config"]))
    model = DualHeadRTDetr(config)
    model.load_state_dict(data["state_dict"], strict=True)
    model.emitter_unit = float(data["emitter_unit"])
    return model, data


def export_weights(model: DualHeadRTDetr, path: Path) -> None:
    """Write a self-contained weights file (config + float32 state dict)."""
    torch.save(
        {
            "format": WEIGHTS_FORMAT,
            "format_version": WEIGHTS_FORMAT_VERSION,
            "hf_config": model.config.to_dict(),
            "emitter_unit": float(model.emitter_unit),
            "state_dict": {
                k: v.detach().float().cpu().clone()
                if v.is_floating_point()
                else v.cpu()
                for k, v in model.state_dict().items()
            },
            "transformers": TRANSFORMERS_VERSION,
        },
        path,
    )


class DualHeadDetector:
    """A bundle's dual-head model, ready for inference (`Detector`)."""

    def __init__(
        self, bundle: Bundle, *, device: str | None = None, batch_size: int = 16
    ) -> None:
        if bundle.model_type != MODEL_TYPE:
            msg = f"bundle model type {bundle.model_type!r} is not {MODEL_TYPE!r}"
            raise ValueError(msg)
        self.device = resolve_device(device)
        apply_vram_limit(None, self.device)
        model, _ = load_weights(bundle.weights_path)
        self.model = cast("DualHeadRTDetr", model.eval().to(self.device))  # type: ignore[arg-type]
        height_in, width_in = (int(v) for v in bundle.model["input_shape"])
        self.input_hw = (height_in, width_in)
        self.image_hw = (bundle.tiling.height_px, bundle.tiling.width_px)
        self.max_det = int(bundle.model.get("max_det", 300))
        self.batch_size = batch_size

    def predict(
        self, images: Sequence[NDArray[np.uint8]], *, score_threshold: float
    ) -> list[pd.DataFrame]:
        """Pixel boxes and emitter rows per image (`Detector.predict`)."""
        height, width = self.image_hw
        out: list[pd.DataFrame] = []
        for start in range(0, len(images), self.batch_size):
            chunk = list(images[start : start + self.batch_size])
            for image in chunk:
                if image.shape != (height, width, 3) or image.dtype != np.uint8:
                    msg = (
                        f"expected ({height}, {width}, 3) uint8 images, got "
                        f"{image.dtype} {image.shape}"
                    )
                    raise ValueError(msg)
            x = to_model_input(chunk, self.input_hw).to(self.device).float() / 255
            with torch.inference_mode():
                outputs = self.model(x)
            out += postprocess(
                outputs,
                input_hw=self.input_hw,
                image_hw=self.image_hw,
                score_threshold=score_threshold,
                max_det=self.max_det,
            )
        return out


# -------------------------------------------------------------- validation


def _prediction_table(
    tables: Sequence[pd.DataFrame],
    geometries: Sequence[ImageGeometry],
    recordings: Sequence[str],
) -> pd.DataFrame:
    """Pixel tables -> a prediction table in time and frequency."""
    parts = []
    for table, g, name in zip(tables, geometries, recordings, strict=True):
        xyxy = table.loc[:, list(PIXEL_COLUMNS[:4])].to_numpy(np.float64)
        t0, t1, f0, f1 = pixel_box_to_timefreq(
            xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3], g
        )
        _, emitter = pixel_to_timefreq(0.0, table["emitter_y"].to_numpy(np.float64), g)
        parts.append(
            pd.DataFrame(
                {
                    "recording": name,
                    "t_start_s": t0,
                    "t_end_s": t1,
                    "f_low_hz": f0,
                    "f_high_hz": f1,
                    "score": table["score"].to_numpy(np.float64),
                    "emitter_hz": emitter,
                }
            )
        )
    columns = ["recording", "t_start_s", "t_end_s", "f_low_hz", "f_high_hz", "score"]
    if not parts:
        return pd.DataFrame(
            {c: pd.Series(dtype=float) for c in [*columns, "emitter_hz"]}
        )
    return pd.concat(parts, ignore_index=True)


def predict_items(
    model: DualHeadRTDetr,
    dataset: Dataset,
    image_ids: Sequence[str],
    *,
    shape: tuple[int, int],
    device: torch.device,
    batch_size: int,
    workers: int,
    amp: bool,
    max_det: int,
    score_threshold: float = VAL_SCORE_FLOOR,
    should_stop: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Predictions of a model on dataset images (``recording`` = image id).

    ``should_stop`` is called before every batch (it raises to abort).
    """
    manifest = dataset.manifest
    tiling = manifest.tiling
    items = make_items(dataset, image_ids)
    loader = torch.utils.data.DataLoader(
        TileDataset(items, shape),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate,
        worker_init_fn=worker_init if workers > 0 else None,
    )
    tables: list[pd.DataFrame] = []
    model.eval()
    with torch.inference_mode():
        for x, _ in loader:
            if should_stop is not None:
                should_stop()
            x = x.to(device, non_blocking=True).float() / 255
            with _autocast(device, amp):
                outputs = model(x)
            tables += postprocess(
                outputs,
                input_hw=shape,
                image_hw=(tiling.height_px, tiling.width_px),
                score_threshold=score_threshold,
                max_det=max_det,
            )
    geometries = [manifest.image(i).geometry for i in image_ids]
    return _prediction_table(tables, geometries, list(image_ids))


def emitter_metrics(
    predictions: pd.DataFrame,
    items: pd.DataFrame,
    dataset: Dataset,
) -> dict[str, float]:
    """Emitter error and nearest-live-track accuracy of the true positives.

    ``items`` is ``EvaluationResult.items``; the live tracks of a chirp come
    from ``live_tracks_hz`` of its box (the emitter is one of them).
    """
    tp = items[(items["outcome"] == "tp") & (items["label"] == "chirp")]
    if len(tp) == 0:
        return {"emitter_mae_hz": math.nan, "assign_accuracy": math.nan, "n_tp": 0.0}
    pred_rows = np.asarray(tp["pred_row"], dtype=np.int64)
    pred_hz = np.asarray(predictions["emitter_hz"], dtype=np.float64)[pred_rows]
    true_hz = np.asarray(tp["true_emitter_hz"], dtype=np.float64)
    live_of: dict[tuple[str, float], NDArray[np.float64]] = {}
    for image_id in sorted(set(tp["recording"].astype(str))):
        rows = dataset.boxes_of(image_id, "chirp")
        for hz, live in zip(rows["emitter_hz"], rows["live_tracks_hz"], strict=True):
            if live is not None:
                live_of[(image_id, float(hz))] = np.asarray(live, np.float64)
    correct, known = 0, 0
    for name, p, t in zip(tp["recording"].astype(str), pred_hz, true_hz, strict=True):
        live = live_of.get((name, float(t)))
        if live is None or len(live) == 0 or not np.isfinite(p):
            continue
        known += 1
        correct += int(np.argmin(np.abs(live - p)) == np.argmin(np.abs(live - t)))
    return {
        "emitter_mae_hz": float(np.nanmean(np.abs(pred_hz - true_hz))),
        "emitter_median_ae_hz": float(np.nanmedian(np.abs(pred_hz - true_hz))),
        "assign_accuracy": correct / known if known else math.nan,
        "n_tp": float(len(tp)),
    }


def _number(value: Any) -> float:
    """``value`` as a float; NaN for None or anything not a number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def validate(
    model: DualHeadRTDetr,
    dataset: Dataset,
    image_ids: Sequence[str],
    *,
    shape: tuple[int, int],
    device: torch.device,
    batch_size: int,
    workers: int,
    amp: bool,
    max_det: int,
    score_threshold: float,
    should_stop: Callable[[], None] | None = None,
) -> dict[str, float]:
    """Detection and emitter metrics on validation images (module docstring)."""
    from chirpdetector.evaluation import EvaluationConfig, evaluate

    predictions = predict_items(
        model,
        dataset,
        image_ids,
        shape=shape,
        device=device,
        batch_size=batch_size,
        workers=workers,
        amp=amp,
        max_det=min(max_det, VAL_MAX_DET),
        should_stop=should_stop,
    )
    truth = dataset_ground_truth(dataset, image_ids)
    result = evaluate(
        predictions,
        truth,
        config=EvaluationConfig(
            thresholds=(score_threshold,),
            reference_threshold=score_threshold,
            n_bootstrap=0,
        ),
    )
    detection = result.metrics()["detection"].get("all", {})
    # undefined rates (e.g. precision without detections) are null in metrics
    out = {
        name: _number(detection.get(key))
        for name, key in (
            ("precision", "precision"),
            ("recall", "recall"),
            ("f1", "f1"),
            ("ap", "average_precision"),
        )
    }
    items = result.items
    if "true_emitter_hz" not in items.columns:
        label_rows = items["label_row"].astype("Int64")
        hz = truth.labels["emitter_hz"].to_numpy(np.float64)
        items = items.assign(
            true_emitter_hz=[
                hz[int(r)] if r is not pd.NA else math.nan for r in label_rows
            ]
        )
    out |= emitter_metrics(predictions, items, dataset)
    ap = out["ap"] if math.isfinite(out["ap"]) else 0.0
    accuracy = out["assign_accuracy"] if math.isfinite(out["assign_accuracy"]) else 0.0
    out["fitness"] = ap * accuracy
    return out


# ---------------------------------------------------------------- training


class ModelEMA:
    """Exponential moving average of a model's weights (ramped-up decay)."""

    def __init__(self, model: nn.Module, decay: float, tau: float = 2000.0) -> None:
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the model's current weights in."""
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        current = model.state_dict()
        for name, value in self.ema.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(d).add_(current[name].detach(), alpha=1 - d)
            else:
                value.copy_(current[name])


def _param_groups(
    model: nn.Module, lr: float, backbone_scale: float, weight_decay: float
) -> list[dict[str, Any]]:
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        backbone = name.startswith("model.backbone")
        decay = p.ndim > 1
        groups.setdefault((backbone, decay), []).append(p)
    out = []
    for (backbone, decay), params in sorted(groups.items()):
        out.append(
            {
                "params": params,
                "lr": lr * (backbone_scale if backbone else 1.0),
                "weight_decay": weight_decay if decay else 0.0,
                "name": f"{'backbone' if backbone else 'head'}{'' if decay else '_nodecay'}",
            }
        )
    return out


def _optimizer(name: str, groups: list[dict[str, Any]], momentum: float) -> Any:
    if name in {"AdamW", "auto"}:
        return torch.optim.AdamW(groups)
    if name == "Adam":
        return torch.optim.Adam(groups)
    if name == "SGD":
        return torch.optim.SGD(groups, momentum=momentum, nesterov=True)
    msg = f"optimizer {name!r} is not supported by {MODEL_TYPE}; use AdamW, Adam or SGD"
    raise ValueError(msg)


def lr_factor(step: int, total: int, warmup: int, lrf: float, schedule: str) -> float:
    """Learning-rate multiplier: linear warm-up, then cosine or linear decay to ``lrf``."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    if schedule == "cosine":
        return lrf + (1 - lrf) * 0.5 * (1 + math.cos(math.pi * progress))
    return 1 - (1 - lrf) * progress


def _seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - global seed for third-party code
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # grid_sample (deformable attention) has no deterministic backward
        torch.use_deterministic_algorithms(True, warn_only=True)


def _subset(ids: Sequence[str], n: int) -> list[str]:
    if n <= 0 or n >= len(ids):
        return list(ids)
    index = np.linspace(0, len(ids) - 1, n).round().astype(int)
    return [str(ids[int(i)]) for i in index]


class Stop(Exception):
    """Raised inside a fit when a stop signal arrived (see `StopSignals`)."""


class StopSignals:
    """Turn SIGTERM and SIGINT into a stop request for the duration of a fit.

    ``systemctl stop`` (the GPU watchdog, a person stopping a guarded job)
    sends SIGTERM to every process of the job. The fit then finishes its
    current step, skips the unfinished epoch and ships the best validated
    checkpoint, so the run's bundle is still written. Outside the main
    thread signals cannot be caught and nothing is installed.
    """

    SIGNALS = (signal.SIGTERM, signal.SIGINT)

    def __init__(self) -> None:
        self.received: str | None = None
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> StopSignals:
        if threading.current_thread() is threading.main_thread():
            for sig in self.SIGNALS:
                self._previous[sig] = signal.signal(sig, self._handle)
        return self

    def __exit__(self, *_: object) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        self._previous.clear()

    def _handle(self, signum: int, _frame: object) -> None:
        first = self.received is None
        self.received = signal.Signals(signum).name
        if not first:
            return
        log.warning("%s received: stopping after this step", self.received)
        # The job's loader workers get the signal too and may die (e.g. while
        # starting, before `worker_init`). torch's SIGCHLD handler would raise
        # about that at any point; from now on worker deaths surface only in
        # the loader call, which the fit treats as the stop. (If the reset
        # races with a pending SIGCHLD, the fit treats that error as the stop.)
        if signal.SIGCHLD not in self._previous:
            try:
                self._previous[signal.SIGCHLD] = signal.signal(
                    signal.SIGCHLD, signal.SIG_DFL
                )
            except OSError:
                log.debug("SIGCHLD handler not reset", exc_info=True)

    def check(self) -> None:
        """Raise `Stop` if a stop signal arrived."""
        if self.received is not None:
            raise Stop(self.received)


def worker_init(_: int) -> None:
    """Data-loader workers leave SIGTERM and SIGINT to the main process."""
    for sig in StopSignals.SIGNALS:
        signal.signal(sig, signal.SIG_IGN)


def _ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def _comparable(config: Mapping[str, Any]) -> dict[str, Any]:
    """A resolved training config without what a resumed fit may change."""
    out = copy.deepcopy(dict(config))
    out["model"] = {k: v for k, v in out["model"].items() if k != "resume"}
    out["optim"] = {k: v for k, v in out["optim"].items() if k != "workers"}
    out.pop("device", None)
    return out


def _save_atomic(data: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


@dataclass
class _Progress:
    """Everything a fit needs to continue after its last validated epoch."""

    epoch: int = 0
    step: int = 0
    best_fitness: float = -math.inf
    best_epoch: int | None = None
    best_metrics: dict[str, float] | None = None
    stale: int = 0
    rows: list[dict[str, Any]] | None = None


def _load_state(
    path: Path,
    request: FitRequest,
    model: DualHeadRTDetr,
    ema: ModelEMA | None,
    optimizer: Any,
    generator: torch.Generator,
    work: Path,
) -> tuple[_Progress, dict[str, Any]]:
    """Restore an interrupted fit's state (`fit`); return its progress."""
    from chirpdetector.data.contract import sha256_file

    path = path.expanduser().resolve()
    if not path.is_file():
        msg = f"training state {path} does not exist"
        raise FileNotFoundError(msg)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != STATE_FORMAT:
        msg = f"{path} is not a {STATE_FORMAT} file"
        raise ValueError(msg)
    if _comparable(state["config"]) != _comparable(request.config.to_dict()):
        msg = f"{path} was made with another training config; resume needs the same one"
        raise ValueError(msg)
    for name, ids in (("train", request.train_ids), ("val", request.val_ids)):
        if state[f"{name}_ids_sha256"] != _ids_sha256(ids):
            msg = f"{path} was trained on other {name} images"
            raise ValueError(msg)
    if state["dataset_sha256"] != request.dataset.manifest.content_sha256:
        msg = f"{path} was trained on another dataset"
        raise ValueError(msg)
    model.load_state_dict(state["model"], strict=True)
    model.emitter_unit = float(state["emitter_unit"])
    if ema is not None:
        ema.ema.load_state_dict(state["ema"], strict=True)
        ema.updates = int(state["ema_updates"])
    optimizer.load_state_dict(state["optimizer"])
    generator.set_state(state["rng"]["generator"])
    torch.set_rng_state(state["rng"]["torch"])
    np.random.set_state(state["rng"]["numpy"])  # noqa: NPY002 - restored global state
    random.setstate(state["rng"]["python"])
    best = path.parent / "best.pt"
    if best.is_file():
        shutil.copyfile(best, work / "best.pt")
    progress = _Progress(
        epoch=int(state["epoch"]),
        step=int(state["step"]),
        best_fitness=float(state["best_fitness"]),
        best_epoch=state["best_epoch"],
        best_metrics=dict(state["best_metrics"] or {}),
        stale=int(state["stale"]),
        rows=list(state["rows"]),
    )
    record = {
        "path": str(path),
        "sha256": sha256_file(path),
        "epoch": progress.epoch,
        "step": progress.step,
    }
    return progress, {**record, "start": state.get("start")}


def fit(request: FitRequest) -> FitOutput:
    """Train one dual-head model on the request's images.

    Writes, in ``out_dir``: ``curves.csv`` (one row per epoch),
    ``dualhead/best.pt`` and ``dualhead/last.pt`` (exported weights of the
    best and the last validated epoch), ``dualhead/state.pt`` (everything
    needed to continue after the last validated epoch: ``[model] resume``)
    and the shipped ``weights.pt`` (= best). A stop signal ends the fit
    early with the best checkpoint (`StopSignals`).
    """
    cfg = request.config
    model_cfg = DualHeadModelConfig(**cfg.model)
    optim = cfg.optim
    if not request.train_ids or not request.val_ids:
        msg = f"{request.name}: needs training and validation images"
        raise ValueError(msg)
    tiling = request.dataset.manifest.tiling
    image_hw = (tiling.height_px, tiling.width_px)
    shape = input_shape(*image_hw)
    device = resolve_device(cfg.device)
    vram = apply_vram_limit(cfg.max_vram_fraction, device)
    _seed(cfg.seed, cfg.deterministic)
    out_dir = request.out_dir
    work = out_dir / "dualhead"
    work.mkdir(parents=True, exist_ok=True)

    unit = image_hw[0] / shape[0]
    if model_cfg.resume is not None:
        # architecture only; the weights come from the state
        model = DualHeadRTDetr(hf_config(model_cfg))
        start: dict[str, Any] = {}
    else:
        model, start = build_model(model_cfg, emitter_unit=unit)
    model = cast("DualHeadRTDetr", model.to(device))  # type: ignore[arg-type]
    criterion = DualHeadLoss(model.config, model_cfg.emitter_loss_weight, unit).to(
        device
    )
    ema = ModelEMA(model, model_cfg.ema_decay) if model_cfg.ema_decay > 0 else None
    groups = _param_groups(
        model, optim.lr0, model_cfg.backbone_lr_scale, optim.weight_decay
    )
    optimizer = _optimizer(optim.optimizer, groups, optim.momentum)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    generator = torch.Generator().manual_seed(cfg.seed)
    progress, resumed = _Progress(rows=[]), None
    if model_cfg.resume is not None:
        progress, resumed = _load_state(
            Path(model_cfg.resume), request, model, ema, optimizer, generator, work
        )
        start = dict(resumed.pop("start") or {})
        model.to(device)  # type: ignore[arg-type]
        if ema is not None:
            ema.ema.to(device)
        log.info(
            "%s: resuming after epoch %d (step %d) from %s",
            request.name,
            progress.epoch,
            progress.step,
            resumed["path"],
        )

    log.info("%s: indexing %d training images", request.name, len(request.train_ids))
    train_items = make_items(request.dataset, request.train_ids)
    loader = torch.utils.data.DataLoader(
        TileDataset(train_items, shape, model_cfg.time_shift_px),
        batch_size=optim.batch,
        shuffle=True,
        drop_last=len(train_items) > optim.batch,
        num_workers=optim.workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init if optim.workers > 0 else None,
        generator=generator,
    )
    val_ids = _subset(request.val_ids, model_cfg.val_max_images)
    steps = len(loader)
    total = optim.epochs * steps
    warmup = round(optim.warmup_epochs * steps)
    curves = out_dir / "curves.csv"
    best_path, last_path = work / "best.pt", work / "last.pt"
    state_path = work / "state.pt"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log.info(
        "%s: %d train / %d val images, batch %d, %d steps per epoch, device %s, "
        "VRAM cap %s",
        request.name,
        len(train_items),
        len(val_ids),
        optim.batch,
        steps,
        device,
        vram,
    )
    rows = list(progress.rows or [])
    step, stopped = progress.step, None
    with StopSignals() as stop:
        for epoch in range(progress.epoch + 1, optim.epochs + 1):
            if optim.patience and progress.stale >= optim.patience:
                break
            model.train()
            t0 = time.time()
            sums: dict[str, float] = dict.fromkeys(
                ("loss", "vfl", "bbox", "giou", "emitter"), 0.0
            )
            iterator = iter(loader)
            try:
                for k in range(steps):
                    stop.check()
                    x, targets = next(iterator)
                    factor = lr_factor(step, total, warmup, optim.lrf, optim.schedule)
                    for group, base in zip(
                        optimizer.param_groups, base_lrs, strict=True
                    ):
                        group["lr"] = base * factor
                    x = x.to(device, non_blocking=True).float() / 255
                    targets = [
                        {key: v.to(device) for key, v in t.items()} for t in targets
                    ]
                    with _autocast(device, optim.amp):
                        outputs = model(x, labels=targets)
                    loss, terms = compute_loss(criterion, outputs, targets)
                    if not torch.isfinite(loss):
                        msg = f"{request.name}: loss is {float(loss)} at step {step}"
                        raise FloatingPointError(msg)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if model_cfg.grad_clip > 0:
                        nn.utils.clip_grad_norm_(
                            model.parameters(), model_cfg.grad_clip
                        )
                    optimizer.step()
                    if ema is not None:
                        ema.update(model)
                    step += 1
                    sums["loss"] += float(loss.detach())
                    for name, value in loss_summary(terms).items():
                        sums[name] += value
                    if (k + 1) % 200 == 0:
                        log.info(
                            "%s epoch %d step %d/%d loss %.3f (%.0f img/s)",
                            request.name,
                            epoch,
                            k + 1,
                            steps,
                            sums["loss"] / (k + 1),
                            (k + 1) * optim.batch / (time.time() - t0),
                        )
                train_s = time.time() - t0
                evaluated = cast(
                    "DualHeadRTDetr", ema.ema if ema is not None else model
                )
                metrics = validate(
                    evaluated,
                    request.dataset,
                    val_ids,
                    shape=shape,
                    device=device,
                    batch_size=max(optim.batch, 16),
                    workers=optim.workers,
                    amp=optim.amp,
                    max_det=model_cfg.max_det,
                    score_threshold=cfg.score_threshold,
                    should_stop=stop.check,
                )
            except Exception as error:
                if not (isinstance(error, Stop) or stop.received):
                    raise
                # a stop, or a loader whose workers the stop signal ended
                signal_name = stop.received or str(error)
                stopped = {"signal": signal_name, "epoch": epoch, "step": step}
                log.warning(
                    "%s: stopped by %s in epoch %d; shipping the best checkpoint "
                    "(epoch %s)",
                    request.name,
                    signal_name,
                    epoch,
                    progress.best_epoch,
                )
                break
            finally:
                del iterator
            export_weights(evaluated, last_path)
            improved = metrics["fitness"] > progress.best_fitness
            if improved:
                progress.best_fitness = metrics["fitness"]
                progress.best_epoch = epoch
                progress.best_metrics = metrics
                export_weights(evaluated, best_path)
                progress.stale = 0
            else:
                progress.stale += 1
            rows.append(
                {
                    "epoch": epoch,
                    **{f"train/{k}": v / max(1, steps) for k, v in sums.items()},
                    **{f"val/{k}": v for k, v in metrics.items()},
                    **{f"lr/{g['name']}": g["lr"] for g in optimizer.param_groups},
                    "time/train_s": train_s,
                    "time/val_s": time.time() - t0 - train_s,
                    "gpu/peak_gib": torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else 0.0,
                }
            )
            _write_curves(rows, curves)
            progress.epoch, progress.step = epoch, step
            _save_atomic(
                {
                    "format": STATE_FORMAT,
                    "config": request.config.to_dict(),
                    "dataset_sha256": request.dataset.manifest.content_sha256,
                    "train_ids_sha256": _ids_sha256(request.train_ids),
                    "val_ids_sha256": _ids_sha256(request.val_ids),
                    "epoch": epoch,
                    "step": step,
                    "model": model.state_dict(),
                    "emitter_unit": float(model.emitter_unit),
                    "ema": None if ema is None else ema.ema.state_dict(),
                    "ema_updates": 0 if ema is None else ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "best_fitness": progress.best_fitness,
                    "best_epoch": progress.best_epoch,
                    "best_metrics": progress.best_metrics,
                    "stale": progress.stale,
                    "rows": rows,
                    "start": start,
                    "rng": {
                        "generator": generator.get_state(),
                        "torch": torch.get_rng_state(),
                        "numpy": np.random.get_state(),  # noqa: NPY002
                        "python": random.getstate(),
                    },
                },
                state_path,
            )
            log.info(
                "%s epoch %d: loss %.3f | P %.3f R %.3f F1 %.3f AP %.3f | emitter "
                "MAE %.1f Hz, assignment %.3f | fitness %.4f%s",
                request.name,
                epoch,
                rows[-1]["train/loss"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["ap"],
                metrics["emitter_mae_hz"],
                metrics["assign_accuracy"],
                metrics["fitness"],
                " (best)" if improved else "",
            )
            if optim.patience and progress.stale >= optim.patience:
                log.info(
                    "%s: no improvement for %d epochs; stopping",
                    request.name,
                    progress.stale,
                )
                break
    if progress.best_epoch is None or not best_path.is_file():
        msg = f"{request.name}: stopped before the first validated epoch; no model"
        raise RuntimeError(msg)
    weights = out_dir / "weights.pt"
    shutil.copyfile(best_path, weights)
    peak = (
        torch.cuda.max_memory_allocated(device) / 2**30
        if device.type == "cuda"
        else None
    )
    model_info = {
        "architecture": model_cfg.architecture,
        "input_shape": list(shape),
        "image_shape": list(image_hw),
        "pad_value": PAD_VALUE,
        "padding": "bottom-right",
        "channels": "window order (0 = reference window)",
        "scaling": "pixel / 255",
        "head": "rt-detr-v2 one-to-one queries + emitter offset",
        "emitter_encoding": "(f_emitter - f_low) / image frequency span, from the "
        "predicted box's lower edge",
        "num_queries": model_cfg.num_queries,
        "max_det": model_cfg.max_det,
        "transformers": TRANSFORMERS_VERSION,
    }
    details = {
        "start_weights": start,
        "resumed_from": resumed,
        "stopped": stopped,
        "selected": "ema" if ema is not None else "model",
        "batch_used": optim.batch,
        "device": str(device),
        "amp": bool(optim.amp and device.type == "cuda"),
        "max_vram_fraction": vram,
        "peak_vram_gib": peak,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "n_train_images": len(request.train_ids),
        "n_val_images": len(request.val_ids),
        "n_val_images_scored": len(val_ids),
        "steps": step,
    }
    best_metrics = progress.best_metrics or {}
    return FitOutput(
        weights=weights,
        model=model_info,
        curves=curves,
        best_epoch=progress.best_epoch,
        epochs_run=progress.epoch,
        metrics={k: float(v) for k, v in best_metrics.items() if math.isfinite(v)},
        details=details,
    )


def _write_curves(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    columns: list[str] = []
    for row in rows:
        columns += [c for c in row if c not in columns]
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
