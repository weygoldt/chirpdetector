"""Read prediction files and apply chirpdetector v1's post-processing.

Three layouts are read by :func:`read_predictions`:

- v2 tables (CSV or Parquet) with the columns of
  :data:`~chirpdetector.evaluation.schema.PREDICTION_COLUMNS`;
- v1 detections, ``chirpdetector_bboxes.csv``, read with
  :func:`gridtools.io.read_v1_detections` (v1's ``track_id`` becomes
  ``emitter_id``; v1 has no emitter probability);
- v1 boxes without emitters (``t1, f1, t2, f2, score``, e.g. the pre-NMS
  boxes stored by ``detector_benchmark/detect.py``), keeping ``batch``.

v1 thresholded its boxes and then ran non-maximum suppression (NMS, IoU
0.2) within each batch of 12 spectrogram windows
(:func:`non_max_suppression`). Greedy NMS commutes with a score threshold:
a box is only ever suppressed by a box with a higher score, which survives
every threshold the lower one survives. NMS once on all boxes followed by a
threshold sweep therefore equals v1's threshold-then-NMS at every threshold.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from gridtools.io import read_v1_detections

from chirpdetector.evaluation.schema import select, validate_predictions

__all__ = ["V1_DETECTIONS_FILE", "non_max_suppression", "read_predictions"]

log = logging.getLogger(__name__)

V1_DETECTIONS_FILE = "chirpdetector_bboxes.csv"
"""File name of v1's detections inside a recording directory."""

_V1_BOX_COLUMNS = {
    "t1": "t_start_s",
    "t2": "t_end_s",
    "f1": "f_low_hz",
    "f2": "f_high_hz",
}


def read_predictions(path: str | Path, *, recording: str | None = None) -> pd.DataFrame:
    """Read predictions in any supported layout (see the module docstring).

    Parameters
    ----------
    path
        A CSV or Parquet file, or a recording directory holding
        ``chirpdetector_bboxes.csv``.
    recording
        Recording of every row. Default: the file's ``recording`` column; for
        ``chirpdetector_bboxes.csv`` the directory's name; otherwise the
        file's stem.

    Returns
    -------
    A validated prediction table
    (:func:`~chirpdetector.evaluation.schema.validate_predictions`) with a
    ``source_row`` column (the row in the file, after v1 sorting).
    """
    path = Path(path)
    if path.is_dir():
        path = path / V1_DETECTIONS_FILE
    raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if {"t1", "f1", "t2", "f2", "score"} <= set(raw.columns):
        if {"track_id", "emitter_eodf"} <= set(raw.columns):
            df = read_v1_detections(path).rename(columns={"track_id": "emitter_id"})
            default = path.parent.name
        else:
            df = raw.rename(columns=_V1_BOX_COLUMNS)
            default = path.stem
    else:
        df = raw
        default = path.stem
    if recording is None and "recording" not in df.columns:
        recording = default
    df = validate_predictions(df, recording=recording)
    df["source_row"] = np.arange(len(df))
    log.info("%s: %d predictions", path, len(df))
    return df


def _iou_one_to_many(
    box: npt.NDArray[np.float64], boxes: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """IoU of one ``(t0, f0, t1, f1)`` box with many; 0 where the union is empty."""
    w = np.maximum(
        0.0, np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0])
    )
    h = np.maximum(
        0.0, np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1])
    )
    inter = w * h
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def _nms_keep(
    boxes: npt.NDArray[np.float64], scores: npt.NDArray[np.float64], iou: float
) -> npt.NDArray[np.bool_]:
    order = np.argsort(-scores, kind="stable")
    keep = np.zeros(len(boxes), dtype=np.bool_)
    suppressed = np.zeros(len(boxes), dtype=np.bool_)
    for a, i in enumerate(order):
        if suppressed[i]:
            continue
        keep[i] = True
        rest = order[a + 1 :]
        rest = rest[~suppressed[rest]]
        if rest.size:
            suppressed[rest[_iou_one_to_many(boxes[i], boxes[rest]) > iou]] = True
    return keep


def non_max_suppression(
    predictions: pd.DataFrame,
    iou_threshold: float,
    *,
    by: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Greedy NMS as in chirpdetector v1 (``torchvision.ops.nms``).

    Boxes are visited in order of decreasing score (ties: row order); a box
    is dropped if its intersection over union with an already kept box,
    in seconds x hertz, exceeds ``iou_threshold``.

    Parameters
    ----------
    predictions
        Prediction table.
    iou_threshold
        v1 used 0.2.
    by
        Columns within whose groups NMS runs. Default: ``recording`` and,
        if present, ``batch`` (v1 ran NMS per batch).

    Returns
    -------
    The kept rows, in their original order.

    Notes
    -----
    v1 computed IoU in float32 (on the GPU when available); this function
    uses float64. Results can differ only for pairs whose IoU lies within
    float32 rounding of the threshold; on the detector_benchmark boxes the
    closest overlapping pair has IoU 0.364.
    """
    if by is None:
        by = [c for c in ("recording", "batch") if c in predictions.columns]
    keep = np.zeros(len(predictions), dtype=np.bool_)
    boxes = predictions[["t_start_s", "f_low_hz", "t_end_s", "f_high_hz"]].to_numpy(
        dtype=np.float64
    )
    scores = predictions["score"].to_numpy(dtype=np.float64)
    if by:
        groups = predictions.groupby(
            list(by), sort=False, dropna=False
        ).indices.values()
    else:
        groups = [np.arange(len(predictions))]
    for rows in groups:
        keep[rows] = _nms_keep(boxes[rows], scores[rows], iou_threshold)
    return select(predictions, keep)
