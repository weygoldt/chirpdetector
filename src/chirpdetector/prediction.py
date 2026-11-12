"""Run a bundle on spectrogram images; boxes in time and frequency.

`predict_images` feeds images (as `to_image` or
`chirpdetector.data.read_image` give them) to the bundle's model and maps
the pixel boxes to absolute time and frequency with each image's
`ImageGeometry` (exact inverse of the mapping the dataset labels were made
with). The result is a prediction table for `chirpdetector.evaluation`,
with the sha256 of the weights in a ``model_sha256`` column, so metrics can
be traced to (and attached only to) the weights that made them. Models
that predict the emitter's baseline row (``emitter_y``) get an
``emitter_hz`` column; `chirpdetector.evaluation.assign_by_frequency` turns
it into a track id.

`predict_dataset` does the same for images of a detector dataset (each
image is one "recording", as in `chirpdetector.data.dataset_ground_truth`).
Streaming inference on whole recordings comes later.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from chirpdetector.bundle import Bundle, load_bundle
from chirpdetector.data.contract import Dataset, read_image
from chirpdetector.models import PIXEL_COLUMNS, Detector, get_backend
from chirpdetector.spectrogram import (
    ImageGeometry,
    pixel_box_to_timefreq,
    pixel_to_timefreq,
)

__all__ = ["PREDICTION_COLUMNS", "load_detector", "predict_dataset", "predict_images"]

PREDICTION_COLUMNS = (
    "recording",
    "image_index",
    "t_start_s",
    "t_end_s",
    "f_low_hz",
    "f_high_hz",
    "score",
    "model_sha256",
)
"""Columns of `predict_images` output (``emitter_hz`` follows if predicted)."""

# A geometry may deviate from the bundle's front end by sample rounding only.
_GEOMETRY_RTOL = 0.02


def load_detector(
    bundle: Bundle | str | Path, *, device: str | None = None
) -> Detector:
    """The bundle's model, loaded by its implementation."""
    bundle = bundle if isinstance(bundle, Bundle) else load_bundle(bundle)
    return get_backend(bundle.model_type).load(bundle, device=device)


def _check_inputs(
    bundle: Bundle,
    images: Sequence[NDArray[np.uint8]],
    geometries: Sequence[ImageGeometry],
) -> None:
    if len(images) != len(geometries):
        msg = f"{len(images)} images but {len(geometries)} geometries"
        raise ValueError(msg)
    shape = (bundle.tiling.height_px, bundle.tiling.width_px, 3)
    dt = bundle.spectrogram.hop_s / bundle.tiling.pixels_per_frame
    df = bundle.spectrogram.df_hz / bundle.tiling.pixels_per_bin
    for k, (image, g) in enumerate(zip(images, geometries, strict=True)):
        if tuple(np.shape(image)) != shape or np.asarray(image).dtype != np.uint8:
            msg = f"image {k}: expected {shape} uint8 (the bundle's tiling), got {np.shape(image)}"
            raise ValueError(msg)
        if (g.height_px, g.width_px) != shape[:2]:
            msg = f"image {k}: geometry is {g.width_px} x {g.height_px} px, image {shape[1]} x {shape[0]}"
            raise ValueError(msg)
        if (
            abs(g.dt_s / dt - 1) > _GEOMETRY_RTOL
            or abs(g.df_hz / df - 1) > _GEOMETRY_RTOL
        ):
            msg = (
                f"image {k}: {g.dt_s:g} s x {g.df_hz:g} Hz per pixel, but the bundle's "
                f"front end gives {dt:g} s x {df:g} Hz; the image was rendered with "
                "another front end"
            )
            raise ValueError(msg)


def predict_images(
    bundle: Bundle | str | Path,
    images: Sequence[NDArray[np.uint8]],
    geometries: Sequence[ImageGeometry],
    *,
    recordings: Sequence[str] | None = None,
    score_threshold: float | None = None,
    device: str | None = None,
    detector: Detector | None = None,
) -> pd.DataFrame:
    """Detect chirps in images; boxes in time and frequency.

    Parameters
    ----------
    bundle
        Bundle (or its directory); loaded and verified if a path.
    images
        ``(H, W, 3)`` uint8 images of the bundle's tiling, in window order.
    geometries
        Placement of each image in time and frequency.
    recordings
        Name of each image's recording (default: ``"image-<k>"``).
    score_threshold
        Keep detections with at least this score; default: the bundle's.
    device
        Torch device for the model.
    detector
        An already loaded model of this bundle (`load_detector`), to avoid
        reloading the weights on every call.

    Returns
    -------
    DataFrame
        `PREDICTION_COLUMNS` (and ``emitter_hz`` if the model predicts the
        emitter's frequency), one row per detection.

    Raises
    ------
    ValueError
        If an image does not have the bundle's size, or its geometry does
        not match the bundle's front end.
    """
    bundle = bundle if isinstance(bundle, Bundle) else load_bundle(bundle)
    _check_inputs(bundle, images, geometries)
    names = (
        [f"image-{k}" for k in range(len(images))]
        if recordings is None
        else [str(r) for r in recordings]
    )
    if len(names) != len(images):
        msg = f"{len(images)} images but {len(names)} recordings"
        raise ValueError(msg)
    threshold = bundle.score_threshold if score_threshold is None else score_threshold
    detector = detector or load_detector(bundle, device=device)
    pixel_tables = detector.predict(list(images), score_threshold=float(threshold))
    if len(pixel_tables) != len(images):
        msg = f"the detector returned {len(pixel_tables)} results for {len(images)} images"
        raise RuntimeError(msg)
    parts = []
    with_emitter = any("emitter_y" in t.columns for t in pixel_tables)
    for k, (table, g) in enumerate(zip(pixel_tables, geometries, strict=True)):
        missing = [c for c in PIXEL_COLUMNS if c not in table.columns]
        if missing:
            msg = f"detector output lacks columns {missing}"
            raise RuntimeError(msg)
        xyxy = table.loc[:, list(PIXEL_COLUMNS[:4])].to_numpy(dtype=np.float64)
        t0, t1, f0, f1 = pixel_box_to_timefreq(
            xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3], g
        )
        part = pd.DataFrame(
            {
                "recording": names[k],
                "image_index": np.full(len(table), k, dtype=np.int64),
                "t_start_s": t0,
                "t_end_s": t1,
                "f_low_hz": f0,
                "f_high_hz": f1,
                "score": table["score"].to_numpy(dtype=np.float64),
                "model_sha256": bundle.weights_sha256,
            }
        )
        if with_emitter:
            if "emitter_y" in table.columns:
                _, emitter = pixel_to_timefreq(
                    0.0, table["emitter_y"].to_numpy(dtype=np.float64), g
                )
                part["emitter_hz"] = emitter
            else:
                part["emitter_hz"] = np.nan
        parts.append(part)
    columns = [*PREDICTION_COLUMNS, *(["emitter_hz"] if with_emitter else [])]
    if not parts:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in columns})
    return pd.concat(parts, ignore_index=True).loc[:, columns]


def predict_dataset(
    bundle: Bundle | str | Path,
    dataset: Dataset,
    image_ids: Iterable[str] | None = None,
    *,
    score_threshold: float | None = None,
    device: str | None = None,
    batch_size: int = 64,
) -> pd.DataFrame:
    """`predict_images` on dataset images; ``recording`` is the image id.

    Refuses a dataset rendered with another front end or tiling than the
    bundle's.
    """
    bundle = bundle if isinstance(bundle, Bundle) else load_bundle(bundle)
    manifest = dataset.manifest
    if manifest.spectrogram != bundle.spectrogram or manifest.tiling != bundle.tiling:
        msg = "the dataset's front end or tiling differs from the bundle's"
        raise ValueError(msg)
    ids = list(manifest.image_ids() if image_ids is None else image_ids)
    detector = load_detector(bundle, device=device)
    parts = []
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        part = predict_images(
            bundle,
            [read_image(dataset.image_path(i)) for i in chunk],
            [manifest.image(i).geometry for i in chunk],
            recordings=chunk,
            score_threshold=score_threshold,
            detector=detector,
        )
        part["image_index"] += start
        parts.append(part)
    if not parts:
        return predict_images(bundle, [], [], detector=detector)
    return pd.concat(parts, ignore_index=True)
