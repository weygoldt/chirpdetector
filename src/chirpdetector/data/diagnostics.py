"""Diagnostics of a built dataset: a contact sheet and per-split counts.

`write_diagnostics` writes into ``DATASET/diagnostics/``:

``contact_sheet.png``
    Random images (training images if there are any) with their boxes:
    solid = chirp, dashed = ignore region (colour by reason).
``counts.json``, ``counts.csv``
    Per split: images, groups, chirp boxes, unique chirps, ignore regions
    by reason, and chirp boxes by family and by visibility SNR (the larger
    of the footprint's baseline and excursion SNR, the quantity that
    decides ``visible``).
``build.json``
    Throughput, memory and other build statistics passed in by the
    builder.

The directory is not part of the dataset contract (not hashed, not read by
trainers).
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from chirpdetector.data.contract import SPLITS, Dataset, read_image
from chirpdetector.spectrogram import timefreq_box_to_pixel

__all__ = ["SNR_BINS_DB", "dataset_counts", "draw_contact_sheet", "write_diagnostics"]

DIAGNOSTICS_DIR = "diagnostics"
SNR_BINS_DB = (-math.inf, 0.0, 5.0, 10.0, 20.0, 30.0, math.inf)
"""Edges of the visibility-SNR bins of the counts."""

# BGR colours
_CHIRP = (0, 255, 255)
_REASON_COLOURS = {
    "invisible": (255, 255, 0),
    "edge": (255, 0, 255),
    "harmonic": (0, 165, 255),
    "unlabelled": (180, 180, 180),
}
_OTHER = (255, 255, 255)


def _dashed_rectangle(
    image: NDArray[np.uint8],
    p0: tuple[int, int],
    p1: tuple[int, int],
    colour: tuple[int, int, int],
    dash: int = 6,
) -> None:
    (x0, y0), (x1, y1) = p0, p1
    for a, b, fixed, horizontal in (
        (x0, x1, y0, True),
        (x0, x1, y1, True),
        (y0, y1, x0, False),
        (y0, y1, x1, False),
    ):
        for s in range(a, b, 2 * dash):
            e = min(s + dash, b)
            if horizontal:
                cv2.line(image, (s, fixed), (e, fixed), colour, 1)
            else:
                cv2.line(image, (fixed, s), (fixed, e), colour, 1)


def _draw_tile(dataset: Dataset, image_id: str) -> NDArray[np.uint8]:
    record = dataset.manifest.image(image_id)
    bgr = np.ascontiguousarray(read_image(dataset.image_path(image_id))[:, :, ::-1])
    for kind in ("ignore", "chirp"):
        rows = dataset.boxes_of(image_id, kind)
        if not len(rows):
            continue
        x0, y0, x1, y1 = timefreq_box_to_pixel(
            rows["t_start_s"].to_numpy(dtype=np.float64),
            rows["t_end_s"].to_numpy(dtype=np.float64),
            rows["f_low_hz"].to_numpy(dtype=np.float64),
            rows["f_high_hz"].to_numpy(dtype=np.float64),
            record.geometry,
        )
        reasons = rows["reason"].tolist()
        for k in range(len(rows)):
            p0 = (round(float(x0[k])), round(float(y0[k])))
            p1 = (round(float(x1[k])) - 1, round(float(y1[k])) - 1)
            if kind == "chirp":
                cv2.rectangle(bgr, p0, p1, _CHIRP, 1)
            else:
                colour = _REASON_COLOURS.get(str(reasons[k]), _OTHER)
                _dashed_rectangle(bgr, p0, p1, colour)
    t0, t1, f0, f1 = record.extent
    text = f"{image_id}  {t0:.2f}-{t1:.2f} s"
    cv2.putText(bgr, text, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(
        bgr, f"{f1:.0f} Hz", (4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1
    )
    cv2.putText(
        bgr,
        f"{f0:.0f} Hz",
        (4, bgr.shape[0] - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
    )
    return bgr


def draw_contact_sheet(
    dataset: Dataset,
    path: str | Path,
    *,
    n: int = 24,
    columns: int = 4,
    split: str | None = None,
    seed: int = 0,
    only_with_boxes: bool = False,
) -> list[str]:
    """Draw ``n`` random images of a split with their boxes into one PNG.

    Chirps are solid yellow boxes; ignore regions dashed (cyan = invisible,
    magenta = edge, orange = harmonic, grey = unlabelled). Returns the ids
    of the drawn images.
    """
    manifest = dataset.manifest
    if split is None:
        split = next((s for s in SPLITS if manifest.image_ids(split=s)), SPLITS[0])
    ids = list(manifest.image_ids(split=split))  # pyright: ignore[reportArgumentType]
    if only_with_boxes:
        with_boxes = set(dataset.boxes["image_id"].astype(str))
        ids = [i for i in ids if i in with_boxes]
    if not ids:
        return []
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(ids), size=min(n, len(ids)), replace=False))
    picked = [ids[k] for k in chosen]
    tiles = [_draw_tile(dataset, i) for i in picked]
    h, w = tiles[0].shape[:2]
    gap = 4
    rows = math.ceil(len(tiles) / columns)
    sheet = np.full(
        (rows * (h + gap) - gap, columns * (w + gap) - gap, 3), 40, dtype=np.uint8
    )
    for k, tile in enumerate(tiles):
        r, c = divmod(k, columns)
        sheet[r * (h + gap) : r * (h + gap) + h, c * (w + gap) : c * (w + gap) + w] = (
            tile
        )
    if not cv2.imwrite(str(path), sheet):
        msg = f"could not write {path}"
        raise OSError(msg)
    return picked


def _visibility_snr(boxes: pd.DataFrame) -> NDArray[np.float64]:
    if "snr_baseline_db" not in boxes.columns:
        return np.full(len(boxes), np.nan)
    a = np.asarray(pd.to_numeric(boxes["snr_baseline_db"], errors="coerce"), float)
    b = np.asarray(pd.to_numeric(boxes["snr_excursion_db"], errors="coerce"), float)
    both = np.stack([a, b], axis=1)
    out = np.full(len(boxes), np.nan)
    finite = np.isfinite(both).any(axis=1)
    out[finite] = np.nanmax(both[finite], axis=1)
    return out


def _bin_label(lo: float, hi: float) -> str:
    if math.isinf(lo):
        return f"<{hi:g}"
    if math.isinf(hi):
        return f">={lo:g}"
    return f"{lo:g}-{hi:g}"


def dataset_counts(dataset: Dataset) -> dict[str, Any]:
    """Per-split counts of images, groups, chirps and ignore regions."""
    manifest = dataset.manifest
    boxes = dataset.boxes
    image_split = {r.id: manifest.splits[r.group] for r in manifest.images}
    split_of = pd.Series(
        [image_split.get(str(i)) for i in boxes["image_id"]], index=boxes.index
    )
    source_of = {r.id: r.source for r in manifest.images}
    out: dict[str, Any] = {}
    for split in SPLITS:
        ids = manifest.image_ids(split=split)
        if not ids:
            continue
        rows = boxes.loc[(split_of == split).to_numpy()]
        chirps = rows.loc[(rows["kind"] == "chirp").to_numpy()]
        ignore = rows.loc[(rows["kind"] == "ignore").to_numpy()]
        entry: dict[str, Any] = {
            "images": len(ids),
            "groups": len(manifest.groups(split)),
            "images_without_chirps": int(
                len(ids) - chirps["image_id"].astype(str).nunique()
            ),
            "chirp_boxes": len(chirps),
            "ignore_boxes": len(ignore),
            "ignore_by_reason": {
                str(k): int(v) for k, v in ignore["reason"].value_counts().items()
            },
        }
        if "chirp_id" in chirps.columns:
            sources = chirps["image_id"].astype(str).map(source_of)
            keys = sources + "#" + chirps["chirp_id"].astype(str)
            entry["unique_chirps"] = int(keys.nunique())
            first = ~keys.duplicated()
            unique = chirps.loc[first.to_numpy()]
        else:
            unique = chirps
        if "family" in unique.columns:
            entry["unique_chirps_by_family"] = {
                str(k): int(v) for k, v in unique["family"].value_counts().items()
            }
        snr = _visibility_snr(unique)
        if np.isfinite(snr).any():
            counts = {}
            for lo, hi in itertools.pairwise(SNR_BINS_DB):
                counts[_bin_label(lo, hi)] = int(((snr >= lo) & (snr < hi)).sum())
            entry["unique_chirps_by_visibility_snr_db"] = counts
            entry["snr_db_quantiles"] = {
                f"q{int(q * 100):02d}": float(np.nanquantile(unique["snr_db"], q))
                for q in (0.05, 0.25, 0.5, 0.75, 0.95)
            }
        if "label_source" in chirps.columns:
            entry["label_source"] = {
                str(k): int(v) for k, v in chirps["label_source"].value_counts().items()
            }
        out[split] = entry
    return out


def _flat_counts(counts: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for split, entry in counts.items():
        for key, value in entry.items():
            if isinstance(value, Mapping):
                for sub, n in value.items():
                    rows.append({"split": split, "count": f"{key}.{sub}", "value": n})
            else:
                rows.append({"split": split, "count": key, "value": value})
    return pd.DataFrame(rows, columns=["split", "count", "value"])


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def write_diagnostics(
    dataset: Dataset,
    *,
    build: Mapping[str, Any] | None = None,
    n_sheet: int = 24,
    seed: int = 0,
) -> Path:
    """Write the contact sheet, counts and build statistics; return the folder."""
    folder = dataset.directory / DIAGNOSTICS_DIR
    folder.mkdir(exist_ok=True)
    counts = dataset_counts(dataset)
    (folder / "counts.json").write_text(json.dumps(_jsonable(counts), indent=1) + "\n")
    _flat_counts(counts).to_csv(folder / "counts.csv", index=False)
    if build is not None:
        (folder / "build.json").write_text(
            json.dumps(_jsonable(build), indent=1) + "\n"
        )
    picked = draw_contact_sheet(
        dataset, folder / "contact_sheet.png", n=n_sheet, seed=seed
    )
    (folder / "contact_sheet.txt").write_text("\n".join(picked) + "\n")
    return folder
