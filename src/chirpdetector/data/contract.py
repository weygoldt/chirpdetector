"""The on-disk detector dataset: the contract between builders and trainers.

Dataset builders (simulated scenes, real hand labels) write this format with
`DatasetWriter`; trainers and evaluation read it with `load_dataset`, which
validates it first (`validate_dataset`). Nothing else is needed to train or
evaluate a detector, whatever its architecture.

Layout
------
::

    DATASET/
      dataset.json        the manifest (`Manifest`)
      boxes.parquet       one row per chirp box and per ignore region
      images/<id>.png     spectrogram images, H x W x 3 uint8
      labels/<id>.txt     YOLO labels of the chirps, ``0 xc yc w h``

* **Images** are ``to_image`` output of the front end in the manifest
  (`chirpdetector.spectrogram`): row 0 is the highest frequency, column 0
  the earliest frame, channel ``c`` the ``c``-th analysis window (0 = the
  reference window). PNG files store channel 0 as red, 1 as green and 2 as
  blue; use `write_image` / `read_image`, never a bare ``cv2.imwrite``
  (OpenCV's BGR order would swap the windows). Every image of a dataset has
  the same size (``tiling.height_px`` x ``tiling.width_px``), so a detector
  never rescales a spectrogram.
* **Geometry.** Each image carries an `ImageGeometry`: the exact map between
  its pixels and absolute time (s, on the clock of its source recording or
  scene) and frequency (Hz). Boxes are stored in time and frequency
  (``boxes.parquet``) and, for chirps, also as YOLO pixel labels derived from
  them; the validator checks that the two agree.
* **Classes.** Exactly one class, ``chirp`` = index 0. Labels with another
  class index are refused (v1 wrote class 1 and only worked because of
  ``single_cls``; audit N3).
* **Ignore regions** are boxes that must count neither as chirps nor as
  background (invisible chirps, harmonic copies, chirps cut by the tile
  edge). They live only in ``boxes.parquet`` (``kind == "ignore"``); the
  YOLO label files hold chirps only.
* **Emitter.** Every chirp box carries ``emitter_hz``, the baseline EOD
  frequency of the fish that emitted it at the chirp's centre time, so that
  models which predict the emitter's frequency (and the evaluation of the
  assignment) can use the same dataset. ``emitter_id`` (the emitter's track
  id within the source) is given where known. ``live_tracks_hz`` lists the
  baseline frequencies of all fish alive at the chirp (the emitter
  included); where it is missing, the source must reference its tracks.
* **Splits** are assigned to *groups*, never to images: every image belongs
  to a source (a simulated scene or a real recording), every source to one
  group, every group to one split (``train``, ``val`` or ``test``).
  Variants of one scene (e.g. clean and noisy renderings) must share a
  group. An image cannot carry a split of its own, and identical images in
  two splits are refused (audit N2).
* **Hashes.** The manifest stores the sha256 of every image and label file
  and of ``boxes.parquet``, and ``content_sha256``, the sha256 of the
  canonical JSON of the manifest without its provenance. It identifies the
  dataset in run directories and model bundles.

``boxes.parquet`` columns
-------------------------
``image_id`` (str), ``kind`` (``chirp`` or ``ignore``), ``box_index`` (0..n-1
per image and kind; for chirps the line of the label file), ``t_start_s``,
``t_end_s``, ``f_low_hz``, ``f_high_hz`` (absolute box), ``t_center_s``
(the chirp's centre time, inside the box), ``emitter_hz`` (float; required
for chirps), ``emitter_id`` (nullable integer), ``live_tracks_hz`` (list of
floats or null), ``reason`` (nullable str, why a region is ignored). Further
columns (per-chirp SNR, ΔF to the nearest fish, overlap flags, template id,
...) are kept as they are.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from chirpdetector.spectrogram import (
    ImageGeometry,
    SpectrogramConfig,
    pixel_box_to_timefreq,
    timefreq_box_to_pixel,
)

if TYPE_CHECKING:
    from chirpdetector.evaluation import GroundTruth

__all__ = [
    "BOXES_FILE",
    "BOX_COLUMNS",
    "BOX_KINDS",
    "CLASSES",
    "DATASET_FORMAT",
    "DATASET_FORMAT_VERSION",
    "IMAGES_DIR",
    "LABELS_DIR",
    "MANIFEST_FILE",
    "SOURCE_KINDS",
    "SPLITS",
    "Dataset",
    "DatasetError",
    "DatasetWriter",
    "ImageRecord",
    "Manifest",
    "Source",
    "Tiling",
    "dataset_ground_truth",
    "load_dataset",
    "read_image",
    "read_yolo_labels",
    "sha256_file",
    "validate_dataset",
    "write_image",
]

Split = Literal["train", "val", "test"]
SourceKind = Literal["simulation", "real", "hybrid"]
BoxKind = Literal["chirp", "ignore"]
ChannelFill = Literal["zeros", "reference"]

DATASET_FORMAT = "chirpdetector-detector-dataset"
DATASET_FORMAT_VERSION = 1
CLASSES: tuple[str, ...] = ("chirp",)
"""The class map: index 0 = ``chirp``, nothing else."""
SPLITS: tuple[Split, ...] = get_args(Split)
SOURCE_KINDS: tuple[SourceKind, ...] = get_args(SourceKind)
BOX_KINDS: tuple[BoxKind, ...] = get_args(BoxKind)
CHANNEL_FILLS: tuple[ChannelFill, ...] = get_args(ChannelFill)

MANIFEST_FILE = "dataset.json"
BOXES_FILE = "boxes.parquet"
IMAGES_DIR = "images"
LABELS_DIR = "labels"
IMAGE_SUFFIX = ".png"
LABEL_SUFFIX = ".txt"

BOX_COLUMNS = (
    "image_id",
    "kind",
    "box_index",
    "t_start_s",
    "t_end_s",
    "f_low_hz",
    "f_high_hz",
    "t_center_s",
    "emitter_hz",
    "emitter_id",
    "live_tracks_hz",
    "reason",
)
"""Columns every ``boxes.parquet`` has, in this order (others follow)."""

_BOX_EDGES = ("t_start_s", "t_end_s", "f_low_hz", "f_high_hz")
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# Geometry may deviate from the config's nominal hop and bin spacing by the
# rounding to whole samples (`SpectrogramConfig.layout`): <= 0.5 sample per
# hop, i.e. 1.1% at 8 kHz with 5 ms; 2% leaves room for that.
_GEOMETRY_RTOL = 0.02
# Boxes may overshoot the image by rounding; more than this is an error.
_EDGE_TOLERANCE_PX = 1e-3
# Disagreement allowed between a YOLO label and its box in boxes.parquet.
_LABEL_TOLERANCE_PX = 1e-4
_HASH_CHUNK = 1 << 20


class DatasetError(ValueError):
    """A dataset violates the contract; the message lists every problem."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        shown = list(self.problems[:50])
        if len(self.problems) > 50:
            shown.append(f"... and {len(self.problems) - 50} more")
        super().__init__("invalid detector dataset:\n  " + "\n  ".join(shown))


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's content."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _strict_keys(
    data: Mapping[str, Any], required: Iterable[str], optional: Iterable[str], what: str
) -> None:
    required, optional = set(required), set(optional)
    missing = sorted(required - set(data))
    unknown = sorted(set(data) - required - optional)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing keys {missing}")
        if unknown:
            parts.append(f"unknown keys {unknown}")
        msg = f"{what}: {'; '.join(parts)}"
        raise ValueError(msg)


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    """``df[name]`` as a Series (a duplicated name raises)."""
    values = df[name]
    if isinstance(values, pd.DataFrame):
        msg = f"duplicate column {name!r}"
        raise ValueError(msg)
    return values


def _finite(value: Any) -> bool:
    return (
        isinstance(value, int | float | np.integer | np.floating)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


# --------------------------------------------------------------------- tiling


@dataclass(frozen=True, slots=True)
class Tiling:
    """How images were cut from the spectrogram and laid out.

    Parameters
    ----------
    tile_duration_s
        Nominal duration of one image (``width_px / pixels_per_frame``
        frames of ``hop_s``).
    overlap_s
        Nominal overlap of consecutive images of one source.
    width_px, height_px
        Size of every image. ``height_px`` is the front end's number of
        frequency bins times ``pixels_per_bin``.
    channels
        Channels stored per image; 3.
    channel_fill
        How channels beyond the front end's windows were filled
        (`chirpdetector.spectrogram.to_three_channels`): ``None`` when the
        front end has exactly 3 windows, else ``"zeros"`` or
        ``"reference"``.
    pixels_per_frame, pixels_per_bin
        Integer upsampling of time and frequency (1 = one pixel per frame
        and bin). The images' geometry already accounts for it.
    """

    tile_duration_s: float
    overlap_s: float
    width_px: int
    height_px: int
    channels: int = 3
    channel_fill: ChannelFill | None = None
    pixels_per_frame: int = 1
    pixels_per_bin: int = 1

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable fields."""
        return {
            "tile_duration_s": float(self.tile_duration_s),
            "overlap_s": float(self.overlap_s),
            "width_px": int(self.width_px),
            "height_px": int(self.height_px),
            "channels": int(self.channels),
            "channel_fill": self.channel_fill,
            "pixels_per_frame": int(self.pixels_per_frame),
            "pixels_per_bin": int(self.pixels_per_bin),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Tiling:
        """Inverse of `to_dict`; unknown keys raise."""
        names = (
            "tile_duration_s",
            "overlap_s",
            "width_px",
            "height_px",
            "channels",
            "channel_fill",
            "pixels_per_frame",
            "pixels_per_bin",
        )
        _strict_keys(data, names[:4], names[4:], "tiling")
        return cls(**data)

    def problems(self, spectrogram: SpectrogramConfig) -> list[str]:
        """Inconsistencies with the front end (empty if none)."""
        out = []
        ints = ("width_px", "height_px", "channels", "pixels_per_frame")
        for name in (*ints, "pixels_per_bin"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                out.append(f"tiling.{name} must be a positive integer, got {value!r}")
        if out:
            return out
        if self.channels != 3:
            out.append(f"tiling.channels must be 3, got {self.channels}")
        n = spectrogram.n_windows
        if n > 3:
            out.append(f"the front end has {n} windows; images hold at most 3")
        elif n == 3 and self.channel_fill is not None:
            out.append("tiling.channel_fill must be null for a 3-window front end")
        elif n < 3 and self.channel_fill not in CHANNEL_FILLS:
            out.append(
                f"a {n}-window front end needs tiling.channel_fill in {CHANNEL_FILLS}"
            )
        expected_h = spectrogram.n_freqs * self.pixels_per_bin
        if self.height_px != expected_h:
            out.append(
                f"tiling.height_px is {self.height_px}, but the front end has "
                f"{spectrogram.n_freqs} bins x {self.pixels_per_bin} = {expected_h}"
            )
        if not (_finite(self.tile_duration_s) and _finite(self.overlap_s)):
            out.append("tiling durations must be finite numbers")
            return out
        nominal = self.width_px / self.pixels_per_frame * spectrogram.hop_s
        if abs(self.tile_duration_s - nominal) > spectrogram.hop_s:
            out.append(
                f"tiling.tile_duration_s is {self.tile_duration_s:g} s, but "
                f"{self.width_px} px at {spectrogram.hop_s:g} s per frame is "
                f"{nominal:g} s"
            )
        if not 0 <= self.overlap_s < self.tile_duration_s:
            out.append("tiling.overlap_s must lie in [0, tile_duration_s)")
        return out


# -------------------------------------------------------------- sources, images


@dataclass(frozen=True, slots=True)
class Source:
    """A simulated scene or real recording that images were cut from.

    Parameters
    ----------
    id
        Unique name (scene id or recording name).
    group
        Splitting unit; every source belongs to exactly one.
    kind
        ``simulation``, ``real`` or ``hybrid`` (simulated fish on real
        background).
    tracks
        Directory of the source's wavetracker tracks, absolute or relative
        to the dataset directory (for assignment by frequency); optional if
        every chirp lists ``live_tracks_hz``.
    samplerate_hz
        Sample rate of the source recording, for the record.
    extra
        Anything else (scene seed, background recording, SNR, ...).
    """

    id: str
    group: str
    kind: SourceKind
    tracks: str | None = None
    samplerate_hz: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON fields (without ``id``, the key in the manifest)."""
        return {
            "group": self.group,
            "kind": self.kind,
            "tracks": self.tracks,
            "samplerate_hz": None
            if self.samplerate_hz is None
            else float(self.samplerate_hz),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, source_id: str, data: Mapping[str, Any]) -> Source:
        """Inverse of `to_dict`; unknown keys raise."""
        _strict_keys(
            data,
            ("group", "kind"),
            ("tracks", "samplerate_hz", "extra"),
            f"source {source_id!r}",
        )
        return cls(id=source_id, **data)


def _geometry_to_dict(geometry: ImageGeometry) -> dict[str, Any]:
    return {
        "width_px": int(geometry.width_px),
        "height_px": int(geometry.height_px),
        "t_first_s": float(geometry.t_first_s),
        "dt_s": float(geometry.dt_s),
        "f_lowest_hz": float(geometry.f_lowest_hz),
        "df_hz": float(geometry.df_hz),
    }


def _geometry_from_dict(data: Mapping[str, Any], what: str) -> ImageGeometry:
    names = ("width_px", "height_px", "t_first_s", "dt_s", "f_lowest_hz", "df_hz")
    _strict_keys(data, names, (), what)
    return ImageGeometry(**data)


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """One image of the dataset.

    The image is ``images/<id>.png`` and its chirp labels ``labels/<id>.txt``;
    the group and split follow from the source.
    """

    id: str
    source: str
    group: str
    geometry: ImageGeometry
    n_chirps: int
    n_ignore: int
    image_sha256: str
    label_sha256: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def image_file(self) -> str:
        """Image path relative to the dataset directory."""
        return f"{IMAGES_DIR}/{self.id}{IMAGE_SUFFIX}"

    @property
    def label_file(self) -> str:
        """YOLO label path relative to the dataset directory."""
        return f"{LABELS_DIR}/{self.id}{LABEL_SUFFIX}"

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(t_start_s, t_end_s, f_low_hz, f_high_hz)`` of the image edges."""
        t0, t1 = self.geometry.time_extent_s
        f0, f1 = self.geometry.freq_extent_hz
        return t0, t1, f0, f1

    def to_dict(self) -> dict[str, Any]:
        """JSON fields; the extent is repeated for readability."""
        t0, t1, f0, f1 = self.extent
        return {
            "id": self.id,
            "source": self.source,
            "group": self.group,
            "t_start_s": t0,
            "t_end_s": t1,
            "f_low_hz": f0,
            "f_high_hz": f1,
            "geometry": _geometry_to_dict(self.geometry),
            "n_chirps": int(self.n_chirps),
            "n_ignore": int(self.n_ignore),
            "image_sha256": self.image_sha256,
            "label_sha256": self.label_sha256,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ImageRecord:
        """Inverse of `to_dict`; unknown keys (e.g. a per-image split) raise."""
        what = f"image {data.get('id', '?')!r}"
        _strict_keys(
            data,
            (
                "id",
                "source",
                "group",
                "t_start_s",
                "t_end_s",
                "f_low_hz",
                "f_high_hz",
                "geometry",
                "n_chirps",
                "n_ignore",
                "image_sha256",
                "label_sha256",
            ),
            ("extra",),
            what,
        )
        record = cls(
            id=str(data["id"]),
            source=str(data["source"]),
            group=str(data["group"]),
            geometry=_geometry_from_dict(data["geometry"], f"{what} geometry"),
            n_chirps=int(data["n_chirps"]),
            n_ignore=int(data["n_ignore"]),
            image_sha256=str(data["image_sha256"]),
            label_sha256=str(data["label_sha256"]),
            extra=dict(data.get("extra", {})),
        )
        stated = (
            data["t_start_s"],
            data["t_end_s"],
            data["f_low_hz"],
            data["f_high_hz"],
        )
        if not np.allclose(stated, record.extent, rtol=1e-12, atol=1e-9):
            msg = f"{what}: stated extent {stated} disagrees with its geometry"
            raise ValueError(msg)
        return record


# -------------------------------------------------------------------- manifest


@dataclass(frozen=True)
class Manifest:
    """Content of ``dataset.json``.

    Attributes
    ----------
    spectrogram
        The front end that rendered every image.
    tiling
        Image size, tile duration and channel layout.
    preprocessing
        Preprocessing applied before the front end, ``{"click_removal":
        None | ClickRemoval.to_dict()}``. Copied into model bundles, so
        that inference preprocesses the same way.
    sources
        Source id -> `Source`.
    splits
        Group -> split. The only place a split is assigned.
    images
        All images, in a fixed order.
    boxes_sha256
        sha256 of ``boxes.parquet``.
    provenance
        Who built the dataset from what (builder, versions, configs,
        seeds); not part of `content_sha256`.
    classes
        Always ``("chirp",)``.
    format_version
        Layout version.
    """

    spectrogram: SpectrogramConfig
    tiling: Tiling
    preprocessing: Mapping[str, Any]
    sources: Mapping[str, Source]
    splits: Mapping[str, Split]
    images: tuple[ImageRecord, ...]
    boxes_sha256: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    classes: tuple[str, ...] = CLASSES
    format_version: int = DATASET_FORMAT_VERSION

    # ------------------------------------------------------------- lookups

    def image(self, image_id: str) -> ImageRecord:
        """The record of one image."""
        index = self._index()
        if image_id not in index:
            msg = f"no image {image_id!r} in the dataset"
            raise KeyError(msg)
        return self.images[index[image_id]]

    def _index(self) -> dict[str, int]:
        cached = self.__dict__.get("_index_cache")
        if cached is None:
            cached = {r.id: i for i, r in enumerate(self.images)}
            object.__setattr__(self, "_index_cache", cached)
        return cached

    def split_of_group(self, group: str) -> Split:
        """Split of a group."""
        return self.splits[group]

    def groups(self, split: Split | None = None) -> tuple[str, ...]:
        """Groups (of one split, or all), sorted."""
        return tuple(sorted(g for g, s in self.splits.items() if split in (None, s)))

    def image_ids(
        self, split: Split | None = None, groups: Iterable[str] | None = None
    ) -> tuple[str, ...]:
        """Ids of the images of a split and/or of some groups, in manifest order."""
        wanted = None if groups is None else set(groups)
        return tuple(
            r.id
            for r in self.images
            if (split is None or self.splits[r.group] == split)
            and (wanted is None or r.group in wanted)
        )

    def group_sizes(self, split: Split | None = None) -> dict[str, int]:
        """Number of images per group (of one split, or all)."""
        sizes = dict.fromkeys(self.groups(split), 0)
        for r in self.images:
            if r.group in sizes:
                sizes[r.group] += 1
        return sizes

    # ---------------------------------------------------------------- JSON

    def content_dict(self) -> dict[str, Any]:
        """Everything hashed into `content_sha256` (no provenance)."""
        return {
            "format": DATASET_FORMAT,
            "format_version": self.format_version,
            "classes": list(self.classes),
            "spectrogram": self.spectrogram.to_dict(),
            "tiling": self.tiling.to_dict(),
            "preprocessing": dict(self.preprocessing),
            "groups": {g: self.splits[g] for g in sorted(self.splits)},
            "sources": {k: self.sources[k].to_dict() for k in sorted(self.sources)},
            "images": [r.to_dict() for r in self.images],
            "boxes_sha256": self.boxes_sha256,
        }

    @property
    def content_sha256(self) -> str:
        """sha256 of the canonical JSON of `content_dict`."""
        cached = self.__dict__.get("_hash_cache")
        if cached is None:
            cached = hashlib.sha256(
                _canonical_json(self.content_dict()).encode()
            ).hexdigest()
            object.__setattr__(self, "_hash_cache", cached)
        return cached

    def to_dict(self) -> dict[str, Any]:
        """``dataset.json`` content."""
        return {
            **self.content_dict(),
            "provenance": dict(self.provenance),
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Manifest:
        """Parse ``dataset.json``.

        Raises
        ------
        ValueError
            On a wrong format or version, unknown or missing keys at any
            level, or a stored ``content_sha256`` that does not match.
        """
        _strict_keys(
            data,
            (
                "format",
                "format_version",
                "classes",
                "spectrogram",
                "tiling",
                "preprocessing",
                "groups",
                "sources",
                "images",
                "boxes_sha256",
            ),
            ("provenance", "content_sha256"),
            "dataset.json",
        )
        if data["format"] != DATASET_FORMAT:
            msg = f"not a {DATASET_FORMAT} manifest (format {data['format']!r})"
            raise ValueError(msg)
        if data["format_version"] != DATASET_FORMAT_VERSION:
            msg = (
                f"dataset format version {data['format_version']!r}; this "
                f"chirpdetector reads version {DATASET_FORMAT_VERSION}"
            )
            raise ValueError(msg)
        groups = data["groups"]
        if not isinstance(groups, Mapping):
            msg = "groups must map group -> split"
            raise ValueError(msg)
        manifest = cls(
            spectrogram=SpectrogramConfig.from_dict(dict(data["spectrogram"])),
            tiling=Tiling.from_dict(data["tiling"]),
            preprocessing=dict(data["preprocessing"]),
            sources={
                str(k): Source.from_dict(str(k), v) for k, v in data["sources"].items()
            },
            splits={str(g): cast("Split", s) for g, s in groups.items()},
            images=tuple(ImageRecord.from_dict(r) for r in data["images"]),
            boxes_sha256=str(data["boxes_sha256"]),
            provenance=dict(data.get("provenance") or {}),
            classes=tuple(data["classes"]),
            format_version=int(data["format_version"]),
        )
        stored = data.get("content_sha256")
        if stored is not None and stored != manifest.content_sha256:
            msg = (
                "content_sha256 does not match the manifest content "
                f"(stored {stored}, computed {manifest.content_sha256}); "
                "the manifest was edited after it was written"
            )
            raise ValueError(msg)
        return manifest

    def write(self, directory: str | Path) -> Path:
        """Write ``dataset.json`` into ``directory``."""
        path = Path(directory) / MANIFEST_FILE
        path.write_text(json.dumps(self.to_dict(), indent=1, allow_nan=False) + "\n")
        return path

    @classmethod
    def read(cls, directory: str | Path) -> Manifest:
        """Read ``dataset.json`` from ``directory``."""
        path = Path(directory) / MANIFEST_FILE
        if not path.is_file():
            msg = f"{path} does not exist"
            raise FileNotFoundError(msg)
        return cls.from_dict(json.loads(path.read_text()))


# ------------------------------------------------------------------ image I/O


def write_image(path: str | Path, image: NDArray[np.uint8]) -> None:
    """Write an ``(H, W, 3)`` uint8 spectrogram image as PNG.

    Channel 0 (the reference window) is stored as red, 1 as green, 2 as
    blue. `read_image` returns the channels in the same order.
    """
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        msg = f"expected an (H, W, 3) uint8 image, got {array.dtype} {array.shape}"
        raise ValueError(msg)
    bgr = np.ascontiguousarray(array[:, :, ::-1])
    if not cv2.imwrite(str(path), bgr):
        msg = f"could not write {path}"
        raise OSError(msg)


def read_image(path: str | Path) -> NDArray[np.uint8]:
    """Read an image written by `write_image`, channels in window order."""
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        msg = f"could not read image {path}"
        raise OSError(msg)
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        msg = f"{path}: expected an (H, W, 3) uint8 image, got {bgr.dtype} {bgr.shape}"
        raise ValueError(msg)
    return cast("NDArray[np.uint8]", np.ascontiguousarray(bgr[:, :, ::-1]))


def _format_labels(xyxy: NDArray[np.float64], width: int, height: int) -> str:
    lines = []
    for x0, y0, x1, y1 in xyxy:
        xc, yc = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
        w, h = (x1 - x0) / width, (y1 - y0) / height
        lines.append(f"0 {xc:.12g} {yc:.12g} {w:.12g} {h:.12g}\n")
    return "".join(lines)


def read_yolo_labels(path: str | Path) -> NDArray[np.float64]:
    """Read a YOLO label file as ``(n, 5)``: class, xc, yc, w, h (normalised).

    Raises
    ------
    ValueError
        On a malformed line. Class values are not checked here (see
        `validate_dataset`).
    """
    rows = []
    for k, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            msg = f"{path}:{k + 1}: expected 5 values, got {len(parts)}"
            raise ValueError(msg)
        try:
            rows.append([float(v) for v in parts])
        except ValueError as err:
            msg = f"{path}:{k + 1}: not a number in {line!r}"
            raise ValueError(msg) from err
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


def labels_to_pixel_boxes(
    labels: NDArray[np.float64], width: int, height: int
) -> NDArray[np.float64]:
    """YOLO ``(n, 5)`` labels -> pixel boxes ``(n, 4)`` ``x_min, y_min, x_max, y_max``."""
    xc, yc = labels[:, 1] * width, labels[:, 2] * height
    w, h = labels[:, 3] * width, labels[:, 4] * height
    return np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], axis=1)


# ----------------------------------------------------------------- the boxes


def _pixel_boxes(boxes: pd.DataFrame, geometry: ImageGeometry) -> NDArray[np.float64]:
    """Time-frequency boxes -> pixel boxes ``(n, 4)`` (x_min, y_min, x_max, y_max)."""
    ts, te, fl, fh = (_column(boxes, c).to_numpy(dtype=np.float64) for c in _BOX_EDGES)
    x0, y0, x1, y1 = timefreq_box_to_pixel(ts, te, fl, fh, geometry)
    return np.stack([x0, y0, x1, y1], axis=1).reshape(-1, 4)


def _normalize_box_table(
    table: pd.DataFrame | None, kind: BoxKind, image_id: str
) -> pd.DataFrame:
    """Box rows of one image in the ``boxes.parquet`` layout."""
    if table is None or len(table) == 0:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in BOX_COLUMNS})
    df = table.reset_index(drop=True).copy()
    required = list(_BOX_EDGES) + (["emitter_hz"] if kind == "chirp" else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = f"{kind} boxes of {image_id!r}: missing columns {missing}"
        raise ValueError(msg)
    for c in ("image_id", "kind", "box_index"):
        if c in df.columns:
            msg = f"{kind} boxes of {image_id!r}: column {c!r} is set by the writer"
            raise ValueError(msg)
    df.insert(0, "image_id", image_id)
    df.insert(1, "kind", kind)
    df.insert(2, "box_index", np.arange(len(df), dtype=np.int64))
    for c in _BOX_EDGES:
        df[c] = df[c].to_numpy(dtype=np.float64)
    if "t_center_s" not in df.columns:
        df["t_center_s"] = 0.5 * (df["t_start_s"] + df["t_end_s"])
    for c, default in (("emitter_hz", np.nan), ("reason", None)):
        if c not in df.columns:
            df[c] = default
    if "emitter_id" not in df.columns:
        df["emitter_id"] = pd.array([pd.NA] * len(df), dtype="Int64")
    if "live_tracks_hz" not in df.columns:
        df["live_tracks_hz"] = None
    df["live_tracks_hz"] = [
        None if v is None else [float(x) for x in np.asarray(v, dtype=np.float64)]
        for v in df["live_tracks_hz"]
    ]
    extra = [c for c in df.columns if c not in BOX_COLUMNS]
    return df.loc[:, [*BOX_COLUMNS, *extra]]


def _normalize_box_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df["image_id"] = df["image_id"].astype(str)
    df["kind"] = df["kind"].astype(str)
    df["box_index"] = df["box_index"].to_numpy(dtype=np.int64)
    for c in (*_BOX_EDGES, "t_center_s", "emitter_hz"):
        numbers = pd.to_numeric(_column(df, c), errors="coerce")
        df[c] = np.asarray(numbers, dtype=np.float64)
    df["emitter_id"] = pd.array(_column(df, "emitter_id").to_numpy(), dtype="Int64")
    df["reason"] = pd.Series(
        [None if v is None or v is pd.NA or v != v else str(v) for v in df["reason"]],
        dtype=object,
    )
    df["live_tracks_hz"] = pd.Series(
        [
            None if v is None else np.asarray(v, dtype=np.float64)
            for v in df["live_tracks_hz"]
        ],
        dtype=object,
    )
    return df


def read_boxes(directory: str | Path) -> pd.DataFrame:
    """``boxes.parquet`` of a dataset, with normalised dtypes."""
    path = Path(directory) / BOXES_FILE
    df = pd.read_parquet(path)
    missing = [c for c in BOX_COLUMNS if c not in df.columns]
    if missing:
        msg = f"{path}: missing columns {missing}"
        raise ValueError(msg)
    return _normalize_box_dtypes(df)


# ---------------------------------------------------------------- the writer


class DatasetWriter:
    """Write a detector dataset image by image.

    Typical use (a builder)::

        writer = DatasetWriter(out, spectrogram=config, tiling=tiling)
        writer.add_source("scene-0001", group="scene-0001", kind="simulation",
                          tracks="/data/scenes/scene-0001")
        writer.assign_split("scene-0001", "train")
        for image, geometry, chirps, ignore in tiles:
            writer.add_image(image_id, image, geometry, source="scene-0001",
                             chirps=chirps, ignore=ignore)
        dataset = writer.close()        # writes boxes.parquet + dataset.json, validates

    ``chirps`` and ``ignore`` are DataFrames with ``t_start_s, t_end_s,
    f_low_hz, f_high_hz`` in absolute time and frequency; chirps also need
    ``emitter_hz`` and may have ``t_center_s``, ``emitter_id``,
    ``live_tracks_hz`` and any further columns. The YOLO labels are derived
    from the chirp boxes with the image's geometry, so the two cannot
    disagree.

    Parameters
    ----------
    directory
        Output directory; must not exist or be empty (unless
        ``overwrite``).
    spectrogram, tiling
        Front end and tiling of every image.
    preprocessing
        Preprocessing before the front end (default: no click removal).
    provenance
        Builder name, versions, configs, seeds; stored in the manifest.
    overwrite
        Delete an existing directory first.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        spectrogram: SpectrogramConfig,
        tiling: Tiling,
        preprocessing: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        self.directory = Path(directory)
        if self.directory.exists() and any(self.directory.iterdir()):
            if not overwrite:
                msg = f"{self.directory} exists and is not empty"
                raise FileExistsError(msg)
            shutil.rmtree(self.directory)
        problems = tiling.problems(spectrogram)
        if problems:
            raise DatasetError(problems)
        (self.directory / IMAGES_DIR).mkdir(parents=True, exist_ok=True)
        (self.directory / LABELS_DIR).mkdir(parents=True, exist_ok=True)
        self.spectrogram = spectrogram
        self.tiling = tiling
        self.preprocessing = dict(preprocessing or {"click_removal": None})
        self.provenance = dict(provenance or {})
        self._sources: dict[str, Source] = {}
        self._splits: dict[str, Split] = {}
        self._images: list[ImageRecord] = []
        self._boxes: list[pd.DataFrame] = []
        self._closed = False

    def add_source(
        self,
        source_id: str,
        *,
        group: str,
        kind: SourceKind,
        tracks: str | Path | None = None,
        samplerate_hz: float | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Source:
        """Register a source (scene or recording) and its group."""
        if source_id in self._sources:
            msg = f"source {source_id!r} added twice"
            raise ValueError(msg)
        source = Source(
            id=source_id,
            group=group,
            kind=kind,
            tracks=None if tracks is None else str(tracks),
            samplerate_hz=samplerate_hz,
            extra=dict(extra or {}),
        )
        self._sources[source_id] = source
        return source

    def assign_split(self, group: str, split: Split) -> None:
        """Put a whole group into a split."""
        if split not in SPLITS:
            msg = f"split must be one of {SPLITS}, got {split!r}"
            raise ValueError(msg)
        if self._splits.get(group, split) != split:
            msg = f"group {group!r} is already in split {self._splits[group]!r}"
            raise ValueError(msg)
        self._splits[group] = split

    def add_image(
        self,
        image_id: str,
        image: NDArray[np.uint8],
        geometry: ImageGeometry,
        *,
        source: str,
        chirps: pd.DataFrame | None = None,
        ignore: pd.DataFrame | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ImageRecord:
        """Write one image, its YOLO labels, and queue its boxes.

        Raises
        ------
        ValueError
            On an unsafe or duplicate id, an unknown source, an image of the
            wrong size, or a box outside the image (beyond rounding; boxes
            within 1e-3 px of an edge are clipped to it).
        """
        if self._closed:
            msg = "the writer is closed"
            raise RuntimeError(msg)
        if not _ID_PATTERN.match(image_id):
            msg = f"image id {image_id!r} must match {_ID_PATTERN.pattern}"
            raise ValueError(msg)
        if source not in self._sources:
            msg = f"unknown source {source!r}; call add_source first"
            raise ValueError(msg)
        shape = (self.tiling.height_px, self.tiling.width_px, 3)
        if tuple(np.shape(image)) != shape:
            msg = f"image {image_id!r} has shape {np.shape(image)}, expected {shape}"
            raise ValueError(msg)
        if (geometry.height_px, geometry.width_px) != shape[:2]:
            msg = f"geometry of {image_id!r} does not match the image size {shape[:2]}"
            raise ValueError(msg)
        chirp_rows = _normalize_box_table(chirps, "chirp", image_id)
        ignore_rows = _normalize_box_table(ignore, "ignore", image_id)
        for rows in (chirp_rows, ignore_rows):
            if len(rows):
                _clip_in_place(rows, geometry, image_id)
        pixel = (
            _pixel_boxes(chirp_rows, geometry) if len(chirp_rows) else np.zeros((0, 4))
        )
        record_id = image_id
        image_path = self.directory / IMAGES_DIR / f"{record_id}{IMAGE_SUFFIX}"
        label_path = self.directory / LABELS_DIR / f"{record_id}{LABEL_SUFFIX}"
        if image_path.exists():
            msg = f"image {image_id!r} written twice"
            raise ValueError(msg)
        write_image(image_path, np.asarray(image, dtype=np.uint8))
        label_path.write_text(_format_labels(pixel, shape[1], shape[0]))
        record = ImageRecord(
            id=record_id,
            source=source,
            group=self._sources[source].group,
            geometry=geometry,
            n_chirps=len(chirp_rows),
            n_ignore=len(ignore_rows),
            image_sha256=sha256_file(image_path),
            label_sha256=sha256_file(label_path),
            extra=dict(extra or {}),
        )
        self._images.append(record)
        self._boxes.extend(r for r in (chirp_rows, ignore_rows) if len(r))
        return record

    def close(self, *, check_files: bool = True) -> Dataset:
        """Write ``boxes.parquet`` and ``dataset.json``; validate; return the dataset."""
        if self._closed:
            msg = "the writer is closed"
            raise RuntimeError(msg)
        boxes = (
            pd.concat(self._boxes, ignore_index=True)
            if self._boxes
            else _normalize_box_table(None, "chirp", "")
        )
        boxes = _normalize_box_dtypes(boxes)
        boxes["live_tracks_hz"] = pd.Series(
            [None if v is None else list(v) for v in boxes["live_tracks_hz"]],
            dtype=object,
        )
        boxes_path = self.directory / BOXES_FILE
        boxes.to_parquet(boxes_path, index=False)
        manifest = Manifest(
            spectrogram=self.spectrogram,
            tiling=self.tiling,
            preprocessing=self.preprocessing,
            sources=dict(self._sources),
            splits=dict(self._splits),
            images=tuple(self._images),
            boxes_sha256=sha256_file(boxes_path),
            provenance=self.provenance,
        )
        manifest.write(self.directory)
        self._closed = True
        return load_dataset(self.directory, check_files=check_files)


def _clip_in_place(rows: pd.DataFrame, geometry: ImageGeometry, image_id: str) -> None:
    t0, t1 = geometry.time_extent_s
    f0, f1 = geometry.freq_extent_hz
    tol_t = _EDGE_TOLERANCE_PX * geometry.dt_s
    tol_f = _EDGE_TOLERANCE_PX * geometry.df_hz
    ts, te = rows["t_start_s"].to_numpy(), rows["t_end_s"].to_numpy()
    fl, fh = rows["f_low_hz"].to_numpy(), rows["f_high_hz"].to_numpy()
    outside = (
        (ts < t0 - tol_t) | (te > t1 + tol_t) | (fl < f0 - tol_f) | (fh > f1 + tol_f)
    )
    if outside.any():
        kind = rows["kind"].iloc[0]
        msg = (
            f"{int(outside.sum())} {kind} box(es) of {image_id!r} extend beyond the "
            f"image ({t0:.6g}-{t1:.6g} s, {f0:.6g}-{f1:.6g} Hz); clip them first"
        )
        raise ValueError(msg)
    rows["t_start_s"] = np.clip(ts, t0, t1)
    rows["t_end_s"] = np.clip(te, t0, t1)
    rows["f_low_hz"] = np.clip(fl, f0, f1)
    rows["f_high_hz"] = np.clip(fh, f0, f1)


# ------------------------------------------------------------ the validator


def _check_geometry(record: ImageRecord, manifest: Manifest) -> list[str]:
    g, cfg, tiling = record.geometry, manifest.spectrogram, manifest.tiling
    out = []
    if (g.width_px, g.height_px) != (tiling.width_px, tiling.height_px):
        out.append(
            f"image {record.id}: geometry is {g.width_px} x {g.height_px} px, "
            f"tiling says {tiling.width_px} x {tiling.height_px}"
        )
    values = (g.t_first_s, g.dt_s, g.f_lowest_hz, g.df_hz)
    if not all(_finite(v) for v in values) or g.dt_s <= 0 or g.df_hz <= 0:
        out.append(f"image {record.id}: geometry must be finite with dt, df > 0")
        return out
    dt = cfg.hop_s / tiling.pixels_per_frame
    df = cfg.df_hz / tiling.pixels_per_bin
    if abs(g.dt_s / dt - 1) > _GEOMETRY_RTOL:
        out.append(
            f"image {record.id}: {g.dt_s:g} s per column, the front end gives {dt:g}"
        )
    if abs(g.df_hz / df - 1) > _GEOMETRY_RTOL:
        out.append(
            f"image {record.id}: {g.df_hz:g} Hz per row, the front end gives {df:g}"
        )
    return out


def _check_boxes(
    boxes: pd.DataFrame, manifest: Manifest, records: Mapping[str, ImageRecord]
) -> list[str]:
    out: list[str] = []
    ids = boxes["image_id"].to_numpy(dtype=object)
    unknown = sorted(set(ids.tolist()) - set(records))
    if unknown:
        out.append(f"boxes.parquet: rows of unknown images {unknown[:10]}")
    kinds = boxes["kind"].to_numpy(dtype=object)
    bad_kinds = sorted(set(kinds.tolist()) - set(BOX_KINDS))
    if bad_kinds:
        out.append(f"boxes.parquet: unknown kinds {bad_kinds}; allowed {BOX_KINDS}")
    edges = boxes.loc[:, list(_BOX_EDGES)].to_numpy(dtype=np.float64)
    tc = boxes["t_center_s"].to_numpy(dtype=np.float64)
    finite = np.isfinite(edges).all(axis=1) & np.isfinite(tc)
    if not finite.all():
        out.append(f"boxes.parquet: {int((~finite).sum())} boxes are not finite")
    flat = (edges[:, 1] <= edges[:, 0]) | (edges[:, 3] <= edges[:, 2])
    if flat[finite].any():
        out.append(
            f"boxes.parquet: {int(flat[finite].sum())} boxes have zero or negative "
            "duration or bandwidth"
        )
    centre_out = (tc < edges[:, 0]) | (tc > edges[:, 1])
    if centre_out[finite].any():
        out.append(
            f"boxes.parquet: {int(centre_out[finite].sum())} t_center_s lie outside "
            "their box"
        )
    chirp = kinds == "chirp"
    emitter = boxes["emitter_hz"].to_numpy(dtype=np.float64)
    bad_emitter = chirp & ~(np.isfinite(emitter) & (emitter > 0))
    if bad_emitter.any():
        out.append(
            f"boxes.parquet: {int(bad_emitter.sum())} chirps without a positive, "
            "finite emitter_hz"
        )
    for k, value in enumerate(boxes["live_tracks_hz"]):
        if value is None:
            if chirp[k]:
                record = records.get(str(ids[k]))
                source = None if record is None else manifest.sources.get(record.source)
                if source is not None and source.tracks is None:
                    out.append(
                        f"chirp {k} of {ids[k]}: neither live_tracks_hz nor tracks "
                        f"of source {source.id!r}"
                    )
            continue
        live = np.asarray(value, dtype=np.float64)
        if live.ndim != 1 or not (np.isfinite(live).all() and (live > 0).all()):
            out.append(f"box {k} of {ids[k]}: live_tracks_hz must be positive numbers")
    # per image: box_index contiguous, counts, inside the image
    for key, rows in boxes.groupby(["image_id", "kind"], sort=False):
        image_id, kind = cast("tuple[str, str]", key)
        record = records.get(str(image_id))
        if record is None or kind not in BOX_KINDS:
            continue
        index = np.sort(rows["box_index"].to_numpy(dtype=np.int64))
        if not np.array_equal(index, np.arange(len(rows))):
            out.append(f"image {image_id}: {kind} box_index is not 0..{len(rows) - 1}")
        expected = record.n_chirps if kind == "chirp" else record.n_ignore
        if len(rows) != expected:
            out.append(
                f"image {image_id}: {len(rows)} {kind} rows, the manifest says {expected}"
            )
        t0, t1, f0, f1 = record.extent
        g = record.geometry
        tol_t, tol_f = _EDGE_TOLERANCE_PX * g.dt_s, _EDGE_TOLERANCE_PX * g.df_hz
        e = rows.loc[:, list(_BOX_EDGES)].to_numpy(dtype=np.float64)
        outside = (
            (e[:, 0] < t0 - tol_t)
            | (e[:, 1] > t1 + tol_t)
            | (e[:, 2] < f0 - tol_f)
            | (e[:, 3] > f1 + tol_f)
        )
        if outside.any():
            out.append(
                f"image {image_id}: {int(outside.sum())} {kind} box(es) outside the "
                "image extent"
            )
    counted = boxes.groupby(["image_id", "kind"]).size().to_dict()
    for record in records.values():
        for kind, n in (("chirp", record.n_chirps), ("ignore", record.n_ignore)):
            if n and (record.id, kind) not in counted:
                out.append(f"image {record.id}: manifest says {n} {kind} rows, found 0")
    return out


def _check_labels(
    record: ImageRecord, label_path: Path, chirps: pd.DataFrame
) -> list[str]:
    try:
        labels = read_yolo_labels(label_path)
    except ValueError as err:
        return [str(err)]
    out = []
    classes = labels[:, 0]
    if np.any(classes != 0):
        bad = sorted({float(c) for c in classes[classes != 0]})
        out.append(
            f"{record.label_file}: class indices {bad}; the only class is 0 = chirp "
            "(v1 wrote 1 and relied on single_cls)"
        )
    coords = labels[:, 1:]
    if np.any(~np.isfinite(coords)) or np.any(coords < 0) or np.any(coords > 1):
        out.append(f"{record.label_file}: coordinates outside [0, 1]")
    if np.any(coords[:, 2:] <= 0):
        out.append(f"{record.label_file}: boxes with zero width or height")
    if len(labels) != record.n_chirps:
        out.append(
            f"{record.label_file}: {len(labels)} lines, the manifest says "
            f"{record.n_chirps} chirps"
        )
        return out
    if len(labels) == 0:
        return out
    g = record.geometry
    from_labels = labels_to_pixel_boxes(labels, g.width_px, g.height_px)
    ordered = chirps.sort_values("box_index", kind="stable")
    from_boxes = _pixel_boxes(ordered, g)
    worst = float(np.max(np.abs(from_labels - from_boxes)))
    if worst > _LABEL_TOLERANCE_PX:
        out.append(
            f"{record.label_file}: labels differ from boxes.parquet by up to "
            f"{worst:.3g} px"
        )
    return out


def validate_dataset(directory: str | Path, *, check_files: bool = True) -> Manifest:
    """Check a detector dataset against the contract.

    Parameters
    ----------
    directory
        Dataset directory.
    check_files
        Also hash and decode every image and parse every label file
        (seconds for thousands of images). Without it, only the manifest
        and ``boxes.parquet`` are checked.

    Returns
    -------
    Manifest
        The parsed manifest.

    Raises
    ------
    DatasetError
        Listing every problem found: unknown keys (e.g. a split per image),
        a class map other than ``["chirp"]``, groups without a valid split,
        sources pointing to unknown groups, a tiling inconsistent with the
        front end, images whose size or geometry disagree with it, boxes
        outside their image or inconsistent with the label files, class
        indices other than 0, chirps without ``emitter_hz``, missing files,
        stale hashes, and identical images in two splits.
    """
    directory = Path(directory)
    try:
        manifest = Manifest.read(directory)
    except (ValueError, KeyError, TypeError, FileNotFoundError) as err:
        raise DatasetError([str(err)]) from err
    problems: list[str] = []
    if manifest.classes != CLASSES:
        problems.append(
            f"classes must be {list(CLASSES)} (index 0 = chirp), got "
            f"{list(manifest.classes)}"
        )
    problems += manifest.tiling.problems(manifest.spectrogram)
    for group, split in manifest.splits.items():
        if split not in SPLITS:
            problems.append(f"group {group!r}: split {split!r} is not one of {SPLITS}")
    used_groups = set()
    for source in manifest.sources.values():
        used_groups.add(source.group)
        if source.group not in manifest.splits:
            problems.append(
                f"source {source.id!r}: group {source.group!r} has no split"
            )
        if source.kind not in SOURCE_KINDS:
            problems.append(f"source {source.id!r}: kind must be one of {SOURCE_KINDS}")
    for group in sorted(set(manifest.splits) - used_groups):
        problems.append(f"group {group!r} has no source")

    records: dict[str, ImageRecord] = {}
    for record in manifest.images:
        if record.id in records:
            problems.append(f"image id {record.id!r} appears twice")
            continue
        records[record.id] = record
        if not _ID_PATTERN.match(record.id):
            problems.append(f"image id {record.id!r} must match {_ID_PATTERN.pattern}")
        source = manifest.sources.get(record.source)
        if source is None:
            problems.append(f"image {record.id}: unknown source {record.source!r}")
        elif record.group != source.group:
            problems.append(
                f"image {record.id}: group {record.group!r} differs from its "
                f"source's group {source.group!r}"
            )
        problems += _check_geometry(record, manifest)

    boxes_path = directory / BOXES_FILE
    boxes = None
    if not boxes_path.is_file():
        problems.append(f"{BOXES_FILE} is missing")
    else:
        if sha256_file(boxes_path) != manifest.boxes_sha256:
            problems.append(f"{BOXES_FILE}: sha256 differs from the manifest")
        try:
            boxes = read_boxes(directory)
        except (ValueError, TypeError) as err:
            problems.append(f"{BOXES_FILE}: {err}")
        if boxes is not None:
            problems += _check_boxes(boxes, manifest, records)

    split_of_hash: dict[str, tuple[str, str]] = {}
    for record in manifest.images:
        split = manifest.splits.get(record.group)
        if split is None:
            continue
        seen = split_of_hash.setdefault(record.image_sha256, (split, record.id))
        if seen[0] != split:
            problems.append(
                f"images {seen[1]} ({seen[0]}) and {record.id} ({split}) are "
                "identical but in different splits"
            )

    if check_files:
        stray = sorted(p.name for p in (directory / IMAGES_DIR).glob("*.npy"))
        if stray:
            problems.append(
                f"{IMAGES_DIR}/ holds .npy files {stray[:5]}; image loaders may read "
                "them instead of the PNGs"
            )
        chirps_of = (
            {str(k): v for k, v in boxes[boxes["kind"] == "chirp"].groupby("image_id")}
            if boxes is not None
            else {}
        )
        empty = (
            boxes.iloc[:0] if boxes is not None else pd.DataFrame(columns=BOX_COLUMNS)
        )
        for record in manifest.images:
            image_path = directory / record.image_file
            label_path = directory / record.label_file
            if not image_path.is_file() or not label_path.is_file():
                problems.append(f"image {record.id}: image or label file missing")
                continue
            if sha256_file(image_path) != record.image_sha256:
                problems.append(
                    f"{record.image_file}: sha256 differs from the manifest"
                )
            if sha256_file(label_path) != record.label_sha256:
                problems.append(
                    f"{record.label_file}: sha256 differs from the manifest"
                )
            try:
                image = read_image(image_path)
            except (OSError, ValueError) as err:
                problems.append(str(err))
            else:
                expected = (manifest.tiling.height_px, manifest.tiling.width_px, 3)
                if image.shape != expected:
                    problems.append(
                        f"{record.image_file}: shape {image.shape}, expected {expected}"
                    )
            chirps = cast("pd.DataFrame", chirps_of.get(record.id, empty))
            problems += _check_labels(record, label_path, chirps)
    if problems:
        raise DatasetError(problems)
    return manifest


# ------------------------------------------------------------ the dataset


@dataclass(frozen=True)
class Dataset:
    """A validated detector dataset: directory, manifest and boxes."""

    directory: Path
    manifest: Manifest
    boxes: pd.DataFrame

    def image_path(self, image_id: str) -> Path:
        """Absolute path of an image."""
        return self.directory / self.manifest.image(image_id).image_file

    def label_path(self, image_id: str) -> Path:
        """Absolute path of an image's YOLO labels."""
        return self.directory / self.manifest.image(image_id).label_file

    def boxes_of(self, image_id: str, kind: BoxKind = "chirp") -> pd.DataFrame:
        """Boxes of one image and kind, ordered by ``box_index``."""
        mask = (_column(self.boxes, "image_id") == image_id) & (
            _column(self.boxes, "kind") == kind
        )
        rows = cast("pd.DataFrame", self.boxes.loc[mask.to_numpy()])
        return rows.sort_values("box_index", kind="stable").reset_index(drop=True)

    def tracks_dir(self, source_id: str) -> Path | None:
        """Tracks directory of a source (resolved against the dataset dir)."""
        tracks = self.manifest.sources[source_id].tracks
        if tracks is None:
            return None
        path = Path(tracks)
        return path if path.is_absolute() else self.directory / path


def load_dataset(directory: str | Path, *, check_files: bool = True) -> Dataset:
    """Validate a dataset and load its manifest and boxes.

    See `validate_dataset` for ``check_files`` and the errors raised.
    """
    directory = Path(directory).resolve()
    manifest = validate_dataset(directory, check_files=check_files)
    return Dataset(directory=directory, manifest=manifest, boxes=read_boxes(directory))


def dataset_ground_truth(
    dataset: Dataset, image_ids: Iterable[str] | None = None
) -> GroundTruth:
    """Ground truth of dataset images for `chirpdetector.evaluation`.

    Every image is one recording (named by its id) with one exhaustively
    labelled region, its extent; its dataset is the source's kind. Chirps
    become ``chirp`` labels (with ``emitter_id`` and an ``emitter_hz``
    column), ignore regions ``ignore`` labels. Images may overlap in time,
    so they are not merged into their source recording here.

    """
    from chirpdetector.evaluation import GroundTruth

    ids = list(dataset.manifest.image_ids() if image_ids is None else image_ids)
    regions, labels = [], []
    for image_id in ids:
        record = dataset.manifest.image(image_id)
        kind = dataset.manifest.sources[record.source].kind
        t0, t1, f0, f1 = record.extent
        regions.append(
            {
                "recording": image_id,
                "dataset": kind,
                "region": 0,
                "t_start_s": t0,
                "t_end_s": t1,
                "f_low_hz": f0,
                "f_high_hz": f1,
            }
        )
        for box_kind in BOX_KINDS:
            rows = dataset.boxes_of(image_id, box_kind)
            for row in rows.to_dict(orient="records"):
                labels.append(
                    {
                        "recording": image_id,
                        "dataset": kind,
                        **{c: row[c] for c in (*_BOX_EDGES, "t_center_s")},
                        "label": box_kind,
                        "emitter_id": row["emitter_id"],
                        "emitter_hz": row["emitter_hz"],
                    }
                )
    region_df = pd.DataFrame(
        regions,
        columns=[
            "recording",
            "dataset",
            "region",
            "t_start_s",
            "t_end_s",
            "f_low_hz",
            "f_high_hz",
        ],
    )
    label_columns = [
        "recording",
        "dataset",
        "t_start_s",
        "t_end_s",
        "f_low_hz",
        "f_high_hz",
        "t_center_s",
        "label",
        "emitter_id",
        "emitter_hz",
    ]
    label_df = pd.DataFrame(labels, columns=label_columns)
    label_df["emitter_id"] = pd.array(
        _column(label_df, "emitter_id").to_numpy(), dtype="Int64"
    )
    return GroundTruth(region_df, label_df)


def pixel_boxes_to_timefreq(
    xyxy: NDArray[np.float64], geometry: ImageGeometry
) -> NDArray[np.float64]:
    """Pixel boxes ``(n, 4)`` -> ``(n, 4)`` ``t_start_s, t_end_s, f_low_hz, f_high_hz``."""
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    t0, t1, f0, f1 = pixel_box_to_timefreq(
        xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3], geometry
    )
    return np.stack([t0, t1, f0, f1], axis=1)
