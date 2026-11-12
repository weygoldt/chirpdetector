"""Model bundles: weights plus everything needed to use and trust them.

A bundle is a directory::

    BUNDLE/
      bundle.json     what the weights are and how to feed them
      weights.pt      the weights (format depends on ``model_type``)
      metrics.json    evaluation of exactly these weights (added later)

``bundle.json`` holds the format version; the model type, which selects the
implementation in `chirpdetector.models`, and its description (e.g.
architecture and input shape); the class map (``["chirp"]``); the
spectrogram front end and tiling the model was trained on; the
preprocessing (click removal on or off, and its config); the default score
threshold; the training provenance (run id, dataset content hash, git SHAs
of chirpdetector and gridtools, ``uv.lock`` hash); and the sha256 of the
weights file.

`load_bundle` recomputes the sha256 of the weights and refuses a bundle
whose weights differ from the recorded hash. ``metrics.json`` is written by
`chirpdetector.evaluation.attach_to_bundle`, which refuses metrics of
predictions made by other weights; `load_bundle` refuses a ``metrics.json``
that names another weight hash. Reported numbers can therefore only belong
to the shipped weights (audit D1, N1).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chirpdetector.data.contract import CLASSES, Tiling, sha256_file
from chirpdetector.spectrogram import SpectrogramConfig

__all__ = [
    "BUNDLE_FILE",
    "BUNDLE_FORMAT",
    "BUNDLE_FORMAT_VERSION",
    "METRICS_FILE",
    "Bundle",
    "BundleError",
    "describe_bundle",
    "load_bundle",
    "write_bundle",
]

BUNDLE_FORMAT = "chirpdetector-model-bundle"
BUNDLE_FORMAT_VERSION = 1
BUNDLE_FILE = "bundle.json"
METRICS_FILE = "metrics.json"
WEIGHTS_FILE = "weights.pt"


class BundleError(ValueError):
    """A bundle is malformed, or its files do not match its record."""


@dataclass(frozen=True)
class Bundle:
    """A loaded, verified model bundle.

    Attributes
    ----------
    path
        Bundle directory.
    model_type
        Implementation key in `chirpdetector.models` (e.g.
        ``"ultralytics-yolo"``).
    model
        Implementation-specific description (architecture, input shape,
        head, ...).
    weights_file
        Weights file name inside the bundle.
    weights_sha256
        sha256 of the weights, verified on load.
    classes
        Class map, ``("chirp",)``.
    spectrogram, tiling
        Front end and image layout the model expects.
    preprocessing
        ``{"click_removal": None | ClickRemoval.to_dict()}``.
    score_threshold
        Default detection threshold.
    training
        Provenance: run id, fit name, dataset content hash, code SHAs, ...
    created_at
        ISO time the bundle was written.
    metrics
        Content of ``metrics.json`` if present (verified to belong to these
        weights), else None.
    """

    path: Path
    model_type: str
    model: Mapping[str, Any]
    weights_file: str
    weights_sha256: str
    classes: tuple[str, ...]
    spectrogram: SpectrogramConfig
    tiling: Tiling
    preprocessing: Mapping[str, Any]
    score_threshold: float
    training: Mapping[str, Any]
    created_at: str
    metrics: Mapping[str, Any] | None = field(default=None, compare=False)

    @property
    def weights_path(self) -> Path:
        """Absolute path of the weights."""
        return self.path / self.weights_file

    def to_dict(self) -> dict[str, Any]:
        """``bundle.json`` content."""
        return {
            "format": BUNDLE_FORMAT,
            "format_version": BUNDLE_FORMAT_VERSION,
            "model_type": self.model_type,
            "model": dict(self.model),
            "weights_file": self.weights_file,
            "weights_sha256": self.weights_sha256,
            "classes": list(self.classes),
            "spectrogram": self.spectrogram.to_dict(),
            "tiling": self.tiling.to_dict(),
            "preprocessing": dict(self.preprocessing),
            "score_threshold": float(self.score_threshold),
            "training": dict(self.training),
            "created_at": self.created_at,
        }


def write_bundle(
    directory: str | Path,
    *,
    weights: str | Path,
    model_type: str,
    model: Mapping[str, Any],
    spectrogram: SpectrogramConfig,
    tiling: Tiling,
    preprocessing: Mapping[str, Any],
    score_threshold: float,
    training: Mapping[str, Any],
) -> Bundle:
    """Create a bundle: copy the weights in and write ``bundle.json``.

    Parameters
    ----------
    directory
        New bundle directory (must not exist or be empty).
    weights
        Weights file to copy (stored as ``weights.pt``).
    model_type, model, spectrogram, tiling, preprocessing, score_threshold, training
        See `Bundle`.

    Returns
    -------
    Bundle
        The bundle, loaded back and verified.
    """
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        msg = f"{directory} exists and is not empty"
        raise FileExistsError(msg)
    if not 0 <= score_threshold <= 1:
        msg = f"score_threshold must lie in [0, 1], got {score_threshold}"
        raise ValueError(msg)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / WEIGHTS_FILE
    shutil.copyfile(weights, target)
    bundle = Bundle(
        path=directory.resolve(),
        model_type=model_type,
        model=dict(model),
        weights_file=WEIGHTS_FILE,
        weights_sha256=sha256_file(target),
        classes=CLASSES,
        spectrogram=spectrogram,
        tiling=tiling,
        preprocessing=dict(preprocessing),
        score_threshold=float(score_threshold),
        training=dict(training),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    (directory / BUNDLE_FILE).write_text(
        json.dumps(bundle.to_dict(), indent=1, allow_nan=False) + "\n"
    )
    return load_bundle(directory)


_REQUIRED = (
    "format",
    "format_version",
    "model_type",
    "model",
    "weights_file",
    "weights_sha256",
    "classes",
    "spectrogram",
    "tiling",
    "preprocessing",
    "score_threshold",
    "training",
    "created_at",
)


def load_bundle(path: str | Path) -> Bundle:
    """Load a bundle and verify it.

    Parameters
    ----------
    path
        Bundle directory (or its ``bundle.json``).

    Returns
    -------
    Bundle

    Raises
    ------
    BundleError
        If ``bundle.json`` is malformed or of another format version, the
        class map is not ``["chirp"]``, the weights are missing or their
        sha256 differs from the recorded one, or ``metrics.json`` belongs
        to other weights.
    FileNotFoundError
        If there is no ``bundle.json``.
    """
    path = Path(path)
    directory = path.parent if path.name == BUNDLE_FILE else path
    record_path = directory / BUNDLE_FILE
    if not record_path.is_file():
        msg = f"{record_path} does not exist"
        raise FileNotFoundError(msg)
    try:
        data = json.loads(record_path.read_text())
    except json.JSONDecodeError as err:
        msg = f"{record_path}: {err}"
        raise BundleError(msg) from err
    missing = [k for k in _REQUIRED if k not in data]
    unknown = sorted(set(data) - set(_REQUIRED))
    if missing or unknown:
        msg = f"{record_path}: missing keys {missing}, unknown keys {unknown}"
        raise BundleError(msg)
    if (
        data["format"] != BUNDLE_FORMAT
        or data["format_version"] != BUNDLE_FORMAT_VERSION
    ):
        msg = (
            f"{record_path}: format {data['format']!r} version "
            f"{data['format_version']!r}; this chirpdetector reads "
            f"{BUNDLE_FORMAT} version {BUNDLE_FORMAT_VERSION}"
        )
        raise BundleError(msg)
    if tuple(data["classes"]) != CLASSES:
        msg = f"{record_path}: classes {data['classes']}, expected {list(CLASSES)}"
        raise BundleError(msg)
    weights = directory / str(data["weights_file"])
    if not weights.is_file():
        msg = f"{record_path}: weights {weights.name} are missing"
        raise BundleError(msg)
    actual = sha256_file(weights)
    if actual != data["weights_sha256"]:
        msg = (
            f"{weights}: sha256 {actual} differs from the bundle's "
            f"{data['weights_sha256']}; the weights were replaced after the bundle "
            "was made"
        )
        raise BundleError(msg)
    try:
        spectrogram = SpectrogramConfig.from_dict(dict(data["spectrogram"]))
        tiling = Tiling.from_dict(data["tiling"])
    except (TypeError, ValueError) as err:
        msg = f"{record_path}: {err}"
        raise BundleError(msg) from err
    metrics = None
    metrics_path = directory / METRICS_FILE
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("weights_sha256") != actual:
            msg = (
                f"{metrics_path} was computed for weights "
                f"{metrics.get('weights_sha256')!r}, not this bundle's {actual}; "
                "re-evaluate the bundle or remove the file"
            )
            raise BundleError(msg)
    return Bundle(
        path=directory.resolve(),
        model_type=str(data["model_type"]),
        model=dict(data["model"]),
        weights_file=str(data["weights_file"]),
        weights_sha256=actual,
        classes=CLASSES,
        spectrogram=spectrogram,
        tiling=tiling,
        preprocessing=dict(data["preprocessing"]),
        score_threshold=float(data["score_threshold"]),
        training=dict(data["training"]),
        created_at=str(data["created_at"]),
        metrics=metrics,
    )


def describe_bundle(bundle: Bundle) -> str:
    """A human-readable summary (``chirpdetector bundle inspect``)."""
    cfg, tiling = bundle.spectrogram, bundle.tiling
    training = bundle.training
    code = training.get("code", {})
    lines = [
        f"bundle      {bundle.path}",
        f"model       {bundle.model_type} {json.dumps(dict(bundle.model))}",
        f"weights     {bundle.weights_file} sha256 {bundle.weights_sha256} (verified)",
        f"classes     {list(bundle.classes)}",
        f"threshold   {bundle.score_threshold:g}",
        (
            f"front end   windows {list(cfg.window_durations_s)} s, hop {cfg.hop_s:g} s, "
            f"{cfg.f_min_hz:g}-{cfg.f_max_hz:g} Hz ({cfg.n_freqs} bins), "
            f"{cfg.aggregation} over electrodes"
        ),
        (
            f"tiling      {tiling.width_px} x {tiling.height_px} px, "
            f"{tiling.tile_duration_s:g} s per tile, overlap {tiling.overlap_s:g} s"
        ),
        f"preprocess  {json.dumps(dict(bundle.preprocessing))}",
        f"run         {training.get('run_id')} ({training.get('fit')})",
        f"dataset     {training.get('dataset_sha256')}",
    ]
    for package, info in code.items():
        if isinstance(info, Mapping):
            dirty = " (dirty)" if info.get("dirty") else ""
            lines.append(f"code        {package} {info.get('sha')}{dirty}")
    if bundle.metrics is None:
        lines.append("metrics     none attached")
    else:
        detection = bundle.metrics.get("metrics", {}).get("detection", {})
        pooled = detection.get("all", {})
        lines.append(
            "metrics     "
            + ", ".join(
                f"{k} {pooled[k]:.3f}"
                for k in ("precision", "recall", "f1")
                if isinstance(pooled.get(k), int | float)
            )
            + f" at threshold {detection.get('reference_threshold')}"
        )
    lines.append(f"created     {bundle.created_at}")
    return "\n".join(lines)
