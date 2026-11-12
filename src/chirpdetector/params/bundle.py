"""Chirp-parameter bundles: decoder weights plus everything needed to use them.

A params bundle is a directory, laid out like a detector bundle
(`chirpdetector.bundle`)::

    BUNDLE/
      bundle.json     format, model, front end, crop, targets, calibration,
                      domain, detector, training provenance
      weights.pt      the decoder's state dict
      metrics.json    evaluation of exactly these weights (added later)
      eval/           figures and report of `chirpdetector evaluate params`

``bundle.json`` (format ``chirpdetector-params-bundle``) holds the model
type (``chirp-params``) and architecture; the spectrogram front end and
preprocessing (always the detector's); the crop geometry
(`chirpdetector.params.crops.CropConfig`); the target definitions and the
standardisation of targets and scalars; the post-hoc calibration
(`chirpdetector.params.calibration`); the validated domain (scalar ranges
seen in training, sigma thresholds); the **detector** whose boxes it was
trained on (weights sha256, path, run id; None for footprint boxes): a
decoder is only valid on that detector's boxes; and the training
provenance (run id, sample-set hash, code SHAs, config).

`load_params_bundle` recomputes the sha256 of the weights and refuses a
bundle whose weights differ, or whose ``metrics.json`` names other
weights.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chirpdetector.bundle import BUNDLE_FILE, METRICS_FILE, WEIGHTS_FILE, BundleError
from chirpdetector.data.contract import sha256_file
from chirpdetector.params.calibration import ParamCalibration
from chirpdetector.params.crops import CropConfig
from chirpdetector.params.model import ModelConfig, Standardization
from chirpdetector.spectrogram import SpectrogramConfig

__all__ = [
    "MODEL_TYPE",
    "PARAMS_BUNDLE_FORMAT",
    "PARAMS_BUNDLE_FORMAT_VERSION",
    "ParamsBundle",
    "attach_params_metrics",
    "describe_params_bundle",
    "is_params_bundle",
    "load_params_bundle",
    "write_params_bundle",
]

PARAMS_BUNDLE_FORMAT = "chirpdetector-params-bundle"
PARAMS_BUNDLE_FORMAT_VERSION = 1
MODEL_TYPE = "chirp-params"

_REQUIRED = (
    "format",
    "format_version",
    "model_type",
    "model",
    "weights_file",
    "weights_sha256",
    "spectrogram",
    "preprocessing",
    "crop",
    "targets",
    "standardization",
    "calibration",
    "domain",
    "detector",
    "training",
    "created_at",
)


@dataclass(frozen=True)
class ParamsBundle:
    """A loaded, verified params bundle (module docstring).

    Attributes
    ----------
    path
        Bundle directory.
    model_type
        ``"chirp-params"``.
    model
        Architecture: ``config`` (`ModelConfig` dict), ``in_channels``,
        ``input_hw``, ``n_scalars``, ``targets``.
    weights_file, weights_sha256
        Weights and their verified sha256.
    spectrogram, preprocessing
        Front end and preprocessing (the detector's).
    crop
        Crop geometry.
    targets
        Target definitions (names, units, model space, ground-truth rule).
    standardization
        Of targets (model space) and scalars.
    calibration
        Post-hoc calibration.
    domain
        Validated domain: ``scalar_range`` (per scalar ``[lo, hi]``),
        ``sigma_threshold`` (per parameter, model space).
    detector
        The detector whose boxes the decoder was trained on, or None.
    training
        Provenance.
    created_at
        ISO time the bundle was written.
    metrics
        ``metrics.json`` if present (verified to belong to these weights).
    """

    path: Path
    model_type: str
    model: Mapping[str, Any]
    weights_file: str
    weights_sha256: str
    spectrogram: SpectrogramConfig
    preprocessing: Mapping[str, Any]
    crop: CropConfig
    targets: Mapping[str, Any]
    standardization: Standardization
    calibration: ParamCalibration
    domain: Mapping[str, Any]
    detector: Mapping[str, Any] | None
    training: Mapping[str, Any]
    created_at: str
    metrics: Mapping[str, Any] | None = field(default=None, compare=False)

    @property
    def weights_path(self) -> Path:
        """Absolute path of the weights."""
        return self.path / self.weights_file

    @property
    def model_config(self) -> ModelConfig:
        """The architecture."""
        return ModelConfig.from_dict(self.model["config"])

    def to_dict(self) -> dict[str, Any]:
        """``bundle.json`` content."""
        return {
            "format": PARAMS_BUNDLE_FORMAT,
            "format_version": PARAMS_BUNDLE_FORMAT_VERSION,
            "model_type": self.model_type,
            "model": dict(self.model),
            "weights_file": self.weights_file,
            "weights_sha256": self.weights_sha256,
            "spectrogram": self.spectrogram.to_dict(),
            "preprocessing": dict(self.preprocessing),
            "crop": self.crop.to_dict(),
            "targets": dict(self.targets),
            "standardization": self.standardization.to_dict(),
            "calibration": self.calibration.to_dict(),
            "domain": dict(self.domain),
            "detector": None if self.detector is None else dict(self.detector),
            "training": dict(self.training),
            "created_at": self.created_at,
        }


def is_params_bundle(path: str | Path) -> bool:
    """Whether ``path`` (a directory or its ``bundle.json``) is a params bundle."""
    path = Path(path)
    record = path if path.name == BUNDLE_FILE else path / BUNDLE_FILE
    try:
        return json.loads(record.read_text()).get("format") == PARAMS_BUNDLE_FORMAT
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def write_params_bundle(
    directory: str | Path,
    *,
    weights: str | Path,
    model: Mapping[str, Any],
    spectrogram: SpectrogramConfig,
    preprocessing: Mapping[str, Any],
    crop: CropConfig,
    targets: Mapping[str, Any],
    standardization: Standardization,
    calibration: ParamCalibration,
    domain: Mapping[str, Any],
    detector: Mapping[str, Any] | None,
    training: Mapping[str, Any],
) -> ParamsBundle:
    """Create a params bundle: copy the weights in, write ``bundle.json``.

    Parameters
    ----------
    directory
        New bundle directory (must not exist or be empty).
    weights
        Weights file to copy (stored as ``weights.pt``).
    model, spectrogram, preprocessing, crop, targets, standardization, calibration, domain, detector, training
        See `ParamsBundle`.

    Returns
    -------
    ParamsBundle
        The bundle, loaded back and verified.
    """
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        msg = f"{directory} exists and is not empty"
        raise FileExistsError(msg)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / WEIGHTS_FILE
    shutil.copyfile(weights, target)
    bundle = ParamsBundle(
        path=directory.resolve(),
        model_type=MODEL_TYPE,
        model=dict(model),
        weights_file=WEIGHTS_FILE,
        weights_sha256=sha256_file(target),
        spectrogram=spectrogram,
        preprocessing=dict(preprocessing),
        crop=crop,
        targets=dict(targets),
        standardization=standardization,
        calibration=calibration,
        domain=dict(domain),
        detector=None if detector is None else dict(detector),
        training=dict(training),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    (directory / BUNDLE_FILE).write_text(
        json.dumps(bundle.to_dict(), indent=1, allow_nan=False) + "\n"
    )
    return load_params_bundle(directory)


def load_params_bundle(path: str | Path) -> ParamsBundle:
    """Load a params bundle and verify it.

    Raises
    ------
    BundleError
        If ``bundle.json`` is malformed or of another format or version,
        the weights are missing or their sha256 differs from the recorded
        one, or ``metrics.json`` belongs to other weights.
    FileNotFoundError
        If there is no ``bundle.json``.
    """
    path = Path(path)
    directory = path.parent if path.name == BUNDLE_FILE else path
    record = directory / BUNDLE_FILE
    if not record.is_file():
        msg = f"{record} does not exist"
        raise FileNotFoundError(msg)
    try:
        data = json.loads(record.read_text())
    except json.JSONDecodeError as err:
        msg = f"{record}: {err}"
        raise BundleError(msg) from err
    if data.get("format") != PARAMS_BUNDLE_FORMAT:
        msg = f"{record}: not a params bundle (format {data.get('format')!r})"
        raise BundleError(msg)
    if data.get("format_version") != PARAMS_BUNDLE_FORMAT_VERSION:
        msg = (
            f"{record}: params bundle version {data.get('format_version')!r}; this "
            f"chirpdetector reads version {PARAMS_BUNDLE_FORMAT_VERSION}"
        )
        raise BundleError(msg)
    missing = [k for k in _REQUIRED if k not in data]
    unknown = sorted(set(data) - set(_REQUIRED))
    if missing or unknown:
        msg = f"{record}: missing keys {missing}, unknown keys {unknown}"
        raise BundleError(msg)
    weights = directory / str(data["weights_file"])
    if not weights.is_file():
        msg = f"{record}: weights {weights.name} are missing"
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
        crop = CropConfig.from_dict(data["crop"])
        standardization = Standardization.from_dict(data["standardization"])
        calibration = ParamCalibration.from_dict(data["calibration"])
        ModelConfig.from_dict(data["model"]["config"])
    except (TypeError, ValueError, KeyError) as err:
        msg = f"{record}: {err}"
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
    return ParamsBundle(
        path=directory.resolve(),
        model_type=str(data["model_type"]),
        model=dict(data["model"]),
        weights_file=str(data["weights_file"]),
        weights_sha256=actual,
        spectrogram=spectrogram,
        preprocessing=dict(data["preprocessing"]),
        crop=crop,
        targets=dict(data["targets"]),
        standardization=standardization,
        calibration=calibration,
        domain=dict(data["domain"]),
        detector=None if data["detector"] is None else dict(data["detector"]),
        training=dict(data["training"]),
        created_at=str(data["created_at"]),
        metrics=metrics,
    )


def attach_params_metrics(bundle: ParamsBundle, metrics: Mapping[str, Any]) -> Path:
    """Write ``metrics.json`` for the bundle's weights (overwrites)."""
    content = {"weights_sha256": bundle.weights_sha256, **dict(metrics)}
    content["weights_sha256"] = bundle.weights_sha256
    path = bundle.path / METRICS_FILE
    path.write_text(json.dumps(content, indent=1, allow_nan=False) + "\n")
    return path


def describe_params_bundle(bundle: ParamsBundle) -> str:
    """A human-readable summary (``chirpdetector bundle inspect``)."""
    cfg = bundle.spectrogram
    training = bundle.training
    detector = bundle.detector
    code = training.get("code", {})
    lines = [
        f"bundle      {bundle.path}",
        f"model       {bundle.model_type} {json.dumps(dict(bundle.model))}",
        f"weights     {bundle.weights_file} sha256 {bundle.weights_sha256} (verified)",
        f"targets     {', '.join(bundle.targets.get('names', []))}",
        (
            f"front end   windows {list(cfg.window_durations_s)} s, hop {cfg.hop_s:g} s, "
            f"{cfg.f_min_hz:g}-{cfg.f_max_hz:g} Hz, {cfg.aggregation} over electrodes"
        ),
        (
            f"crop        {bundle.crop.size[0]} x {bundle.crop.size[1]}, margin "
            f"{bundle.crop.margin_rel:g}, canvas {bundle.crop.canvas_s:g} s"
        ),
        f"calibration {bundle.calibration.method}",
        (
            "detector    none (footprint boxes)"
            if detector is None
            else f"detector    {detector.get('weights_sha256')} ({detector.get('path')})"
        ),
        f"run         {training.get('run_id')}",
        f"samples     {training.get('samples_sha256')}",
    ]
    for package, info in code.items():
        if isinstance(info, Mapping):
            dirty = " (dirty)" if info.get("dirty") else ""
            lines.append(f"code        {package} {info.get('sha')}{dirty}")
    if bundle.metrics is None:
        lines.append("metrics     none attached")
    else:
        summary = bundle.metrics.get("summary", {})
        parts = [
            f"{name} {value}"
            for name, value in summary.items()
            if isinstance(value, str)
        ]
        lines.append("metrics     " + ("; ".join(parts) if parts else "attached"))
    lines.append(f"created     {bundle.created_at}")
    return "\n".join(lines)
