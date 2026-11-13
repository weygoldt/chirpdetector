"""Run configuration of the production pipeline (TOML).

A run config names the models, the thresholds, the track source, the
compute budget and the validation outputs. Every key has a default except
the run name; unknown keys are refused, so a typo cannot silently fall back
to a default. Example (all keys)::

    [run]
    name = "v2_2019_competition"        # OUT/<name>/...
    out = "/mnt/data1/chirp_v2/detections"

    [models]
    detector = "/mnt/data1/chirp_v2/runs/dualhead_r18_sim_v4_r1/final/bundle"
    decoder = "/mnt/data1/chirp_v2/params/runs/v4-dualhead-sim-v4/bundle"
    allow_detector_mismatch = false

    [detection]
    store_threshold = 0.05      # every detection down to this score is stored
    operating_threshold = 0.5   # marked in the table (column `operating`)
    postprocess = true

    [assignment]
    tracks = "."                # wavetracker arrays, relative to the recording dir
    tolerance_hz = 20.0
    max_gap_s = 10.0
    baseline_window_s = 1.0
    baseline_guard_s = 0.0

    [compute]
    device = "cuda"
    max_vram_fraction = 0.4
    chunk_tiles = 48
    click_workers = 2
    clean_processes = 4
    prefetch = 5

    [validation]
    qc = true
    windows = 10
    window_s = 10.0
    candidates_per_cell = 6
    seed = 0

Models
------
``detector`` and ``decoder`` are bundle directories (relative to the config
file) or references to published models (`chirpdetector.pretrained`):
``"default"`` or ``"pretrained:NAME"``. Without ``detector``, the default
pretrained detector is used. Without ``decoder``, the published decoder
trained on the detector's boxes is used if there is one (for the default
detector: the default decoder), else none; ``decoder = "none"`` switches
it off. References resolve to ``"pretrained:<versioned name>"`` in the
resolved config, so a run made with one default model is not resumed with
another. Pretrained models are downloaded when the run loads them.

`default_config` is the config of ``chirpdetector run --out OUT
RECORDING...`` without a config file: pretrained models, every default.

The resolved config (defaults filled in) is written next to every output
and hashed (`config_hash`); a finished recording is only skipped on resume
if it was made with the same models and detection settings.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from chirpdetector import pretrained
from chirpdetector.pretrained import DEFAULT, PREFIX, Kind

__all__ = [
    "AssignmentConfig",
    "ComputeConfig",
    "ConfigError",
    "DetectionConfig",
    "ModelsConfig",
    "PipelineConfig",
    "ValidationConfig",
    "default_config",
    "load_config",
]

DETECTOR_BATCH = 16
"""Batch size of the detectors at inference; chunks hold whole batches."""


class ConfigError(ValueError):
    """A run config is malformed."""


@dataclass(frozen=True)
class ModelsConfig:
    """Model bundles.

    Attributes
    ----------
    detector
        Detector bundle directory, or ``"pretrained:NAME"`` (module
        docstring).
    decoder
        Chirp-parameter decoder bundle or ``"pretrained:NAME"`` (None: no
        chirp parameters).
    allow_detector_mismatch
        Use a decoder trained on another detector's boxes (not validated).
    """

    detector: Path | str
    decoder: Path | str | None = None
    allow_detector_mismatch: bool = False


@dataclass(frozen=True)
class DetectionConfig:
    """Thresholds and post-processing.

    Attributes
    ----------
    store_threshold
        Detections down to this score are stored, so that the operating
        threshold can change later without a re-run.
    operating_threshold
        Score at which a detection counts as a chirp (column ``operating``;
        the compatibility export and the QC use it). None: the detector
        bundle's threshold.
    postprocess
        Flag harmonic copies and duplicates (`chirpdetector.postprocess`
        with its default rules, the assignment's ``max_gap_s`` and the
        pipeline's assignment as ``emitter_id``); flagged rows are kept and
        marked, never dropped.
    """

    store_threshold: float = 0.05
    operating_threshold: float | None = None
    postprocess: bool = True


@dataclass(frozen=True)
class AssignmentConfig:
    """Assignment of detections to wavetracker tracks.

    See `chirpdetector.pipeline.assign` for the rule.

    Attributes
    ----------
    tracks
        Directory with the wavetracker arrays, relative to the recording
        directory ("." = the recording directory itself), or an absolute
        path template with ``{recording}``; "none" disables assignment.
    tolerance_hz
        Largest distance between the predicted emitter frequency and the
        assigned track's baseline.
    max_gap_s
        A track is live at a time if its detections around it are at most
        this far apart. 10 s: the competition tracks of the 33 trials'
        fish are in gaps longer than 1 s for 5.1 % of their time (median
        3.4 %, up to 27 %), longer than 10 s for 2.4 %; within 10 s a
        fish's baseline hardly moves, and the tolerance still applies to
        the interpolated baseline.
    baseline_window_s, baseline_guard_s
        The track's baseline at a chirp is the mean of its detections in
        ``[onset - window, onset - guard)`` (interpolated at the onset if
        it has none there).
    """

    tracks: str = "."
    tolerance_hz: float = 20.0
    max_gap_s: float = 10.0
    baseline_window_s: float = 1.0
    baseline_guard_s: float = 0.0


@dataclass(frozen=True)
class ComputeConfig:
    """Compute budget.

    Attributes
    ----------
    device
        Torch device of the spectrogram and the models.
    max_vram_fraction
        Cap on this process's share of GPU memory (the smaller of this and
        ``CHIRP_MAX_VRAM_FRACTION`` applies).
    chunk_tiles
        Detector tiles per chunk (a multiple of the detector batch, 16):
        48 tiles of 3.2 s with 1 s overlap are 105.6 s of recording.
    click_workers
        Threads per chunk for click removal (one per electrode at most).
    clean_processes
        Worker processes for click removal (0: threads in the main
        process, which compete with the GPU loop for the GIL).
    prefetch
        Chunks read and cleaned ahead of the GPU, in parallel.
    """

    device: str = "cuda"
    max_vram_fraction: float | None = 0.4
    chunk_tiles: int = 48
    click_workers: int = 2
    clean_processes: int = 4
    prefetch: int = 5


@dataclass(frozen=True)
class ValidationConfig:
    """Validation outputs.

    Attributes
    ----------
    qc
        Per-recording QC report (PNG + Markdown).
    windows
        Annotated spectrogram windows per recording (0: none).
    window_s
        Duration of a window.
    candidates_per_cell
        Chirps sampled per (score bin, ΔF bin) cell for the run's
        validation candidates file (0: no file).
    seed
        Seed of all random sampling.
    """

    qc: bool = True
    windows: int = 10
    window_s: float = 10.0
    candidates_per_cell: int = 6
    seed: int = 0


@dataclass(frozen=True)
class PipelineConfig:
    """A full run configuration."""

    name: str
    out: Path
    models: ModelsConfig
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    assignment: AssignmentConfig = field(default_factory=AssignmentConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def __post_init__(self) -> None:
        _validate(self)

    @property
    def run_dir(self) -> Path:
        """``out / name``."""
        return self.out / self.name

    def to_dict(self) -> dict[str, Any]:
        """JSON-able resolved config (paths as strings)."""
        return json.loads(json.dumps(asdict(self), default=str))

    def detection_hash(self) -> str:
        """sha256 of what determines the detection table.

        Models, thresholds, post-processing, assignment and the chunking
        (which does not change results but is recorded); not the run name,
        output root, device or validation settings.
        """
        d = self.to_dict()
        relevant = {
            k: d[k] for k in ("models", "detection", "assignment") if k in d
        } | {"chunk_tiles": d["compute"]["chunk_tiles"]}
        text = json.dumps(relevant, sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()

    def with_overrides(
        self,
        *,
        name: str | None = None,
        out: Path | None = None,
        **sections: dict[str, Any],
    ) -> PipelineConfig:
        """A copy with the run name, output root or some section keys replaced."""
        kwargs: dict[str, Any] = {}
        if name is not None:
            kwargs["name"] = name
        if out is not None:
            kwargs["out"] = out
        for section, values in sections.items():
            kwargs[section] = replace(getattr(self, section), **values)
        return replace(self, **kwargs)


def _validate(cfg: PipelineConfig) -> None:
    if not cfg.name or "/" in cfg.name or cfg.name in (".", ".."):
        msg = f"run name must be a plain directory name, got {cfg.name!r}"
        raise ConfigError(msg)
    det = cfg.detection
    if not 0 < det.store_threshold <= 1:
        msg = f"store_threshold must lie in (0, 1], got {det.store_threshold}"
        raise ConfigError(msg)
    if det.operating_threshold is not None and not (
        det.store_threshold <= det.operating_threshold <= 1
    ):
        msg = (
            f"operating_threshold ({det.operating_threshold}) must lie in "
            f"[store_threshold, 1] ([{det.store_threshold}, 1])"
        )
        raise ConfigError(msg)
    a = cfg.assignment
    for key in ("tolerance_hz", "max_gap_s", "baseline_window_s"):
        value = getattr(a, key)
        if not (math.isfinite(value) and value > 0):
            msg = f"assignment.{key} must be positive, got {value}"
            raise ConfigError(msg)
    if not 0 <= a.baseline_guard_s < a.baseline_window_s:
        msg = "assignment.baseline_guard_s must lie in [0, baseline_window_s)"
        raise ConfigError(msg)
    c = cfg.compute
    if c.chunk_tiles < DETECTOR_BATCH or c.chunk_tiles % DETECTOR_BATCH:
        msg = f"compute.chunk_tiles must be a multiple of {DETECTOR_BATCH}, got {c.chunk_tiles}"
        raise ConfigError(msg)
    if c.click_workers < 1 or c.prefetch < 1 or c.clean_processes < 0:
        msg = (
            "compute.click_workers and compute.prefetch must be at least 1, "
            "compute.clean_processes at least 0"
        )
        raise ConfigError(msg)
    if c.max_vram_fraction is not None and not 0 < c.max_vram_fraction <= 1:
        msg = f"compute.max_vram_fraction must lie in (0, 1], got {c.max_vram_fraction}"
        raise ConfigError(msg)
    v = cfg.validation
    if v.windows < 0 or v.candidates_per_cell < 0 or not v.window_s > 0:
        msg = "validation: windows and candidates_per_cell >= 0, window_s > 0"
        raise ConfigError(msg)


def _section[T](cls: type[T], data: dict[str, Any], name: str) -> T:
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - known
    if unknown:
        msg = f"[{name}]: unknown keys {sorted(unknown)} (known: {sorted(known)})"
        raise ConfigError(msg)
    return cls(**data)


def config_from_dict(
    data: dict[str, Any], base_dir: Path | None = None
) -> PipelineConfig:
    """Build a config from parsed TOML.

    Relative model paths are resolved against ``base_dir`` (the config
    file's directory).
    """
    data = {k: dict(v) if isinstance(v, dict) else v for k, v in data.items()}
    unknown = set(data) - {
        "run",
        "models",
        "detection",
        "assignment",
        "compute",
        "validation",
    }
    if unknown:
        msg = f"unknown sections {sorted(unknown)}"
        raise ConfigError(msg)
    run = dict(data.get("run", {}))
    if set(run) - {"name", "out"}:
        msg = f"[run]: unknown keys {sorted(set(run) - {'name', 'out'})}"
        raise ConfigError(msg)
    if "name" not in run:
        msg = "[run] needs a name"
        raise ConfigError(msg)
    base = base_dir or Path.cwd()

    def path(value: object) -> Path:
        p = Path(str(value)).expanduser()
        return p if p.is_absolute() else (base / p).resolve()

    models = dict(data.get("models", {}))
    detector = _pretrained(models.get("detector", DEFAULT), "detector")
    models["detector"] = path(models["detector"]) if detector is None else detector
    if "decoder" not in models:
        models["decoder"] = _decoder_for(detector)
    elif models["decoder"] in (None, "", "none"):
        models["decoder"] = None
    else:
        decoder = _pretrained(models["decoder"], "decoder")
        models["decoder"] = path(models["decoder"]) if decoder is None else decoder
    return PipelineConfig(
        name=str(run["name"]),
        out=path(run.get("out", "/mnt/data1/chirp_v2/detections")),
        models=_section(ModelsConfig, models, "models"),
        detection=_section(DetectionConfig, data.get("detection", {}), "detection"),
        assignment=_section(AssignmentConfig, data.get("assignment", {}), "assignment"),
        compute=_section(ComputeConfig, data.get("compute", {}), "compute"),
        validation=_section(ValidationConfig, data.get("validation", {}), "validation"),
    )


def _pretrained(value: object, kind: Kind) -> str | None:
    """``"pretrained:<name>"`` for a model reference, None for a path."""
    try:
        name = pretrained.reference(value, kind)
    except (KeyError, ValueError) as err:
        msg = f"[models] {kind}: {err.args[0] if err.args else err}"
        raise ConfigError(msg) from err
    return None if name is None else f"{PREFIX}{name}"


def _decoder_for(detector: str | None) -> str | None:
    """The published decoder trained on a published detector's boxes, if any."""
    if detector is None:
        return None
    name = detector.removeprefix(PREFIX)
    decoders = [
        m
        for m in pretrained.list_models()
        if m.kind == "decoder" and m.detector == name
    ]
    default = pretrained.ALIASES.get("decoder")
    decoders.sort(key=lambda m: m.name != default)  # the default decoder first
    return f"{PREFIX}{decoders[0].name}" if decoders else None


def default_config(out: str | Path, *, name: str = "default") -> PipelineConfig:
    """The built-in run config: pretrained models and every default.

    The device is CUDA if available, else the CPU.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return config_from_dict(
        {"run": {"name": name, "out": str(out)}, "compute": {"device": device}}
    )


def load_config(path: str | Path) -> PipelineConfig:
    """Read and validate a run config TOML file."""
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)
    return config_from_dict(data, base_dir=path.parent.resolve())
