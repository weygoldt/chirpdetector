"""Train the chirp-parameter decoder: samples, fit, calibration, bundle.

Typical use::

    chirpdetector train params params.toml \\
        --scenes /mnt/data1/chirp_v2/scenes/train_v3 \\
        --detector /mnt/data1/chirp_v2/runs/yolo26s_sim_v3_r1/final/bundle \\
        --run-dir /mnt/data1/chirp_v2/params/runs/p1

(run it through ``scripts/guarded``; the GPU is shared). `train_params`

1. finds the scenes and their splits (by default the ``train``/``val``/
   ``test`` directory each scene lies in, as the detector datasets built
   with ``--split-dir``; ``split`` can give seed ranges instead);
2. extracts (or reuses) the sample set (`chirpdetector.params.samples`):
   with ``boxes.mode = "detector"`` the detector bundle is run over every
   scene and its predictions matched to the true chirps, so the decoder
   learns on the boxes it will see;
3. trains `ParamNet` on the usable ``train`` samples (beta-NLL in model
   space, AdamW, one-cycle learning rate, AMP on CUDA), keeps the epoch
   with the smallest mean validation error on the ``val`` samples and stops
   after ``optim.patience`` epochs without improvement (``test`` samples are
   never read);
4. fits the calibration (`chirpdetector.params.calibration`) on the
   ``val`` samples at the detector's operating threshold, and the domain
   (scalar ranges seen in training, sigma thresholds);
5. writes the params bundle, which records the detector's weight hash.

Configuration (TOML; every key optional, these are the defaults)::

    seed = 0
    device = "auto"            # "cpu", "cuda", "cuda:1"
    max_vram_fraction = 0.1    # optional; CHIRP_MAX_VRAM_FRACTION also applies
    workers = 4                # sample extraction processes
    scene_background = "hybrid"  # only scenes on real background segments
                                 # ("any": also synthetic-noise scenes)
    split = "directory"        # split = the scene's train/val/test directory;
                               # or seed ranges "train=0:1600,val=1600:1800,test=1800:2000"

    [boxes]                    # chirpdetector.params.samples.BoxSource
    mode = "detector"          # or "footprint"
    score_threshold = 0.25     # detector boxes kept for the sample set

    [crop]                     # chirpdetector.params.crops.CropConfig
    size = [96, 96]
    margin_rel = 0.15
    canvas_s = 3.2

    [model]                    # chirpdetector.params.model.ModelConfig
    widths = [32, 48, 64]

    [optim]
    epochs = 40
    batch = 256
    lr = 0.002
    weight_decay = 0.0001
    patience = 8
    beta = 0.5                 # beta-NLL
    amp = true
    brightness = 0.08          # uniform offset in [-b, b] added to crops
    min_score = 0.25           # train on boxes with at least this score

    [calibration]              # chirpdetector.params.calibration.CalibrationConfig
    method = "inverse-median"
    min_score = 0.5            # fit on val boxes at the operating threshold

Run directory::

    RUN/
      run.json             run id, sample set, detector, best epoch, bundle
      config.json          resolved config (config.toml: the input file)
      provenance.json      code SHAs, uv.lock hash, versions, hardware
      samples/             sample set (unless --samples points elsewhere)
      curves.csv           per-epoch losses, validation errors, learning rate
      val_predictions.parquet  raw validation predictions (calibration input)
      bundle/              the params bundle
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
import tomllib
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from chirpdetector.bundle import Bundle, load_bundle
from chirpdetector.data.scenes import SplitRule
from chirpdetector.params.bundle import ParamsBundle, write_params_bundle
from chirpdetector.params.calibration import CalibrationConfig, fit_calibration
from chirpdetector.params.crops import SCALAR_NAMES, CropConfig
from chirpdetector.params.model import (
    LOG_OFFSETS,
    ModelConfig,
    ParamNet,
    Standardization,
    beta_nll,
    to_model_space,
)
from chirpdetector.params.samples import (
    SPLIT_BY_DIRECTORY,
    BoxSource,
    SampleSet,
    discover_scenes,
    extract_samples,
    parse_split,
)
from chirpdetector.params.truth import PARAMETERS, TRUTH_PAD_S, TRUTH_VERSION
from chirpdetector.resources import apply_vram_limit
from chirpdetector.training import provenance

__all__ = [
    "OptimConfig",
    "ParamsRunResult",
    "ParamsTrainingConfig",
    "train_params",
]

log = logging.getLogger(__name__)

RUN_FILE = "run.json"
TARGET_DEFINITIONS = {
    "names": list(PARAMETERS),
    "units": {
        "height_hz": "Hz",
        "fwhm_s": "s",
        "width10_s": "s",
        "undershoot_hz": "Hz",
    },
    "log_offsets": dict(LOG_OFFSETS),
    "model_space": {p: f"log({p} + {LOG_OFFSETS[p]:g})" for p in PARAMETERS},
    "ground_truth": (
        "per-EOD-period frequency of the emitter (chirpdetector.params.truth): "
        "cycle boundaries at the 2 pi crossings of SceneModel.phase with only this "
        "chirp (the emitter's other chirps omitted, phase='natural'), "
        "delta_k = 1/T_k minus the period mean of the baseline (with drift); "
        "height = max delta_k; fwhm/width10 from the crossings of 50 %/10 % of the "
        "height bracketing the peak, interpolated between period midpoints; "
        "undershoot = max(0, -min delta_k after the peak)"
    ),
    "truth_pad_s": TRUTH_PAD_S,
    "truth_version": TRUTH_VERSION,
}


def _from_table(cls: type, data: Any, what: str) -> Any:
    if not isinstance(data, Mapping):
        msg = f"[{what}] must be a table, got {data!r}"
        raise ValueError(msg)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        msg = f"unknown keys in [{what}]: {unknown}; known: {sorted(known)}"
        raise ValueError(msg)
    return cls(**data)


@dataclass(frozen=True)
class OptimConfig:
    """Optimisation of the decoder (module docstring)."""

    epochs: int = 40
    batch: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 8
    beta: float = 0.5
    amp: bool = True
    brightness: float = 0.08
    min_score: float | None = 0.25

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch < 2 or self.patience < 0:
            msg = "optim.epochs >= 1, optim.batch >= 2 and optim.patience >= 0 required"
            raise ValueError(msg)
        if not (self.lr > 0 and self.weight_decay >= 0 and 0 <= self.beta <= 1):
            msg = "optim.lr > 0, weight_decay >= 0 and 0 <= beta <= 1 required"
            raise ValueError(msg)
        if not 0 <= self.brightness < 1:
            msg = f"optim.brightness must lie in [0, 1), got {self.brightness}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ParamsTrainingConfig:
    """Everything that determines a decoder training run (module docstring).

    ``scene_background`` filters the scenes (`chirpdetector.params.samples.
    discover_scenes`); the default, ``"hybrid"``, keeps only scenes on real
    background segments (no synthetic-noise-only scenes in training,
    validation or test). ``split`` assigns scenes to splits:
    ``"directory"`` (default) by the ``train``/``val``/``test`` directory
    they lie in, or seed ranges such as
    ``"train=0:1600,val=1600:1800,test=1800:2000"``.
    ``calibration_min_score`` is the score threshold of the validation
    boxes the calibration is fitted on (None: the detector's own threshold).
    """

    seed: int = 0
    device: str = "auto"
    max_vram_fraction: float | None = None
    workers: int = 4
    scene_background: str = "hybrid"
    split: str = SPLIT_BY_DIRECTORY
    boxes: BoxSource = field(default_factory=BoxSource)
    crop: CropConfig = field(default_factory=CropConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    calibration_min_score: float | None = None

    def __post_init__(self) -> None:
        from chirpdetector.params.samples import BACKGROUNDS

        if self.scene_background not in BACKGROUNDS:
            msg = (
                f"scene_background must be one of {BACKGROUNDS}, got "
                f"{self.scene_background!r}"
            )
            raise ValueError(msg)
        try:
            parse_split(self.split)
        except ValueError as err:
            msg = f"split: {err}"
            raise ValueError(msg) from err

    @property
    def split_rule(self) -> SplitRule | None:
        """`split` parsed (None: by directory)."""
        return parse_split(self.split)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ParamsTrainingConfig:
        """From a parsed TOML document (or `to_dict` output)."""
        data = dict(data)
        calibration = dict(data.pop("calibration", {}) or {})
        min_score = calibration.pop(
            "min_score", data.pop("calibration_min_score", None)
        )
        top = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - top)
        if unknown:
            msg = f"unknown top-level keys: {unknown}; known: {sorted(top)}"
            raise ValueError(msg)
        return cls(
            boxes=BoxSource.from_dict(data.pop("boxes", {}) or {}),
            crop=CropConfig.from_dict(data.pop("crop", {}) or {}),
            model=ModelConfig.from_dict(data.pop("model", {}) or {}),
            optim=_from_table(OptimConfig, data.pop("optim", {}) or {}, "optim"),
            calibration=_from_table(CalibrationConfig, calibration, "calibration"),
            calibration_min_score=min_score,
            **data,
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> ParamsTrainingConfig:
        """Load a TOML file."""
        with Path(path).open("rb") as f:
            return cls.from_dict(tomllib.load(f))

    def to_dict(self) -> dict[str, Any]:
        """The resolved config, in the TOML layout."""
        return {
            "seed": self.seed,
            "device": self.device,
            "max_vram_fraction": self.max_vram_fraction,
            "workers": self.workers,
            "scene_background": self.scene_background,
            "split": self.split,
            "boxes": self.boxes.to_dict(),
            "crop": self.crop.to_dict(),
            "model": self.model.to_dict(),
            "optim": asdict(self.optim),
            "calibration": {
                **asdict(self.calibration),
                "fit_from": dict(self.calibration.fit_from),
                "min_score": self.calibration_min_score,
            },
        }


@dataclass(frozen=True)
class ParamsRunResult:
    """Outcome of `train_params`."""

    run_dir: Path
    run_id: str
    bundle: Path
    best_epoch: int
    epochs_run: int
    validation: Mapping[str, Any]


# ----------------------------------------------------------------- helpers


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _targets(rows: pd.DataFrame) -> npt.NDArray[np.float64]:
    return to_model_space(rows[[f"true_{p}" for p in PARAMETERS]].to_numpy(float))


@dataclass
class _Split:
    """A split's samples in memory."""

    rows: pd.DataFrame
    crops: torch.Tensor  # uint8 (n, C, H, W), CPU
    scalars: npt.NDArray[np.float32]  # standardised
    targets: npt.NDArray[np.float32]  # standardised model space


def _load_split(
    samples: SampleSet, rows: pd.DataFrame, standardization: Standardization
) -> _Split:
    inputs = samples.inputs(rows)
    return _Split(
        rows=rows.reset_index(drop=True),
        crops=torch.from_numpy(inputs.crops),
        scalars=standardization.scalars(inputs.scalars),
        targets=standardization.targets(_targets(rows)),
    )


@torch.no_grad()
def _predict(
    model: ParamNet, split: _Split, device: torch.device, batch: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    model.eval()
    means, log_vars = [], []
    for start in range(0, len(split.rows), batch):
        x = split.crops[start : start + batch].to(device).float().div_(255.0)
        s = torch.from_numpy(split.scalars[start : start + batch]).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            m, lv = model(x, s)
        means.append(m.float().cpu().numpy())
        log_vars.append(lv.float().cpu().numpy())
    if not means:
        empty = np.zeros((0, len(PARAMETERS)), dtype=np.float32)
        return empty, empty.copy()
    return np.concatenate(means), np.concatenate(log_vars)


def _prepare_run_dir(
    run_dir: Path, config: ParamsTrainingConfig, config_file: Path | None
) -> str:
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = config.to_dict()
    if (run_dir / RUN_FILE).is_file():
        run = json.loads((run_dir / RUN_FILE).read_text())
        if run.get("bundle"):
            msg = f"{run_dir} already holds a trained decoder; use a new run dir"
            raise FileExistsError(msg)
        if json.loads((run_dir / "config.json").read_text()) != resolved:
            msg = f"{run_dir} holds an unfinished run with another config"
            raise ValueError(msg)
        return str(run["run_id"])
    run_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    _write_json(run_dir / "config.json", resolved)
    if config_file is not None:
        shutil.copyfile(config_file, run_dir / "config.toml")
    _write_json(run_dir / "provenance.json", provenance.collect(run_id=run_id))
    _write_json(run_dir / RUN_FILE, {"run_id": run_id, "bundle": None})
    return run_id


def _fit(
    model: ParamNet,
    train: _Split,
    val: _Split,
    config: ParamsTrainingConfig,
    device: torch.device,
    curves_path: Path,
    target_sd: Sequence[float],
) -> tuple[dict[str, torch.Tensor], int, int, list[dict[str, Any]]]:
    """Training loop with early stopping.

    Returns the state of the epoch with the smallest mean validation error
    (model space), its index, the number of epochs run and the curves.
    """
    optim = config.optim
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    opt = torch.optim.AdamW(
        model.parameters(), lr=optim.lr, weight_decay=optim.weight_decay
    )
    steps_per_epoch = math.ceil(len(train.rows) / optim.batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=optim.lr, total_steps=optim.epochs * steps_per_epoch, pct_start=0.15
    )
    amp = optim.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)
    sd = np.asarray(target_sd, dtype=np.float64)
    best_score, best_epoch, waited = math.inf, -1, 0
    best_state: dict[str, torch.Tensor] = {}
    curves: list[dict[str, Any]] = []
    started = time.perf_counter()
    epoch = 0
    for epoch in range(optim.epochs):
        model.train()
        total, n_batches = 0.0, 0
        order = rng.permutation(len(train.rows))
        for start in range(0, len(order), optim.batch):
            idx = np.sort(order[start : start + optim.batch])
            if idx.size < 2:
                continue
            index = torch.from_numpy(idx)
            x = train.crops[index].to(device, non_blocking=True).float().div_(255.0)
            if optim.brightness > 0:
                offset = (torch.rand(len(idx), 1, 1, 1, device=device) - 0.5) * (
                    2 * optim.brightness
                )
                x = (x + offset).clamp_(0.0, 1.0)
            s = torch.from_numpy(train.scalars[idx]).to(device)
            y = torch.from_numpy(train.targets[idx]).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                mean, log_var = model(x, s)
            loss = beta_nll(mean.float(), log_var.float(), y, beta=optim.beta)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            schedule.step()
            total += float(loss.item())
            n_batches += 1
        mean, log_var = _predict(model, val, device, 4 * optim.batch)
        err = np.abs(mean - val.targets) * sd  # model-space units
        mae = np.nanmean(err, axis=0)
        nll = float(
            np.nanmean(0.5 * (log_var + (mean - val.targets) ** 2 / np.exp(log_var)))
        )
        score = float(np.mean(mae))
        curves.append(
            {
                "epoch": epoch,
                "train_loss": total / max(1, n_batches),
                "val_nll": nll,
                "val_score": score,
                **{
                    f"val_mae_{p}": float(v)
                    for p, v in zip(PARAMETERS, mae, strict=True)
                },
                "lr": float(schedule.get_last_lr()[0]),
                "seconds": time.perf_counter() - started,
            }
        )
        pd.DataFrame(curves).to_csv(curves_path, index=False)
        log.info(
            "epoch %d: train %.4f, val nll %.4f, val mae (log) %s",
            epoch,
            curves[-1]["train_loss"],
            nll,
            " ".join(f"{v:.4f}" for v in mae),
        )
        if score < best_score:
            best_score, best_epoch, waited = score, epoch, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            waited += 1
            if optim.patience and waited >= optim.patience:
                log.info("no improvement for %d epochs: stopping", waited)
                break
    return best_state, best_epoch, epoch + 1, curves


def _spearman(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    from scipy.stats import spearmanr

    return float(np.asarray(spearmanr(a, b)[0]))


def _validation_summary(
    true_log: npt.NDArray[np.float64],
    mean_log: npt.NDArray[np.float64],
    cal_log: npt.NDArray[np.float64],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for j, p in enumerate(PARAMETERS):
        ok = np.isfinite(true_log[:, j])
        raw_err = mean_log[ok, j] - true_log[ok, j]
        cal_err = cal_log[ok, j] - true_log[ok, j]
        out[p] = {
            "n": int(ok.sum()),
            "median_abs_log_error_raw": float(np.median(np.abs(raw_err))),
            "median_abs_log_error_calibrated": float(np.median(np.abs(cal_err))),
            "spearman_error_vs_true_raw": _spearman(true_log[ok, j], raw_err),
            "spearman_error_vs_true_calibrated": _spearman(true_log[ok, j], cal_err),
        }
    return out


# ----------------------------------------------------------------- training


def train_params(
    config: ParamsTrainingConfig | str | Path,
    scenes: Sequence[str | Path],
    run_dir: str | Path,
    *,
    detector: Bundle | str | Path | None = None,
    samples_dir: str | Path | None = None,
) -> ParamsRunResult:
    """Train a decoder and write its bundle (module docstring).

    Parameters
    ----------
    config
        Training config, or the path of its TOML file.
    scenes
        Scene roots (`chirpdetector.params.samples.discover_scenes`).
    run_dir
        Run directory (created; an unfinished one with the same config is
        resumed, a finished one refused).
    detector
        Detector bundle (required for ``boxes.mode = "detector"``; its
        front end and preprocessing are used either way).
    samples_dir
        Sample-set directory (default ``RUN/samples``); an existing one is
        reused if it was made with the same settings.

    Returns
    -------
    ParamsRunResult
    """
    config_file = None
    if not isinstance(config, ParamsTrainingConfig):
        config_file = Path(config)
        config = ParamsTrainingConfig.from_toml(config_file)
    detector_bundle = (
        None
        if detector is None
        else (detector if isinstance(detector, Bundle) else load_bundle(detector))
    )
    run_dir = Path(run_dir).resolve()
    run_id = _prepare_run_dir(run_dir, config, config_file)
    device = _device(config.device)
    if device.type == "cuda":
        apply_vram_limit(config.max_vram_fraction, device)
    refs = discover_scenes(
        scenes, config.split_rule, background=config.scene_background
    )
    samples = extract_samples(
        refs,
        Path(samples_dir) if samples_dir is not None else run_dir / "samples",
        crop=config.crop,
        boxes=config.boxes,
        detector=detector_bundle,
        workers=config.workers,
        device=str(device),
    )
    train_rows = samples.select("train", usable=True, min_score=config.optim.min_score)
    val_rows = samples.select("val", usable=True, min_score=config.optim.min_score)
    if len(train_rows) < 2 or len(val_rows) < 2:
        msg = f"too few usable samples: {len(train_rows)} train, {len(val_rows)} val"
        raise ValueError(msg)
    standardization = Standardization.fit(
        _targets(train_rows), train_rows[list(SCALAR_NAMES)].to_numpy(float)
    )
    log.info("loading %d train and %d val samples", len(train_rows), len(val_rows))
    train = _load_split(samples, train_rows, standardization)
    val = _load_split(samples, val_rows, standardization)
    channels, height, width = samples.crops.shape[1:]
    model = ParamNet(
        int(channels),
        (int(height), int(width)),
        len(SCALAR_NAMES),
        len(PARAMETERS),
        config.model,
    ).to(device)
    best_state, best_epoch, epochs_run, _ = _fit(
        model,
        train,
        val,
        config,
        device,
        run_dir / "curves.csv",
        standardization.target_sd,
    )
    model.load_state_dict(best_state)
    weights = run_dir / "best.pt"
    torch.save(best_state, weights)

    # calibration on val boxes at the operating threshold
    threshold = config.calibration_min_score
    if threshold is None:
        threshold = (
            detector_bundle.score_threshold if detector_bundle is not None else None
        )
    cal_rows = samples.select("val", usable=True, min_score=threshold)
    cal = _load_split(samples, cal_rows, standardization)
    mean_z, log_var = _predict(model, cal, device, 4 * config.optim.batch)
    mean_log, sigma_log = standardization.unscale(mean_z, log_var)
    true_log = _targets(cal.rows)
    calibration = fit_calibration(
        true_log,
        mean_log,
        sigma_log,
        config.calibration,
        fitted_on={"split": "val", "min_score": threshold, "selection": "usable"},
    )
    cal_mean, cal_sigma = calibration.apply(mean_log, sigma_log)
    predictions = pd.DataFrame({"row": cal.rows["row"].to_numpy()})
    for j, p in enumerate(PARAMETERS):
        predictions[f"true_log_{p}"] = true_log[:, j]
        predictions[f"mean_log_{p}"] = mean_log[:, j]
        predictions[f"sigma_log_{p}"] = sigma_log[:, j]
    predictions.to_parquet(run_dir / "val_predictions.parquet", index=False)
    train_scalars = train_rows[list(SCALAR_NAMES)].to_numpy(float)
    domain = {
        "scalar_range": {
            name: [
                float(np.quantile(train_scalars[:, k], 0.001)),
                float(np.quantile(train_scalars[:, k], 0.999)),
            ]
            for k, name in enumerate(SCALAR_NAMES)
        },
        "sigma_threshold": {
            p: float(np.nanquantile(cal_sigma[:, j], config.calibration.flag_quantile))
            for j, p in enumerate(PARAMETERS)
        },
        "sigma_threshold_quantile": config.calibration.flag_quantile,
    }
    validation = _validation_summary(true_log, mean_log, cal_mean)
    fit_provenance = json.loads((run_dir / "provenance.json").read_text())
    n_weights = sum(p.numel() for p in model.parameters())
    bundle = write_params_bundle(
        run_dir / "bundle",
        weights=weights,
        model={
            "architecture": "ParamNet",
            "config": config.model.to_dict(),
            "in_channels": int(channels),
            "input_hw": [int(height), int(width)],
            "n_scalars": len(SCALAR_NAMES),
            "scalars": list(SCALAR_NAMES),
            "targets": list(PARAMETERS),
            "n_weights": int(n_weights),
        },
        spectrogram=samples_spectrogram(samples),
        preprocessing=dict(samples.manifest["preprocessing"]),
        crop=config.crop,
        targets=TARGET_DEFINITIONS,
        standardization=standardization,
        calibration=calibration,
        domain=domain,
        detector=samples.manifest.get("detector"),
        training={
            "run_id": run_id,
            "samples_path": str(samples.directory),
            "samples_sha256": dict(samples.manifest["files"]),
            "sample_counts": samples.manifest["counts"],
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "n_calibration": len(cal_rows),
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
            "validation": validation,
            "config": config.to_dict(),
            "code": fit_provenance.get("code"),
            "uv_lock": fit_provenance.get("uv_lock"),
        },
    )
    weights.unlink(missing_ok=True)
    result = ParamsRunResult(
        run_dir=run_dir,
        run_id=run_id,
        bundle=bundle.path,
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        validation=validation,
    )
    _write_json(
        run_dir / RUN_FILE,
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "samples": str(samples.directory),
            "detector": samples.manifest.get("detector"),
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
            "validation": validation,
            "bundle": str(bundle.path),
        },
    )
    return result


def samples_spectrogram(samples: SampleSet) -> Any:
    """The front end recorded in a sample set's manifest."""
    from chirpdetector.spectrogram import SpectrogramConfig

    return SpectrogramConfig.from_dict(dict(samples.manifest["spectrogram"]))


def load_run_bundle(run_dir: str | Path) -> ParamsBundle:
    """The params bundle of a finished run directory."""
    from chirpdetector.params.bundle import load_params_bundle

    run = json.loads((Path(run_dir) / RUN_FILE).read_text())
    if not run.get("bundle"):
        msg = f"{run_dir} has no bundle yet"
        raise FileNotFoundError(msg)
    return load_params_bundle(run["bundle"])
