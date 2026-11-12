"""Train detectors: grouped cross-validation, final fit, run directory, bundles.

`train_detector` trains fresh models on a detector dataset (see
`chirpdetector.data.contract`) with the implementation the config names
(`chirpdetector.models`):

* with ``cv.n_folds = k >= 2``: one model per fold of the ``train`` groups
  (grouped k-fold, `chirpdetector.training.folds`), each from scratch (or
  from the same pretrained weights), validated on its held-out groups;
* with ``cv.final_fit``: one model on all ``train`` groups, validated
  (early stopping, selection of the best epoch) on the ``val`` groups.
  Its bundle is the run's result.

``test`` groups are never read. A single fit can be run on its own
(``fold=k`` or ``fold="final"``), e.g. folds in parallel processes into
the same run directory; the run directory then refuses fits made with a
different config or dataset.

Run directory
-------------
::

    RUN/
      run.json             index: run id, dataset hash, fits, CV summary, bundle
      config.json          resolved training config (config.toml: the input file)
      provenance.json      code SHAs (+ dirty flags), uv.lock hash, versions, hardware
      dataset.json         copy of the dataset manifest
      folds.json           group -> fold
      fold-0/ ... final/   one directory per fit:
        fit.json           groups, image counts, best epoch, validation metrics
        provenance.json    as above, at the time of this fit
        curves.csv         per-epoch losses, metrics and learning rates
        bundle/            weights + bundle.json (`chirpdetector.bundle`)
        ultralytics/       the backend's own run directory (checkpoints, args)

GPU memory: run training through ``scripts/guarded`` (or set
``max_vram_fraction``); see `chirpdetector.resources`.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from chirpdetector.bundle import write_bundle
from chirpdetector.data.contract import MANIFEST_FILE, Dataset, load_dataset
from chirpdetector.models import FitRequest, get_backend
from chirpdetector.resources import VRAM_ENV
from chirpdetector.training import provenance
from chirpdetector.training.config import DetectorTrainingConfig
from chirpdetector.training.folds import assign_folds, fold_groups

__all__ = ["FitResult", "RunResult", "plan_fits", "train_detector"]

log = logging.getLogger(__name__)

FINAL = "final"
RUN_FILE = "run.json"
FIT_FILE = "fit.json"


@dataclass(frozen=True)
class FitResult:
    """One trained model of a run."""

    name: str
    train_groups: tuple[str, ...]
    val_groups: tuple[str, ...]
    n_train_images: int
    n_val_images: int
    best_epoch: int | None
    epochs_run: int
    metrics: Mapping[str, Any]
    bundle: Path
    weights_sha256: str
    curves: Path | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON form (``fit.json``)."""
        return {
            "name": self.name,
            "train_groups": list(self.train_groups),
            "val_groups": list(self.val_groups),
            "n_train_images": self.n_train_images,
            "n_val_images": self.n_val_images,
            "best_epoch": self.best_epoch,
            "epochs_run": self.epochs_run,
            "validation_metrics": dict(self.metrics),
            "bundle": str(self.bundle),
            "weights_sha256": self.weights_sha256,
            "curves": None if self.curves is None else str(self.curves),
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FitResult:
        """Inverse of `to_dict`."""
        return cls(
            name=data["name"],
            train_groups=tuple(data["train_groups"]),
            val_groups=tuple(data["val_groups"]),
            n_train_images=int(data["n_train_images"]),
            n_val_images=int(data["n_val_images"]),
            best_epoch=data["best_epoch"],
            epochs_run=int(data["epochs_run"]),
            metrics=dict(data["validation_metrics"]),
            bundle=Path(data["bundle"]),
            weights_sha256=data["weights_sha256"],
            curves=None if data.get("curves") is None else Path(data["curves"]),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class RunResult:
    """All fits of a run directory so far.

    Attributes
    ----------
    run_dir, run_id
        Where and which run.
    dataset_sha256
        Content hash of the dataset.
    fits
        Every completed fit (folds first, then ``final``).
    cv
        Mean, standard deviation and per-fold values of each validation
        metric over the completed folds (validation of fold models, not of
        the final model).
    bundle
        Bundle of the final fit, if it exists.
    """

    run_dir: Path
    run_id: str
    dataset_sha256: str
    fits: tuple[FitResult, ...]
    cv: Mapping[str, Any]
    bundle: Path | None

    def to_dict(self) -> dict[str, Any]:
        """JSON form (``run.json``)."""
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "dataset_sha256": self.dataset_sha256,
            "fits": [f.to_dict() for f in self.fits],
            "cv": dict(self.cv),
            "bundle": None if self.bundle is None else str(self.bundle),
        }


def _write_json(path: Path, data: Any) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=1, allow_nan=False)
            f.write("\n")
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def plan_fits(
    dataset: Dataset,
    config: DetectorTrainingConfig,
    fold: int | Literal["final"] | None = None,
) -> tuple[dict[str, int], list[tuple[str, tuple[str, ...], tuple[str, ...]]]]:
    """Fold assignment and the fits to run: ``(name, train groups, val groups)``.

    Raises
    ------
    ValueError
        If the requested fold does not exist, there are too few ``train``
        groups for the folds, or the final fit has no ``val`` groups.
    """
    manifest = dataset.manifest
    n_folds = config.cv.n_folds
    assignment = (
        assign_folds(manifest.group_sizes("train"), n_folds, config.seed)
        if n_folds >= 2
        else {}
    )
    if fold is None:
        names: list[str] = [f"fold-{k}" for k in range(n_folds if n_folds >= 2 else 0)]
        if config.cv.final_fit:
            names.append(FINAL)
    elif fold == FINAL:
        names = [FINAL]
    else:
        if not isinstance(fold, int) or not 0 <= fold < n_folds:
            msg = f"fold must be 'final' or in 0..{n_folds - 1}, got {fold!r}"
            raise ValueError(msg)
        names = [f"fold-{fold}"]
    if not names:
        msg = "nothing to train: cv.n_folds is 0 and cv.final_fit is false"
        raise ValueError(msg)
    fits = []
    for name in names:
        if name == FINAL:
            train, val = manifest.groups("train"), manifest.groups("val")
            if not val:
                msg = (
                    "the final fit needs 'val' groups for early stopping; the dataset "
                    "has none (assign some groups to 'val', or set cv.final_fit = false)"
                )
                raise ValueError(msg)
        else:
            train, val = fold_groups(assignment, int(name.split("-")[1]))
        if not train:
            msg = f"{name}: no training groups"
            raise ValueError(msg)
        fits.append((name, train, val))
    return assignment, fits


def _prepare_run_dir(
    run_dir: Path,
    config: DetectorTrainingConfig,
    config_file: Path | None,
    dataset: Dataset,
    assignment: Mapping[str, int],
) -> str:
    """Create or reuse a run directory; return the run id."""
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = config.to_dict()
    config_path = run_dir / "config.json"
    if config_path.exists():
        stored = json.loads(config_path.read_text())
        run = json.loads((run_dir / RUN_FILE).read_text())
        problems = []
        if stored != resolved:
            problems.append("a different training config")
        if run["dataset_sha256"] != dataset.manifest.content_sha256:
            problems.append("a different dataset")
        if problems:
            msg = f"{run_dir} holds a run with {' and '.join(problems)}; use a new run dir"
            raise ValueError(msg)
        return str(run["run_id"])
    run_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    _write_json(config_path, resolved)
    if config_file is not None:
        shutil.copyfile(config_file, run_dir / "config.toml")
    shutil.copyfile(dataset.directory / MANIFEST_FILE, run_dir / "dataset.json")
    _write_json(run_dir / "folds.json", dict(sorted(assignment.items())))
    _write_json(
        run_dir / "provenance.json",
        provenance.collect(
            run_id=run_id,
            dataset={
                "path": str(dataset.directory),
                "content_sha256": dataset.manifest.content_sha256,
            },
        ),
    )
    _write_json(
        run_dir / RUN_FILE,
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "dataset_sha256": dataset.manifest.content_sha256,
            "fits": [],
            "cv": {},
            "bundle": None,
        },
    )
    return run_id


def _cv_summary(fits: list[FitResult]) -> dict[str, Any]:
    folds = [f for f in fits if f.name != FINAL]
    if not folds:
        return {}
    keys = sorted(set.intersection(*(set(f.metrics) for f in folds)))
    out: dict[str, Any] = {"n_folds": len(folds), "metrics": {}}
    for key in keys:
        values = np.array([float(f.metrics[key]) for f in folds])
        out["metrics"][key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "per_fold": {f.name: float(f.metrics[key]) for f in folds},
        }
    return out


def _collect(run_dir: Path, run_id: str, dataset_sha256: str) -> RunResult:
    fits = []
    for path in sorted(run_dir.glob(f"*/{FIT_FILE}")):
        fits.append(FitResult.from_dict(json.loads(path.read_text())))
    fits.sort(key=lambda f: (f.name == FINAL, f.name))
    final = next((f for f in fits if f.name == FINAL), None)
    result = RunResult(
        run_dir=run_dir,
        run_id=run_id,
        dataset_sha256=dataset_sha256,
        fits=tuple(fits),
        cv=_cv_summary(fits),
        bundle=None if final is None else final.bundle,
    )
    _write_json(run_dir / RUN_FILE, result.to_dict())
    return result


def train_detector(
    dataset_dir: str | Path,
    config: DetectorTrainingConfig | str | Path,
    run_dir: str | Path,
    *,
    fold: int | Literal["final"] | None = None,
    check_files: bool = True,
) -> RunResult:
    """Train detectors on a dataset and write a run directory (module docstring).

    Parameters
    ----------
    dataset_dir
        Detector dataset; validated first.
    config
        Training config, or the path of its TOML file.
    run_dir
        Run directory (created; an existing one must hold the same config
        and dataset).
    fold
        Train only this fold (``0 .. n_folds - 1``) or only the final fit
        (``"final"``); default: all folds, then the final fit.
    check_files
        Hash and decode every dataset file before training.

    Returns
    -------
    RunResult
        All fits in the run directory, including earlier ones.
    """
    config_file = None
    if not isinstance(config, DetectorTrainingConfig):
        config_file = Path(config)
        config = DetectorTrainingConfig.from_toml(config_file)
    dataset = load_dataset(dataset_dir, check_files=check_files)
    assignment, fits = plan_fits(dataset, config, fold)
    run_dir = Path(run_dir).resolve()
    run_id = _prepare_run_dir(run_dir, config, config_file, dataset, assignment)
    backend = get_backend(config.model_type)
    if (
        config.device != "cpu"
        and os.environ.get(VRAM_ENV) is None
        and (config.max_vram_fraction is None)
    ):
        log.warning(
            "GPU memory is not capped: run through scripts/guarded or set "
            "max_vram_fraction (the GPU is shared)"
        )
    manifest = dataset.manifest
    for name, train_groups, val_groups in fits:
        fit_dir = run_dir / name
        if (fit_dir / FIT_FILE).exists():
            msg = f"{fit_dir} is already trained; use a new run dir to retrain"
            raise FileExistsError(msg)
        if fit_dir.exists():
            shutil.rmtree(fit_dir)  # an interrupted fit
        fit_dir.mkdir(parents=True)
        log.info(
            "%s: %d train groups, %d val groups",
            name,
            len(train_groups),
            len(val_groups),
        )
        _write_json(
            fit_dir / "provenance.json", provenance.collect(run_id=run_id, fit=name)
        )
        request = FitRequest(
            dataset=dataset,
            train_ids=manifest.image_ids(
                "train" if name == FINAL else None, train_groups
            ),
            val_ids=manifest.image_ids(None, val_groups),
            config=config,
            out_dir=fit_dir,
            name=name,
        )
        output = backend.fit(request)
        bundle = write_bundle(
            fit_dir / "bundle",
            weights=output.weights,
            model_type=config.model_type,
            model=output.model,
            spectrogram=manifest.spectrogram,
            tiling=manifest.tiling,
            preprocessing=manifest.preprocessing,
            score_threshold=config.score_threshold,
            training={
                "run_id": run_id,
                "fit": name,
                "dataset_sha256": manifest.content_sha256,
                "train_groups": list(train_groups),
                "val_groups": list(val_groups),
                "best_epoch": output.best_epoch,
                "epochs_run": output.epochs_run,
                "validation_metrics": dict(output.metrics),
                "config": config.to_dict(),
                "code": provenance.code_state(),
                "uv_lock": provenance.uv_lock(),
            },
        )
        if output.weights.resolve() != bundle.weights_path:
            output.weights.unlink(missing_ok=True)  # the bundle holds the copy
        result = FitResult(
            name=name,
            train_groups=train_groups,
            val_groups=val_groups,
            n_train_images=len(request.train_ids),
            n_val_images=len(request.val_ids),
            best_epoch=output.best_epoch,
            epochs_run=output.epochs_run,
            metrics=dict(output.metrics),
            bundle=bundle.path,
            weights_sha256=bundle.weights_sha256,
            curves=output.curves,
            details=dict(output.details),
        )
        _write_json(fit_dir / FIT_FILE, result.to_dict())
        log.info(
            "%s: bundle %s (weights %s)", name, bundle.path, bundle.weights_sha256[:12]
        )
    return _collect(run_dir, run_id, manifest.content_sha256)
