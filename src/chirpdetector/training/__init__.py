"""Train chirp detectors with provenance, grouped CV and model bundles.

Typical use::

    from chirpdetector.training import DetectorTrainingConfig, train_detector

    config = DetectorTrainingConfig.from_toml("train.toml")
    result = train_detector("/mnt/data1/chirp_v2/datasets/d1", config, "runs/r1")
    result.bundle                     # the final model's bundle

or ``chirpdetector train detector train.toml --dataset DIR --run-dir DIR``.

Run long jobs, and anything on the GPU, through the workspace's
``scripts/guarded`` wrapper::

    scripts/guarded --name train-r1 --mem 24G --vram 0.6 -- \\
        uv run chirpdetector train detector train.toml --dataset D --run-dir R

It caps RAM in a systemd scope and sets ``CHIRP_MAX_VRAM_FRACTION``, which
training and inference apply with
``torch.cuda.set_per_process_memory_fraction`` before building a model
(the smaller of it and the config's ``max_vram_fraction``). Batch sizes
are fixed numbers in the config, chosen to fit under that cap; AutoBatch is
not available. If ultralytics still runs out of memory it halves the batch
and retries; ``fit.json`` records the batch actually used.

Modules
-------
config
    `DetectorTrainingConfig` (TOML): model, optimisation, augmentation,
    cross-validation, seed, device, GPU memory cap.
folds
    Grouped k-fold assignment.
detector
    `train_detector`, the run directory layout, `RunResult`.
provenance
    Git SHAs, ``uv.lock`` hash, versions and hardware of a run.
"""

from chirpdetector.training.config import (
    AugmentConfig,
    CVConfig,
    DetectorTrainingConfig,
    OptimConfig,
)
from chirpdetector.training.detector import (
    FitResult,
    RunResult,
    plan_fits,
    train_detector,
)
from chirpdetector.training.folds import assign_folds, fold_groups

__all__ = [
    "AugmentConfig",
    "CVConfig",
    "DetectorTrainingConfig",
    "FitResult",
    "OptimConfig",
    "RunResult",
    "assign_folds",
    "fold_groups",
    "plan_fits",
    "train_detector",
]
