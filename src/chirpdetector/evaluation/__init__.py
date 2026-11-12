"""Evaluate chirp detection and emitter assignment against ground truth."""

from chirpdetector.evaluation.detection import (
    average_precision,
    bootstrap_regions,
    clopper_pearson,
    confidence_intervals,
    pr_curve,
    summarize,
)
from chirpdetector.evaluation.matching import (
    MatchConfig,
    Scope,
    admissible_pairs,
    assign,
)
from chirpdetector.evaluation.predictions import non_max_suppression, read_predictions
from chirpdetector.evaluation.schema import (
    LABEL_VALUES,
    GroundTruth,
    read_ground_truth,
    simulation_ground_truth,
    validate_predictions,
)

__all__ = [
    "LABEL_VALUES",
    "GroundTruth",
    "MatchConfig",
    "Scope",
    "admissible_pairs",
    "assign",
    "average_precision",
    "bootstrap_regions",
    "clopper_pearson",
    "confidence_intervals",
    "non_max_suppression",
    "pr_curve",
    "read_ground_truth",
    "read_predictions",
    "simulation_ground_truth",
    "summarize",
    "validate_predictions",
]
