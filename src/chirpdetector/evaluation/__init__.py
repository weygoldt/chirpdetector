"""Evaluate chirp detection and emitter assignment against ground truth."""

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
    "non_max_suppression",
    "read_ground_truth",
    "read_predictions",
    "simulation_ground_truth",
    "validate_predictions",
]
