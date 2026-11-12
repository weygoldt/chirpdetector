"""Evaluate chirp detection and emitter assignment against ground truth.

Every reported number is computed here, from the exact predictions (or the
weights that made them) being reported; :func:`attach_to_bundle` stores the
sha256 of what was evaluated next to the metrics.

Typical use::

    from chirpdetector.evaluation import (
        GroundTruth, evaluate, read_ground_truth, read_predictions,
    )

    truth = GroundTruth.concat(read_ground_truth(f) for f in label_files)
    predictions = pd.concat(read_predictions(f) for f in prediction_files)
    result = evaluate(predictions, truth, tracks={"2019-12-09-10_00": tracks})
    result.write("out/")

Modules
-------
schema
    Prediction and ground-truth tables; readers for audian labels, the
    labelling tool's files and simulated scenes (``chirps.csv``).
predictions
    Readers for v1/v2 prediction files; v1's non-maximum suppression.
matching
    The matching rule (default: ``detector_benchmark``'s) and scopes
    (exhaustive regions, non-exhaustive spot labels).
detection
    Precision, recall, F1, Clopper-Pearson and bootstrap CIs, PR curves, AP.
assignment
    Emitter accuracy, coverage, calibration and end-to-end metrics by ΔF.
report
    :func:`evaluate`, :class:`EvaluationResult`, :func:`attach_to_bundle`.
"""

from chirpdetector.evaluation.assignment import (
    assign_by_frequency,
    delta_f_at,
    df_bins,
)
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
from chirpdetector.evaluation.report import (
    DEFAULT_THRESHOLDS,
    EvaluationConfig,
    EvaluationResult,
    attach_to_bundle,
    evaluate,
    parse_thresholds,
    sha256_file,
)
from chirpdetector.evaluation.schema import (
    LABEL_VALUES,
    GroundTruth,
    read_ground_truth,
    simulation_ground_truth,
    validate_predictions,
)

__all__ = [
    "DEFAULT_THRESHOLDS",
    "LABEL_VALUES",
    "EvaluationConfig",
    "EvaluationResult",
    "GroundTruth",
    "MatchConfig",
    "Scope",
    "admissible_pairs",
    "assign",
    "assign_by_frequency",
    "attach_to_bundle",
    "average_precision",
    "bootstrap_regions",
    "clopper_pearson",
    "confidence_intervals",
    "delta_f_at",
    "df_bins",
    "evaluate",
    "non_max_suppression",
    "parse_thresholds",
    "pr_curve",
    "read_ground_truth",
    "read_predictions",
    "sha256_file",
    "simulation_ground_truth",
    "summarize",
    "validate_predictions",
]
