"""Chirp parameters from the spectrogram: a second-stage decoder.

For every detected chirp, the decoder estimates its **height** (largest
frequency excursion above the emitter's baseline, Hz), its **duration**
(full width at half height, ``fwhm_s``, and width at 10 % of the height,
``width10_s``) and its **undershoot** (deepest drop below the baseline
after the peak, Hz), each with a calibrated standard deviation.

Ground truth is the simulator's noise-free per-EOD-period frequency of the
emitter (`chirpdetector.params.truth`), never a spectrogram quantity. The
decoder sees the detector's own input spectrogram around the detection box
(`chirpdetector.params.crops`) and is trained on the boxes of the detector
that feeds it: a params bundle is only valid with that detector.

Modules
-------
truth
    Ground-truth definitions on the per-period frequency.
crops
    `decoder_inputs`: the one definition of the decoder's input.
model
    `ParamNet` (CNN with heteroscedastic Gaussian heads), target space.
samples
    Training and evaluation samples from simulated scenes (detector or
    footprint boxes, matching, truth, crops).
calibration
    Post-hoc removal of the decoder's shrinkage; sigma rescaling.
bundle
    Params bundles (weights, front end, crop, calibration, detector hash).
training
    `train_params` and its config; ``chirpdetector train params``.
inference
    `ParamsEstimator`: calibrated parameters for detection boxes
    (`chirpdetector.inference.predict_span` uses it).
evaluation, report
    Accuracy, calibration and distribution fidelity on test scenes, the
    real-data agreement check; ``chirpdetector evaluate params``.
"""

from chirpdetector.params.bundle import (
    ParamsBundle,
    is_params_bundle,
    load_params_bundle,
)
from chirpdetector.params.calibration import CalibrationConfig, ParamCalibration
from chirpdetector.params.crops import CropConfig, DecoderInputs, decoder_inputs
from chirpdetector.params.inference import PARAM_COLUMNS, ParamsEstimator
from chirpdetector.params.truth import PARAMETERS, ChirpParams, scene_truth

__all__ = [
    "PARAMETERS",
    "PARAM_COLUMNS",
    "CalibrationConfig",
    "ChirpParams",
    "CropConfig",
    "DecoderInputs",
    "ParamCalibration",
    "ParamsBundle",
    "ParamsEstimator",
    "decoder_inputs",
    "is_params_bundle",
    "load_params_bundle",
    "scene_truth",
]
