"""Production pipeline: detect, decode and assign chirps in whole recordings.

``chirpdetector run CONFIG [RECORDINGS...] [--dataset ROOT]`` runs it;
``chirpdetector render`` re-renders any window from a run's outputs;
``chirpdetector check-reference`` compares the chunked pipeline with
`chirpdetector.inference.predict_span` on a part of a recording.

Modules
-------
config
    The run config (TOML) and its validation.
stream
    Chunked detection over a whole recording, equal to `predict_span` on
    all of it; prefetching of reads and click removal.
assign
    Assignment to wavetracker tracks by the predicted emitter frequency,
    with a calibrated confidence.
outputs
    Output layout, the chirp table's columns, metadata, legacy arrays.
run
    Resumable runs over many recordings, manifest, log, progress.
qc, windows, candidates
    Per-recording QC report, annotated sample windows (and on-demand
    rendering), validation candidates for the labelling tool.
check
    Comparison with the reference path.
"""

from chirpdetector.pipeline.assign import (
    ASSIGN_COLUMNS,
    SIM_V4_DUALHEAD,
    ErrorModel,
    assign_detections,
)
from chirpdetector.pipeline.config import PipelineConfig, load_config
from chirpdetector.pipeline.outputs import legacy_arrays, model_tag, read_chirps
from chirpdetector.pipeline.run import (
    discover_recordings,
    load_models,
    process_recording,
    run_pipeline,
)
from chirpdetector.pipeline.stream import RecordingSlice, detect_recording

__all__ = [
    "ASSIGN_COLUMNS",
    "SIM_V4_DUALHEAD",
    "ErrorModel",
    "PipelineConfig",
    "RecordingSlice",
    "assign_detections",
    "detect_recording",
    "discover_recordings",
    "legacy_arrays",
    "load_config",
    "load_models",
    "model_tag",
    "process_recording",
    "read_chirps",
    "run_pipeline",
]
