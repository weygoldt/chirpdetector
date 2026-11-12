"""Spectrogram front end shared by dataset building, training and inference."""

from chirpdetector.spectrogram.compute import (
    Spectrogram,
    compute_spectrogram,
    compute_spectrogram_span,
)
from chirpdetector.spectrogram.config import SpectrogramConfig
from chirpdetector.spectrogram.framing import (
    SpanPlan,
    complete_frames,
    context_samples,
    frame_times_s,
    n_frames_total,
    plan_span,
)
from chirpdetector.spectrogram.normalize import normalize, reference_level_db

__all__ = [
    "SpanPlan",
    "Spectrogram",
    "SpectrogramConfig",
    "complete_frames",
    "compute_spectrogram",
    "compute_spectrogram_span",
    "context_samples",
    "frame_times_s",
    "n_frames_total",
    "normalize",
    "plan_span",
    "reference_level_db",
]
