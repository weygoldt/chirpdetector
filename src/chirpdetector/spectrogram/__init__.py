"""Spectrogram front end shared by dataset building, training and inference."""

from chirpdetector.spectrogram.config import SpectrogramConfig
from chirpdetector.spectrogram.framing import (
    SpanPlan,
    complete_frames,
    context_samples,
    frame_times_s,
    n_frames_total,
    plan_span,
)

__all__ = [
    "SpanPlan",
    "SpectrogramConfig",
    "complete_frames",
    "context_samples",
    "frame_times_s",
    "n_frames_total",
    "plan_span",
]
