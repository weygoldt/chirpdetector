"""Spectrogram front end shared by dataset building, training and inference.

Typical use::

    config = SpectrogramConfig()
    spec = compute_spectrogram(signal, config, t0_s=t0)  # (n_samples, n_electrodes)
    image = to_image(normalize(spec, config))            # (H, W, C) uint8
    geometry = ImageGeometry.from_spectrogram(spec)
    x, y = timefreq_to_pixel(t_s, f_hz, geometry)

Long recordings, chunk by chunk (frames identical to the whole)::

    plan = plan_span(start, stop, n_samples_total, config)
    spec = compute_spectrogram_span(recording[plan.read_slice], plan, config)
"""

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
from chirpdetector.spectrogram.image import (
    ImageGeometry,
    pixel_box_to_timefreq,
    pixel_to_timefreq,
    timefreq_box_to_pixel,
    timefreq_to_pixel,
    to_image,
    to_three_channels,
)
from chirpdetector.spectrogram.normalize import normalize, reference_level_db

__all__ = [
    "ImageGeometry",
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
    "pixel_box_to_timefreq",
    "pixel_to_timefreq",
    "plan_span",
    "reference_level_db",
    "timefreq_box_to_pixel",
    "timefreq_to_pixel",
    "to_image",
    "to_three_channels",
]
