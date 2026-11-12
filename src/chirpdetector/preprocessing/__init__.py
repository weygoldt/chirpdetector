"""Preprocessing of raw grid recordings before the spectrogram.

Click removal (`ClickRemoval`, `remove_clicks`) repairs short broadband
impulses that v1 often mistook for chirps. Long recordings, chunk by chunk
(identical to cleaning the whole)::

    plan = plan_span(start, stop, recording.n_samples, spec_config)
    samples, mask = read_clean(recording, plan.read_start, plan.read_stop)
    spec = compute_spectrogram_span(samples, plan, spec_config)
"""

from chirpdetector.preprocessing.clicks import (
    CleanPlan,
    ClickRemoval,
    SampleSource,
    clean_channel,
    clean_span,
    click_spans,
    plan_clean,
    read_clean,
    remove_clicks,
)

__all__ = [
    "CleanPlan",
    "ClickRemoval",
    "SampleSource",
    "clean_channel",
    "clean_span",
    "click_spans",
    "plan_clean",
    "read_clean",
    "remove_clicks",
]
