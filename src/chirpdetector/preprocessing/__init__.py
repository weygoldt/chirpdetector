"""Preprocessing of raw grid recordings before the spectrogram.

Click removal (`ClickRemoval`, `remove_clicks`) repairs short broadband
impulses on multi-electrode recordings. It always uses an electrode-
coincidence guard and refuses recordings with fewer than
``ClickRemoval().min_electrodes`` electrodes (9 by default; single-channel
data never), see `can_remove_clicks`. Long recordings, chunk by chunk
(identical to cleaning the whole)::

    plan = plan_span(start, stop, recording.n_samples, spec_config)
    samples, mask = read_clean(recording, plan.read_start, plan.read_stop)
    spec = compute_spectrogram_span(samples, plan, spec_config)
"""

from chirpdetector.preprocessing.clicks import (
    MAX_CHIRP_ELECTRODES,
    MIN_ISOLATED_THRESHOLD,
    CleanPlan,
    ClickRemoval,
    SampleSource,
    TooFewElectrodesError,
    can_remove_clicks,
    clean_span,
    click_spans,
    plan_clean,
    read_clean,
    remove_clicks,
)

__all__ = [
    "MAX_CHIRP_ELECTRODES",
    "MIN_ISOLATED_THRESHOLD",
    "CleanPlan",
    "ClickRemoval",
    "SampleSource",
    "TooFewElectrodesError",
    "can_remove_clicks",
    "clean_span",
    "click_spans",
    "plan_clean",
    "read_clean",
    "remove_clicks",
]
