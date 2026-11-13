"""Check the chunked pipeline against `predict_span` on a part of a recording.

The part ``[start, start + duration)`` is presented as a recording of its
own (`RecordingSlice`), so `predict_span(slice, 0, duration)` tiles exactly
the frames the pipeline tiles; both use the same models, threshold and
decoder. The detections are compared row by row (sorted by centre time);
``identical`` means equal counts and bit-equal boxes, scores and emitter
frequencies. Post-processing and assignment are not part of the
comparison (they are deterministic functions of the detections).
"""

from __future__ import annotations

import inspect
import time
from typing import Any

import numpy as np
import pandas as pd

from chirpdetector.inference.spans import predict_span
from chirpdetector.pipeline.stream import (
    RecordingSlice,
    SampleRecording,
    detect_recording,
)

__all__ = ["compare_with_reference"]

_COMPARE = ("t_start_s", "t_end_s", "f_low_hz", "f_high_hz", "score", "emitter_hz")


def compare_with_reference(
    bundle: Any,
    recording: SampleRecording,
    start_s: float,
    duration_s: float,
    *,
    detector: Any,
    estimator: Any | None,
    store_threshold: float,
    chunk_tiles: int,
    click_workers: int = 4,
    prefetch: int = 3,
    clean_processes: int = 0,
    device: str | None = None,
) -> dict[str, Any]:
    """Run both paths on a slice and compare (module docstring)."""
    fs = float(recording.samplerate_hz)
    a = round(start_s * fs)
    b = min(int(recording.n_samples), a + round(duration_s * fs))
    part = RecordingSlice(recording, a, b)
    kwargs: dict[str, Any] = {}
    params = inspect.signature(predict_span).parameters
    if "postprocess" in params:  # post-processing is compared separately
        kwargs["postprocess"] = False
    t = time.perf_counter()
    ref = predict_span(
        bundle,
        part,
        0.0,
        part.n_samples / fs,
        name="slice",
        score_threshold=store_threshold,
        detector=detector,
        device=device,
        params=estimator,
        **kwargs,
    )
    ref_s = time.perf_counter() - t
    t = time.perf_counter()
    ours = detect_recording(
        bundle,
        part,
        name="slice",
        detector=detector,
        estimator=estimator,
        store_threshold=store_threshold,
        chunk_tiles=chunk_tiles,
        click_workers=click_workers,
        prefetch=prefetch,
        clean_processes=clean_processes,
        device=device,
    ).detections
    ours_s = time.perf_counter() - t
    report: dict[str, Any] = {
        "start_s": start_s,
        "duration_s": part.n_samples / fs,
        "chunk_tiles": chunk_tiles,
        "clean_processes": clean_processes,
        "device": device,
        "n_reference": len(ref),
        "n_pipeline": len(ours),
        "reference_s": ref_s,
        "pipeline_s": ours_s,
    }
    same_n = len(ref) == len(ours)
    report["same_count"] = same_n
    if not same_n or not len(ref):
        report["identical"] = same_n
        return report
    r = ref.reset_index(drop=True)
    o = ours.reset_index(drop=True)
    diffs = {}
    exact = True
    for col in [*_COMPARE, *_param_columns(r, o)]:
        if col not in r.columns or col not in o.columns:
            continue
        x = r[col].to_numpy(dtype=np.float64)
        y = o[col].to_numpy(dtype=np.float64)
        both_nan = np.isnan(x) & np.isnan(y)
        d = np.where(both_nan, 0.0, np.abs(x - y))
        diffs[col] = float(np.nanmax(d)) if len(d) else 0.0
        if col in _COMPARE:
            exact &= bool(np.array_equal(x, y, equal_nan=True))
    report["max_abs_diff"] = diffs
    report["tiles_equal"] = bool(
        np.array_equal(r["tile"].to_numpy(), o["tile"].to_numpy())
    )
    if "params_flag" in r.columns:
        report["params_flag_equal"] = bool(
            (r["params_flag"].astype(str) == o["params_flag"].astype(str)).all()
        )
    report["identical"] = exact and report["tiles_equal"]
    return report


def _param_columns(r: pd.DataFrame, o: pd.DataFrame) -> list[str]:
    skip = {*_COMPARE, "tile", "image_index", "span_id", "click_fraction_box"}
    return [
        c
        for c in r.columns
        if c in o.columns and c not in skip and pd.api.types.is_float_dtype(r[c])
    ]
