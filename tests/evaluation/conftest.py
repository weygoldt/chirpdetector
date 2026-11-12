"""Builders for hand-made predictions and ground truth."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chirpdetector.evaluation import GroundTruth

HALF_WIDTH_S = 0.0625  # exact in binary, so centre times survive (t0 + t1) / 2


def make_predictions(
    rows: Sequence[tuple[Any, ...]], recording: str = "rec"
) -> pd.DataFrame:
    """Predictions from ``(t_center_s, f_low_hz, f_high_hz, score[, emitter_id[, emitter_prob]])``."""
    out = []
    for row in rows:
        tc, f0, f1, score, *rest = row
        entry = {
            "t_start_s": tc - HALF_WIDTH_S,
            "t_end_s": tc + HALF_WIDTH_S,
            "f_low_hz": f0,
            "f_high_hz": f1,
            "score": score,
            "recording": recording,
        }
        if rest:
            entry["emitter_id"] = rest[0]
        if len(rest) > 1:
            entry["emitter_prob"] = rest[1]
        out.append(entry)
    columns = ["t_start_s", "t_end_s", "f_low_hz", "f_high_hz", "score", "recording"]
    df = pd.DataFrame(out, columns=[*columns, "emitter_id", "emitter_prob"])
    if df["emitter_id"].isna().to_numpy().all():
        df = df.drop(columns=["emitter_id"])
    if df["emitter_prob"].isna().to_numpy().all():
        df = df.drop(columns=["emitter_prob"])
    return df


def make_truth(
    regions: Sequence[tuple[float, float, float, float]] = (),
    labels: Sequence[tuple[Any, ...]] = (),
    recording: str = "rec",
    dataset: str = "ds",
) -> GroundTruth:
    """Ground truth from regions ``(t0, t1, f0, f1)`` and labels
    ``(t_center_s, f_low_hz, f_high_hz[, label[, emitter_id]])``."""
    region_df = pd.DataFrame(
        {
            "recording": recording,
            "dataset": dataset,
            "region": np.arange(len(regions)),
            "t_start_s": [r[0] for r in regions],
            "t_end_s": [r[1] for r in regions],
            "f_low_hz": [r[2] for r in regions],
            "f_high_hz": [r[3] for r in regions],
        }
    )
    label_df = pd.DataFrame(
        {
            "recording": recording,
            "dataset": dataset,
            "t_start_s": [r[0] - HALF_WIDTH_S for r in labels],
            "t_end_s": [r[0] + HALF_WIDTH_S for r in labels],
            "f_low_hz": [r[1] for r in labels],
            "f_high_hz": [r[2] for r in labels],
            "label": [r[3] if len(r) > 3 else "chirp" for r in labels],
            "emitter_id": pd.array(
                [r[4] if len(r) > 4 else pd.NA for r in labels], dtype="Int64"
            ),
        }
    )
    return GroundTruth(region_df, label_df)


@pytest.fixture
def truth_factory():
    """:func:`make_truth`."""
    return make_truth


@pytest.fixture
def predictions_factory():
    """:func:`make_predictions`."""
    return make_predictions
