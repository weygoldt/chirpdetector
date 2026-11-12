"""Assignment and end-to-end metrics on synthetic matched sets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gridtools.io import load_tracks

from chirpdetector.evaluation import delta_f_at, df_bins
from chirpdetector.evaluation.assignment import (
    assignment_summary,
    assignment_table,
    bin_names,
    confusion,
    coverage_accuracy,
    end_to_end,
    reliability,
)

NAMES = bin_names()


def test_delta_f_bins_on_a_frequency_ramp(
    make_ramp_trial: Callable[[Path], Path], tmp_path: Path
) -> None:
    tracks = load_tracks(make_ramp_trial(tmp_path / "ramp"))
    t = 0.03 + 0.1 * np.arange(100)
    delta, live = delta_f_at(tracks, t)
    assert live.tolist()[:2] == [1, 1]
    assert np.isnan(delta[:2]).all()
    np.testing.assert_allclose(delta[2:], 2 + 20 * t[2:])
    counts = Counter(df_bins(delta, live).tolist())
    assert counts == {
        "single": 2,
        "0-10": 2,
        "10-25": 8,
        "25-50": 12,
        "50-100": 25,
        "100+": 51,
    }
    assert NAMES == ["0-10", "10-25", "25-50", "50-100", "100+", "single", "unknown"]


def _items(rows: list[dict]) -> pd.DataFrame:
    base = {
        "recording": "r",
        "dataset": "d",
        "scope": "region",
        "region": 0,
        "outcome": "tp",
        "label_row": 0,
        "pred_row": 0,
        "label": "chirp",
        "true_emitter_id": pd.NA,
        "pred_emitter_id": pd.NA,
        "emitter_prob": np.nan,
        "df_bin": "0-10",
    }
    df = pd.DataFrame([{**base, **r} for r in rows])
    for column in (
        "region",
        "label_row",
        "pred_row",
        "true_emitter_id",
        "pred_emitter_id",
    ):
        df[column] = df[column].astype("Int64")
    df["emitter_prob"] = df["emitter_prob"].astype(float)
    return df


def _pair(
    i: int, true: int, pred: int | None, prob: float, df_bin: str = "0-10"
) -> dict:
    return {
        "label_row": i,
        "pred_row": i,
        "true_emitter_id": true,
        "pred_emitter_id": pd.NA if pred is None else pred,
        "emitter_prob": prob,
        "df_bin": df_bin,
    }


def test_assignment_table_selects_scored_pairs_once() -> None:
    items = _items(
        [
            _pair(0, 1, 1, 0.9),
            {**_pair(1, 1, 1, 0.9), "label": "chirp_emitter_unclear"},
            {**_pair(2, 1, 1, 0.9), "outcome": "fn", "pred_row": pd.NA},
            {**_pair(3, 1, 1, 0.9), "true_emitter_id": pd.NA},
            {**_pair(0, 2, 1, 0.9), "scope": "spot", "region": pd.NA, "label_row": 9},
            {**_pair(5, 2, 1, 0.9), "scope": "spot", "region": pd.NA},
        ]
    )
    table = assignment_table(items)
    assert table["pred_row"].tolist() == [0, 5]
    assert table["scope"].tolist() == ["region", "spot"]  # region match wins
    assert table["correct"].tolist() == [True, False]


def test_coverage_accuracy_and_ece_by_hand() -> None:
    pairs = (
        [_pair(i, 1, 1, 0.95) for i in range(3)]  # high bin: 3 of 4 right
        + [_pair(3, 1, 2, 0.95)]
        + [_pair(4 + i, 1, 1, 0.55) for i in range(3)]  # mid bin: 3 of 4 right
        + [_pair(7, 1, 2, 0.55)]
        + [_pair(8, 1, None, 0.3), _pair(9, 1, None, 0.2)]  # abstentions
    )
    table = assignment_table(_items(pairs))
    summary = assignment_summary(table, NAMES).set_index("df_bin")
    row = summary.loc["all"]
    assert (row["n"], row["n_assigned"], row["n_correct"]) == (10, 8, 6)
    assert row["coverage"] == 0.8
    assert row["accuracy"] == 0.75
    assert row["accuracy_all"] == 0.6
    assert row["ece"] == pytest.approx(0.5 * 0.2 + 0.5 * 0.2)
    assert row["mean_confidence"] == pytest.approx(0.75)
    bins = reliability(table, NAMES)
    pooled = bins[bins["df_bin"] == "all"]
    assert pooled["bin"].tolist() == [5, 9]
    assert pooled["accuracy"].tolist() == [0.75, 0.75]


def test_perfect_calibration_has_zero_ece() -> None:
    pairs = [_pair(i, 1, 1, 0.75) for i in range(3)] + [_pair(3, 1, 2, 0.75)]
    summary = assignment_summary(assignment_table(_items(pairs)), NAMES)
    assert summary.loc[0, "ece"] == pytest.approx(0.0)


def test_coverage_accuracy_curve() -> None:
    pairs = [
        _pair(0, 1, 1, 0.9),
        _pair(1, 1, 2, 0.8),
        _pair(2, 1, 1, 0.7),
        _pair(3, 1, None, 0.1),
    ]
    curve = coverage_accuracy(assignment_table(_items(pairs)), NAMES)
    pooled = curve[curve["df_bin"] == "all"]
    assert pooled["prob_threshold"].tolist() == [0.9, 0.8, 0.7]
    assert pooled["coverage"].tolist() == [0.25, 0.5, 0.75]
    assert pooled["accuracy"].tolist() == [1.0, 0.5, 2 / 3]


def test_curve_needs_probabilities() -> None:
    pairs = [_pair(0, 1, 1, np.nan), _pair(1, 1, 1, 0.9)]
    assert coverage_accuracy(assignment_table(_items(pairs)), NAMES).empty


def test_metrics_are_stratified_by_delta_f() -> None:
    pairs = [
        _pair(0, 1, 1, 0.9, "100+"),
        _pair(1, 1, 1, 0.9, "100+"),
        _pair(2, 1, 2, 0.9, "0-10"),
        _pair(3, 1, 1, 0.9, "single"),
    ]
    summary = assignment_summary(assignment_table(_items(pairs)), NAMES)
    assert summary["df_bin"].tolist() == ["all", "0-10", "100+", "single"]
    assert summary["accuracy"].tolist() == [0.75, 0.0, 1.0, 1.0]


def test_confusion_counts() -> None:
    pairs = [
        _pair(0, 1, 1, 0.9),
        _pair(1, 1, 2, 0.9),
        _pair(2, 1, 2, 0.9),
        _pair(3, 2, None, 0.1),
    ]
    table = confusion(assignment_table(_items(pairs)))
    counts = {
        (
            int(r["true_emitter_id"]),
            None if pd.isna(r["pred_emitter_id"]) else int(r["pred_emitter_id"]),
        ): r["n"]
        for r in table.to_dict(orient="records")
    }
    assert counts == {(1, 1): 1, (1, 2): 2, (2, None): 1}


def test_end_to_end_counts_only_emitter_labelled_regions() -> None:
    items = _items(
        [
            # region 0: emitter-labelled; one unclear chirp is left out
            _pair(0, 1, 1, 0.9),
            _pair(1, 1, 2, 0.9),
            {
                **_pair(2, 1, 1, 0.9),
                "label": "chirp_emitter_unclear",
                "true_emitter_id": pd.NA,
            },
            {**_pair(3, 1, None, 0.9), "outcome": "fn", "pred_row": pd.NA},
            {**_pair(4, 1, 1, 0.9), "outcome": "fp", "label_row": pd.NA, "label": None},
            # region 1: chirps without emitters, not eligible
            {**_pair(5, 1, 1, 0.9), "region": 1, "true_emitter_id": pd.NA},
            {**_pair(6, 1, 1, 0.9), "region": 1, "outcome": "fp", "label": None},
            # spot: a correct pair and a prediction on a not_chirp label
            {**_pair(7, 3, 3, 0.9), "scope": "spot", "region": pd.NA},
            {
                **_pair(8, 3, 3, 0.9),
                "scope": "spot",
                "region": pd.NA,
                "outcome": "fp",
                "label": "not_chirp",
                "true_emitter_id": pd.NA,
            },
        ]
    )
    e2e = end_to_end(items, NAMES)
    pooled = e2e[(e2e["level"] == "all") & (e2e["df_bin"] == "all")].set_index("scope")
    region = pooled.loc["region"]
    assert (region["labels"], region["detections"], region["tp"]) == (3, 3, 1)
    assert region["n_regions"] == 1
    spot = pooled.loc["spot"]
    assert (spot["labels"], spot["detections"], spot["tp"]) == (1, 2, 1)
    assert spot["precision"] == 0.5
