"""Prediction readers and v1's non-maximum suppression."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chirpdetector.evaluation import non_max_suppression, read_predictions
from chirpdetector.evaluation.schema import select


def test_reads_v1_detections_from_a_trial_directory(synthetic_trial: Path) -> None:
    df = read_predictions(synthetic_trial)
    assert len(df) == 8
    assert set(df["recording"]) == {synthetic_trial.name}
    assert df["emitter_id"].dtype == "Int64"
    assert (df["emitter_id"].to_numpy() == 0).all()  # v1 track_id
    assert np.isnan(df["emitter_prob"]).all()
    assert df["t_start_s"].is_monotonic_increasing


def test_reads_v2_tables_from_csv_and_parquet(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "t_start_s": [1.0, 2.0],
            "t_end_s": [1.1, 2.1],
            "f_low_hz": [700.0, 710.0],
            "f_high_hz": [800.0, 810.0],
            "score": [0.9, 0.4],
            "emitter_id": pd.array([3, pd.NA], dtype="Int64"),
            "emitter_prob": [0.8, np.nan],
        }
    )
    table.to_csv(tmp_path / "rec_a.csv", index=False)
    table.assign(recording="rec_b").to_parquet(tmp_path / "x.parquet")
    a = read_predictions(tmp_path / "rec_a.csv")
    b = read_predictions(tmp_path / "x.parquet")
    assert set(a["recording"]) == {"rec_a"}  # file stem
    assert set(b["recording"]) == {"rec_b"}  # column
    for df in (a, b):
        assert df["emitter_id"].tolist()[0] == 3
        assert pd.isna(df["emitter_id"].tolist()[1])
        assert df["emitter_prob"].tolist()[0] == 0.8


def test_reads_v1_boxes_and_keeps_the_batch(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "recording": ["r", "r"],
            "batch": [3.0, 4.0],
            "t1": [1.0, 2.0],
            "f1": [700.0, 700.0],
            "t2": [1.1, 2.1],
            "f2": [800.0, 800.0],
            "score": [0.9, 0.3],
        }
    ).to_csv(tmp_path / "boxes.csv", index=False)
    df = read_predictions(tmp_path / "boxes.csv")
    assert df["t_end_s"].tolist() == [1.1, 2.1]
    assert df["batch"].tolist() == [3.0, 4.0]
    assert set(df["recording"]) == {"r"}


def _boxes(rows: list[tuple[float, float, float, float, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["t_start_s", "f_low_hz", "t_end_s", "f_high_hz", "score", "batch"],
    ).assign(recording="r")


def test_nms_keeps_the_best_of_overlapping_boxes_per_batch() -> None:
    df = _boxes(
        [
            (0.0, 0.0, 1.0, 1.0, 0.6, 0),
            (0.05, 0.0, 1.05, 1.0, 0.9, 0),  # IoU 0.9 with the first
            (0.05, 0.0, 1.05, 1.0, 0.5, 1),  # same box, other batch
            (5.0, 0.0, 6.0, 1.0, 0.1, 0),
        ]
    )
    kept = non_max_suppression(df, 0.2)
    assert kept.index.tolist() == [1, 2, 3]
    assert len(non_max_suppression(df, 0.2, by=["recording"])) == 2


def test_nms_suppresses_only_above_the_threshold() -> None:
    df = _boxes([(0.0, 0.0, 2.0, 1.0, 0.9, 0), (1.0, 0.0, 3.0, 1.0, 0.8, 0)])
    assert len(non_max_suppression(df, 1.0 / 3.0)) == 2  # IoU == 1/3
    assert len(non_max_suppression(df, 0.3)) == 1


@pytest.mark.parametrize("seed", range(5))
def test_nms_commutes_with_a_score_threshold(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 60
    t0 = rng.uniform(0, 5, n)
    f0 = rng.uniform(600, 900, n)
    df = _boxes(
        list(
            zip(
                t0,
                f0,
                t0 + rng.uniform(0.05, 0.3, n),
                f0 + rng.uniform(50, 200, n),
                np.round(rng.uniform(0.25, 1.0, n), 2),  # ties on purpose
                rng.integers(0, 3, n),
                strict=True,
            )
        )
    )
    once = non_max_suppression(df, 0.2)
    for threshold in (0.3, 0.5, 0.75):
        after = once[once["score"] >= threshold].index
        subset = select(df, df["score"].to_numpy() >= threshold)
        before = non_max_suppression(subset, 0.2).index
        assert after.tolist() == before.tolist()
