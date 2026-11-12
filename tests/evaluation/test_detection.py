"""Detection metrics against known counts and against scipy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest

from chirpdetector.evaluation import (
    average_precision,
    bootstrap_regions,
    clopper_pearson,
    confidence_intervals,
    pr_curve,
    summarize,
)
from chirpdetector.evaluation.detection import add_rates
from chirpdetector.evaluation.schema import ints, select


def test_rates_from_known_counts() -> None:
    out = add_rates(
        pd.DataFrame({"labels": [10, 0, 4], "detections": [8, 0, 0], "tp": [6, 0, 0]})
    )
    assert out["fn"].tolist() == [4, 0, 4]
    assert out["fp"].tolist() == [2, 0, 0]
    assert out.loc[0, ["precision", "recall", "f1"]].tolist() == [0.75, 0.6, 12 / 18]
    assert np.isnan(out.loc[1, ["precision", "recall", "f1"]].to_numpy(float)).all()
    assert np.isnan(out.loc[2, "precision"])
    assert out.loc[2, ["recall", "f1"]].tolist() == [0.0, 0.0]


def test_summary_pools_counts_not_ratios() -> None:
    per_region = pd.DataFrame(
        {
            "dataset": ["a", "a", "b"],
            "recording": ["r1", "r2", "r3"],
            "threshold": 0.5,
            "labels": [10, 90, 5],
            "detections": [10, 50, 5],
            "tp": [10, 45, 0],
        }
    )
    summary = summarize(per_region)
    assert summary["level"].tolist() == ["recording"] * 3 + ["dataset"] * 2 + ["all"]
    pooled = summary[summary["level"] == "all"].iloc[0]
    assert pooled["tp"] == 55
    assert pooled["precision"] == 55 / 65
    assert pooled["recall"] == 55 / 105
    a = summary[(summary["level"] == "dataset") & (summary["dataset"] == "a")].iloc[0]
    assert a["recall"] == 55 / 100
    assert pd.isna(pooled["dataset"])
    assert pd.isna(pooled["recording"])
    assert summarize(per_region.iloc[:0]).empty


@pytest.mark.parametrize(
    ("k", "n"), [(0, 10), (3, 10), (10, 10), (256, 256), (256, 335), (1, 1)]
)
@pytest.mark.parametrize("alpha", [0.05, 0.1])
def test_clopper_pearson_equals_scipy(k: int, n: int, alpha: float) -> None:
    ci = binomtest(k, n).proportion_ci(confidence_level=1 - alpha, method="exact")
    lo, hi = clopper_pearson(k, n, alpha)
    assert lo == pytest.approx(ci.low, rel=1e-12, abs=1e-15)
    assert hi == pytest.approx(ci.high, rel=1e-12, abs=1e-15)


def test_clopper_pearson_edge_cases() -> None:
    assert clopper_pearson(0, 10)[0] == 0.0
    assert clopper_pearson(10, 10)[1] == 1.0
    assert all(np.isnan(clopper_pearson(0, 0)))
    with pytest.raises(ValueError, match="0 <= k <= n"):
        clopper_pearson(5, 3)


def _regions(n: int = 6, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = rng.integers(10, 100, n)
    tp = (labels * rng.uniform(0.5, 0.95, n)).astype(int)
    return pd.DataFrame(
        {
            "dataset": ["a"] * (n - 2) + ["b"] * 2,
            "labels": labels,
            "detections": tp + rng.integers(0, 3, n),
            "tp": tp,
        }
    )


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    df = _regions()
    a = bootstrap_regions(df, rng=np.random.default_rng(3), n_boot=500)
    b = bootstrap_regions(df, rng=np.random.default_rng(3), n_boot=500)
    c = bootstrap_regions(df, rng=np.random.default_rng(4), n_boot=500)
    assert a == b
    assert a != c
    assert a["recall_lo"] <= a["recall_hi"]


def test_bootstrap_draws_like_detector_benchmark() -> None:
    # detector_benchmark/evaluate.py:bootstrap, verbatim apart from names
    df = _regions(n=5)  # odd: catches a single batched draw
    rng = np.random.default_rng(0)
    stats = []
    for _ in range(300):
        s = df.iloc[rng.integers(0, len(df), len(df))]
        stats.append([s.tp.sum() / s.detections.sum(), s.tp.sum() / s.labels.sum()])
    lo, hi = np.nanpercentile(np.array(stats), [2.5, 97.5], axis=0)
    ours = bootstrap_regions(df, rng=np.random.default_rng(0), n_boot=300)
    assert (ours["precision_lo"], ours["precision_hi"]) == (lo[0], hi[0])
    assert (ours["recall_lo"], ours["recall_hi"]) == (lo[1], hi[1])


def test_confidence_intervals_per_group_with_fresh_generators() -> None:
    df = _regions()
    table = confidence_intervals(df, n_boot=200, seed=7)
    assert table.index.tolist() == ["all", "a", "b"]
    alone = confidence_intervals(
        select(df, df["dataset"].to_numpy() == "b"), n_boot=200, seed=7
    )
    assert table.loc["b"].equals(alone.loc["b"])
    k, n = int(ints(df["tp"]).sum()), int(ints(df["labels"]).sum())
    assert (
        table.loc["all", "recall_lo"],
        table.loc["all", "recall_hi"],
    ) == clopper_pearson(k, n)
    assert table.loc["all", "n_regions"] == 6


def _events(scores, tp, det) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score": scores,
            "detections": det,
            "tp": tp,
            "ignored": 0,
            "fp_negative": 0,
            "dataset": "d",
            "recording": "r",
        }
    )


def test_pr_curve_and_average_precision_by_hand() -> None:
    # 4 labels; detections at scores 0.9 (TP), 0.8 (FP), 0.7 (TP), 0.6 (TP).
    events = _events([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 1], [1, 1, 1, 1])
    labels = pd.DataFrame(
        {"dataset": ["d"], "recording": ["r"], "labels": [4], "negatives": [0]}
    )
    curve = pr_curve(events, labels)
    pooled = curve[curve["level"] == "all"]
    assert pooled["threshold"].tolist() == [0.9, 0.8, 0.7, 0.6]
    assert pooled["recall"].tolist() == [0.25, 0.25, 0.5, 0.75]
    assert pooled["precision"].tolist() == [1.0, 0.5, 2 / 3, 0.75]
    ap = average_precision(curve)
    expected = 0.25 * 1.0 + 0.25 * (2 / 3) + 0.25 * 0.75
    assert ap.loc[ap["level"] == "all", "average_precision"].item() == pytest.approx(
        expected
    )
    assert len(ap) == 3  # recording, dataset, all


def test_average_precision_of_a_perfect_detector_is_one() -> None:
    events = _events([0.9, 0.8], [1, 1], [1, 1])
    labels = pd.DataFrame(
        {"dataset": ["d"], "recording": ["r"], "labels": [2], "negatives": [0]}
    )
    ap = average_precision(pr_curve(events, labels))
    assert (ap["average_precision"].to_numpy() == 1.0).all()


def test_pr_curve_keeps_recordings_without_predictions() -> None:
    events = _events([0.9], [1], [1])
    labels = pd.DataFrame(
        {
            "dataset": ["d", "d"],
            "recording": ["r", "q"],
            "labels": [1, 3],
            "negatives": [0, 0],
        }
    )
    curve = pr_curve(events, labels)
    assert set(curve.loc[curve["level"] == "recording", "recording"]) == {"r"}
    pooled = curve[curve["level"] == "all"].iloc[0]
    assert pooled["labels"] == 4
    assert pooled["recall"] == 0.25
