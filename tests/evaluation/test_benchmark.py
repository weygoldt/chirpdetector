"""Acceptance: reproduce ``detector_benchmark`` number for number.

The benchmark (``chirp_competition/detector_benchmark``) stored the frozen v1
detector's boxes before NMS in ``results/boxes/<recording>.csv`` and scored
them with ``evaluate.py`` into ``summary.csv``, ``per_region.csv`` and
``confidence_intervals_threshold_0.5.csv``. These tests rebuild all three
from those boxes and the four hand-label files with this package and require
exact equality (no tolerance).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gridtools.io import load_tracks
from scipy.optimize import linear_sum_assignment

from chirpdetector.evaluation import (
    EvaluationConfig,
    EvaluationResult,
    GroundTruth,
    MatchConfig,
    Scope,
    evaluate,
    non_max_suppression,
    read_ground_truth,
    read_predictions,
)
from chirpdetector.evaluation.matching import MatchMethod
from chirpdetector.evaluation.schema import (
    floats,
    select,
    strings,
    validate_predictions,
)

BENCHMARK = Path("/home/weygoldt/wrk/analyses/chirp_competition/detector_benchmark")
RESULTS = BENCHMARK / "results"
LABEL_FILES = [
    Path(
        "/mnt/data2/2019_tube_competition/2019-12-09-10_00/traces-grid1-editable-labels.csv"
    ),
    Path(
        "/mnt/data2/2019_tube_competition/2020-06-03-10_10/traces-grid1-editable-labels.csv"
    ),
    Path(
        "/mnt/data2/2022_tube_competition/raw/2022-05-12-10_00/traces-grid1-editable-labels.csv"
    ),
    Path(
        "/mnt/data2/2022_tube_competition/raw/2022-06-01-10_00/traces-grid1-editable-labels.csv"
    ),
]
COUNTS = ["labels", "detections", "tp", "fn", "fp"]
RATES = ["precision", "recall", "f1"]


def _inputs() -> tuple[GroundTruth, pd.DataFrame]:
    if not RESULTS.exists():
        pytest.skip(f"{RESULTS} not available")
    truth = GroundTruth.concat(read_ground_truth(f) for f in LABEL_FILES)
    boxes = pd.concat(
        [
            read_predictions(RESULTS / "boxes" / f"{f.parent.name}.csv")
            for f in LABEL_FILES
        ],
        ignore_index=True,
    )
    return truth, non_max_suppression(boxes, 0.2)  # v1: NMS per batch at IoU 0.2


@pytest.fixture(scope="module")
def inputs() -> tuple[GroundTruth, pd.DataFrame]:
    return _inputs()


@pytest.fixture(scope="module")
def result(inputs) -> EvaluationResult:
    truth, preds = inputs
    return evaluate(preds, truth)  # all defaults: the benchmark's settings


def _keyed(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["level", "dataset", "recording", "threshold"]
    out = df.assign(
        dataset=df["dataset"].fillna(""), recording=df["recording"].fillna("")
    )
    return out.sort_values(keys).set_index(keys)


def _assert_identical(
    ours: pd.DataFrame, theirs: pd.DataFrame, columns: list[str]
) -> None:
    assert ours.index.equals(theirs.index)
    for column in columns:
        a = ours[column].to_numpy(dtype=np.float64)
        b = theirs[column].to_numpy(dtype=np.float64)
        np.testing.assert_array_equal(a, b, err_msg=column)  # NaN == NaN here


@pytest.mark.data
def test_reproduces_summary_csv(result: EvaluationResult) -> None:
    theirs = _keyed(pd.read_csv(RESULTS / "summary.csv"))
    ours = _keyed(result.summary)
    assert len(theirs) == 105  # 4 recordings + 2 datasets + pooled, 15 thresholds
    _assert_identical(ours, theirs, COUNTS + RATES)


@pytest.mark.data
def test_reproduces_per_region_csv(result: EvaluationResult) -> None:
    keys = ["recording", "region", "threshold"]
    theirs = pd.read_csv(RESULTS / "per_region.csv").sort_values(keys).set_index(keys)
    ours = result.per_region.astype({"region": int}).sort_values(keys).set_index(keys)
    _assert_identical(ours, theirs, ["labels", "detections", "tp"])
    np.testing.assert_array_equal(ours["t_start_s"], theirs["t_start"])
    np.testing.assert_array_equal(ours["t_end_s"], theirs["t_end"])


@pytest.mark.data
def test_reproduces_the_confidence_interval_table(result: EvaluationResult) -> None:
    theirs = pd.read_csv(
        RESULTS / "confidence_intervals_threshold_0.5.csv", index_col=0
    )
    ours = result.intervals.loc[theirs.index, theirs.columns]
    np.testing.assert_array_equal(
        ours.to_numpy(np.float64), theirs.to_numpy(np.float64)
    )


@pytest.mark.data
def test_headline_numbers(result: EvaluationResult) -> None:
    pooled = result.metrics()["detection"]["all"]
    assert (pooled["labels"], pooled["detections"], pooled["tp"]) == (335, 256, 256)
    assert pooled["precision"] == 1.0
    assert round(pooled["recall"], 3) == 0.764
    assert round(pooled["f1"], 3) == 0.866
    assert [round(v, 3) for v in pooled["precision_ci"]] == [0.986, 1.0]
    assert [round(v, 3) for v in pooled["recall_ci"]] == [0.715, 0.809]


def _benchmark_match(labels: pd.DataFrame, dets: pd.DataFrame, tol: float = 0.1):
    """``detector_benchmark/evaluate.py:match``, verbatim apart from names."""
    if len(labels) == 0 or len(dets) == 0:
        return np.array([], int), np.array([], int)
    dt = np.abs(labels.tc.to_numpy()[:, None] - dets.tc.to_numpy()[None])
    fo = np.minimum(
        labels.f_high_hz.to_numpy()[:, None], dets.f2.to_numpy()[None]
    ) - np.maximum(labels.f_low_hz.to_numpy()[:, None], dets.f1.to_numpy()[None])
    ok = (dt <= tol) & (fo > 0)
    rows, cols = linear_sum_assignment(np.where(ok, dt, 1e6))
    good = ok[rows, cols]
    return rows[good], cols[good]


@pytest.mark.data
@pytest.mark.parametrize("method", ["hungarian", "greedy"])
def test_matched_pairs_equal_the_benchmark(inputs, method: MatchMethod) -> None:
    """Not just counts: the same label-prediction pairs at every threshold."""
    truth, preds = inputs
    preds = validate_predictions(preds)
    config = MatchConfig(method=method)
    label_rec = np.asarray(strings(truth.labels["recording"]))
    label_region = floats(truth.labels["region"])
    pred_rec = np.asarray(strings(preds["recording"]))
    n_pairs = 0
    for region in truth.regions.to_dict(orient="records"):
        name, t0, t1 = region["recording"], region["t_start_s"], region["t_end_s"]
        f_low, f_high = region["f_low_hz"], region["f_high_hz"]
        labels = select(
            truth.labels, (label_rec == name) & (label_region == region["region"])
        )
        rec = select(preds, pred_rec == name)
        scope = Scope(labels, rec, config, (t0, t1, f_low, f_high))
        tc = 0.5 * (floats(rec["t_start_s"]) + floats(rec["t_end_s"]))
        f1, f2 = floats(rec["f_low_hz"]), floats(rec["f_high_hz"])
        lab = labels.assign(tc=labels["t_center_s"])
        for threshold in np.round(np.arange(0.25, 0.951, 0.05), 2):
            m = scope.match(threshold)
            ours = {
                (scope.label_t_s[a], scope.pred_t_s[b])
                for a, b in zip(*m.tp, strict=True)
            }
            keep = (
                (floats(rec["score"]) >= threshold)
                & (tc >= t0)
                & (tc <= t1)
                & (f1 < f_high)
                & (f2 > f_low)
            )
            dets = pd.DataFrame({"tc": tc[keep], "f1": f1[keep], "f2": f2[keep]})
            li, di = _benchmark_match(lab, dets)
            theirs = set(
                zip(floats(lab["tc"])[li], floats(dets["tc"])[di], strict=True)
            )
            assert ours == theirs, (name, region["region"], threshold)
            n_pairs += len(ours)
    summary = pd.read_csv(RESULTS / "summary.csv")
    assert n_pairs == summary.loc[summary["level"] == "all", "tp"].sum()


@pytest.mark.data
def test_v1_production_detections_with_tracks() -> None:
    """End to end on v1's own output (chirpdetector_bboxes.csv) with tracks.

    For 2020-06-03 the benchmark's replay equals the production output, so
    the counts at 0.5 equal the benchmark's (81 of 88). For 2019-12-09 the
    replay has 6 extra boxes, 2 of them in a labelled region (benchmark
    README), so production has 2 fewer detections there.
    """
    truth = GroundTruth.concat(read_ground_truth(f) for f in LABEL_FILES[:2])
    dirs = [f.parent for f in LABEL_FILES[:2]]
    preds = pd.concat([read_predictions(d) for d in dirs], ignore_index=True)
    tracks = {d.name: load_tracks(d) for d in dirs}
    result = evaluate(
        preds, truth, tracks=tracks, config=EvaluationConfig(n_bootstrap=100)
    )
    rec = result.metrics()["detection"]["recordings"]
    assert (rec["2020-06-03-10_10"]["tp"], rec["2020-06-03-10_10"]["detections"]) == (
        81,
        81,
    )
    assert rec["2019-12-09-10_00"]["labels"] == 204
    assert rec["2019-12-09-10_00"]["detections"] == 145 - 2
    assert set(result.items["df_bin"]) - {"unknown"} == set(result.items["df_bin"])
    assert result.items["n_live_tracks"].notna().to_numpy().all()
    assert result.assignment.empty  # no emitter labels exist yet
