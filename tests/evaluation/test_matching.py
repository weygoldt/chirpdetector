"""The matching rule and its scopes, on hand-built cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linear_sum_assignment

from chirpdetector.evaluation import MatchConfig, Scope, admissible_pairs, assign
from chirpdetector.evaluation.schema import validate_predictions

REGION = (0.0, 100.0, 600.0, 1000.0)
METHODS = ("hungarian", "greedy")


def _scope(truth, preds: pd.DataFrame, config: MatchConfig | None = None, spot=False):
    config = config or MatchConfig()
    preds = validate_predictions(preds)
    labels = truth.labels
    if spot:
        return Scope(labels[labels["region"].isna()], preds, config, None)
    return Scope(labels[labels["region"].notna()], preds, config, REGION)


def _pairs(scope: Scope, threshold: float = 0.0) -> list[tuple[float, float]]:
    """Matched (label centre, prediction score) pairs."""
    m = scope.match(threshold)
    return sorted(
        (float(scope.label_t_s[a]), float(scope.score[b]))
        for a, b in zip(*m.tp, strict=True)
    )


def test_tolerance_is_inclusive_and_bands_must_overlap() -> None:
    li, pj, dt = admissible_pairs(
        np.array([1.0, 1.0, 1.0]),
        (np.array([700.0, 700, 700]), np.array([800.0, 800, 800])),
        np.array([1.125, 1.0, 1.0]),
        (np.array([700.0, 800, 799]), np.array([800.0, 900, 900])),
        time_tolerance_s=0.125,
    )
    pairs = set(zip(li.tolist(), pj.tolist(), strict=True))
    assert (0, 0) in pairs  # |Δt| == tolerance
    assert (0, 1) not in pairs  # bands only touch at 800 Hz
    assert (0, 2) in pairs
    assert dt[(li == 0) & (pj == 0)][0] == 0.125
    li2, _, _ = admissible_pairs(
        np.array([1.0]),
        (np.array([700.0]), np.array([800.0])),
        np.array([1.0]),
        (np.array([800.0]), np.array([900.0])),
        time_tolerance_s=0.1,
        require_frequency_overlap=False,
    )
    assert len(li2) == 1


@pytest.mark.parametrize("method", METHODS)
def test_tie_between_two_predictions_goes_to_the_higher_score(
    method, truth_factory, predictions_factory
) -> None:
    truth = truth_factory(regions=[REGION], labels=[(10.0, 700, 800)])
    preds = predictions_factory([(9.9375, 700, 800, 0.6), (10.0625, 700, 800, 0.9)])
    scope = _scope(truth, preds, MatchConfig(method=method))
    assert _pairs(scope) == [(10.0, 0.9)]
    counts = scope.match(0.0).counts(scope)
    assert (counts["tp"], counts["detections"]) == (1, 2)  # the other one is a FP


@pytest.mark.parametrize("method", METHODS)
def test_tie_between_two_labels_goes_to_the_earlier_label(
    method, truth_factory, predictions_factory
) -> None:
    truth = truth_factory(
        regions=[REGION], labels=[(10.125, 700, 800), (10.0, 700, 800)]
    )
    preds = predictions_factory([(10.0625, 700, 800, 0.8)])
    scope = _scope(truth, preds, MatchConfig(method=method))
    assert _pairs(scope) == [(10.0, 0.8)]


def test_hungarian_finds_more_pairs_than_greedy(
    truth_factory, predictions_factory
) -> None:
    # All admissible pairs are 62.5 ms apart; greedy takes (P1, L1) first (higher
    # score, earlier label), which strands P2 and L2.
    truth = truth_factory(
        regions=[REGION], labels=[(10.0, 700, 800), (10.125, 700, 800)]
    )
    preds = predictions_factory([(10.0625, 700, 800, 0.9), (9.9375, 700, 800, 0.8)])
    hungarian = _scope(truth, preds, MatchConfig(time_tolerance_s=0.07))
    greedy = _scope(truth, preds, MatchConfig(time_tolerance_s=0.07, method="greedy"))
    assert len(_pairs(hungarian)) == 2
    assert len(_pairs(greedy)) == 1


def test_predictions_outside_the_region_are_not_counted(
    truth_factory, predictions_factory
) -> None:
    truth = truth_factory(
        regions=[REGION], labels=[(50.0, 700, 800), (150.0, 700, 800)]
    )
    preds = predictions_factory(
        [(50.0, 700, 800, 0.9), (150.0, 700, 800, 0.9), (170.0, 700, 800, 0.9)]
    )
    counts = _scope(truth, preds).match(0.5).counts(_scope(truth, preds))
    assert counts == {
        "labels": 1,
        "detections": 1,
        "tp": 1,
        "ignored": 0,
        "negatives": 0,
        "fp_negative": 0,
    }
    spot = _scope(truth, preds, spot=True)  # the label at 150 s is a spot label
    assert spot.match(0.5).counts(spot)["tp"] == 1
    assert spot.match(0.5).counts(spot)["detections"] == 1  # 170 s: not scored


@pytest.mark.parametrize(("band_filter", "fp"), [(True, 0), (False, 1)])
def test_harmonic_boxes_are_dropped_by_the_region_band(
    band_filter, fp, truth_factory, predictions_factory
) -> None:
    truth = truth_factory(regions=[REGION], labels=[(10.0, 650, 800)])
    preds = predictions_factory([(10.0, 650, 800, 0.9), (10.0, 1300, 1600, 0.9)])
    scope = _scope(truth, preds, MatchConfig(band_filter=band_filter))
    counts = scope.match(0.5).counts(scope)
    assert counts["tp"] == 1
    assert counts["detections"] - counts["tp"] == fp


@pytest.mark.parametrize(
    ("edge_tolerance", "tp", "detections"), [(False, 0, 0), (True, 1, 1)]
)
def test_label_near_the_region_edge(
    edge_tolerance, tp, detections, truth_factory, predictions_factory
) -> None:
    truth = truth_factory(regions=[REGION], labels=[(99.96875, 700, 800)])
    preds = predictions_factory([(100.03125, 700, 800, 0.9), (100.5, 700, 800, 0.9)])
    scope = _scope(truth, preds, MatchConfig(edge_tolerance=edge_tolerance))
    counts = scope.match(0.5).counts(scope)
    assert (counts["tp"], counts["detections"], counts["labels"]) == (tp, detections, 1)


def test_ignore_and_not_chirp_labels_in_a_region(
    truth_factory, predictions_factory
) -> None:
    truth = truth_factory(
        regions=[REGION],
        labels=[
            (10.0, 700, 800, "chirp"),
            (20.0, 700, 800, "ignore"),
            (30.0, 700, 800, "not_chirp"),
            (40.0, 700, 800, "not_chirp"),
            (50.0, 700, 800, "chirp_emitter_unclear"),
        ],
    )
    preds = predictions_factory(
        [
            (10.0, 700, 800, 0.9),
            (20.0, 700, 800, 0.9),
            (30.0, 700, 800, 0.9),
            (50.0, 700, 800, 0.9),
        ]
    )
    scope = _scope(truth, preds)
    counts = scope.match(0.5).counts(scope)
    assert counts == {
        "labels": 2,  # chirp + chirp_emitter_unclear
        "detections": 3,  # the prediction on the ignore label is not counted
        "tp": 2,
        "ignored": 1,
        "negatives": 2,
        "fp_negative": 1,
    }


def test_spot_labels_score_only_what_was_labelled(
    truth_factory, predictions_factory
) -> None:
    truth = truth_factory(
        labels=[
            (10.0, 700, 800, "chirp", 1),
            (20.0, 700, 800, "chirp", 2),
            (30.0, 700, 800, "not_chirp"),
            (40.0, 700, 800, "not_chirp"),
        ]
    )
    preds = predictions_factory(
        [(10.0, 700, 800, 0.9), (30.0, 700, 800, 0.9), (60.0, 700, 800, 0.9)]
    )
    scope = _scope(truth, preds, spot=True)
    counts = scope.match(0.5).counts(scope)
    assert counts == {
        "labels": 2,
        "detections": 2,  # TP + the prediction on a not_chirp label
        "tp": 1,
        "ignored": 0,
        "negatives": 2,
        "fp_negative": 1,
    }


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("seed", range(8))
def test_count_events_equal_direct_matching(
    method, seed, truth_factory, predictions_factory
) -> None:
    rng = np.random.default_rng(seed)
    n_lab, n_pred = 25, 40
    kinds = rng.choice(
        ["chirp", "chirp", "chirp_emitter_unclear", "ignore", "not_chirp"], n_lab
    )
    lab_t = np.sort(rng.uniform(0, 8, n_lab))
    truth = truth_factory(
        regions=[(0.0, 8.0, 600.0, 1000.0)],
        labels=[(t, 700, 800, k) for t, k in zip(lab_t, kinds, strict=True)],
    )
    pred_t = np.concatenate(
        [lab_t + rng.normal(0, 0.06, n_lab), rng.uniform(-0.2, 8.2, n_pred - n_lab)]
    )
    f0 = rng.choice([650.0, 700.0, 1300.0], n_pred)
    scores = np.round(rng.uniform(0.2, 1.0, n_pred), 1)  # many ties
    preds = predictions_factory(
        [(t, f, f + 100, s) for t, f, s in zip(pred_t, f0, scores, strict=True)]
    )
    config = MatchConfig(method=method, edge_tolerance=bool(seed % 2))
    region = (0.0, 8.0, 600.0, 1000.0)
    scope = Scope(truth.labels, validate_predictions(preds), config, region)
    thresholds = np.array([0.0, 0.2, 0.3, 0.45, 0.5, 0.7, 0.9, 1.0, 1.1])
    from_events = scope.counts_at(thresholds)
    for row in from_events.to_dict(orient="records"):
        direct = scope.match(row["threshold"]).counts(scope)
        assert {k: row[k] for k in direct} == direct


@pytest.mark.parametrize("seed", range(20))
def test_hungarian_matches_as_many_pairs_as_the_benchmark(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_lab, n_pred = int(rng.integers(1, 12)), int(rng.integers(1, 12))
    lab_t = rng.uniform(0, 1, n_lab)
    pred_t = rng.uniform(0, 1, n_pred)
    band = (np.full(n_lab, 700.0), np.full(n_lab, 800.0))
    pband = (np.full(n_pred, 700.0), np.full(n_pred, 800.0))
    li, pj, dt = admissible_pairs(lab_t, band, pred_t, pband, time_tolerance_s=0.1)
    chosen = assign(li, pj, dt, np.argsort(np.argsort(lab_t)), np.arange(n_pred))
    # detector_benchmark/evaluate.py:match
    d = np.abs(lab_t[:, None] - pred_t[None])
    rows, cols = linear_sum_assignment(np.where(d <= 0.1, d, 1e6))
    ok = d[rows, cols] <= 0.1
    assert chosen.sum() == ok.sum()
    assert np.isclose(dt[chosen].sum(), d[rows[ok], cols[ok]].sum())
    assert len(set(li[chosen])) == len(set(pj[chosen])) == chosen.sum()
