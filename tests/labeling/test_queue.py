"""Queue sampling: stratification, determinism, resume, duplicates."""

from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gridtools.io import load_tracks, read_v1_detections

from chirpdetector.labeling import queue as queue_module
from chirpdetector.labeling.queue import (
    QueueConfig,
    binned_detections,
    build_queue,
    load_or_build_queue,
)

# bins of the ramp trial's 100 detections (see conftest._ramp_trial)
RAMP_BIN_COUNTS = {
    "0-10": 2,
    "10-25": 8,
    "25-50": 12,
    "50-100": 25,
    "100+": 51,
    "single": 2,
}


@pytest.fixture
def ramp(tmp_path: Path, make_ramp_trial: Callable[[Path], Path]) -> Path:
    return make_ramp_trial(tmp_path / "2001-01-01-10_00")


def test_bins_follow_delta_f_and_single_tracks_get_their_own_bin(ramp: Path) -> None:
    binned = binned_detections(
        load_tracks(ramp), read_v1_detections(ramp), QueueConfig()
    )
    assert dict(Counter(binned["df_bin"])) == RAMP_BIN_COUNTS
    single = binned[binned["df_bin"] == "single"]
    assert (np.asarray(single["n_candidates"]) == 1).all()
    assert np.isnan(np.asarray(single["delta_f_hz"], dtype=float)).all()


def test_stratified_sample_takes_up_to_per_bin(ramp: Path) -> None:
    queue = build_queue([ramp], QueueConfig(per_bin=5, seed=3))
    expected = {b: min(5, n) for b, n in RAMP_BIN_COUNTS.items()}
    assert queue.counts_per_bin() == expected
    assert len({i.item_id for i in queue.items}) == len(queue)
    (trial,) = queue.trials
    assert trial.kept_per_bin == RAMP_BIN_COUNTS


def test_queue_items_never_carry_v1_assignment(ramp: Path) -> None:
    queue = build_queue([ramp], QueueConfig(per_bin=5))
    fields = set(queue_module.QueueItem.__dataclass_fields__)
    assert not fields & {"track_id", "emitter_hz", "emitter_eodf", "score"}
    assert "track_id" not in str(queue.to_json()["items"])


def test_same_seed_same_queue_other_seed_other_queue(ramp: Path) -> None:
    config = QueueConfig(per_bin=5, seed=11)
    a = build_queue([ramp], config)
    b = build_queue([ramp], config)
    c = build_queue([ramp], QueueConfig(per_bin=5, seed=12))
    assert a.items == b.items
    assert a.queue_id == b.queue_id
    assert a.items != c.items
    assert a.queue_id != c.queue_id


def test_adding_a_trial_keeps_the_sample_of_the_others(
    ramp: Path, tmp_path: Path
) -> None:
    other = tmp_path / "2001-01-02-10_00"
    shutil.copytree(ramp, other)
    config = QueueConfig(per_bin=5, seed=2)
    alone = build_queue([ramp], config)
    both = build_queue([other, ramp], config)
    assert {i.item_id for i in alone.items} == {
        i.item_id for i in both.items if i.trial == ramp.name
    }
    assert {i.trial for i in both.items} == {ramp.name, other.name}


def test_resume_loads_instead_of_resampling(
    ramp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    config = QueueConfig(per_bin=5, seed=1)
    first = load_or_build_queue([ramp], config, out)
    assert (out / "queues" / f"{first.queue_id}.json").exists()

    def no_resampling(*args: object, **kwargs: object) -> None:
        raise AssertionError("queue was sampled again")

    monkeypatch.setattr(queue_module, "build_queue", no_resampling)
    again = load_or_build_queue([ramp], config, out)
    assert again == first


def test_changed_detections_give_a_new_queue(ramp: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    config = QueueConfig(per_bin=5)
    first = load_or_build_queue([ramp], config, out)
    csv = ramp / "chirpdetector_bboxes.csv"
    df = pd.read_csv(csv)
    df.loc[0, "score"] = 0.123
    df.to_csv(csv, index=False)
    assert load_or_build_queue([ramp], config, out).queue_id != first.queue_id


def test_duplicates_are_dropped_keeping_the_higher_score(ramp: Path) -> None:
    detections = read_v1_detections(ramp)
    duplicate = detections.iloc[[50]].copy()
    duplicate["t_start_s"] += 0.005
    duplicate["score"] = detections["score"].iloc[50] - 0.1
    with_dup = pd.concat([detections, duplicate], ignore_index=True)
    binned = binned_detections(load_tracks(ramp), with_dup, QueueConfig())
    assert len(binned) == len(detections)
    assert 50 in binned.index
    assert len(with_dup) - 1 not in binned.index


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="per_bin"):
        QueueConfig(per_bin=0)
    with pytest.raises(ValueError, match="bin_edges_hz"):
        QueueConfig(bin_edges_hz=(5.0, 10.0))
