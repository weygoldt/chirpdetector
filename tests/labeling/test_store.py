"""Persistence: event log, derived CSV, override, undo, restart, lock."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from gridtools.io import AUDIAN_COLUMNS, read_labels

from chirpdetector.labeling.queue import QueueItem
from chirpdetector.labeling.store import LABEL_COLUMNS, LabelStore


def _item(det_index: int, trial: str = "2000-01-01-10_00") -> QueueItem:
    t = 1.0 + det_index
    return QueueItem(
        item_id=f"{trial}#{det_index}",
        trial=trial,
        det_index=det_index,
        t_start_s=t,
        t_end_s=t + 0.1,
        f_low_hz=690.0,
        f_high_hz=800.0,
        n_candidates=2,
        delta_f_hz=60.0,
        df_bin="50-100",
    )


def _events(out: Path) -> list[dict]:
    return [
        json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()
    ]


def _csv(out: Path, trial: str = "2000-01-01-10_00") -> pd.DataFrame:
    return read_labels(out / "labels" / f"{trial}.csv")


def test_click_is_logged_and_written_to_csv(tmp_path: Path) -> None:
    with LabelStore(tmp_path, "pw", "2.0.0") as store:
        store.record(
            _item(3),
            "q1",
            "chirp",
            emitter_id=7,
            clicked_hz=701.5,
            clicked_t_s=4.05,
            candidates=[(7, 700.0), (8, 760.0)],
        )
    (event,) = _events(tmp_path)
    assert event["type"] == "click"
    assert event["emitter_id"] == 7
    assert event["candidates"] == [[7, 700.0], [8, 760.0]]
    df = _csv(tmp_path)
    assert list(df.columns) == [*AUDIAN_COLUMNS, *LABEL_COLUMNS]
    row = df.iloc[0]
    assert (row.t_start_s, row.t_end_s, row.f_low_hz, row.f_high_hz) == (
        4.0,
        4.1,
        690.0,
        800.0,
    )
    assert row.category == "chirp"
    assert row.label == "chirp"
    assert row.emitter_id == 7
    assert row.clicked_hz == pytest.approx(701.5)
    assert row.labeler == "pw"
    assert row.queue_id == "q1"
    assert row.tool_version == "2.0.0"
    assert row.det_index == 3


def test_not_chirp_and_unclear_have_no_emitter(tmp_path: Path) -> None:
    with LabelStore(tmp_path, "pw", "2") as store:
        store.record(_item(1), "q", "not_chirp")
        store.record(_item(2), "q", "chirp_emitter_unclear")
    df = pd.read_csv(
        tmp_path / "labels" / "2000-01-01-10_00.csv", dtype={"emitter_id": "Int64"}
    )
    assert list(df.label) == ["not_chirp", "chirp_emitter_unclear"]
    assert list(df.category) == ["not_chirp", "chirp"]
    assert df.emitter_id.isna().all()
    assert df.clicked_hz.isna().all()


def test_clicking_again_overrides_and_undo_restores(tmp_path: Path) -> None:
    with LabelStore(tmp_path, "pw", "2") as store:
        store.record(_item(1), "q", "chirp", emitter_id=1)
        store.record(_item(1), "q", "not_chirp")
        assert len(_csv(tmp_path)) == 1
        assert _csv(tmp_path).label.iloc[0] == "not_chirp"

        undone = store.undo()
        assert undone is not None
        assert undone.label == "not_chirp"
        decision = store.current(_item(1).item_id)
        assert decision is not None
        assert decision.label == "chirp"
        assert _csv(tmp_path).label.iloc[0] == "chirp"

        store.undo()
        assert store.current(_item(1).item_id) is None
        assert len(_csv(tmp_path)) == 0
        assert store.undo() is None
    types = [e["type"] for e in _events(tmp_path)]
    assert types == ["click", "click", "undo", "undo"]


def test_restart_reproduces_state(tmp_path: Path) -> None:
    with LabelStore(tmp_path, "pw", "2") as store:
        store.record(_item(1), "q", "chirp", emitter_id=4)
        store.record(_item(2), "q", "not_chirp")
        store.record(_item(3), "q", "chirp", emitter_id=5)
        store.undo()
    (tmp_path / "labels" / "2000-01-01-10_00.csv").unlink()
    with LabelStore(tmp_path, "pw", "2") as store:
        first = store.current(_item(1).item_id)
        assert first is not None
        assert first.emitter_id == 4
        assert store.current(_item(3).item_id) is None
        assert len(_csv(tmp_path)) == 2  # rebuilt on start
        undone = store.undo()  # continues the history of the first session
        assert undone is not None
        assert undone.item_id == _item(2).item_id


def test_labelers_are_independent(tmp_path: Path) -> None:
    with LabelStore(tmp_path, "a", "2") as store:
        store.record(_item(1), "q", "chirp", emitter_id=1)
    with LabelStore(tmp_path, "b", "2") as store:
        assert store.current(_item(1).item_id) is None
        store.record(_item(1), "q", "not_chirp")
        assert store.undo() is not None
        assert store.undo() is None  # cannot undo labeller a's click
    assert list(_csv(tmp_path).labeler) == ["a"]


def test_second_session_on_the_same_directory_is_refused(tmp_path: Path) -> None:
    with (
        LabelStore(tmp_path, "pw", "2"),
        pytest.raises(RuntimeError, match="another labelling session"),
    ):
        LabelStore(tmp_path, "pw", "2")


def test_torn_last_line_is_ignored(tmp_path: Path) -> None:
    with LabelStore(tmp_path, "pw", "2") as store:
        store.record(_item(1), "q", "chirp", emitter_id=1)
    with (tmp_path / "events.jsonl").open("a") as f:
        f.write('{"type": "click", "seq": 1, "lab')
    with LabelStore(tmp_path, "pw", "2") as store:
        store.record(_item(2), "q", "not_chirp")
    with LabelStore(tmp_path, "pw", "2") as store:
        assert store.current(_item(2).item_id) is not None
    assert len(_csv(tmp_path)) == 2


def test_unknown_label_is_rejected(tmp_path: Path) -> None:
    with (
        LabelStore(tmp_path, "pw", "2") as store,
        pytest.raises(ValueError, match="label"),
    ):
        store.record(_item(1), "q", "maybe")
