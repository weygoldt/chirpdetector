"""Persistence: event log, derived CSV, override, undo, restart, lock."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from gridtools.io import AUDIAN_COLUMNS, read_labels

from chirpdetector.labeling.queue import QueueItem
from chirpdetector.labeling.store import (
    EXTRA_COLUMNS,
    LABEL_COLUMNS,
    SKIP,
    LabelStore,
)


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
    # the original columns stay first and in order; new ones are appended
    assert list(df.columns) == [*AUDIAN_COLUMNS, *LABEL_COLUMNS, *EXTRA_COLUMNS]
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


def test_new_fields_are_stored_and_written(tmp_path: Path) -> None:
    with LabelStore(tmp_path, "pw", "2") as store:
        store.record(
            _item(3),
            "q1",
            "chirp",
            emitter_id=7,
            clicked_hz=701.5,
            clicked_t_s=4.05,
            emitter_baseline_hz=700.25,
            click_offset_hz=1.25,
            via="mouse",
            frontend="browser",
            window_s=0.2048,
            prediction_shown=False,
            recording_dir="/rec",
            tracks_dir="/trk",
        )
    row = _csv(tmp_path).iloc[0]
    assert row.emitter_baseline_hz == pytest.approx(700.25)
    assert row.click_offset_hz == pytest.approx(1.25)
    assert (row.via, row.frontend, row.window_s) == ("mouse", "browser", 0.2048)
    assert not row.prediction_shown
    assert (row.n_candidates, row.delta_f_hz) == (2, 60.0)
    assert (row.recording_dir, row.tracks_dir) == ("/rec", "/trk")
    assert pd.isna(row.candidate_id)


def test_unknown_frontend_is_rejected(tmp_path: Path) -> None:
    with (
        LabelStore(tmp_path, "pw", "2") as store,
        pytest.raises(ValueError, match="frontend"),
    ):
        store.record(_item(1), "q", "chirp", frontend="excel")


def test_old_events_without_the_new_fields_replay(tmp_path: Path) -> None:
    """Logs written by the first version (no new fields) still load."""
    old = {
        "type": "click",
        "seq": 0,
        "labeler": "pw",
        "labeled_at": "2026-11-12T10:00:00.000+00:00",
        "tool_version": "2.0.0.dev0",
        "queue_id": "q",
        "item_id": "2000-01-01-10_00#1",
        "trial": "2000-01-01-10_00",
        "det_index": 1,
        "df_bin": "50-100",
        "box": [2.0, 2.1, 690.0, 800.0],
        "label": "chirp",
        "emitter_id": 4,
        "clicked_hz": 701.0,
        "clicked_t_s": 2.05,
        "candidates": [[4, 700.0]],
    }
    (tmp_path / "events.jsonl").write_text(json.dumps(old) + "\n")
    with LabelStore(tmp_path, "pw", "2") as store:
        decision = store.current(old["item_id"])
        assert decision is not None
        assert decision.emitter_id == 4
        assert decision.emitter_baseline_hz is None
        assert decision.frontend == ""
        store.record(_item(2), "q", "not_chirp", frontend="mpl")
    df = _csv(tmp_path)
    assert list(df.label) == ["chirp", "not_chirp"]
    assert df.prediction_shown.isna().all()


def test_skip_is_logged_undoable_and_never_written(tmp_path: Path) -> None:
    with LabelStore(tmp_path, "pw", "2") as store:
        store.record(_item(1), "q", SKIP)
        assert store.skipped(_item(1).item_id)
        assert store.current(_item(1).item_id) is None
        assert not (tmp_path / "labels" / "2000-01-01-10_00.csv").exists()
        store.record(_item(1), "q", "not_chirp")  # deciding later replaces the skip
        assert not store.skipped(_item(1).item_id)
        assert len(_csv(tmp_path)) == 1
        store.undo()
        assert store.skipped(_item(1).item_id)
        undone = store.undo()
        assert undone is not None
        assert undone.label == SKIP
        assert not store.skipped(_item(1).item_id)
    assert [e["type"] for e in _events(tmp_path)] == [SKIP, "click", "undo", "undo"]
    with LabelStore(tmp_path, "pw", "2") as store:  # replay
        assert not store.skipped(_item(1).item_id)
        assert store.decisions() == []
