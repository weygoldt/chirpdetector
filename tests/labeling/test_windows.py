"""Window items: several marks per item, then done (item_kind = window)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from gridtools.io import read_labels
from typer.testing import CliRunner

from chirpdetector.cli import app
from chirpdetector.labeling.candidates import load_or_build_candidate_queue
from chirpdetector.labeling.queue import LabelQueue
from chirpdetector.labeling.server import LabelServer
from chirpdetector.labeling.session import LabelSession
from chirpdetector.labeling.store import WINDOW_DONE, LabelStore

from .test_server import Client

REC = "2000-01-01-10_00"


def _write(path: Path, recording: Path) -> Path:
    """A window (zone 3-5 s around the synthetic chirp at 4 s) and a box."""
    df = pd.DataFrame(
        {
            "recording": str(recording),
            "t_start_s": [3.0, 5.93],
            "t_end_s": [5.0, 6.07],
            "f_low_hz": [650.0, 690.0],
            "f_high_hz": [900.0, 800.0],
            "candidate_id": ["w1", "box"],
            "item_kind": ["window", ""],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def window_csv(trial: Path, tmp_path: Path) -> Path:
    return _write(tmp_path / "cands" / "windows.csv", trial)


@pytest.fixture
def window_queue(window_csv: Path, tmp_path: Path) -> LabelQueue:
    return load_or_build_candidate_queue(
        window_csv, None, tmp_path / "out", shuffle=False
    )


@pytest.fixture
def wsession(window_queue: LabelQueue, tmp_path: Path) -> Iterator[LabelSession]:
    with LabelStore(tmp_path / "out", "tester", "2") as store:
        s = LabelSession(window_queue, store, background=False, blind=True)
        yield s
        s.shutdown()


def _csv(session: LabelSession) -> pd.DataFrame:
    return read_labels(session.store.labels_dir / f"{REC}.csv")


def test_item_kind_column_makes_a_window(window_queue: LabelQueue) -> None:
    window, box = window_queue.items
    assert window.is_window
    assert not box.is_window
    assert window.candidate_columns["candidate_item_kind"] == "window"


def test_marks_stay_on_the_window_until_done(wsession: LabelSession) -> None:
    first = wsession.click(0, 4.0, 701.0, elapsed_s=1.0)
    assert first.ok
    assert first.next_index == 0  # stays on the window
    assert wsession.status(0) == "open"
    mark = first.decision
    assert mark is not None
    assert mark.parent_item_id == f"{REC}#w1"
    assert mark.item_id.startswith(f"{REC}#w1@")
    assert mark.box == (4.0, 4.0, 701.0, 701.0)
    assert (mark.label, mark.emitter_id) == ("chirp", 0)
    assert mark.emitter_baseline_hz == pytest.approx(700.0)  # track at the click
    second = wsession.click(0, 4.6, 759.0)
    assert second.ok
    assert [m.emitter_id for m in wsession.marks(0)] == [0, 1]
    # an unclear mark needs the chirp's position
    assert not wsession.mark(0, "chirp_emitter_unclear").ok
    unclear = wsession.mark(0, "chirp_emitter_unclear", t_s=3.5, f_hz=820.0)
    assert unclear.ok
    assert unclear.decision.emitter_id is None  # pyright: ignore[reportOptionalMemberAccess]
    assert len(wsession.marks(0)) == 3
    done = wsession.done(0, elapsed_s=6.5)
    assert done.ok
    assert "3 chirps" in done.message
    assert done.next_index == 1
    assert wsession.status(0) == "labelled"
    assert wsession.progress().labelled == 1


def test_window_refuses_marks_outside_the_zone_and_number_keys(
    wsession: LabelSession,
) -> None:
    outside = wsession.click(0, 5.5, 701.0)
    assert not outside.ok
    assert "outside the zone" in outside.message
    assert not wsession.mark(0, "chirp_emitter_unclear", t_s=2.9, f_hz=800.0).ok
    assert not wsession.pick(0, 1).ok
    assert wsession.marks(0) == []
    assert wsession.store.undo() is None  # nothing was stored


def test_not_chirp_finishes_a_window_without_marks(wsession: LabelSession) -> None:
    result = wsession.mark(0, "not_chirp", via="mouse")
    assert result.ok
    assert wsession.decision(0).label == WINDOW_DONE  # pyright: ignore[reportOptionalMemberAccess]
    assert "0 chirps" in result.message
    rows = _csv(wsession)
    assert list(rows["label"]) == [WINDOW_DONE]
    assert list(rows["category"]) == ["labeled_region"]


def test_ordinary_items_keep_their_semantics(wsession: LabelSession) -> None:
    assert not wsession.done(1).ok  # not a window
    result = wsession.click(1, 6.0, 701.0)
    assert result.ok
    assert result.decision.parent_item_id == ""  # pyright: ignore[reportOptionalMemberAccess]
    assert wsession.status(1) == "labelled"


def test_label_file_rows_of_a_window(wsession: LabelSession) -> None:
    wsession.click(0, 4.0, 701.0)
    wsession.mark(0, "chirp_emitter_unclear", t_s=4.5, f_hz=830.0)
    wsession.done(0)
    rows = _csv(wsession).sort_values("t_start_s").reset_index(drop=True)
    region = rows[rows["label"] == WINDOW_DONE].iloc[0]
    assert (region.category, region.kind) == ("labeled_region", "span")
    assert pd.isna(region.parent_item_id)  # empty in the file
    assert (region.t_start_s, region.t_end_s) == (3.0, 5.0)
    marks = rows[rows["parent_item_id"] == f"{REC}#w1"]
    assert list(marks["label"]) == ["chirp", "chirp_emitter_unclear"]
    assert set(marks["kind"]) == {"point"}
    assert set(marks["category"]) == {"chirp"}
    assert list(marks["t_start_s"]) == list(marks["t_end_s"]) == [4.0, 4.5]
    assert marks["emitter_id"].iloc[0] == 0
    assert pd.isna(marks["emitter_id"].iloc[1])
    assert set(marks["candidate_item_kind"]) == {"window"}


def test_undo_and_clear(wsession: LabelSession) -> None:
    wsession.click(0, 4.0, 701.0)
    wsession.click(0, 4.6, 759.0)
    wsession.done(0)
    undone = wsession.undo()  # reverts the done
    assert undone.ok
    assert undone.index == 0
    assert wsession.status(0) == "open"
    assert len(wsession.marks(0)) == 2
    undone = wsession.undo()  # reverts the last mark
    assert undone.index == 0
    assert [m.emitter_id for m in wsession.marks(0)] == [0]
    wsession.click(0, 4.6, 759.0)
    wsession.done(0)
    cleared = wsession.clear(0)
    assert cleared.ok
    assert cleared.next_index == 0
    assert wsession.marks(0) == []
    assert wsession.status(0) == "open"
    assert (
        not (wsession.store.labels_dir / f"{REC}.csv").exists() or _csv(wsession).empty
    )


def test_marks_survive_a_restart(window_queue: LabelQueue, tmp_path: Path) -> None:
    with LabelStore(tmp_path / "out", "tester", "2") as store:
        s = LabelSession(window_queue, store, background=False)
        s.click(0, 4.0, 701.0)
        s.click(0, 4.6, 759.0)
        s.shutdown()
    with LabelStore(tmp_path / "out", "tester", "2") as store:
        s = LabelSession(window_queue, store, background=False)
        assert [m.clicked_t_s for m in s.marks(0)] == [4.0, 4.6]
        assert s.status(0) == "open"
        assert s.first_index() == 0
        s.shutdown()


def test_server_sends_zone_and_marks(window_queue: LabelQueue, tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "out", "tester", "2")
    server = LabelServer(
        LabelSession(window_queue, store, frontend="browser", blind=True), port=0
    )
    server.start()
    try:
        client = Client(server)
        status, view = client.json("/api/view?index=0&window=0.2048")
        assert status == 200
        assert view["item"]["kind"] == "window"
        assert view["item"]["zone"] == [3.0, 5.0]
        assert "hint_box" not in view["item"]
        assert view["marks"] == []
        rect = view["panels"]["overview"]["extent"]
        w, h = 1000.0, 400.0
        # a middle click (unclear) at 4.2 s, 800 Hz of the overview
        x = (4.2 - rect[0]) / (rect[1] - rect[0]) * w
        y = (rect[3] - 800.0) / (rect[3] - rect[2]) * h
        _, result = client.json(
            "/api/action",
            {
                "index": 0,
                "window_s": 0.2048,
                "action": "unclear",
                "via": "mouse",
                "panel": "overview",
                "x": x,
                "y": y,
                "width": w,
                "height": h,
            },
        )
        assert result["ok"], result
        assert result["next_index"] == 0
        assert result["t_s"] == pytest.approx(4.2, abs=1e-6)
        _, view = client.json("/api/view?index=0&window=0.2048")
        assert len(view["marks"]) == 1
        mark = view["marks"][0]
        assert mark["label"] == "chirp_emitter_unclear"
        assert mark["t_s"] == pytest.approx(4.2, abs=1e-6)
        _, result = client.json("/api/action", {"index": 0, "action": "done"})
        assert result["ok"]
        assert result["next_index"] == 1
        _, result = client.json("/api/action", {"index": 0, "action": "clear"})
        assert result["ok"]
        _, view = client.json("/api/view?index=0&window=0.2048")
        assert (view["marks"], view["status"]) == ([], "open")
        # ordinary items in a blind session still get the hint and no zone
        _, box = client.json("/api/view?index=1&window=0.2048")
        assert "zone" not in box["item"]
        assert "hint_box" in box["item"]
        assert box["marks"] == []
    finally:
        server.shutdown()
        store.close()


def test_page_draws_zone_and_marks() -> None:
    from importlib import resources

    html = (
        resources.files("chirpdetector.labeling").joinpath("labeller.html").read_text()
    )
    assert "if (it.zone)" in html
    assert 'k === "Enter"' in html
    assert 'action: "clear"' in html
    assert "v.marks" in html


def test_cli_needs_the_browser_for_windows(window_csv: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "label-emitters",
            "--candidates",
            str(window_csv),
            "--out",
            str(tmp_path / "out_cli"),
            "--labeler",
            "tester",
        ],
    )
    assert result.exit_code != 0
    assert "--serve" in result.output


def test_events_of_marks_parse_back(wsession: LabelSession, tmp_path: Path) -> None:
    wsession.click(0, 4.0, 701.0)
    events = [
        json.loads(line)
        for line in (tmp_path / "out" / "events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["parent_item_id"] == f"{REC}#w1"
