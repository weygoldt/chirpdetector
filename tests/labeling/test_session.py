"""The labelling session both frontends drive: actions, skip, undo, resume."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from gridtools.io import read_labels

from chirpdetector.evaluation import read_ground_truth
from chirpdetector.labeling.candidates import load_or_build_candidate_queue
from chirpdetector.labeling.session import LabelSession
from chirpdetector.labeling.store import LabelStore

from .conftest import write_candidates


def _csv(session: LabelSession) -> pd.DataFrame:
    return read_labels(session.store.labels_dir / "2000-01-01-10_00.csv")


def test_click_snaps_and_stores_the_track_baseline(session: LabelSession) -> None:
    # item 0 is the synthetic chirp of fish 0 (700 Hz); click 1.3 Hz above
    result = session.click(0, 3.85, 701.3, prediction_shown=False)
    assert result.ok
    d = result.decision
    assert d is not None
    assert (d.label, d.emitter_id, d.via, d.frontend) == ("chirp", 0, "mouse", "mpl")
    assert d.emitter_baseline_hz == 700.0  # from the track, not the click
    assert d.clicked_hz == 701.3
    assert d.click_offset_hz == pytest.approx(1.3)
    assert d.window_s == 0.2048
    assert d.candidates == ((0, 700.0), (1, 760.0))
    assert result.next_index == 1
    row = _csv(session).iloc[0]
    assert row.candidate_id == "chirp"
    assert row.candidate_emitter_hz == pytest.approx(700.4)
    assert row.candidate_source == "model-x"
    assert row.candidate_tile == 3


def test_click_far_from_every_track_is_rejected(session: LabelSession) -> None:
    result = session.click(0, 4.0, 730.0)
    assert not result.ok
    assert "no track within 15 Hz" in result.message
    assert session.decision(0) is None
    assert session.store.undo() is None  # nothing was stored


def test_number_keys_pick_tracks_in_frequency_order(session: LabelSession) -> None:
    result = session.pick(0, 2, prediction_shown=True)
    assert result.ok
    d = result.decision
    assert d is not None
    assert (d.emitter_id, d.emitter_baseline_hz, d.via) == (1, 760.0, "key")
    assert d.clicked_hz is None
    assert d.prediction_shown
    bad = session.pick(3, 2)  # the item at 0.5 s has a single track
    assert not bad.ok
    assert "no track 2 (1 live)" in bad.message


def test_marks_skip_and_the_order_of_items(session: LabelSession) -> None:
    assert session.first_index() == 0
    r = session.mark(0, "not_chirp")
    assert (r.ok, r.next_index) == (True, 1)
    r = session.skip(1)
    assert r.next_index == 2
    assert session.status(1) == "skipped"
    r = session.mark(2, "chirp_emitter_unclear", via="mouse")
    assert r.next_index == 3
    r = session.pick(3, 1)
    assert r.next_index == 1  # only the skipped item is left
    p = session.progress()
    assert (p.labelled, p.skipped, p.open) == (3, 1, 0)
    r = session.click(1, 1.5, 759.0)
    assert r.next_index is None  # all labelled
    assert session.skip(0).message.endswith("kept its label")
    assert session.status(0) == "labelled"
    with pytest.raises(ValueError, match="mark takes"):
        session.mark(0, "chirp")


def test_undo_reverts_and_returns_the_item(session: LabelSession) -> None:
    session.pick(0, 1)
    session.skip(1)
    r = session.undo()
    assert (r.ok, r.index, r.next_index) == (True, 1, 1)
    assert session.status(1) == "open"
    r = session.undo()
    assert r.index == 0
    assert session.decision(0) is None
    assert not session.undo().ok


def test_window_toggle(session: LabelSession) -> None:
    assert session.windows == (0.2048, 0.0512)
    assert session.toggle_window(0.2048) == 0.0512
    assert session.toggle_window(0.0512) == 0.2048
    view = session.view(0, 0.0512)
    assert view.window_s == 0.0512
    assert session.mark(0, "not_chirp", window_s=0.0512).decision.window_s == 0.0512  # pyright: ignore[reportOptionalMemberAccess]


def test_same_id_with_another_box_is_refused(
    trial: Path, tmp_path: Path, candidate_queue
) -> None:
    out = tmp_path / "out"
    with LabelStore(out, "tester", "2") as store:
        LabelSession(candidate_queue, store, background=False).pick(0, 1)
    moved = write_candidates(
        tmp_path / "moved" / "c.csv",
        trial,
        candidate_id=["chirp", "a", "b", "c"],
        t_start_s=[3.0, 1.43, 5.93, 0.43],
    )
    other = load_or_build_candidate_queue(moved, None, out, shuffle=False)
    with (
        LabelStore(out, "tester", "2") as store,
        pytest.raises(ValueError, match="labelled before with the box"),
    ):
        LabelSession(other, store, background=False)


def test_same_box_in_another_file_reuses_its_label(
    trial: Path, tmp_path: Path, candidate_queue
) -> None:
    out = tmp_path / "out"
    with LabelStore(out, "tester", "2") as store:
        LabelSession(candidate_queue, store, background=False).mark(1, "not_chirp")
    again = write_candidates(
        tmp_path / "again" / "c.csv", trial, candidate_id=["chirp", "a", "b", "c"]
    )
    queue = load_or_build_candidate_queue(again, None, out, shuffle=False)
    with LabelStore(out, "tester", "2") as store:
        s = LabelSession(queue, store, background=False)
        assert s.status(1) == "labelled"
        assert s.first_index() == 0


def test_labels_are_read_by_the_evaluation(session: LabelSession) -> None:
    """label-emitters files of candidates are spot labels with emitters."""
    session.click(0, 3.85, 700.5)
    session.mark(1, "not_chirp")
    session.mark(2, "chirp_emitter_unclear")
    path = session.store.labels_dir / "2000-01-01-10_00.csv"
    truth = read_ground_truth(path)
    labels = truth.labels.sort_values("t_start_s")
    assert set(labels["recording"]) == {"2000-01-01-10_00"}
    assert labels["label"].tolist() == ["not_chirp", "chirp", "chirp_emitter_unclear"]
    assert labels["emitter_id"].tolist()[1] == 0
    assert pd.isna(labels["region"].to_numpy()).all()


def test_queue_file_names_its_origin(session: LabelSession, tmp_path: Path) -> None:
    queue = session.queue
    data = json.loads(
        (tmp_path / "out" / "queues" / f"{queue.queue_id}.json").read_text()
    )
    assert data["origin"]["kind"] == "candidates"
    assert data["items"][0]["candidate"][0] == ["candidate_id", "chirp"]
