"""Blind labelling: no box, no model output, time per item."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from chirpdetector.labeling.queue import LabelQueue
from chirpdetector.labeling.server import LabelServer
from chirpdetector.labeling.session import LabelSession
from chirpdetector.labeling.single import SingleLabeller
from chirpdetector.labeling.store import LabelStore

from .test_server import Client

# the candidates fixture's first row: box 3.93-4.07 s x 680-820 Hz, score
# 0.9, emitter 700.4 Hz, source model-x; none of it may reach the page
LEAKED_KEYS = {
    "score",
    "emitter_hz",
    "source",
    "t_start_s",
    "t_end_s",
    "f_low_hz",
    "f_high_hz",
}
LEAKED_VALUES = {3.93, 4.07, 680.0, 820.0, 0.9, 700.4, "model-x"}


def _walk(value: object) -> Iterator[tuple[str | None, object]]:
    """Every (key, value) pair of a JSON document."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield None, v
            yield from _walk(v)


@pytest.fixture
def blind_session(
    candidate_queue: LabelQueue, tmp_path: Path
) -> Iterator[LabelSession]:
    with LabelStore(tmp_path / "out", "tester", "2") as store:
        s = LabelSession(candidate_queue, store, background=False, blind=True)
        yield s
        s.shutdown()


def test_server_never_sends_box_or_model_output(
    candidate_queue: LabelQueue, tmp_path: Path
) -> None:
    store = LabelStore(tmp_path / "out", "tester", "2")
    server = LabelServer(
        LabelSession(candidate_queue, store, frontend="browser", blind=True), port=0
    )
    server.start()
    try:
        client = Client(server)
        _, state = client.json("/api/state")
        assert state["blind"] is True
        status, raw, _ = client.request("/api/view?index=0&window=0.2048")
        assert status == 200
        view = json.loads(raw)
        assert view["prediction"] is None
        assert set(view["item"]) == {
            "item_id",
            "trial",
            "t_center_s",
            "df_bin",
            "delta_f_hz",
            "n_candidates",
            "hint_box",  # fixed size, never the box's edges (checked below)
        }
        assert view["item"]["t_center_s"] == pytest.approx(4.0)
        # track ids and gaps would tell simulated from real background fish
        assert [t["track_id"] for t in view["tracks"]] == [None, None]
        assert {t["gap_s"] for t in view["tracks"]} == {0.0}
        for key, value in _walk(view):
            assert key not in LEAKED_KEYS, key
            if isinstance(value, (int, float, str)) and not isinstance(value, bool):
                assert value not in LEAKED_VALUES, (key, value)
        # the page cannot switch the prediction on: the server forces it off
        _, result = client.json(
            "/api/action",
            {
                "index": 0,
                "window_s": 0.2048,
                "action": "pick",
                "key": 1,
                "prediction_shown": True,
                "elapsed_s": 3.25,
            },
        )
        assert result["ok"]
        assert result["decision"]["emitter_id"] is None
        assert "id" not in result["decision"]["text"]
        assert "id" not in result["message"]
        decision = server.session.decision(0)
        assert decision is not None
        assert (decision.blind, decision.prediction_shown, decision.elapsed_s) == (
            True,
            False,
            3.25,
        )
        _, bad = client.json(
            "/api/action",
            {"index": 1, "window_s": 0.2048, "action": "not_chirp", "elapsed_s": "nan"},
        )
        assert bad["ok"]
        assert server.session.decision(1).elapsed_s is None  # pyright: ignore[reportOptionalMemberAccess]
    finally:
        server.shutdown()
        store.close()


def test_page_draws_a_marker_and_ignores_e_when_blind() -> None:
    from importlib import resources

    html = (
        resources.files("chirpdetector.labeling").joinpath("labeller.html").read_text()
    )
    assert "if (S.state.blind)" in html
    assert 'k === "e" && S.state.blind' in html
    assert "elapsed_s: elapsed" in html


def test_matplotlib_blind_view(blind_session: LabelSession) -> None:
    lab = SingleLabeller(blind_session, figure=plt.figure(figsize=(16, 9)))
    lab.fig.canvas.draw()
    try:
        assert not lab.box.get_visible()
        assert not lab.box_lines.get_visible()
        assert lab.hint.get_visible()  # fixed-size hint at the candidate
        assert (lab.hint.get_width(), lab.hint.get_height()) == pytest.approx(
            (0.3, 200.0)
        )
        for mark in lab.centre_marks.values():
            assert mark.get_visible()
            xs = np.asarray(mark.get_xdata(), dtype=float)
            np.testing.assert_allclose(xs[np.isfinite(xs)], 4.0)
        info = lab.info.get_text()
        assert "t 4.00 s" in info
        assert "3.930" not in info  # no box edges
        for leak in ("700.4", "model", "score", "e: model"):
            assert leak not in info
        assert "(id" not in info
        labels = [t.get_text() for t in lab._overlays if hasattr(t, "get_text")]
        assert labels
        assert not any("id" in text for text in labels)
        lab.on_key(SimpleNamespace(key="e"))
        assert np.asarray(lab.pred_lines["zoom"].get_xydata()).size == 0
        assert "stays hidden" in lab.note.get_text()
        lab._shown_at -= 2.0  # two seconds on the item
        lab.on_key(SimpleNamespace(key="1"))
        decision = blind_session.decision(0)
        assert decision is not None
        assert decision.blind
        assert decision.prediction_shown is False
        assert decision.elapsed_s is not None
        assert decision.elapsed_s >= 2.0
    finally:
        plt.close(lab.fig)


def test_window_toggle_keeps_the_clock(blind_session: LabelSession) -> None:
    lab = SingleLabeller(blind_session, figure=plt.figure())
    try:
        lab._shown_at -= 5.0
        lab.on_key(SimpleNamespace(key="w"))
        lab.on_key(SimpleNamespace(key="x"))
        decision = blind_session.decision(0)
        assert decision is not None
        assert decision.elapsed_s is not None
        assert decision.elapsed_s >= 5.0
    finally:
        plt.close(lab.fig)


def test_label_file_records_blind_and_time(blind_session: LabelSession) -> None:
    from gridtools.io import read_labels

    blind_session.mark(0, "not_chirp", elapsed_s=1.5)
    row = read_labels(blind_session.store.labels_dir / "2000-01-01-10_00.csv").iloc[0]
    assert row.blind
    assert row.elapsed_s == pytest.approx(1.5)


def test_blind_hint_is_a_fixed_size_box_at_the_candidate(
    candidate_queue: LabelQueue, tmp_path: Path
) -> None:
    """The hint shows where (centre time, near the lower edge), never the
    candidate box's own edges or size."""
    from chirpdetector.labeling.view import ViewConfig, hint_box

    store = LabelStore(tmp_path / "out", "tester", "2")
    session = LabelSession(candidate_queue, store, frontend="browser", blind=True)
    server = LabelServer(session, port=0)
    server.start()
    try:
        client = Client(server)
        status, raw, _ = client.request("/api/view?index=0&window=0.2048")
        assert status == 200
        view = json.loads(raw)
        t0, t1, f0, f1 = view["item"]["hint_box"]
        assert (t1 - t0, f1 - f0) == pytest.approx((0.3, 200.0))
        assert 0.5 * (t0 + t1) == pytest.approx(4.0)  # the candidate's centre
        assert f0 == pytest.approx(680.0 - 30.0)
        values = {v for _, v in _walk(view) if isinstance(v, float)}
        assert not values & {3.93, 4.07, 820.0}
        item = session.item(0)
        assert hint_box(item, ViewConfig()) == pytest.approx((t0, t1, f0, f1))
    finally:
        server.shutdown()
        store.close()


def test_blind_hint_can_be_switched_off(
    candidate_queue: LabelQueue, tmp_path: Path
) -> None:
    from chirpdetector.labeling.view import ViewConfig

    with LabelStore(tmp_path / "out", "tester", "2") as store:
        session = LabelSession(
            candidate_queue,
            store,
            background=False,
            blind=True,
            config=ViewConfig(blind_hint=False),
        )
        lab = SingleLabeller(session, figure=plt.figure(figsize=(16, 9)))
        try:
            lab.fig.canvas.draw()
            assert not lab.hint.get_visible()
        finally:
            plt.close(lab.fig)
            session.shutdown()
