"""The matplotlib labeller, driven with synthetic mouse and key events (Agg)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from gridtools.io import read_labels
from matplotlib.backend_bases import MouseButton
from matplotlib.colors import to_rgba

from chirpdetector.labeling.gui import FRAME_COLOURS, Labeller, nearest_track
from chirpdetector.labeling.queue import LabelQueue, QueueConfig, build_queue
from chirpdetector.labeling.store import LabelStore

GRID = (2, 2)


@pytest.fixture
def trial(tmp_path: Path, make_synthetic_trial: Callable[[Path], Path]) -> Path:
    return make_synthetic_trial(tmp_path / "2000-01-01-10_00")


@pytest.fixture
def queue(trial: Path) -> LabelQueue:
    return build_queue([trial], QueueConfig(per_bin=50, seed=0))


@pytest.fixture
def session(
    trial: Path, queue: LabelQueue, tmp_path: Path
) -> Iterator[tuple[Labeller, LabelStore]]:
    store = LabelStore(tmp_path / "out", "tester", "2.0.0")
    labeller = Labeller(
        queue,
        {trial.name: trial},
        store,
        grid=GRID,
        figure=plt.figure(),
        background=False,
    )
    yield labeller, store
    plt.close(labeller.fig)
    store.close()


def click(
    labeller: Labeller, slot: int, button: MouseButton, t_s: float, f_hz: float
) -> None:
    ax = labeller.slots[slot].ax
    labeller.on_click(
        SimpleNamespace(inaxes=ax, button=button, xdata=t_s, ydata=f_hz, dblclick=False)
    )


def key(labeller: Labeller, name: str) -> None:
    labeller.on_key(SimpleNamespace(key=name))


def slot_of_chirp(labeller: Labeller) -> int:
    """Go to the page with the synthetic chirp; return its slot."""
    index = next(i for i, it in enumerate(labeller.queue.items) if it.t_start_s == 3.93)
    labeller.show_page(index // labeller.per_page)
    return index % labeller.per_page


def test_nearest_track_within_tolerance() -> None:
    baselines = [(5, 700.0), (9, 760.0)]
    assert nearest_track(baselines, 703.0, 15.0) == 5
    assert nearest_track(baselines, 752.0, 15.0) == 9
    assert nearest_track(baselines, 730.0, 15.0) is None
    assert nearest_track([], 700.0, 15.0) is None


def test_left_click_assigns_the_nearest_track(session) -> None:
    labeller, store = session
    slot = slot_of_chirp(labeller)
    item = labeller.page_items(labeller.page)[slot]
    click(labeller, slot, MouseButton.LEFT, 4.0, 702.0)
    decision = store.current(item.item_id)
    assert decision is not None
    assert (decision.label, decision.emitter_id) == ("chirp", 0)
    assert (decision.clicked_t_s, decision.clicked_hz) == (4.0, 702.0)
    spine = labeller.slots[slot].ax.spines["top"]
    assert spine.get_edgecolor() == pytest.approx(to_rgba(FRAME_COLOURS["chirp"]))
    assert np.asarray(labeller.slots[slot].marker.get_xydata()).tolist() == [
        [4.0, 702.0]
    ]

    click(labeller, slot, MouseButton.LEFT, 4.0, 757.0)  # override: other fish
    decision = store.current(item.item_id)
    assert decision is not None
    assert decision.emitter_id == 1


def test_left_click_far_from_tracks_is_a_chirp_without_emitter(session) -> None:
    labeller, store = session
    slot = slot_of_chirp(labeller)
    click(labeller, slot, MouseButton.LEFT, 4.0, 730.0)
    decision = store.current(labeller.page_items(labeller.page)[slot].item_id)
    assert decision is not None
    assert (decision.label, decision.emitter_id, decision.clicked_hz) == (
        "chirp",
        None,
        730.0,
    )
    assert labeller.slots[slot].marker.get_marker() == "x"


@pytest.mark.parametrize(
    ("button", "label"),
    [(MouseButton.RIGHT, "not_chirp"), (MouseButton.MIDDLE, "chirp_emitter_unclear")],
)
def test_right_and_middle_click(session, button: MouseButton, label: str) -> None:
    labeller, store = session
    click(labeller, 0, button, 1.0, 700.0)
    decision = store.current(labeller.page_items(0)[0].item_id)
    assert decision is not None
    assert (decision.label, decision.emitter_id, decision.clicked_hz) == (
        label,
        None,
        None,
    )
    spine = labeller.slots[0].ax.spines["left"]
    assert spine.get_edgecolor() == pytest.approx(to_rgba(FRAME_COLOURS[label]))


def test_clicks_outside_cutouts_are_ignored(session) -> None:
    labeller, store = session
    labeller.on_click(SimpleNamespace(inaxes=None, button=1, xdata=None, ydata=None))
    labeller.on_click(
        SimpleNamespace(inaxes=labeller.slots[0].ax, button=8, xdata=1.0, ydata=1.0)
    )
    assert store.undo() is None


def test_keys_page_and_undo(session) -> None:
    labeller, store = session
    assert labeller.n_pages == 2  # 8 detections on 2x2 pages
    key(labeller, " ")
    assert labeller.page == 1
    item = labeller.page_items(1)[0]
    click(labeller, 0, MouseButton.RIGHT, 1.0, 700.0)
    key(labeller, "n")  # last page: stays
    assert labeller.page == 1
    key(labeller, "b")
    assert labeller.page == 0
    key(labeller, "u")  # undo jumps to the page of the undone click
    assert labeller.page == 1
    assert store.current(item.item_id) is None
    assert labeller.slots[0].ax.spines["top"].get_linewidth() < 1
    key(labeller, "h")
    assert labeller.help.get_visible()


def test_unclicked_items_stay_open_and_resume_starts_there(
    trial: Path, queue: LabelQueue, tmp_path: Path
) -> None:
    out = tmp_path / "resume"
    with LabelStore(out, "tester", "2") as store:
        labeller = Labeller(
            queue,
            {trial.name: trial},
            store,
            grid=GRID,
            figure=plt.figure(),
            background=False,
        )
        for slot in range(4):
            click(labeller, slot, MouseButton.RIGHT, 1.0, 700.0)
        key(labeller, "n")
        click(labeller, 0, MouseButton.MIDDLE, 1.0, 700.0)
        plt.close(labeller.fig)
    with LabelStore(out, "tester", "2") as store:
        labeller = Labeller(
            queue,
            {trial.name: trial},
            store,
            grid=GRID,
            figure=plt.figure(),
            background=False,
        )
        assert labeller.page == 1  # first page with an unlabelled item
        done, per_bin = labeller.progress()
        assert done == 5
        assert sum(n for _, n in per_bin.values()) == len(queue)
        width = labeller.slots[0].ax.spines["top"].get_linewidth()
        assert width > 1  # the middle click is still shown
        plt.close(labeller.fig)
    labels = read_labels(out / "labels" / f"{trial.name}.csv")
    assert sorted(labels.label) == ["chirp_emitter_unclear"] + ["not_chirp"] * 4


def test_image_extents_are_the_requested_windows(session) -> None:
    labeller, _ = session
    for page in range(labeller.n_pages):
        labeller.show_page(page)
        for slot, cut in zip(labeller.slots, labeller.cutouts, strict=False):
            assert cut is not None
            item = cut.item
            n_freq, n_time = cut.db.shape
            extent = slot.image.get_extent()
            t0, t1, f0, f1 = cut.window
            assert (t0, t1) == pytest.approx(
                (item.t_start_s - 0.15, item.t_end_s + 0.15)
            )
            low = min(item.f_low_hz, *(f for _, f in cut.baselines)) - 40.0
            high = max(item.f_high_hz, *(f for _, f in cut.baselines)) + 40.0
            assert (f0, f1) == pytest.approx((low, high))
            dt, df = (extent[1] - extent[0]) / n_time, (extent[3] - extent[2]) / n_freq
            np.testing.assert_allclose(extent[:2], (t0, t1), atol=dt)
            np.testing.assert_allclose(extent[2:], (f0, f1), atol=df)
            assert slot.ax.get_xlim() == pytest.approx(extent[:2])


def test_page_state_is_blind_to_v1_assignment(
    tmp_path: Path, make_synthetic_trial: Callable[[Path], Path]
) -> None:
    """Swapping v1's emitter in the detections changes nothing that is shown."""
    pages = []
    for k, (track_id, emitter_hz) in enumerate([(0.0, 700.0), (1.0, 760.0)]):
        trial = make_synthetic_trial(tmp_path / f"v{k}" / "2000-01-01-10_00")
        csv = trial / "chirpdetector_bboxes.csv"
        df = pd.read_csv(csv)
        df["track_id"], df["emitter_eodf"] = track_id, emitter_hz
        df.to_csv(csv, index=False)
        q = build_queue([trial], QueueConfig(per_bin=50, seed=0))
        with LabelStore(tmp_path / f"out{k}", "t", "2") as store:
            labeller = Labeller(
                q,
                {trial.name: trial},
                store,
                grid=GRID,
                figure=plt.figure(),
                background=False,
            )
            cutouts = labeller.cutouts
            ticks = [np.asarray(s.ticks.get_xydata()) for s in labeller.slots]
            colours = {s.ticks.get_color() for s in labeller.slots}
            plt.close(labeller.fig)
        pages.append((q.items, cutouts, ticks))
        assert len(colours) == 1
    (items_a, cut_a, ticks_a), (items_b, cut_b, ticks_b) = pages
    assert items_a == items_b
    for ta, tb in zip(ticks_a, ticks_b, strict=True):
        np.testing.assert_array_equal(ta, tb)  # NaN-separated segments
    for a, b in zip(cut_a, cut_b, strict=True):
        assert a is not None
        assert b is not None
        np.testing.assert_array_equal(a.db, b.db)
        assert (a.baselines, a.extent, a.clim) == (b.baselines, b.extent, b.clim)
    names = {f.name for f in dataclasses.fields(cut_a[0])}  # pyright: ignore[reportArgumentType]
    assert not names & {"track_id", "emitter_hz", "emitter_eodf", "score"}
