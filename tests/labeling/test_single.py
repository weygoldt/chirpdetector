"""The matplotlib single-candidate labeller (Agg, real mouse events)."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent

from chirpdetector.labeling.session import LabelSession
from chirpdetector.labeling.single import SingleLabeller, _maximise


@pytest.fixture
def labeller(session: LabelSession) -> Iterator[SingleLabeller]:
    lab = SingleLabeller(session, figure=plt.figure(figsize=(16, 9), dpi=100))
    lab.fig.canvas.draw()
    yield lab
    plt.close(lab.fig)


def mouse(lab: SingleLabeller, kind: str, panel: str, t: float, f: float, button=None):
    """Send a mouse event at the display position of ``(t, f)`` in a panel."""
    x, y = lab.axes[panel].transData.transform((t, f))
    event = MouseEvent(kind, lab.fig.canvas, x, y, button=button)
    lab.fig.canvas.callbacks.process(kind, event)
    return event


def key(lab: SingleLabeller, name: str) -> None:
    lab.on_key(SimpleNamespace(key=name))


def test_images_fill_their_axes_exactly(labeller: SingleLabeller) -> None:
    view = labeller.view
    assert view is not None
    for name, ax in labeller.axes.items():
        extent = view.panel(name).extent
        assert tuple(labeller.images[name].get_extent()) == pytest.approx(extent)
        assert ax.get_xlim() == pytest.approx(extent[:2])
        assert ax.get_ylim() == pytest.approx(extent[2:])


def test_click_maps_pixels_to_hz_and_snaps(labeller: SingleLabeller) -> None:
    session = labeller.session
    assert labeller.index == 0
    event = mouse(labeller, "button_press_event", "zoom", 3.85, 701.2, MouseButton.LEFT)
    # matplotlib's inverse transform gives back the clicked data point
    assert (event.xdata, event.ydata) == pytest.approx((3.85, 701.2), abs=1e-9)
    decision = session.decision(0)
    assert decision is not None
    assert (decision.label, decision.emitter_id, decision.via) == ("chirp", 0, "mouse")
    assert decision.clicked_hz == pytest.approx(701.2, abs=1e-9)
    assert decision.emitter_baseline_hz == 700.0
    assert decision.click_offset_hz == pytest.approx(1.2, abs=1e-9)
    assert labeller.index == 1  # moved on


def test_one_pixel_is_a_fraction_of_a_hz_in_the_zoom(labeller: SingleLabeller) -> None:
    ax = labeller.axes["zoom"]
    (_, y0), (_, y1) = ax.transData.transform([(3.9, 700.0), (3.9, 701.0)])
    assert y1 - y0 > 5  # pixels per Hz at 16 x 9 in, 100 dpi; tracks 60 Hz apart
    ax = labeller.axes["overview"]
    (_, y0), (_, y1) = ax.transData.transform([(3.9, 700.0), (3.9, 701.0)])
    assert y1 - y0 < 5


def test_hover_highlights_the_track_a_click_would_snap_to(
    labeller: SingleLabeller,
) -> None:
    mouse(labeller, "motion_notify_event", "zoom", 3.8, 757.0)
    assert labeller.hover is not None
    assert labeller.hover.track.track_id == 1
    for line in labeller.highlight.values():
        np.testing.assert_allclose(np.asarray(line.get_xydata())[:, 1], 760.0)
    assert "-> 2 (id 1)" in labeller.readout.get_text()
    mouse(labeller, "motion_notify_event", "zoom", 3.8, 730.0)
    assert labeller.hover is None
    assert "no track within" in labeller.readout.get_text()
    assert all(
        np.asarray(ln.get_xydata()).size == 0 for ln in labeller.highlight.values()
    )


def test_a_click_off_the_tracks_is_rejected(labeller: SingleLabeller) -> None:
    mouse(labeller, "button_press_event", "overview", 3.95, 790.0, MouseButton.LEFT)
    assert labeller.session.decision(0) is None
    assert labeller.index == 0
    assert "no track within" in labeller.note.get_text()


@pytest.mark.parametrize(
    ("button", "label"),
    [(MouseButton.RIGHT, "not_chirp"), (MouseButton.MIDDLE, "chirp_emitter_unclear")],
)
def test_right_and_middle_buttons(labeller: SingleLabeller, button, label) -> None:
    mouse(labeller, "button_press_event", "zoom", 3.9, 700.0, button)
    decision = labeller.session.decision(0)
    assert decision is not None
    assert (decision.label, decision.emitter_id, decision.via) == (label, None, "mouse")


def test_keys(labeller: SingleLabeller) -> None:
    s = labeller.session
    key(labeller, "2")  # item 0: track 2 = id 1
    assert s.decision(0).emitter_id == 1  # pyright: ignore[reportOptionalMemberAccess]
    assert labeller.index == 1
    key(labeller, "0")
    assert s.decision(1).label == "chirp_emitter_unclear"  # pyright: ignore[reportOptionalMemberAccess]
    key(labeller, "x")
    assert s.decision(2).label == "not_chirp"  # pyright: ignore[reportOptionalMemberAccess]
    key(labeller, " ")
    assert s.status(3) == "skipped"
    assert labeller.index == 3  # the skipped item is all that is left
    key(labeller, "u")
    assert s.status(3) == "open"
    key(labeller, "backspace")
    assert s.decision(2) is None
    assert labeller.index == 2
    key(labeller, "left")
    assert labeller.index == 1
    key(labeller, "right")
    assert labeller.index == 2
    key(labeller, "9")
    assert "no track 9" in labeller.note.get_text()
    key(labeller, "h")
    assert labeller.help.get_visible()
    key(labeller, "escape")
    assert not labeller.help.get_visible()


def test_window_toggle_rerenders(labeller: SingleLabeller) -> None:
    before = labeller.view
    key(labeller, "w")
    assert labeller.window_s == 0.0512
    assert labeller.view is not None
    assert labeller.view is not before
    assert labeller.view.window_s == 0.0512
    key(labeller, "1")
    assert labeller.session.decision(0).window_s == 0.0512  # pyright: ignore[reportOptionalMemberAccess]


def test_prediction_is_hidden_until_asked_for(labeller: SingleLabeller) -> None:
    line = labeller.pred_lines["zoom"]
    assert np.asarray(line.get_xydata()).size == 0
    assert "700.4" not in labeller.info.get_text()
    key(labeller, "e")
    assert np.asarray(line.get_xydata())[0, 1] == pytest.approx(700.4)
    assert "model: 700.4 Hz" in labeller.info.get_text()
    key(labeller, "1")
    decision = labeller.session.decision(0)
    assert decision is not None
    assert decision.prediction_shown
    key(labeller, "x")  # item 1 has no prediction: not "shown"
    decision = labeller.session.decision(1)
    assert decision is not None
    assert decision.prediction_shown is False


def test_resume_starts_at_the_first_open_item(
    session: LabelSession,
) -> None:
    session.pick(0, 1)
    session.skip(1)
    lab = SingleLabeller(session, figure=plt.figure())
    assert lab.index == 2
    plt.close(lab.fig)


def test_maximise_is_harmless_without_a_window(labeller: SingleLabeller) -> None:
    _maximise(labeller.fig)
