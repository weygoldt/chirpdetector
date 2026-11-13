"""Matplotlib single-candidate labeller: one chirp per maximised window.

The window shows a :class:`~chirpdetector.labeling.view.CandidateView`:
the overview (whole chirp) on top and the baseline zoom below, with every
live track marked by short dashed ticks at both edges and its number. The
track a left click would snap to is highlighted while the mouse moves. Keys
and mouse buttons are listed in :data:`HELP`; the actions themselves are
:class:`~chirpdetector.labeling.session.LabelSession`'s, so this window and
the browser frontend label the same way.

Hover highlighting is blitted (only the highlight is redrawn), so the
mouse stays responsive on a full-screen figure.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from chirpdetector.labeling.session import ActionResult, LabelSession, describe
from chirpdetector.labeling.view import (
    PANELS,
    CandidateView,
    Snap,
    colour_lut,
    colourize,
    hint_box,
    prediction_of,
    snap,
)

__all__ = ["HELP", "SingleLabeller", "run_single"]

log = logging.getLogger(__name__)

HELP = """\
left click on a baseline  chirp from that fish: snaps to the highlighted
                          track (nearest within the tolerance)
1-9                       chirp from the track with that number
0 / middle click          chirp, emitter unclear
x / right click           not a chirp
s / space                 skip (comes back after the open items)
u / backspace             undo the last action (jumps to its item)
left / right arrow        previous / next item
w                         overview window 204.8 / 51.2 ms (zoom: 204.8)
e                         show / hide the model's prediction
f                         full screen
h                         toggle this help
q                         quit (every action is already saved)"""

DECISION_COLOURS = {
    "chirp": "#2ecc40",
    "not_chirp": "#ff4136",
    "chirp_emitter_unclear": "#ff851b",
    "skipped": "#aaaaaa",
}
TRACK_COLOUR = "#d0d0d0"
HIGHLIGHT_COLOUR = "#00e5ff"
PREDICTION_COLOUR = "#ff4fd8"
_TICK_FRACTION = 0.1
_KEYS_NUMBERS = {str(k): k for k in range(1, 10)}


class SingleLabeller:
    """The single-candidate figure and its event handlers.

    Parameters
    ----------
    session
        Queue, store and rendering.
    figure
        Figure to draw into; a new pyplot figure if None.
    """

    def __init__(self, session: LabelSession, *, figure: Figure | None = None) -> None:
        self.session = session
        self.window_s = session.config.window_s
        self.show_prediction = False
        self.index = session.first_index()
        self.view: CandidateView | None = None
        self.hover: Snap | None = None
        self.message = ""
        self._lut = colour_lut()
        self._background: Any = None
        self._overlays: list[Any] = []

        if figure is None:
            import matplotlib.pyplot as plt

            figure = plt.figure(figsize=(16, 9))
        self.fig = figure
        self.fig.set_facecolor("#1b1b1b")
        self.axes: dict[str, Axes] = {
            "overview": self.fig.add_axes((0.05, 0.585, 0.935, 0.33)),
            "zoom": self.fig.add_axes((0.05, 0.05, 0.935, 0.49)),
        }
        self.images = {}
        self.highlight: dict[str, Line2D] = {}
        self.pred_lines: dict[str, Line2D] = {}
        self.markers: dict[str, Line2D] = {}
        for name, ax in self.axes.items():
            ax.set_facecolor("black")
            ax.tick_params(colors="0.8", labelsize=8)
            ax.set_ylabel("Hz", color="0.8", fontsize=8)
            self.images[name] = ax.imshow(
                np.zeros((2, 2, 3), dtype=np.uint8),
                origin="lower",
                aspect="auto",
                interpolation="antialiased",
            )
            (self.highlight[name],) = ax.plot(
                [], [], color=HIGHLIGHT_COLOUR, lw=1.3, ls="--", animated=True
            )
            (self.pred_lines[name],) = ax.plot(
                [], [], color=PREDICTION_COLOUR, lw=2.0, marker="<", ms=9
            )
            (self.markers[name],) = ax.plot([], [], ls="", ms=11, mew=2.0, mfc="none")
        self.axes["zoom"].set_xlabel("time (s)", color="0.8", fontsize=8)
        self.box = Rectangle((0, 0), 0, 0, fill=False, edgecolor="white", lw=0.8)
        self.axes["overview"].add_patch(self.box)
        (self.box_lines,) = self.axes["zoom"].plot(
            [], [], color="white", lw=0.6, ls=":"
        )
        # blind mode: a faint fixed-size hint box (where to look, not the shape)
        self.hint = Rectangle(
            (0, 0), 0, 0, fill=False, edgecolor="white", lw=0.8, ls="--", alpha=0.45
        )
        self.axes["overview"].add_patch(self.hint)
        # blind mode: a short tick at the box centre on the top and bottom edges
        self.centre_marks = {
            name: ax.plot([], [], color="white", lw=2.0, solid_capstyle="butt")[0]
            for name, ax in self.axes.items()
        }
        self._shown_at = time.perf_counter()
        text = {"color": "0.9", "family": "monospace", "fontsize": 9}
        self.status = self.fig.text(0.005, 0.995, "", va="top", **text)
        self.info = self.fig.text(0.005, 0.965, "", va="top", **text)
        self.note = self.fig.text(0.05, 0.555, "", va="center", **text)
        self.readout = self.fig.text(
            0.985, 0.555, "", va="center", ha="right", animated=True, **text
        )
        self.help = self.fig.text(
            0.5,
            0.5,
            HELP,
            ha="center",
            va="center",
            family="monospace",
            fontsize=11,
            visible=False,
            zorder=10,
            bbox={"facecolor": "white", "alpha": 0.95, "edgecolor": "0.3"},
        )
        canvas = self.fig.canvas
        manager = canvas.manager
        default_keys = getattr(manager, "key_press_handler_id", None)
        if default_keys is not None:  # matplotlib's own keys (pan, save, ...)
            canvas.mpl_disconnect(default_keys)
        canvas.mpl_connect("button_press_event", self.on_click)
        canvas.mpl_connect("key_press_event", self.on_key)
        canvas.mpl_connect("motion_notify_event", self.on_move)
        canvas.mpl_connect("draw_event", self._on_draw)
        canvas.mpl_connect("close_event", lambda _: self.session.shutdown())
        self.show(self.index)

    # drawing ---------------------------------------------------------------

    def show(self, index: int) -> None:
        """Show an item (rendered now unless prefetched)."""
        index = min(max(index, 0), len(self.session) - 1)
        if index != self.index or self.view is None:  # a window toggle keeps the clock
            self._shown_at = time.perf_counter()
        self.index = index
        self.hover = None
        try:
            self.view = self.session.view(self.index, self.window_s)
        except Exception as err:
            log.exception("could not render item %d", self.index)
            self.view = None
            self.message = f"render error: {err}  (x: not a chirp, s: skip)"
        for artist in self._overlays:
            artist.remove()
        self._overlays = []
        if self.view is not None:
            self._draw_view(self.view)
        self._decorate()
        self._update_text()
        self.fig.canvas.draw_idle()

    def _draw_view(self, view: CandidateView) -> None:
        item = view.item
        for name in PANELS:
            panel = view.panel(name)
            ax = self.axes[name]
            image = self.images[name]
            image.set_data(colourize(panel, self._lut))
            image.set_extent(panel.extent)
            ax.set_xlim(panel.extent[0], panel.extent[1])
            ax.set_ylim(panel.extent[2], panel.extent[3])
            t0, t1 = panel.extent[0], panel.extent[1]
            length = _TICK_FRACTION * (t1 - t0)
            xs: list[float] = []
            ys: list[float] = []
            for track in view.tracks:
                fa, fb = track.frequency_at(t0), track.frequency_at(t1)
                xs += [t0, t0 + length, math.nan, t1 - length, t1, math.nan]
                ys += [fa, track.frequency_at(t0 + length), math.nan]
                ys += [track.frequency_at(t1 - length), fb, math.nan]
                label = f"{track.key}"
                if name == "zoom":
                    if not self.session.blind:
                        label += f"  id {track.track_id}"
                    label += f"  {track.baseline_hz:.1f} Hz"
                    if track.gap_s > 1.0 and not self.session.blind:
                        label += f"  (gap {track.gap_s:.1f} s)"
                self._overlays.append(
                    ax.text(
                        t0 + 0.2 * length,
                        fa,
                        label,
                        color="black",
                        fontsize=9 if name == "zoom" else 8,
                        va="center",
                        ha="left",
                        bbox={
                            "boxstyle": "round,pad=0.15",
                            "facecolor": TRACK_COLOUR,
                            "edgecolor": "none",
                            "alpha": 0.85,
                        },
                        clip_on=True,
                    )
                )
            (ticks,) = ax.plot(xs, ys, color=TRACK_COLOUR, lw=1.0, ls="--")
            self._overlays.append(ticks)
        blind = self.session.blind
        show_hint = blind and self.session.config.blind_hint
        self.hint.set_visible(show_hint)
        if show_hint:
            h0, h1, g0, g1 = hint_box(item, self.session.config)
            self.hint.set_bounds(h0, g0, h1 - h0, g1 - g0)
        self.box.set_visible(not blind)
        self.box_lines.set_visible(not blind)
        self.box.set_bounds(
            item.t_start_s,
            item.f_low_hz,
            item.t_end_s - item.t_start_s,
            item.f_high_hz - item.f_low_hz,
        )
        lo, hi = view.zoom.extent[2], view.zoom.extent[3]
        self.box_lines.set_data(
            [item.t_start_s, item.t_start_s, math.nan, item.t_end_s, item.t_end_s],
            [lo, hi, math.nan, lo, hi],
        )
        for name, mark in self.centre_marks.items():
            f0, f1 = view.panel(name).extent[2:]
            tick = 0.06 * (f1 - f0)
            tc = item.t_center_s
            mark.set_visible(blind)
            mark.set_data(
                [tc, tc, math.nan, tc, tc], [f1, f1 - tick, math.nan, f0, f0 + tick]
            )

    def _decorate(self) -> None:
        """Frames, decision markers and the prediction from the current state."""
        state = self.session.status(self.index)
        decision = self.session.decision(self.index)
        colour = (
            DECISION_COLOURS[decision.label]
            if decision
            else (DECISION_COLOURS["skipped"] if state == "skipped" else "0.35")
        )
        width = 3.0 if state != "open" else 0.8
        for name, ax in self.axes.items():
            for spine in ax.spines.values():
                spine.set_edgecolor(colour)
                spine.set_linewidth(width)
            marker = self.markers[name]
            point = None
            if decision is not None:
                t_click, f_click = decision.clicked_t_s, decision.clicked_hz
                if t_click is not None and f_click is not None:
                    point = (t_click, f_click)
                elif decision.emitter_baseline_hz is not None:
                    point = (decision.box[0], decision.emitter_baseline_hz)
            if point is not None:
                marker.set_data([point[0]], [point[1]])
                marker.set_marker("o" if decision and decision.via == "mouse" else "D")
                marker.set_markeredgecolor(colour)
            else:
                marker.set_data([], [])
            self._draw_prediction(name)

    def _draw_prediction(self, name: str) -> None:
        line = self.pred_lines[name]
        prediction = prediction_of(self.session.item(self.index))
        if not self._prediction_visible() or prediction is None or self.view is None:
            line.set_data([], [])
            return
        f = prediction.emitter_hz
        if f is None and prediction.track_id is not None:
            track = next(
                (t for t in self.view.tracks if t.track_id == prediction.track_id), None
            )
            f = track.baseline_hz if track else None
        if f is None:
            line.set_data([], [])
            return
        t0, t1 = self.view.panel(name).extent[:2]
        line.set_data([t1 - 0.06 * (t1 - t0), t1], [f, f])

    def _update_text(self) -> None:
        session = self.session
        p = session.progress()
        bins = "  ".join(f"{b}:{d}/{n}" for b, (d, n) in p.per_bin.items() if n)
        self.status.set_text(
            f"{self.index + 1}/{p.total}   labelled {p.labelled}   skipped "
            f"{p.skipped}   open {p.open}   [ΔF Hz {bins}]   "
            f"{session.store.labeler}   window {1e3 * self.window_s:.1f} ms   h: help"
        )
        item = session.item(self.index)
        delta = (
            f"ΔF {item.delta_f_hz:.1f} Hz" if item.delta_f_hz is not None else "ΔF -"
        )
        tracks = (
            self.view.tracks if self.view is not None else session.tracks(self.index)
        )
        listed = "   ".join(
            f"{t.key}: {t.baseline_hz:.1f} Hz"
            + (
                ""
                if session.blind
                else f" (id {t.track_id}"
                + (f", gap {t.gap_s:.1f} s)" if t.gap_s > 1.0 else ")")
            )
            for t in tracks
        )
        decision = session.decision(self.index)
        state = (
            describe(decision, tracks, ids=not session.blind)
            if decision
            else ("skipped" if session.status(self.index) == "skipped" else "open")
        )
        when = (
            f"t {item.t_center_s:.2f} s"
            if session.blind
            else f"{item.t_start_s:.3f}-{item.t_end_s:.3f} s"
        )
        text = (
            f"{item.trial}   {when}   {delta} "
            f"({item.df_bin})   tracks {listed or 'none'}   [{state}]"
        )
        prediction = None if session.blind else prediction_of(item)
        if self.show_prediction and prediction is not None:
            parts = ["model:"]
            if prediction.emitter_hz is not None:
                parts.append(f"{prediction.emitter_hz:.1f} Hz")
            if prediction.track_id is not None:
                parts.append(f"id {prediction.track_id}")
            if prediction.score is not None:
                parts.append(f"score {prediction.score:.2f}")
            if prediction.source:
                parts.append(prediction.source)
            text += "   " + " ".join(parts)
        elif prediction is not None:
            text += "   (e: model's prediction)"
        self.info.set_text(text)
        self.note.set_text(self.message)
        manager = self.fig.canvas.manager
        if manager is not None:
            manager.set_window_title(f"label-emitters {p.labelled}/{p.total}")

    def _on_draw(self, _event: Any) -> None:
        canvas = self.fig.canvas
        if canvas.supports_blit:
            self._background = canvas.copy_from_bbox(self.fig.bbox)  # pyright: ignore[reportAttributeAccessIssue]
        self._draw_animated()
        self.session.prefetch(self.session.next_index(self.index), self.window_s)

    def _draw_animated(self) -> None:
        for name, ax in self.axes.items():
            ax.draw_artist(self.highlight[name])
        self.fig.draw_artist(self.readout)

    def _update_hover(self, t_s: float | None, f_hz: float | None) -> None:
        view = self.view
        found = None
        readout = ""
        if view is not None and t_s is not None and f_hz is not None:
            found = snap(view.tracks, t_s, f_hz, self.session.tolerance_hz)
            readout = f"{f_hz:.1f} Hz  "
            if found is not None:
                readout += (
                    f"-> {found.track.key}"
                    + ("" if self.session.blind else f" (id {found.track.track_id})")
                    + "  "
                    f"{found.offset_hz:+.1f} Hz"
                )
            else:
                readout += f"no track within {self.session.tolerance_hz:g} Hz"
        self.hover = found
        for line in self.highlight.values():
            if found is None:
                line.set_data([], [])
            else:
                line.set_data(found.track.times_s, found.track.freqs_hz)
        self.readout.set_text(readout)
        canvas = self.fig.canvas
        if self._background is not None and canvas.supports_blit:
            canvas.restore_region(self._background)  # pyright: ignore[reportAttributeAccessIssue]
            self._draw_animated()
            canvas.blit(self.fig.bbox)

    # events ----------------------------------------------------------------

    def _panel_of(self, ax: object) -> str | None:
        return next((n for n, a in self.axes.items() if a is ax), None)

    def on_move(self, event: Any) -> None:
        """Highlight the track a click would snap to."""
        if self._panel_of(event.inaxes) is None:
            if self.hover is not None or self.readout.get_text():
                self._update_hover(None, None)
            return
        self._update_hover(event.xdata, event.ydata)

    def on_click(self, event: Any) -> None:
        """Left: snap to a track; middle: unclear; right: not a chirp."""
        if getattr(event, "dblclick", False) or event.button is None:
            return
        button = int(event.button)
        if button == 1:
            if self._panel_of(event.inaxes) is None or event.xdata is None:
                return
            result = self.session.click(
                self.index,
                float(event.xdata),
                float(event.ydata),
                window_s=self.window_s,
                prediction_shown=self._prediction_visible(),
                elapsed_s=self._elapsed(),
            )
        elif button == 2:
            result = self._mark("chirp_emitter_unclear", via="mouse")
        elif button == 3:
            result = self._mark("not_chirp", via="mouse")
        else:
            return
        self._after(result)

    def on_key(self, event: Any) -> None:
        """Keys of :data:`HELP`."""
        key = event.key
        if key is None:
            return
        if key in _KEYS_NUMBERS:
            self._after(
                self.session.pick(
                    self.index,
                    _KEYS_NUMBERS[key],
                    window_s=self.window_s,
                    prediction_shown=self._prediction_visible(),
                    elapsed_s=self._elapsed(),
                )
            )
        elif key == "0":
            self._after(self._mark("chirp_emitter_unclear"))
        elif key == "x":
            self._after(self._mark("not_chirp"))
        elif key in ("s", " ", "space"):
            self._after(self.session.skip(self.index, elapsed_s=self._elapsed()))
        elif key in ("u", "backspace", "ctrl+z"):
            self._after(self.session.undo())
        elif key in ("right", "n"):
            self.message = ""
            self.show((self.index + 1) % len(self.session))
        elif key in ("left", "p"):
            self.message = ""
            self.show((self.index - 1) % len(self.session))
        elif key == "w":
            self.window_s = self.session.toggle_window(self.window_s)
            self.message = f"window {1e3 * self.window_s:.1f} ms"
            self.show(self.index)
        elif key == "e" and self.session.blind:
            self.message = "blind: the model's prediction stays hidden"
            self._update_text()
            self.fig.canvas.draw_idle()
        elif key == "e":
            self.show_prediction = not self.show_prediction
            self._decorate()
            self._update_text()
            self.fig.canvas.draw_idle()
        elif key == "f":
            manager = self.fig.canvas.manager
            if manager is not None:
                manager.full_screen_toggle()
        elif key in ("h", "?", "escape"):
            self.help.set_visible(key != "escape" and not self.help.get_visible())
            self.fig.canvas.draw_idle()
        elif key == "q":
            import matplotlib.pyplot as plt

            plt.close(self.fig)

    def _prediction_visible(self) -> bool:
        return (
            not self.session.blind
            and self.show_prediction
            and prediction_of(self.session.item(self.index)) is not None
        )

    def _elapsed(self) -> float:
        """Seconds since the current item was shown."""
        return time.perf_counter() - self._shown_at

    def _mark(self, label: str, via: str = "key") -> ActionResult:
        return self.session.mark(
            self.index,
            label,
            window_s=self.window_s,
            prediction_shown=self._prediction_visible(),
            via=via,
            elapsed_s=self._elapsed(),
        )

    def _after(self, result: ActionResult) -> None:
        self.message = result.message
        if not result.ok:
            self._update_text()
            self.fig.canvas.draw_idle()
            return
        if result.next_index is None:
            p = self.session.progress()
            self.message += (
                f"   -- all {p.total} labelled (arrows to review, q to quit)"
            )
            self.show(self.index)
        else:
            self.show(result.next_index)


def _maximise(fig: Figure) -> None:
    """Maximise the figure's window (Qt, Tk or GTK); best effort."""
    manager = fig.canvas.manager
    window = getattr(manager, "window", None)
    if window is None:
        return
    try:
        if hasattr(window, "showMaximized"):  # Qt
            window.showMaximized()
        elif hasattr(window, "wm_attributes"):  # Tk
            try:
                window.wm_attributes("-zoomed", True)
            except Exception:
                window.state("zoomed")
        elif hasattr(window, "maximize"):  # GTK
            window.maximize()
    except Exception:  # pragma: no cover - depends on the window system
        log.debug("could not maximise the window", exc_info=True)


def run_single(session: LabelSession) -> None:
    """Open the maximised labelling window and block until it is closed.

    Raises
    ------
    RuntimeError
        If matplotlib has no interactive backend (e.g. no display).
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.registry import BackendFilter, backend_registry

    backend = matplotlib.get_backend().lower()
    if backend in backend_registry.list_builtin(BackendFilter.NON_INTERACTIVE):
        msg = f"matplotlib backend {backend!r} is not interactive; is a display available? (try --serve)"
        raise RuntimeError(msg)
    with plt.rc_context({"toolbar": "None", "text.hinting": "no_hinting"}):
        labeller = SingleLabeller(session)
        _maximise(labeller.fig)
        plt.show()
    session.shutdown()
