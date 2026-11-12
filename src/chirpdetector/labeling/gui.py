"""Matplotlib grid labeller: many cutouts per page, one click per chirp.

One figure shows a page of cutouts (default 4 x 6). Each cutout is a v1
detection with its box outlined and short dashed ticks at the left and
right edges marking every live track's baseline (all in one neutral colour,
unnumbered, so nothing hints at v1's emitter).

Mouse (inside a cutout; clicking again overrides):

- left: chirp; the emitter is the live track whose baseline at the chirp is
  nearest to the clicked frequency, if within ``tolerance_hz`` (else NA);
- right: not a chirp;
- middle: chirp, emitter unclear.

Keys: space / enter / n next page, p / b previous page, u undo the last
click, h help, q quit. Every click is saved immediately.

The next and previous pages are rendered (and colour-mapped) in a
background thread while the current one is labelled, and the figure's
artists are updated in place.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import numpy.typing as npt
from gridtools.io import Recording, Tracks, load_tracks, open_recording
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from chirpdetector.labeling.queue import LabelQueue, QueueItem
from chirpdetector.labeling.render import Cutout, CutoutConfig, render_cutout
from chirpdetector.labeling.store import LabelStore

__all__ = ["HELP", "Labeller", "nearest_track", "run"]

log = logging.getLogger(__name__)

HELP = """\
left click     chirp; emitter = track nearest the click
right click    not a chirp
middle click   chirp, emitter unclear
               (click again to override)
space/enter/n  next page
p / b          previous page
u              undo last click
h              toggle this help
q              quit (every click is already saved)"""

FRAME_COLOURS = {
    "chirp": "#2ecc40",
    "not_chirp": "#ff4136",
    "chirp_emitter_unclear": "#ff851b",
}
_BUTTON_LABELS = {1: "chirp", 2: "chirp_emitter_unclear", 3: "not_chirp"}
_TICK_COLOUR = "#bbbbbb"
_TICK_FRACTION = 0.15


def nearest_track(
    baselines: Sequence[tuple[int, float]], f_hz: float, tolerance_hz: float
) -> int | None:
    """Track whose baseline is nearest to ``f_hz``, if within the tolerance.

    Parameters
    ----------
    baselines
        ``(track_id, baseline_hz)`` pairs.
    f_hz
        Clicked frequency.
    tolerance_hz
        Maximum distance.

    Returns
    -------
    The track id, or None if no baseline is within ``tolerance_hz``.
    """
    if not baselines:
        return None
    ident, f = min(baselines, key=lambda p: abs(p[1] - f_hz))
    return ident if abs(f - f_hz) <= tolerance_hz else None


@dataclass(frozen=True)
class _Page:
    """Rendered cutouts of a page and their colour-mapped images."""

    cutouts: list[Cutout | None]
    rgba: list[npt.NDArray[np.uint8] | None]


@dataclass
class _Slot:
    ax: Axes
    image: AxesImage
    box: Rectangle
    ticks: Line2D
    marker: Line2D
    title: Text


class Labeller:
    """The labelling figure and its event handlers.

    Parameters
    ----------
    queue
        Items to label, in order.
    trial_dirs
        Directory of every trial in the queue, by trial name.
    store
        Where clicks are saved.
    config
        Cutout rendering.
    grid
        ``(rows, columns)`` of cutouts per page.
    tolerance_hz
        Maximum distance between a left click and a track baseline for the
        click to name that track as the emitter.
    figure
        Figure to draw into; a new pyplot figure if None.
    background
        Render neighbouring pages in a background thread.
    """

    def __init__(
        self,
        queue: LabelQueue,
        trial_dirs: Mapping[str, Path],
        store: LabelStore,
        *,
        config: CutoutConfig | None = None,
        grid: tuple[int, int] = (4, 6),
        tolerance_hz: float = 15.0,
        figure: Figure | None = None,
        background: bool = True,
    ) -> None:
        missing = {t.name for t in queue.trials} - set(trial_dirs)
        if missing:
            msg = f"no directory given for trials {sorted(missing)}"
            raise ValueError(msg)
        self.queue = queue
        self.trial_dirs = dict(trial_dirs)
        self.store = store
        self.config = config or CutoutConfig()
        self.rows, self.cols = grid
        self.per_page = self.rows * self.cols
        self.n_pages = max(1, math.ceil(len(queue) / self.per_page))
        self.tolerance_hz = tolerance_hz
        self._index = {item.item_id: i for i, item in enumerate(queue.items)}
        self._trials: dict[str, tuple[Recording, Tracks]] = {}
        self._trial_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(1) if background else None
        self._pages: dict[int, Future[_Page]] = {}
        self._cmap = matplotlib.colormaps["magma"]  # a copy, used by the worker
        self.page = -1
        self.cutouts: list[Cutout | None] = []
        self.last_switch_ms = math.nan
        self._message = ""

        if figure is None:
            import matplotlib.pyplot as plt

            figure = plt.figure(figsize=(3 * self.cols, 2.4 * self.rows))
        self.fig = figure
        self.slots = self._build_axes()
        self.status = self.fig.text(
            0.004, 0.996, "", va="top", fontsize=8, family="monospace"
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
        canvas.mpl_connect("close_event", lambda _: self.shutdown())
        canvas.mpl_connect("draw_event", lambda _: self._prefetch())
        self.show_page(self._first_open_page())

    # pages -----------------------------------------------------------------

    def page_items(self, page: int) -> tuple[QueueItem, ...]:
        """Queue items on a page."""
        return self.queue.items[page * self.per_page : (page + 1) * self.per_page]

    def _first_open_page(self) -> int:
        for i, item in enumerate(self.queue.items):
            if self.store.current(item.item_id) is None:
                return i // self.per_page
        return 0

    def _trial(self, name: str) -> tuple[Recording, Tracks]:
        with self._trial_lock:
            if name not in self._trials:
                path = self.trial_dirs[name]
                self._trials[name] = (open_recording(path), load_tracks(path))
            return self._trials[name]

    def _render_page(self, page: int) -> _Page:
        """Render and colour-map a page (runs in the worker thread).

        Colour mapping here rather than in the figure leaves only RGBA
        resampling to the main-thread draw.
        """
        cutouts: list[Cutout | None] = []
        rgba: list[npt.NDArray[np.uint8] | None] = []
        for item in self.page_items(page):
            try:
                rec, tracks = self._trial(item.trial)
                cut = render_cutout(rec, tracks, item, self.config)
            except Exception:
                log.exception("could not render %s", item.item_id)
                cutouts.append(None)
                rgba.append(None)
                continue
            lo, hi = cut.clim
            scaled = np.clip((cut.db - lo) / (hi - lo), 0.0, 1.0)
            cutouts.append(cut)
            rgba.append(np.asarray(self._cmap(scaled, bytes=True), dtype=np.uint8))
        return _Page(cutouts, rgba)

    def _page_future(self, page: int) -> Future[_Page]:
        if page not in self._pages:
            if self._executor is not None:
                self._pages[page] = self._executor.submit(self._render_page, page)
            else:
                future: Future[_Page] = Future()
                future.set_result(self._render_page(page))
                self._pages[page] = future
        return self._pages[page]

    def _prefetch(self) -> None:
        """Render the neighbouring pages in the background.

        Called after each draw, so the worker does not compete with the
        draw of a new page for the interpreter.
        """
        if self._executor is None:
            return
        for page in (self.page + 1, self.page - 1):
            if 0 <= page < self.n_pages:
                self._page_future(page)

    def show_page(self, page: int) -> None:
        """Show a page (rendering it now unless it was prefetched)."""
        tic = time.perf_counter()
        page = min(max(page, 0), self.n_pages - 1)
        rendered = self._page_future(page).result()
        self.cutouts = rendered.cutouts
        self.page = page
        items = self.page_items(page)
        for i, slot in enumerate(self.slots):
            if i < len(items):
                slot.ax.set_visible(True)
                index = i + page * self.per_page
                self._draw_cutout(
                    slot, index, items[i], self.cutouts[i], rendered.rgba[i]
                )
                self._decorate(i)
            else:
                slot.ax.set_visible(False)
        for stale in [p for p in self._pages if abs(p - page) > 2]:
            self._pages.pop(stale).cancel()
        self._update_status()
        self.last_switch_ms = 1e3 * (time.perf_counter() - tic)
        log.debug("page %d shown in %.0f ms", page + 1, self.last_switch_ms)
        self.fig.canvas.draw_idle()

    def next_page(self) -> None:
        """Go to the next page (unclicked cutouts stay in the queue)."""
        if self.page + 1 >= self.n_pages:
            self._message = "last page"
            self._update_status()
            self.fig.canvas.draw_idle()
            return
        self._message = ""
        self.show_page(self.page + 1)

    def previous_page(self) -> None:
        """Go to the previous page."""
        self._message = ""
        self.show_page(max(self.page - 1, 0))

    # drawing ---------------------------------------------------------------

    def _build_axes(self) -> list[_Slot]:
        axes = self.fig.subplots(
            self.rows,
            self.cols,
            squeeze=False,
            gridspec_kw={
                "left": 0.003,
                "right": 0.997,
                "bottom": 0.003,
                "top": 0.975,
                "wspace": 0.03,
                "hspace": 0.03,
            },
        )
        slots = []
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            image = ax.imshow(
                np.zeros((2, 2, 4), dtype=np.uint8),
                origin="lower",
                aspect="auto",
                interpolation="none",  # Agg scales the RGBA directly
            )
            box = Rectangle((0, 0), 0, 0, fill=False, edgecolor="white", linewidth=0.7)
            ax.add_patch(box)
            (ticks,) = ax.plot([], [], ls="--", color=_TICK_COLOUR, lw=0.9)
            (marker,) = ax.plot([], [], ls="", ms=8, mew=1.6, mfc="none")
            title = ax.text(
                0.02,
                0.97,
                "",
                transform=ax.transAxes,
                va="top",
                fontsize=7,
                color="white",
            )
            slots.append(_Slot(ax, image, box, ticks, marker, title))
        return slots

    def _draw_cutout(
        self,
        slot: _Slot,
        index: int,
        item: QueueItem,
        cut: Cutout | None,
        rgba: npt.NDArray[np.uint8] | None,
    ) -> None:
        if cut is None or rgba is None:
            slot.image.set_data(np.zeros((2, 2, 4), dtype=np.uint8))
            extent = (item.t_start_s, item.t_end_s, item.f_low_hz, item.f_high_hz)
            slot.ticks.set_data([], [])
            title = f"{index + 1}  render error"
        else:
            slot.image.set_data(rgba)
            extent = cut.extent
            t0, t1 = extent[0], extent[1]
            length = _TICK_FRACTION * (t1 - t0)
            xs: list[float] = []
            ys: list[float] = []
            for _, f in cut.baselines:
                xs += [t0, t0 + length, math.nan, t1 - length, t1, math.nan]
                ys += [f, f, math.nan, f, f, math.nan]
            slot.ticks.set_data(xs, ys)
            n = len(cut.baselines)
            spread = (
                f"ΔF {item.delta_f_hz:.0f} Hz"
                if item.delta_f_hz is not None
                else f"{n} track{'s' * (n != 1)}"
            )
            title = f"{index + 1}  {spread}"
        slot.image.set_extent(extent)
        slot.ax.set_xlim(extent[0], extent[1])
        slot.ax.set_ylim(extent[2], extent[3])
        slot.box.set_bounds(
            item.t_start_s,
            item.f_low_hz,
            item.t_end_s - item.t_start_s,
            item.f_high_hz - item.f_low_hz,
        )
        slot.title.set_text(title)

    def _decorate(self, i: int) -> None:
        """Frame and click marker of slot ``i`` from the stored label."""
        slot = self.slots[i]
        decision = self.store.current(self.page_items(self.page)[i].item_id)
        colour = FRAME_COLOURS[decision.label] if decision else "0.6"
        width = 3.0 if decision else 0.6
        for spine in slot.ax.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(width)
        if (
            decision
            and decision.clicked_hz is not None
            and decision.clicked_t_s is not None
        ):
            slot.marker.set_data([decision.clicked_t_s], [decision.clicked_hz])
            slot.marker.set_marker("o" if decision.emitter_id is not None else "x")
            slot.marker.set_markeredgecolor(colour)
        else:
            slot.marker.set_data([], [])

    def progress(self) -> tuple[int, dict[str, tuple[int, int]]]:
        """Labelled items in the queue, overall and ``{bin: (labelled, total)}``."""
        per_bin: dict[str, tuple[int, int]] = dict.fromkeys(self.queue.bins, (0, 0))
        done = 0
        for item in self.queue.items:
            labelled = self.store.current(item.item_id) is not None
            done += labelled
            n_done, n_total = per_bin[item.df_bin]
            per_bin[item.df_bin] = (n_done + labelled, n_total + 1)
        return done, per_bin

    def _update_status(self) -> None:
        done, per_bin = self.progress()
        bins = "  ".join(f"{b}:{d}/{n}" for b, (d, n) in per_bin.items() if n)
        text = (
            f"page {self.page + 1}/{self.n_pages}   labelled {done}/{len(self.queue)}"
            f"   [ΔF Hz {bins}]   {self.store.labeler}   h: help"
        )
        if self._message:
            text += f"   -- {self._message}"
        self.status.set_text(text)
        manager = self.fig.canvas.manager
        if manager is not None:
            manager.set_window_title(
                f"label-emitters: {done}/{len(self.queue)}  [{bins}]"
            )

    # events ----------------------------------------------------------------

    def on_click(self, event: Any) -> None:
        """Label the cutout under the mouse (left / right / middle button)."""
        if event.inaxes is None or getattr(event, "dblclick", False):
            return
        slot = next((i for i, s in enumerate(self.slots) if s.ax is event.inaxes), None)
        items = self.page_items(self.page)
        if slot is None or slot >= len(items) or event.button is None:
            return
        label = _BUTTON_LABELS.get(int(event.button))
        if label is None:
            return
        cut = self.cutouts[slot]
        baselines = cut.baselines if cut is not None else ()
        emitter, clicked_hz, clicked_t_s = None, None, None
        if label == "chirp" and event.xdata is not None and event.ydata is not None:
            clicked_t_s, clicked_hz = float(event.xdata), float(event.ydata)
            emitter = nearest_track(baselines, clicked_hz, self.tolerance_hz)
        self.store.record(
            items[slot],
            self.queue.queue_id,
            label,
            emitter_id=emitter,
            clicked_hz=clicked_hz,
            clicked_t_s=clicked_t_s,
            candidates=baselines,
        )
        self._message = ""
        self._decorate(slot)
        self._update_status()
        self.fig.canvas.draw_idle()

    def on_key(self, event: Any) -> None:
        """Page, undo, help and quit keys."""
        key = event.key
        if key in (" ", "space", "enter", "n"):
            self.next_page()
        elif key in ("p", "b"):
            self.previous_page()
        elif key == "u":
            self.undo()
        elif key == "h":
            self.help.set_visible(not self.help.get_visible())
            self.fig.canvas.draw_idle()
        elif key == "q":
            import matplotlib.pyplot as plt

            plt.close(self.fig)

    def undo(self) -> None:
        """Revert the last click and show its page."""
        decision = self.store.undo()
        if decision is None:
            self._message = "nothing to undo"
        else:
            index = self._index.get(decision.item_id)
            self._message = f"undid {decision.label} on {index + 1 if index is not None else decision.item_id}"
            if index is not None and index // self.per_page != self.page:
                self.show_page(index // self.per_page)
                return
            if index is not None:
                self._decorate(index - self.page * self.per_page)
        self._update_status()
        self.fig.canvas.draw_idle()

    def shutdown(self) -> None:
        """Stop background rendering."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)


def run(
    queue: LabelQueue,
    trial_dirs: Mapping[str, Path],
    store: LabelStore,
    *,
    config: CutoutConfig | None = None,
    grid: tuple[int, int] = (4, 6),
    tolerance_hz: float = 15.0,
) -> None:
    """Open the labelling window and block until it is closed.

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
        msg = f"matplotlib backend {backend!r} is not interactive; is a display available?"
        raise RuntimeError(msg)
    # no toolbar (no pan/zoom modes to fall into); unhinted text draws faster
    with plt.rc_context({"toolbar": "None", "text.hinting": "no_hinting"}):
        labeller = Labeller(
            queue,
            trial_dirs,
            store,
            config=config,
            grid=grid,
            tolerance_hz=tolerance_hz,
        )
        plt.show()
    labeller.shutdown()
