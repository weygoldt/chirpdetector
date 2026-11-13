"""A labelling session: queue position, actions, undo, rendering cache.

The matplotlib window and the browser server both drive a
:class:`LabelSession`, so the queue order, the snapping, the stored fields
and undo are identical, and a session started in one frontend resumes in
the other (same queue file, same output directory).

Actions on an item (every one is saved at once):

- ``click(t, f)``: chirp; the emitter is the live track snapped to the
  click (:func:`chirpdetector.labeling.view.snap`); rejected (nothing
  stored) if no track is within the tolerance;
- ``pick(key)``: chirp; the emitter is the track numbered ``key``;
- ``mark(label)``: ``not_chirp`` or ``chirp_emitter_unclear``;
- ``skip()``: no decision now; the item comes back after all open items;
- ``undo()``: reverts this labeller's latest action (any item).

After an action the frontends move to :meth:`LabelSession.next_index`: the
next open item in queue order (wrapping around), then skipped items.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gridtools.io import Recording, Tracks, load_tracks, open_recording

from chirpdetector.labeling.queue import LabelQueue, QueueItem
from chirpdetector.labeling.store import LABELS, SKIP, Decision, LabelStore
from chirpdetector.labeling.view import (
    CandidateView,
    TrackView,
    ViewConfig,
    render_view,
    snap,
    track_views,
)

__all__ = [
    "ALT_WINDOW_S",
    "ActionResult",
    "LabelSession",
    "Progress",
    "TrialSources",
]

log = logging.getLogger(__name__)

ALT_WINDOW_S = (0.2048, 0.0512)
"""The two windows the ``w`` key toggles between (if the start window is
one of them; otherwise the start window and 0.2048 s)."""

_CACHE_SIZE = 24


class TrialSources:
    """Opened recordings and tracks by trial name (thread-safe, lazy).

    Parameters
    ----------
    paths
        ``{trial: (recording_dir, tracks_dir)}``.
    """

    def __init__(self, paths: Mapping[str, tuple[Path, Path]]) -> None:
        self.paths = {k: (Path(r), Path(t)) for k, (r, t) in paths.items()}
        self._lock = threading.Lock()
        self._recordings: dict[str, Recording] = {}
        self._tracks: dict[str, Tracks] = {}

    @classmethod
    def from_queue(
        cls, queue: LabelQueue, overrides: Mapping[str, Path] | None = None
    ) -> TrialSources:
        """Paths of the queue's trials; ``overrides`` (trial -> directory
        holding recording and tracks) take precedence."""
        paths = {t.name: (t.recording_dir, t.tracks_dir) for t in queue.trials}
        for name, directory in (overrides or {}).items():
            paths[name] = (Path(directory), Path(directory))
        return cls(paths)

    def recording(self, trial: str) -> Recording:
        """The trial's recording (opened on first use)."""
        with self._lock:
            if trial not in self._recordings:
                self._recordings[trial] = open_recording(self.paths[trial][0])
            return self._recordings[trial]

    def tracks(self, trial: str) -> Tracks:
        """The trial's tracks (loaded on first use)."""
        with self._lock:
            if trial not in self._tracks:
                self._tracks[trial] = load_tracks(self.paths[trial][1])
            return self._tracks[trial]


@dataclass(frozen=True)
class ActionResult:
    """Outcome of an action.

    Parameters
    ----------
    ok
        False if the action was rejected (nothing was stored).
    message
        What happened, for the status line.
    index
        The item acted on (for undo: the item whose decision was reverted).
    decision
        The stored (or, for undo, reverted) decision.
    next_index
        Where to go now (None: nothing left to label).
    """

    ok: bool
    message: str
    index: int | None = None
    decision: Decision | None = None
    next_index: int | None = None


@dataclass(frozen=True)
class Progress:
    """Counts over the queue for the status line."""

    total: int
    labelled: int
    skipped: int
    per_bin: dict[str, tuple[int, int]]

    @property
    def open(self) -> int:
        """Items neither labelled nor skipped."""
        return self.total - self.labelled - self.skipped


def describe(
    decision: Decision, view_tracks: tuple[TrackView, ...] = (), *, ids: bool = True
) -> str:
    """Short text of a decision, e.g. ``chirp -> 2 (id 7)`` (``chirp -> 2``
    without ``ids``, as blind sessions show it)."""
    if decision.label == SKIP:
        return "skipped"
    if decision.label == "chirp" and decision.emitter_id is not None:
        key = next(
            (t.key for t in view_tracks if t.track_id == decision.emitter_id), None
        )
        if not ids:
            return f"chirp -> {key}" if key is not None else "chirp"
        number = f"{key} " if key is not None else ""
        return f"chirp -> {number}(id {decision.emitter_id})"
    return {
        "chirp": "chirp, no emitter",
        "not_chirp": "not a chirp",
        "chirp_emitter_unclear": "chirp, emitter unclear",
    }[decision.label]


class LabelSession:
    """Queue, store and rendering shared by the frontends.

    Parameters
    ----------
    queue
        Items to label, in order.
    store
        Where decisions are saved.
    config
        View rendering.
    tolerance_hz
        Largest distance of a click from a track for it to snap.
    frontend
        Stored with every decision (``mpl`` or ``browser``).
    sources
        Recordings and tracks; default: the queue's trial paths.
    background
        Render prefetched views in a worker thread.
    blind
        Blind labelling (``--blind``): the frontends draw no box (only a
        time marker at its centre), never show or send the candidate's
        score, source or predicted emitter, and show tracks by their
        on-screen number only (no track ids, no track gaps: in simulated
        scenes they tell simulated from real background fish); every
        decision records ``blind`` and ``prediction_shown`` False.

    Raises
    ------
    ValueError
        If an item id of the queue was labelled before (in this output
        directory) with a different box.
    """

    def __init__(
        self,
        queue: LabelQueue,
        store: LabelStore,
        *,
        config: ViewConfig | None = None,
        tolerance_hz: float = 15.0,
        frontend: str = "mpl",
        sources: TrialSources | None = None,
        background: bool = True,
        blind: bool = False,
    ) -> None:
        if not tolerance_hz > 0:
            msg = f"tolerance_hz must be > 0, got {tolerance_hz}"
            raise ValueError(msg)
        self.queue = queue
        self.store = store
        self.config = config or ViewConfig()
        self.tolerance_hz = float(tolerance_hz)
        self.frontend = frontend
        self.blind = bool(blind)
        self.sources = sources or TrialSources.from_queue(queue)
        start = self.config.window_s
        other = ALT_WINDOW_S[1] if start == ALT_WINDOW_S[0] else ALT_WINDOW_S[0]
        self.windows = (start, other)
        self._lock = threading.RLock()
        self._cache: OrderedDict[tuple[int, float], Future[CandidateView]] = (
            OrderedDict()
        )
        self._tracks: dict[int, tuple[TrackView, ...]] = {}
        self._executor = ThreadPoolExecutor(2) if background else None
        missing = {i.trial for i in queue.items} - set(self.sources.paths)
        if missing:
            msg = f"no recording given for trials {sorted(missing)}"
            raise ValueError(msg)
        self._check_boxes()

    # state -----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.queue)

    def item(self, index: int) -> QueueItem:
        """The item at a queue position."""
        return self.queue.items[index]

    def index_of(self, item_id: str) -> int | None:
        """Queue position of an item id."""
        for i, item in enumerate(self.queue.items):
            if item.item_id == item_id:
                return i
        return None

    def _check_boxes(self) -> None:
        latest_by_item = self.store.latest_by_item()
        for item in self.queue.items:
            latest = latest_by_item.get(item.item_id)
            if latest is not None and not np.allclose(
                latest.box, item.box, rtol=0, atol=1e-6
            ):
                msg = (
                    f"{item.item_id} was labelled before with the box {latest.box}, "
                    f"but the queue has {item.box}: candidate ids must identify "
                    "their box (use other candidate_id values or another --out)"
                )
                raise ValueError(msg)

    def decision(self, index: int) -> Decision | None:
        """Current label of an item (None if open or skipped)."""
        with self._lock:
            return self.store.current(self.item(index).item_id)

    def status(self, index: int) -> str:
        """``labelled``, ``skipped`` or ``open``."""
        with self._lock:
            item_id = self.item(index).item_id
            if self.store.current(item_id) is not None:
                return "labelled"
            return "skipped" if self.store.skipped(item_id) else "open"

    def next_index(self, after: int | None = None) -> int | None:
        """Next open item after ``after`` (wrapping), then next skipped one.

        ``after`` itself is considered last. None if everything is labelled.
        """
        n = len(self.queue)
        if n == 0:
            return None
        start = -1 if after is None else after
        order = [(start + k) % n for k in range(1, n + 1)]
        with self._lock:
            states = {i: self.status(i) for i in order}
        for wanted in ("open", "skipped"):
            for i in order:
                if states[i] == wanted:
                    return i
        return None

    def first_index(self) -> int:
        """Where to start: the first open item (or skipped, or 0)."""
        found = self.next_index(None)
        return 0 if found is None else found

    def progress(self) -> Progress:
        """Counts over the queue."""
        per_bin: dict[str, tuple[int, int]] = dict.fromkeys(self.queue.bins, (0, 0))
        labelled = skipped = 0
        with self._lock:
            for i, item in enumerate(self.queue.items):
                state = self.status(i)
                labelled += state == "labelled"
                skipped += state == "skipped"
                done, total = per_bin.get(item.df_bin, (0, 0))
                per_bin[item.df_bin] = (done + (state == "labelled"), total + 1)
        return Progress(len(self.queue), labelled, skipped, per_bin)

    def toggle_window(self, window_s: float) -> float:
        """The other of the two windows."""
        return self.windows[1] if window_s == self.windows[0] else self.windows[0]

    # rendering -------------------------------------------------------------

    def tracks(self, index: int) -> tuple[TrackView, ...]:
        """Numbered live tracks of an item (cheap; no spectrogram)."""
        with self._lock:
            if index not in self._tracks:
                item = self.item(index)
                self._tracks[index] = track_views(
                    self.sources.tracks(item.trial), item, self.config
                )
            return self._tracks[index]

    def _render(self, index: int, window_s: float) -> CandidateView:
        item = self.item(index)
        return render_view(
            self.sources.recording(item.trial),
            self.sources.tracks(item.trial),
            item,
            self.config,
            window_s,
        )

    def _future(self, index: int, window_s: float) -> Future[CandidateView]:
        key = (index, float(window_s))
        with self._lock:
            future = self._cache.get(key)
            if future is None:
                if self._executor is not None:
                    future = self._executor.submit(self._render, index, window_s)
                else:
                    future = Future()
                    try:
                        future.set_result(self._render(index, window_s))
                    except Exception as err:
                        future.set_exception(err)
                self._cache[key] = future
                while len(self._cache) > _CACHE_SIZE:
                    _, old = self._cache.popitem(last=False)
                    old.cancel()
            else:
                self._cache.move_to_end(key)
            return future

    def view(self, index: int, window_s: float | None = None) -> CandidateView:
        """Rendered view of an item (from the cache if prefetched).

        Raises
        ------
        Exception
            Whatever rendering raised (e.g. an unreadable recording).
        """
        window = self.config.window_s if window_s is None else float(window_s)
        return self._future(index, window).result()

    def prefetch(self, index: int | None, window_s: float | None = None) -> None:
        """Start rendering an item in the background."""
        if index is None or self._executor is None:
            return
        window = self.config.window_s if window_s is None else float(window_s)
        self._future(index, window)

    def shutdown(self) -> None:
        """Stop background rendering."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    # actions ---------------------------------------------------------------

    def _record(
        self,
        index: int,
        label: str,
        window_s: float | None,
        *,
        emitter_id: int | None = None,
        clicked_hz: float | None = None,
        clicked_t_s: float | None = None,
        emitter_baseline_hz: float | None = None,
        click_offset_hz: float | None = None,
        via: str = "",
        prediction_shown: bool | None = None,
        elapsed_s: float | None = None,
    ) -> ActionResult:
        item = self.item(index)
        if self.blind:
            prediction_shown = False
        recording_dir, tracks_dir = self.sources.paths[item.trial]
        if window_s is None and label != SKIP:
            window_s = self.config.window_s
        with self._lock:
            decision = self.store.record(
                item,
                self.queue.queue_id,
                label,
                emitter_id=emitter_id,
                clicked_hz=clicked_hz,
                clicked_t_s=clicked_t_s,
                candidates=[(t.track_id, t.baseline_hz) for t in self.tracks(index)],
                emitter_baseline_hz=emitter_baseline_hz,
                click_offset_hz=click_offset_hz,
                via=via,
                frontend=self.frontend,
                window_s=window_s,
                prediction_shown=prediction_shown,
                recording_dir=recording_dir,
                tracks_dir=tracks_dir,
                elapsed_s=elapsed_s,
                blind=self.blind,
            )
            next_index = self.next_index(index)
        text = describe(decision, self.tracks(index), ids=not self.blind)
        return ActionResult(
            ok=True,
            message=f"#{index + 1}: {text}",
            index=index,
            decision=decision,
            next_index=next_index,
        )

    def click(
        self,
        index: int,
        t_s: float,
        f_hz: float,
        *,
        window_s: float | None = None,
        prediction_shown: bool = False,
        elapsed_s: float | None = None,
    ) -> ActionResult:
        """Chirp from the track snapped to a click at ``(t_s, f_hz)``.

        ``elapsed_s`` (all actions): seconds since the frontend showed the
        item, stored with the decision.
        """
        tracks = self.tracks(index)
        found = snap(tracks, t_s, f_hz, self.tolerance_hz)
        if found is None:
            return ActionResult(
                ok=False,
                message=(
                    f"no track within {self.tolerance_hz:g} Hz of {f_hz:.1f} Hz: "
                    "click on a baseline, or press 0 (unclear) / x (not a chirp)"
                ),
                index=index,
            )
        return self._record(
            index,
            "chirp",
            window_s,
            emitter_id=found.track.track_id,
            clicked_hz=f_hz,
            clicked_t_s=t_s,
            emitter_baseline_hz=found.track.baseline_hz,
            click_offset_hz=found.offset_hz,
            via="mouse",
            prediction_shown=prediction_shown,
            elapsed_s=elapsed_s,
        )

    def pick(
        self,
        index: int,
        key: int,
        *,
        window_s: float | None = None,
        prediction_shown: bool = False,
        elapsed_s: float | None = None,
    ) -> ActionResult:
        """Chirp from the track numbered ``key`` on screen."""
        tracks = self.tracks(index)
        track = next((t for t in tracks if t.key == key), None)
        if track is None:
            return ActionResult(
                ok=False,
                message=f"no track {key} ({len(tracks)} live)",
                index=index,
            )
        return self._record(
            index,
            "chirp",
            window_s,
            emitter_id=track.track_id,
            emitter_baseline_hz=track.baseline_hz,
            via="key",
            prediction_shown=prediction_shown,
            elapsed_s=elapsed_s,
        )

    def mark(
        self,
        index: int,
        label: str,
        *,
        window_s: float | None = None,
        prediction_shown: bool = False,
        via: str = "key",
        elapsed_s: float | None = None,
    ) -> ActionResult:
        """``not_chirp`` or ``chirp_emitter_unclear``."""
        if label not in LABELS or label == "chirp":
            msg = f"mark takes not_chirp or chirp_emitter_unclear, got {label!r}"
            raise ValueError(msg)
        return self._record(
            index,
            label,
            window_s,
            via=via,
            prediction_shown=prediction_shown,
            elapsed_s=elapsed_s,
        )

    def skip(self, index: int, *, elapsed_s: float | None = None) -> ActionResult:
        """Move on without deciding (logged unless the item is labelled)."""
        if self.status(index) == "labelled":
            return ActionResult(
                ok=True,
                message=f"#{index + 1}: kept its label",
                index=index,
                next_index=self.next_index(index),
            )
        return self._record(index, SKIP, None, elapsed_s=elapsed_s)

    def undo(self) -> ActionResult:
        """Revert this labeller's latest action."""
        with self._lock:
            decision = self.store.undo()
        if decision is None:
            return ActionResult(ok=False, message="nothing to undo")
        index = self.index_of(decision.item_id)
        where = f"#{index + 1}" if index is not None else decision.item_id
        tracks = self.tracks(index) if index is not None else ()
        return ActionResult(
            ok=True,
            message=f"undid {describe(decision, tracks, ids=not self.blind)} on {where}",
            index=index,
            decision=decision,
            next_index=index,
        )
