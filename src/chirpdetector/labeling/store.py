"""Label persistence: an append-only event log and derived audian CSV files.

Every click is appended to ``<out>/events.jsonl`` and fsynced before the
call returns, so killing the labeller loses nothing. Undo appends an
``undo`` event naming the click it reverts; the log is never edited
(except that a line cut short by a crash is dropped on start). The
current label of an item is the latest click that has not been undone, per
labeller.

A *skip* (the labeller moved on without deciding) is logged as a ``skip``
event, can be undone like a click, and never appears in the label files.

From the log, one audian label file per trial (recording) is derived,
``<out>/labels/<trial>.csv`` (rewritten atomically after every event and
rebuilt on start), one row per labelled item and labeller. Columns: the
audian columns with the box, then :data:`LABEL_COLUMNS`:

``label``
    ``chirp``, ``not_chirp`` or ``chirp_emitter_unclear``.
``emitter_id``
    Track id of the emitter (Int64; NA unless a track was picked).
``clicked_hz``, ``clicked_t_s``
    Where the labeller clicked (NA for keyboard picks).
``labeler``, ``labeled_at`` (UTC, ISO 8601), ``tool_version``, ``queue_id``,
``item_id``, ``det_index``, ``df_bin``
    Provenance (``det_index``: row of the v1 detection or candidate).

then :data:`EXTRA_COLUMNS`:

``emitter_baseline_hz``
    The emitter track's frequency at the box start (the baseline before
    the chirp), interpolated from the track, not from the click.
``click_offset_hz``
    ``clicked_hz`` minus the emitter track's frequency at ``clicked_t_s``
    (how far from the track the click landed).
``via``
    ``mouse`` or ``key`` (how the label was given).
``frontend``
    ``mpl``, ``mpl-grid`` or ``browser``.
``window_s``
    Spectrogram window shown when labelling.
``prediction_shown``
    Whether the model's predicted emitter was visible (boolean; NA for
    older labels).
``n_candidates``, ``delta_f_hz``
    Live tracks at the box centre and the frequency difference of the two
    closest of them.
``recording_dir``, ``tracks_dir``
    Where the recording and its tracks were read from.
``elapsed_s``
    Seconds from showing the item to this decision (NA for older labels).
``blind``
    Whether the session was blind (``--blind``: no box, no model output).
``candidate_id``, ``candidate_score``, ``candidate_emitter_hz``,
``candidate_track_id``, ``candidate_source``
    The candidates-file columns (NA for v1 detections), followed by any
    extra ``candidate_<column>``.

One labelling session per output directory at a time (enforced with a lock
on the log).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from gridtools.io import write_labels

from chirpdetector.labeling.queue import CandidateValue, QueueItem

__all__ = [
    "EXTRA_COLUMNS",
    "FRONTENDS",
    "LABELS",
    "LABEL_COLUMNS",
    "SKIP",
    "Decision",
    "LabelStore",
    "decision_table",
]

log = logging.getLogger(__name__)

LABELS = ("chirp", "not_chirp", "chirp_emitter_unclear")
"""Possible labels: left, right and middle click."""

LABEL_COLUMNS = (
    "label",
    "emitter_id",
    "clicked_hz",
    "clicked_t_s",
    "labeler",
    "labeled_at",
    "tool_version",
    "queue_id",
    "item_id",
    "det_index",
    "df_bin",
)
"""Columns written after the audian columns, in order (since the first version)."""

EXTRA_COLUMNS = (
    "emitter_baseline_hz",
    "click_offset_hz",
    "via",
    "frontend",
    "window_s",
    "prediction_shown",
    "n_candidates",
    "delta_f_hz",
    "recording_dir",
    "tracks_dir",
    "elapsed_s",
    "blind",
    "candidate_id",
    "candidate_score",
    "candidate_emitter_hz",
    "candidate_track_id",
    "candidate_source",
)
"""Columns written after :data:`LABEL_COLUMNS` (extra ``candidate_*`` follow)."""

SKIP = "skip"
"""Pseudo-label of a skip event (never written to the label files)."""

FRONTENDS = ("mpl", "mpl-grid", "browser")
"""Values of ``frontend``."""

_CANDIDATE_DTYPES = {
    "candidate_score": "Float64",
    "candidate_emitter_hz": "Float64",
    "candidate_track_id": "Int64",
}


@dataclass(frozen=True)
class Decision:
    """One click (or skip), as stored in the event log.

    Fields after ``candidates`` were added with the single-candidate
    frontends; events written before have their defaults.
    """

    seq: int
    labeler: str
    labeled_at: str
    tool_version: str
    queue_id: str
    item_id: str
    trial: str
    det_index: int
    df_bin: str
    box: tuple[float, float, float, float]
    label: str
    emitter_id: int | None
    clicked_hz: float | None
    clicked_t_s: float | None
    candidates: tuple[tuple[int, float], ...]
    emitter_baseline_hz: float | None = None
    click_offset_hz: float | None = None
    via: str = ""
    frontend: str = ""
    window_s: float | None = None
    prediction_shown: bool | None = None
    n_candidates: int | None = None
    delta_f_hz: float | None = None
    recording_dir: str = ""
    tracks_dir: str = ""
    candidate: tuple[tuple[str, CandidateValue], ...] = ()
    elapsed_s: float | None = None
    blind: bool | None = None

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> Decision:
        """Parse a ``click`` or ``skip`` event."""
        values = {f.name: event[f.name] for f in fields(cls) if f.name in event}
        values["box"] = tuple(values["box"])
        values["candidates"] = tuple(
            (int(i), float(f)) for i, f in values["candidates"]
        )
        values["candidate"] = tuple((str(k), v) for k, v in values.get("candidate", ()))
        return cls(**values)

    @property
    def is_skip(self) -> bool:
        """True for a skip (no label)."""
        return self.label == SKIP


class LabelStore:
    """Event log and label files of one output directory.

    Parameters
    ----------
    out_dir
        Output directory (created if needed).
    labeler
        Name recorded with every click; undo and :meth:`current` only see
        this labeller's clicks.
    tool_version
        Version recorded with every click.

    Raises
    ------
    RuntimeError
        If another session holds the directory.
    ValueError
        If the log is corrupt (other than a torn last line).
    """

    def __init__(self, out_dir: str | Path, labeler: str, tool_version: str) -> None:
        if not labeler:
            msg = "labeler must not be empty"
            raise ValueError(msg)
        self.out_dir = Path(out_dir)
        self.labeler = labeler
        self.tool_version = tool_version
        self.labels_dir = self.out_dir / "labels"
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.out_dir / "events.jsonl"
        self._file = self.events_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._file.close()
            msg = f"another labelling session is using {self.out_dir}"
            raise RuntimeError(msg) from None
        self._stacks: dict[tuple[str, str], list[Decision]] = {}
        self._history: dict[str, list[Decision]] = {}
        self._next_seq = 0
        self._replay()
        for trial in {d.trial for stack in self._stacks.values() for d in stack}:
            self._write_csv(trial)

    def close(self) -> None:
        """Release the directory."""
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> LabelStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _replay(self) -> None:
        self._file.seek(0)
        text = self._file.read()
        complete, _, tail = text.rpartition("\n")
        if tail:  # a write cut short by a crash: never a whole event
            log.warning("%s: dropping incomplete last line %r", self.events_path, tail)
            self._file.truncate(len(complete.encode("utf-8")) + (1 if complete else 0))
        for n, line in enumerate(complete.split("\n"), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                msg = f"{self.events_path}:{n}: corrupt event"
                raise ValueError(msg) from None
            self._apply(event)
            self._next_seq = max(self._next_seq, int(event["seq"]) + 1)

    def _apply(self, event: dict[str, Any]) -> Decision | None:
        if event["type"] in ("click", SKIP):
            decision = Decision.from_event(event)
            self._stacks.setdefault((decision.labeler, decision.item_id), []).append(
                decision
            )
            self._history.setdefault(decision.labeler, []).append(decision)
            return decision
        if event["type"] == "undo":
            target = int(event["undoes"])
            history = self._history.get(event["labeler"], [])
            for i in range(len(history) - 1, -1, -1):
                if history[i].seq == target:
                    decision = history.pop(i)
                    self._stacks[(decision.labeler, decision.item_id)].remove(decision)
                    return decision
            log.warning("undo of unknown click %d ignored", target)
            return None
        msg = f"unknown event type {event['type']!r}"
        raise ValueError(msg)

    def _append(self, event: dict[str, Any]) -> None:
        self._file.write(json.dumps(event) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())
        self._next_seq += 1

    def record(
        self,
        item: QueueItem,
        queue_id: str,
        label: str,
        *,
        emitter_id: int | None = None,
        clicked_hz: float | None = None,
        clicked_t_s: float | None = None,
        candidates: Sequence[tuple[int, float]] = (),
        emitter_baseline_hz: float | None = None,
        click_offset_hz: float | None = None,
        via: str = "",
        frontend: str = "",
        window_s: float | None = None,
        prediction_shown: bool | None = None,
        recording_dir: str | Path = "",
        tracks_dir: str | Path = "",
        elapsed_s: float | None = None,
        blind: bool | None = None,
    ) -> Decision:
        """Store a click on ``item`` (overrides earlier clicks on it).

        Parameters
        ----------
        item
            The labelled detection.
        queue_id
            Queue the item was shown from.
        label
            One of :data:`LABELS`, or :data:`SKIP`.
        emitter_id
            Track id of the emitter, if known.
        clicked_hz, clicked_t_s
            Where the labeller clicked.
        candidates
            ``(track_id, baseline_hz)`` of the live tracks shown.
        emitter_baseline_hz
            The emitter track's frequency at the box start.
        click_offset_hz
            Click frequency minus the emitter track's frequency at the
            click time.
        via
            ``mouse`` or ``key``.
        frontend
            One of :data:`FRONTENDS`.
        window_s
            Spectrogram window shown.
        prediction_shown
            Whether the model's predicted emitter was visible.
        recording_dir, tracks_dir
            Where the item's recording and tracks were read from.
        elapsed_s
            Seconds from showing the item to this decision.
        blind
            Whether the session was blind.

        Returns
        -------
        The stored decision.
        """
        if label not in (*LABELS, SKIP):
            msg = f"label must be one of {LABELS} or {SKIP!r}, got {label!r}"
            raise ValueError(msg)
        if frontend and frontend not in FRONTENDS:
            msg = f"frontend must be one of {FRONTENDS}, got {frontend!r}"
            raise ValueError(msg)

        def opt(value: float | None) -> float | None:
            return None if value is None else float(value)

        decision = Decision(
            seq=self._next_seq,
            labeler=self.labeler,
            labeled_at=datetime.now(UTC).isoformat(timespec="milliseconds"),
            tool_version=self.tool_version,
            queue_id=queue_id,
            item_id=item.item_id,
            trial=item.trial,
            det_index=item.det_index,
            df_bin=item.df_bin,
            box=item.box,
            label=label,
            emitter_id=None if emitter_id is None else int(emitter_id),
            clicked_hz=opt(clicked_hz),
            clicked_t_s=opt(clicked_t_s),
            candidates=tuple((int(i), float(f)) for i, f in candidates),
            emitter_baseline_hz=opt(emitter_baseline_hz),
            click_offset_hz=opt(click_offset_hz),
            via=via,
            frontend=frontend,
            window_s=opt(window_s),
            prediction_shown=None
            if prediction_shown is None
            else bool(prediction_shown),
            n_candidates=int(item.n_candidates),
            delta_f_hz=opt(item.delta_f_hz),
            recording_dir=str(recording_dir),
            tracks_dir=str(tracks_dir),
            candidate=item.candidate,
            elapsed_s=opt(elapsed_s),
            blind=None if blind is None else bool(blind),
        )
        event_type = SKIP if label == SKIP else "click"
        self._append({"type": event_type, **asdict(decision)})
        self._apply({"type": event_type, **asdict(decision)})
        if label != SKIP:
            self._write_csv(item.trial)
        return decision

    def undo(self) -> Decision | None:
        """Revert this labeller's latest click that is not yet undone.

        Returns
        -------
        The reverted decision, or None if there is nothing to undo.
        """
        history = self._history.get(self.labeler, [])
        if not history:
            return None
        event = {
            "type": "undo",
            "seq": self._next_seq,
            "labeler": self.labeler,
            "labeled_at": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "tool_version": self.tool_version,
            "undoes": history[-1].seq,
        }
        self._append(event)
        decision = self._apply(event)
        if decision is not None and not decision.is_skip:
            self._write_csv(decision.trial)
        return decision

    def current(self, item_id: str) -> Decision | None:
        """This labeller's current label on an item (None if open or skipped)."""
        stack = self._stacks.get((self.labeler, item_id))
        return stack[-1] if stack and not stack[-1].is_skip else None

    def skipped(self, item_id: str) -> bool:
        """True if this labeller's latest action on the item was a skip."""
        stack = self._stacks.get((self.labeler, item_id))
        return bool(stack) and stack[-1].is_skip

    def latest(self, item_id: str) -> Decision | None:
        """Latest decision on an item by any labeller (skips included)."""
        return self.latest_by_item().get(item_id)

    def latest_by_item(self) -> dict[str, Decision]:
        """Latest decision on every item by any labeller (skips included)."""
        out: dict[str, Decision] = {}
        for (_, item_id), stack in self._stacks.items():
            if stack and (item_id not in out or stack[-1].seq > out[item_id].seq):
                out[item_id] = stack[-1]
        return out

    def decisions(self) -> list[Decision]:
        """Every current label of every labeller (no skips), in log order."""
        return sorted(
            (s[-1] for s in self._stacks.values() if s and not s[-1].is_skip),
            key=lambda d: d.seq,
        )

    def _write_csv(self, trial: str) -> None:
        rows = sorted(
            (
                stack[-1]
                for stack in self._stacks.values()
                if stack and stack[-1].trial == trial and not stack[-1].is_skip
            ),
            key=lambda d: (d.box[0], d.item_id, d.labeler),
        )
        write_labels(decision_table(rows), self.labels_dir / f"{trial}.csv")


def decision_table(rows: Sequence[Decision]) -> pd.DataFrame:
    """Label-file table of decisions (audian columns first)."""
    n = len(rows)
    candidate = [dict(d.candidate) for d in rows]
    extra_keys: list[str] = []
    for c in candidate:
        for k in c:
            if k not in EXTRA_COLUMNS and k not in extra_keys:
                extra_keys.append(k)

    def floats(values: Sequence[float | None]) -> Any:
        return pd.array(list(values), dtype="Float64")

    table: dict[str, Any] = {
        "category": ["not_chirp" if d.label == "not_chirp" else "chirp" for d in rows],
        "kind": ["span"] * n,
        "channel": pd.array([pd.NA] * n, dtype="Int64"),
        "t_start_s": [d.box[0] for d in rows],
        "t_end_s": [d.box[1] for d in rows],
        "f_low_hz": [d.box[2] for d in rows],
        "f_high_hz": [d.box[3] for d in rows],
        "note": [""] * n,
        "label": [d.label for d in rows],
        "emitter_id": pd.array([d.emitter_id for d in rows], dtype="Int64"),
        "clicked_hz": floats([d.clicked_hz for d in rows]),
        "clicked_t_s": floats([d.clicked_t_s for d in rows]),
        "labeler": [d.labeler for d in rows],
        "labeled_at": [d.labeled_at for d in rows],
        "tool_version": [d.tool_version for d in rows],
        "queue_id": [d.queue_id for d in rows],
        "item_id": [d.item_id for d in rows],
        "det_index": pd.array([d.det_index for d in rows], dtype="Int64"),
        "df_bin": [d.df_bin for d in rows],
        "emitter_baseline_hz": floats([d.emitter_baseline_hz for d in rows]),
        "click_offset_hz": floats([d.click_offset_hz for d in rows]),
        "via": [d.via for d in rows],
        "frontend": [d.frontend for d in rows],
        "window_s": floats([d.window_s for d in rows]),
        "prediction_shown": pd.array(
            [d.prediction_shown for d in rows], dtype="boolean"
        ),
        "n_candidates": pd.array([d.n_candidates for d in rows], dtype="Int64"),
        "delta_f_hz": floats([d.delta_f_hz for d in rows]),
        "recording_dir": [d.recording_dir for d in rows],
        "tracks_dir": [d.tracks_dir for d in rows],
        "elapsed_s": floats([d.elapsed_s for d in rows]),
        "blind": pd.array([d.blind for d in rows], dtype="boolean"),
    }
    for key in (*EXTRA_COLUMNS[EXTRA_COLUMNS.index("candidate_id") :], *extra_keys):
        values = [c.get(key) for c in candidate]
        dtype = _CANDIDATE_DTYPES.get(key)
        if dtype is not None:
            table[key] = pd.array(values, dtype=dtype)
        elif key == "candidate_id" or key == "candidate_source":
            table[key] = ["" if v is None else str(v) for v in values]
        else:
            table[key] = pd.array(values, dtype=object)
    return pd.DataFrame(table)
