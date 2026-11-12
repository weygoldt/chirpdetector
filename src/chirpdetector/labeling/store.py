"""Label persistence: an append-only event log and derived audian CSV files.

Every click is appended to ``<out>/events.jsonl`` and fsynced before the
call returns, so killing the labeller loses nothing. Undo appends an
``undo`` event naming the click it reverts; the log is never edited
(except that a line cut short by a crash is dropped on start). The
current label of an item is the latest click that has not been undone, per
labeller.

From the log, one audian label file per trial is derived,
``<out>/labels/<trial>.csv`` (rewritten atomically after every event and
rebuilt on start). Columns: the audian columns with the v1 box, then
``label`` (``chirp``, ``not_chirp``, ``chirp_emitter_unclear``),
``emitter_id`` (Int64 track id; NA unless a left click landed within the
tolerance of a live track), ``clicked_hz``, ``clicked_t_s``, ``labeler``,
``labeled_at`` (UTC, ISO 8601), ``tool_version``, ``queue_id``, ``item_id``,
``det_index`` and ``df_bin``.

One labelling session per output directory at a time (enforced with a lock
on the log).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from gridtools.io import write_labels

from chirpdetector.labeling.queue import QueueItem

__all__ = ["LABELS", "LABEL_COLUMNS", "Decision", "LabelStore"]

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
"""Columns written after the audian columns, in order."""


@dataclass(frozen=True)
class Decision:
    """One click, as stored in the event log."""

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

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> Decision:
        """Parse a ``click`` event."""
        fields = {k: event[k] for k in cls.__dataclass_fields__}
        fields["box"] = tuple(fields["box"])
        fields["candidates"] = tuple(
            (int(i), float(f)) for i, f in fields["candidates"]
        )
        return cls(**fields)


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
        if event["type"] == "click":
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
    ) -> Decision:
        """Store a click on ``item`` (overrides earlier clicks on it).

        Parameters
        ----------
        item
            The labelled detection.
        queue_id
            Queue the item was shown from.
        label
            One of :data:`LABELS`.
        emitter_id
            Track id of the emitter, if known.
        clicked_hz, clicked_t_s
            Where the labeller clicked.
        candidates
            ``(track_id, baseline_hz)`` of the live tracks shown.

        Returns
        -------
        The stored decision.
        """
        if label not in LABELS:
            msg = f"label must be one of {LABELS}, got {label!r}"
            raise ValueError(msg)
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
            clicked_hz=None if clicked_hz is None else float(clicked_hz),
            clicked_t_s=None if clicked_t_s is None else float(clicked_t_s),
            candidates=tuple((int(i), float(f)) for i, f in candidates),
        )
        self._append({"type": "click", **asdict(decision)})
        self._apply({"type": "click", **asdict(decision)})
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
        if decision is not None:
            self._write_csv(decision.trial)
        return decision

    def current(self, item_id: str) -> Decision | None:
        """This labeller's current decision on an item."""
        stack = self._stacks.get((self.labeler, item_id))
        return stack[-1] if stack else None

    def _write_csv(self, trial: str) -> None:
        rows = sorted(
            (
                stack[-1]
                for stack in self._stacks.values()
                if stack and stack[-1].trial == trial
            ),
            key=lambda d: (d.box[0], d.item_id, d.labeler),
        )
        df = pd.DataFrame(
            {
                "category": [
                    "not_chirp" if d.label == "not_chirp" else "chirp" for d in rows
                ],
                "kind": ["span"] * len(rows),
                "channel": pd.array([pd.NA] * len(rows), dtype="Int64"),
                "t_start_s": [d.box[0] for d in rows],
                "t_end_s": [d.box[1] for d in rows],
                "f_low_hz": [d.box[2] for d in rows],
                "f_high_hz": [d.box[3] for d in rows],
                "note": [""] * len(rows),
                "label": [d.label for d in rows],
                "emitter_id": pd.array([d.emitter_id for d in rows], dtype="Int64"),
                "clicked_hz": pd.array([d.clicked_hz for d in rows], dtype="Float64"),
                "clicked_t_s": pd.array([d.clicked_t_s for d in rows], dtype="Float64"),
                "labeler": [d.labeler for d in rows],
                "labeled_at": [d.labeled_at for d in rows],
                "tool_version": [d.tool_version for d in rows],
                "queue_id": [d.queue_id for d in rows],
                "item_id": [d.item_id for d in rows],
                "det_index": pd.array([d.det_index for d in rows], dtype="Int64"),
                "df_bin": [d.df_bin for d in rows],
            }
        )
        write_labels(df, self.labels_dir / f"{trial}.csv")
