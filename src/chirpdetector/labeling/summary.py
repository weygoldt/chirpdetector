"""Counts of labels: by decision, by candidate source and by ΔF bin.

Reads the label files of an output directory (``<out>/labels/*.csv``), so
it works while a labelling session is running. Decisions are
``chirp`` (emitter known), ``chirp_no_emitter`` (a chirp label without a
track, from the grid view), ``chirp_emitter_unclear`` and ``not_chirp``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from chirpdetector.labeling.queue import LabelQueue

__all__ = ["DECISIONS", "LabelSummary", "read_label_files", "summarize"]

DECISIONS = ("chirp", "chirp_no_emitter", "chirp_emitter_unclear", "not_chirp")
"""Decision columns of the summary tables, in order."""


def read_label_files(out_dir: str | Path) -> pd.DataFrame:
    """All label rows of an output directory (empty table if none)."""
    files = sorted((Path(out_dir) / "labels").glob("*.csv"))
    frames = [
        pd.read_csv(f, dtype={"emitter_id": "Int64", "candidate_source": str})
        for f in files
    ]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=["label", "emitter_id", "df_bin", "labeler"])
    return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class LabelSummary:
    """Count tables (rows x :data:`DECISIONS` + ``total``)."""

    by_decision: pd.DataFrame
    by_source: pd.DataFrame
    by_bin: pd.DataFrame
    open_items: int | None

    def to_text(self) -> str:
        """Plain-text rendering for the terminal."""

        def text(table: pd.DataFrame) -> str:
            return table.to_string() if len(table) else "  (no labels)"

        parts = [
            "by decision\n" + text(self.by_decision),
            "by source\n" + text(self.by_source),
            "by ΔF bin (Hz)\n" + text(self.by_bin),
        ]
        if self.open_items is not None:
            parts.append(f"not labelled yet (in the queue): {self.open_items}")
        return "\n\n".join(parts)


def _strings(df: pd.DataFrame, column: str, fill: str = "") -> list[str]:
    if column not in df.columns:
        return [fill] * len(df)
    return [
        fill if v is None or (isinstance(v, float) and v != v) or v is pd.NA else str(v)
        for v in df[column].to_numpy(dtype=object)
    ]


def _table(
    keys: Sequence[str], decisions: Sequence[str], order: Sequence[str] = ()
) -> pd.DataFrame:
    rows = list(order) + sorted(set(keys) - set(order))
    counts = Counter(zip(keys, decisions, strict=True))
    table = pd.DataFrame(
        [[counts[(r, d)] for d in DECISIONS] for r in rows],
        index=pd.Index(rows),
        columns=pd.Index(list(DECISIONS)),
        dtype="int64",
    )
    table["total"] = table.sum(axis=1)
    return table


def summarize(
    out_dir: str | Path,
    *,
    labeler: str | None = None,
    queue: LabelQueue | str | Path | None = None,
) -> LabelSummary:
    """Count the labels of an output directory.

    Parameters
    ----------
    out_dir
        The labeller's output directory.
    labeler
        Only this labeller's labels.
    queue
        Only labels of this queue (a queue, its id, or its JSON file); also
        reports how many of its items have no label yet.

    Returns
    -------
    Tables by decision, by ``candidate_source`` (``v1`` for v1 detections,
    ``-`` for candidates without a source) and by ``df_bin``.
    """
    labels = read_label_files(out_dir)
    loaded: LabelQueue | None = None
    if isinstance(queue, LabelQueue):
        loaded = queue
    elif queue is not None:
        path = Path(queue)
        if not path.suffix:
            path = Path(out_dir) / "queues" / f"{queue}.json"
        loaded = LabelQueue.from_json(json.loads(path.read_text()))
    keep = np.ones(len(labels), dtype=bool)
    if labeler is not None:
        keep &= np.array(
            [v == labeler for v in _strings(labels, "labeler")], dtype=bool
        )
    if loaded is not None:
        keep &= np.array(
            [v == loaded.queue_id for v in _strings(labels, "queue_id")], dtype=bool
        )
    label = [v for v, k in zip(_strings(labels, "label"), keep, strict=True) if k]
    emitter = [
        v for v, k in zip(_strings(labels, "emitter_id"), keep, strict=True) if k
    ]
    decision = [
        "chirp_no_emitter" if lab == "chirp" and not e else lab
        for lab, e in zip(label, emitter, strict=True)
    ]
    source = [
        (s or "-") if c else "v1"
        for s, c, k in zip(
            _strings(labels, "candidate_source"),
            _strings(labels, "candidate_id"),
            keep,
            strict=True,
        )
        if k
    ]
    bins = [v for v, k in zip(_strings(labels, "df_bin"), keep, strict=True) if k]
    open_items = None
    if loaded is not None:
        done = {v for v, k in zip(_strings(labels, "item_id"), keep, strict=True) if k}
        open_items = sum(item.item_id not in done for item in loaded.items)
    return LabelSummary(
        by_decision=_table(["all"] * len(decision), decision, ["all"]),
        by_source=_table(source, decision),
        by_bin=_table(bins, decision, loaded.bins if loaded is not None else ()),
        open_items=open_items,
    )
