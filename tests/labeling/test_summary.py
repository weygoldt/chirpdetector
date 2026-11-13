"""Counts by decision, source and ΔF bin."""

from __future__ import annotations

from pathlib import Path

from chirpdetector.labeling.session import LabelSession
from chirpdetector.labeling.summary import DECISIONS, summarize


def test_counts_by_decision_source_and_bin(
    session: LabelSession, tmp_path: Path
) -> None:
    session.pick(0, 1)
    session.mark(1, "not_chirp")
    session.mark(2, "chirp_emitter_unclear")
    session.skip(3)
    summary = summarize(tmp_path / "out", queue=session.queue.queue_id)
    assert list(summary.by_decision.columns) == [*DECISIONS, "total"]
    row = summary.by_decision.loc["all"]
    assert (row.chirp, row.not_chirp, row.chirp_emitter_unclear, row.total) == (
        1,
        1,
        1,
        3,
    )
    assert summary.by_source.loc["model-x", "total"] == 3
    assert summary.by_bin.loc["50-100", "total"] == 3
    assert summary.by_bin.loc["single", "total"] == 0  # skipped, not labelled
    assert summary.open_items == 1
    text = summary.to_text()
    assert "by ΔF bin" in text
    assert "not labelled yet (in the queue): 1" in text
    assert (
        summarize(tmp_path / "out", labeler="nobody").by_decision.loc["all", "total"]
        == 0
    )


def test_empty_directory(tmp_path: Path) -> None:
    summary = summarize(tmp_path)
    assert summary.by_decision.loc["all", "total"] == 0
    assert summary.by_source.empty
    assert summary.open_items is None
