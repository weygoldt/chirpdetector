"""Fixtures of the labelling tests: candidates files on the synthetic trial."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd
import pytest

from chirpdetector.labeling.candidates import load_or_build_candidate_queue
from chirpdetector.labeling.queue import LabelQueue
from chirpdetector.labeling.session import LabelSession
from chirpdetector.labeling.store import LabelStore

# boxes of the synthetic trial (tests/conftest.py): fish 0 at 700 Hz chirps
# at 4 s; fish 1 (760 Hz) starts at 1 s, so the box at 0.5 s has one track
CHIRP = (3.93, 4.07, 680.0, 820.0)
BOXES = [
    CHIRP,
    (1.43, 1.57, 690.0, 800.0),
    (5.93, 6.07, 690.0, 800.0),
    (0.43, 0.57, 690.0, 800.0),
]


def write_candidates(
    path: Path, recording: Path, **columns: list[object] | object
) -> Path:
    """A candidates CSV/Parquet with the synthetic trial's four boxes."""
    t0, t1, f0, f1 = (list(v) for v in zip(*BOXES, strict=True))
    df = pd.DataFrame(
        {
            "recording": str(recording),
            "t_start_s": t0,
            "t_end_s": t1,
            "f_low_hz": f0,
            "f_high_hz": f1,
            **columns,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


@pytest.fixture
def trial(tmp_path: Path, make_synthetic_trial: Callable[[Path], Path]) -> Path:
    return make_synthetic_trial(tmp_path / "rec" / "2000-01-01-10_00")


@pytest.fixture
def candidates_csv(trial: Path, tmp_path: Path) -> Path:
    return write_candidates(
        tmp_path / "cands" / "candidates.csv",
        trial,
        candidate_id=["chirp", "a", "b", "c"],
        score=[0.9, 0.6, 0.55, 0.7],
        emitter_hz=[700.4, None, 759.0, None],
        source="model-x",
        tile=[3, 1, 5, 0],
    )


@pytest.fixture
def candidate_queue(candidates_csv: Path, tmp_path: Path) -> LabelQueue:
    return load_or_build_candidate_queue(
        candidates_csv, None, tmp_path / "out", shuffle=False
    )


@pytest.fixture
def session(candidate_queue: LabelQueue, tmp_path: Path) -> Iterator[LabelSession]:
    with LabelStore(tmp_path / "out", "tester", "2.0.0") as store:
        s = LabelSession(candidate_queue, store, background=False)
        yield s
        s.shutdown()
