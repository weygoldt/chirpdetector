"""Candidates files: format, validation and the queue built from them."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chirpdetector.labeling.candidates import (
    CANDIDATE_FIELDS,
    build_candidate_queue,
    default_candidate_id,
    load_or_build_candidate_queue,
    read_candidates,
)
from chirpdetector.labeling.queue import QueueConfig

from .conftest import BOXES, CHIRP, write_candidates


def test_minimal_file_reads_with_defaults(trial: Path, tmp_path: Path) -> None:
    path = write_candidates(tmp_path / "c.csv", trial)
    df = read_candidates(path)
    assert len(df) == len(BOXES)
    assert set(df["recording"].to_numpy()) == {str(trial)}
    assert set(df["tracks"].to_numpy()) == {str(trial)}
    assert np.isnan(df["score"].to_numpy(dtype=float)).all()
    assert pd.isna(df["track_id"].to_numpy()).all()
    assert set(df["source"].to_numpy()) == {""}
    ids = df["candidate_id"].tolist()
    assert ids[0] == default_candidate_id(trial.name, CHIRP)
    assert len(set(ids)) == len(ids)


def test_parquet_and_csv_give_the_same_table(trial: Path, tmp_path: Path) -> None:
    kw = {"score": [0.9, 0.8, 0.7, 0.6], "track_id": [0, None, 1, 0]}
    a = read_candidates(write_candidates(tmp_path / "c.csv", trial, **kw))
    b = read_candidates(write_candidates(tmp_path / "c.parquet", trial, **kw))
    pd.testing.assert_frame_equal(a, b)
    assert a["track_id"].tolist()[:2] == [0, pd.NA]


def test_relative_paths_are_relative_to_the_file(trial: Path, tmp_path: Path) -> None:
    tracks = tmp_path / "tracks" / trial.name
    tracks.mkdir(parents=True)
    for name in ("times.npy", "fund_v.npy", "ident_v.npy", "idx_v.npy"):
        shutil.copy(trial / name, tracks / name)
    path = write_candidates(
        tmp_path / "rec" / "c.csv",
        Path(trial.name),
        tracks=f"../tracks/{trial.name}",
    )
    df = read_candidates(path)
    assert Path(df["recording"].iloc[0]) == trial
    assert Path(df["tracks"].iloc[0]).resolve() == tracks.resolve()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"t_end_s": [3.0, 1.57, 6.07, 0.57]}, "t_end_s <= t_start_s"),
        ({"f_high_hz": [600.0, 800.0, 800.0, 800.0]}, "f_high_hz <= f_low_hz"),
        ({"t_start_s": ["x", 1.43, 5.93, 0.43]}, "t_start_s is not a number"),
        ({"emitter_hz": [-5.0, None, None, None]}, "emitter_hz must be positive"),
        ({"track_id": [1.5, None, None, None]}, "track_id is not an integer"),
        ({"track_id": ["fish", None, None, None]}, "track_id is not a number"),
    ],
)
def test_invalid_rows_are_listed(
    trial: Path, tmp_path: Path, change: dict, message: str
) -> None:
    path = write_candidates(tmp_path / "c.csv", trial)
    df = pd.read_csv(path)
    for column, values in change.items():
        df[column] = values
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message) as info:
        read_candidates(path)
    assert "rows 0" in str(info.value)


def test_missing_columns_and_paths_fail_early(trial: Path, tmp_path: Path) -> None:
    path = write_candidates(tmp_path / "c.csv", trial)
    df = pd.read_csv(path)
    df.drop(columns="f_low_hz").to_csv(tmp_path / "missing.csv", index=False)
    with pytest.raises(ValueError, match="f_low_hz"):
        read_candidates(tmp_path / "missing.csv")
    df.assign(recording=str(tmp_path / "nowhere")).to_csv(
        tmp_path / "x.csv", index=False
    )
    with pytest.raises(FileNotFoundError, match="nowhere"):
        read_candidates(tmp_path / "x.csv")
    df.assign(tracks=str(tmp_path)).to_csv(tmp_path / "t.csv", index=False)
    with pytest.raises(FileNotFoundError, match="lacks"):
        read_candidates(tmp_path / "t.csv")
    (tmp_path / "c.txt").write_text("")
    with pytest.raises(ValueError, match=r"\.csv or \.parquet"):
        read_candidates(tmp_path / "c.txt")


def test_ids_must_be_unique_and_without_hash(trial: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        read_candidates(
            write_candidates(
                tmp_path / "a.csv", trial, candidate_id=["a", "a", "b", "c"]
            )
        )
    with pytest.raises(ValueError, match="'#'"):
        read_candidates(
            write_candidates(
                tmp_path / "b.csv", trial, candidate_id=["a#1", "b", "c", "d"]
            )
        )
    path = write_candidates(tmp_path / "c.csv", trial)
    df = pd.read_csv(path)
    pd.concat([df, df.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="same box is listed twice"):
        read_candidates(path)


def test_two_recordings_with_one_name_are_refused(trial: Path, tmp_path: Path) -> None:
    other = tmp_path / "elsewhere" / trial.name
    shutil.copytree(trial, other)
    path = write_candidates(tmp_path / "c.csv", trial)
    df = pd.read_csv(path)
    df.loc[1, "recording"] = str(other)
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="different recordings"):
        read_candidates(path)


def test_queue_keeps_every_row_with_candidate_columns(candidate_queue) -> None:
    queue = candidate_queue
    assert len(queue) == len(BOXES)
    assert queue.origin["kind"] == "candidates"
    first = queue.items[0]  # --keep-order
    assert first.item_id == "2000-01-01-10_00#chirp"
    assert (first.det_index, first.box) == (0, CHIRP)
    columns = first.candidate_columns
    assert list(columns)[: len(CANDIDATE_FIELDS)] == list(CANDIDATE_FIELDS)
    assert columns["candidate_score"] == pytest.approx(0.9)
    assert columns["candidate_emitter_hz"] == pytest.approx(700.4)
    assert columns["candidate_track_id"] is None
    assert columns["candidate_source"] == "model-x"
    assert columns["candidate_tile"] == 3
    assert queue.items[1].candidate_columns["candidate_emitter_hz"] is None
    # ΔF bins as in the v1 queue: two tracks 60 Hz apart, one at 0.5 s
    assert [i.df_bin for i in queue.items] == ["50-100", "50-100", "50-100", "single"]
    assert queue.items[3].n_candidates == 1


def test_queue_is_shuffled_by_seed_and_resumes(
    candidates_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chirpdetector.labeling import candidates as module

    out = tmp_path / "out2"
    a = load_or_build_candidate_queue(candidates_csv, QueueConfig(seed=1), out)
    b = build_candidate_queue(candidates_csv, QueueConfig(seed=1))
    c = build_candidate_queue(candidates_csv, QueueConfig(seed=1), shuffle=False)
    assert a.items == b.items
    assert a.queue_id == b.queue_id != c.queue_id
    assert sorted(i.item_id for i in a.items) == sorted(i.item_id for i in c.items)
    assert (out / "queues" / f"{a.queue_id}.json").exists()

    def no_rebuild(*args: object, **kwargs: object) -> None:
        raise AssertionError("queue was rebuilt")

    monkeypatch.setattr(module, "build_candidate_queue", no_rebuild)
    again = load_or_build_candidate_queue(candidates_csv, QueueConfig(seed=1), out)
    assert again == a


def test_changed_file_gives_a_new_queue(candidates_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "out3"
    first = load_or_build_candidate_queue(candidates_csv, None, out)
    df = pd.read_csv(candidates_csv)
    df.loc[0, "score"] = 0.1
    df.to_csv(candidates_csv, index=False)
    assert (
        load_or_build_candidate_queue(candidates_csv, None, out).queue_id
        != first.queue_id
    )
