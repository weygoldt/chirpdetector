"""Runs: output schema, resume, failures, legacy export, render, candidates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from chirpdetector.bundle import Bundle
from chirpdetector.cli import app
from chirpdetector.pipeline import run as run_module
from chirpdetector.pipeline.candidates import CANDIDATE_COLUMNS
from chirpdetector.pipeline.config import (
    ConfigError,
    PipelineConfig,
    config_from_dict,
    load_config,
)
from chirpdetector.pipeline.outputs import (
    COLUMN_DOCS,
    atomic_dir,
    legacy_arrays,
    model_tag,
    read_chirps,
    recording_start,
)
from chirpdetector.pipeline.run import discover_recordings, run_pipeline

from .conftest import make_models


def _config(tmp_path: Path, bundle: Bundle, **sections: dict) -> PipelineConfig:
    data = {
        "run": {"name": "test_run", "out": str(tmp_path / "out")},
        "models": {"detector": str(bundle.path)},
        "detection": {"store_threshold": 0.05, "operating_threshold": 0.5},
        "assignment": {"tolerance_hz": 15.0, "max_gap_s": 1.0},
        "compute": {
            "device": "cpu",
            "chunk_tiles": 16,
            "click_workers": 1,
            "prefetch": 2,
        },
        "validation": {"windows": 2, "window_s": 2.0, "candidates_per_cell": 3},
    }
    for key, values in sections.items():
        data[key] = {**data.get(key, {}), **values}
    return config_from_dict(data)


# ------------------------------------------------------------------ config


def test_config_file_round_trip(tmp_path: Path, stub_bundle: Bundle) -> None:
    path = tmp_path / "run.toml"
    path.write_text(
        f"""
[run]
name = "r1"
out = "{tmp_path / "out"}"
[models]
detector = "{stub_bundle.path}"
[detection]
store_threshold = 0.05
[compute]
chunk_tiles = 32
"""
    )
    cfg = load_config(path)
    assert cfg.run_dir == tmp_path / "out" / "r1"
    assert cfg.assignment.tolerance_hz == 20.0  # default
    assert cfg.detection.operating_threshold is None
    same = load_config(path)
    assert cfg.detection_hash() == same.detection_hash()
    other = cfg.with_overrides(assignment={"tolerance_hz": 25.0})
    assert other.detection_hash() != cfg.detection_hash()
    # name, output root, device and validation do not change the hash
    moved = cfg.with_overrides(name="r2", compute={"device": "cuda:1"})
    assert moved.detection_hash() == cfg.detection_hash()


@pytest.mark.parametrize(
    ("section", "values", "match"),
    [
        ("detection", {"store_treshold": 0.1}, "unknown keys"),
        ("compute", {"chunk_tiles": 20}, "multiple of 16"),
        (
            "detection",
            {"store_threshold": 0.5, "operating_threshold": 0.3},
            "operating_threshold",
        ),
        ("assignment", {"tolerance_hz": -1.0}, "positive"),
    ],
)
def test_config_refuses_bad_values(
    tmp_path: Path, stub_bundle: Bundle, section: str, values: dict, match: str
) -> None:
    with pytest.raises(ConfigError, match=match):
        _config(tmp_path, stub_bundle, **{section: values})


# ----------------------------------------------------------------- outputs


def test_model_tag_and_start_time(stub_bundle: Bundle, box_only_bundle: Bundle) -> None:
    assert model_tag(stub_bundle) == "rtdetr"
    assert model_tag(box_only_bundle) == "yolo"
    start = recording_start(
        Path("/x/2020-03-10-09_56"),
        {"StartDate": "2020-03-10", "StartTime": "10: 0:00"},
    )
    assert start == {
        "fishgrid": "2020-03-10T10:00:00",
        "dirname": "2020-03-10T09:56:00",
        "agree": False,
    }
    unknown = recording_start(
        Path("/x/2019-11-25-09_59"), {"StartDate": "~", "StartTime": "~"}
    )
    assert unknown["fishgrid"] is None
    assert unknown["dirname"] == "2019-11-25T09:59:00"


def test_legacy_arrays_keep_assigned_chirps_only() -> None:
    table = pd.DataFrame(
        {
            "t_center_s": [5.0, 1.0, 3.0, 4.0, 2.0],
            "chirp": [True, True, True, False, True],
            "assign_status": [
                "assigned",
                "assigned",
                "beyond_tolerance",
                "assigned",
                "assigned",
            ],
            "track_id": [3.0, 1044.0, np.nan, 3.0, 3.0],
        }
    )
    times, ids = legacy_arrays(table)
    assert times.dtype == np.float64
    assert ids.dtype == np.float64
    np.testing.assert_array_equal(times, [1.0, 2.0, 5.0])
    np.testing.assert_array_equal(ids, [1044.0, 3.0, 3.0])
    empty_t, empty_i = legacy_arrays(table.iloc[:0])
    assert empty_t.shape == (0,)
    assert empty_i.dtype == np.float64


def test_atomic_dir_keeps_the_old_result_on_error(tmp_path: Path) -> None:
    final = tmp_path / "rec"
    with atomic_dir(final) as tmp:
        (tmp / "a.txt").write_text("1")
    assert (final / "a.txt").read_text() == "1"

    def fail() -> None:
        with atomic_dir(final) as tmp:
            (tmp / "a.txt").write_text("2")
            raise RuntimeError

    with pytest.raises(RuntimeError):
        fail()
    assert (final / "a.txt").read_text() == "1"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rec"]


# --------------------------------------------------------------------- run


def test_run_writes_the_documented_outputs(
    tmp_path: Path, stub_bundle: Bundle, trial_dirs: list[Path]
) -> None:
    cfg = _config(tmp_path, stub_bundle)
    summary = run_pipeline(cfg, trial_dirs, models=make_models(stub_bundle))
    assert summary.done == [p.name for p in trial_dirs]
    run_dir = cfg.run_dir
    manifest = json.loads((run_dir / "run.json").read_text())
    assert {r["status"] for r in manifest["recordings"].values()} == {"complete"}
    assert (run_dir / "run.log").read_text().count("done in") == 2
    rec = run_dir / trial_dirs[0].name
    names = {p.name for p in rec.iterdir()}
    assert {
        "chirps_rtdetr.parquet",
        "chunks_rtdetr.parquet",
        "chirp_times_rtdetr.npy",
        "chirp_ids_rtdetr.npy",
        "meta.json",
        "qc",
        "windows",
    } <= names
    table, file_meta = read_chirps(rec / "chirps_rtdetr.parquet")
    assert file_meta["weights_sha256"] == stub_bundle.weights_sha256
    assert file_meta["model_tag"] == "rtdetr"
    assert len(table)
    # every documented column that applies is there, in order
    documented = [c for c in COLUMN_DOCS if c in table.columns]
    assert table.columns[: len(documented)].tolist() == documented
    missing = set(COLUMN_DOCS) - set(table.columns)
    assert missing <= {
        c
        for c in COLUMN_DOCS
        if c.startswith(("height", "fwhm", "width10", "undershoot", "params_"))
    }
    assert (table["chirp_id"].to_numpy() == np.arange(len(table))).all()
    assert (np.diff(table["t_center_s"]) >= 0).all()
    assert table["track_id"].dtype == np.float64
    assert set(table["track_id"].dropna()) <= {0.0, 1.0}
    assert set(table["assign_status"]) <= {
        "assigned",
        "beyond_tolerance",
        "no_live_track",
    }
    np.testing.assert_array_equal(table["operating"], table["score"] >= 0.5)
    # legacy arrays: the assigned chirps, float64
    times = np.load(rec / "chirp_times_rtdetr.npy")
    ids = np.load(rec / "chirp_ids_rtdetr.npy")
    sel = table.loc[table["chirp"] & (table["assign_status"] == "assigned")]
    np.testing.assert_array_equal(times, sel["t_center_s"].to_numpy())
    np.testing.assert_array_equal(ids, sel["track_id"].to_numpy())
    meta = json.loads((rec / "meta.json").read_text())
    assert meta["status"] == "complete"
    assert meta["inputs"]["n_electrodes"] == 4
    assert meta["tracks"]["status"] == "loaded"
    assert len(meta["tracks"]["sha256"]) == 64
    assert meta["click_removal"]["applied"] is False
    assert meta["counts"]["legacy"] == len(times)
    assert meta["models"]["detector"]["weights_sha256"] == stub_bundle.weights_sha256
    assert "chirpdetector" in meta["code"]["git"]
    # validation outputs
    assert (rec / "qc" / "qc.png").exists()
    assert (rec / "qc" / "qc.md").read_text().startswith("# QC")
    windows = pd.read_csv(rec / "windows" / "windows.csv")
    assert len(windows) == 2
    cand = pd.read_csv(run_dir / "candidates.csv")
    assert tuple(cand.columns) == CANDIDATE_COLUMNS
    assert (
        cand[["recording", "t_start_s", "t_end_s", "f_low_hz", "f_high_hz"]]
        .notna()
        .to_numpy()
        .all()
    )
    assert set(cand["recording"]) <= {str(p) for p in trial_dirs}
    assert cand["candidate_id"].is_unique


def test_run_resumes_and_survives_a_failing_recording(
    tmp_path: Path,
    stub_bundle: Bundle,
    trial_dirs: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(tmp_path, stub_bundle, validation={"windows": 0, "qc": False})
    models = make_models(stub_bundle)
    real = run_module.process_recording
    calls: list[str] = []

    def crashing(path: Path, *args, **kwargs):
        calls.append(path.name)
        if path.name == trial_dirs[0].name:
            msg = "disk on fire"
            raise OSError(msg)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(run_module, "process_recording", crashing)
    first = run_pipeline(cfg, trial_dirs, models=models)
    assert first.failed == {trial_dirs[0].name: "OSError('disk on fire')"}
    assert first.done == [trial_dirs[1].name]
    assert not (cfg.run_dir / trial_dirs[0].name).exists()
    manifest = json.loads((cfg.run_dir / "run.json").read_text())
    assert manifest["recordings"][trial_dirs[0].name]["status"] == "failed"
    assert "disk on fire" in (cfg.run_dir / "run.log").read_text()
    # a crash inside the writer leaves no partial directory behind
    assert not [p for p in cfg.run_dir.iterdir() if p.name.startswith(".")]
    monkeypatch.setattr(run_module, "process_recording", real)
    second = run_pipeline(cfg, trial_dirs, models=models)
    assert second.skipped == [trial_dirs[1].name]
    assert second.done == [trial_dirs[0].name]
    third = run_pipeline(cfg, trial_dirs, models=models)
    assert third.skipped == [p.name for p in trial_dirs]
    # other detection settings in the same run directory are refused
    other = cfg.with_overrides(assignment={"tolerance_hz": 30.0})
    with pytest.raises(ValueError, match="other models or detection settings"):
        run_pipeline(other, trial_dirs, models=models)


def test_interrupted_writer_leaves_old_result(
    tmp_path: Path,
    stub_bundle: Bundle,
    trial_dirs: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(tmp_path, stub_bundle, validation={"windows": 0, "qc": False})
    models = make_models(stub_bundle)
    run_pipeline(cfg, trial_dirs[:1], models=models)
    rec = cfg.run_dir / trial_dirs[0].name
    before = (rec / "meta.json").read_text()

    def boom(*args, **kwargs):
        msg = "killed while writing"
        raise RuntimeError(msg)

    monkeypatch.setattr(run_module, "write_json", boom)
    with pytest.raises(RuntimeError):
        run_module.process_recording(trial_dirs[0], cfg, models)
    assert (rec / "meta.json").read_text() == before
    assert not [p for p in cfg.run_dir.iterdir() if p.name.startswith(".")]


def test_box_only_model_is_not_assigned(
    tmp_path: Path, box_only_bundle: Bundle, trial_dirs: list[Path]
) -> None:
    cfg = _config(tmp_path, box_only_bundle, validation={"windows": 0, "qc": False})
    run_pipeline(cfg, trial_dirs[:1], models=make_models(box_only_bundle))
    rec = cfg.run_dir / trial_dirs[0].name
    table, _ = read_chirps(rec / "chirps_yolo.parquet")
    assert len(table)
    assert set(table["assign_status"]) == {"no_emitter"}
    assert table["emitter_hz"].isna().to_numpy().all()
    assert np.load(rec / "chirp_ids_yolo.npy").shape == (0,)
    meta = json.loads((rec / "meta.json").read_text())
    assert meta["assignment"]["supported"] is False


def test_discover_recordings(tmp_path: Path, trial_dirs: list[Path]) -> None:
    (trial_dirs[0].parent / "notes").mkdir()
    found = discover_recordings([], trial_dirs[0].parent)
    assert found == sorted(trial_dirs)
    with pytest.raises(FileNotFoundError):
        discover_recordings([trial_dirs[0].parent / "notes"])


# ------------------------------------------------------------------ render


def test_render_round_trip(
    tmp_path: Path, stub_bundle: Bundle, trial_dirs: list[Path]
) -> None:
    cfg = _config(tmp_path, stub_bundle, validation={"windows": 0, "qc": False})
    run_pipeline(cfg, trial_dirs[:1], models=make_models(stub_bundle))
    name = trial_dirs[0].name
    result = CliRunner().invoke(
        app,
        ["render", str(cfg.run_dir), name, "3.0", "5.0", "--out", str(tmp_path / "r")],
    )
    assert result.exit_code == 0, result.output
    pngs = list((tmp_path / "r").glob("*.png"))
    npzs = list((tmp_path / "r").glob("*.npz"))
    assert len(pngs) == 1
    assert len(npzs) == 1
    data = np.load(npzs[0])
    table, _ = read_chirps(cfg.run_dir / name / "chirps_rtdetr.parquet")
    inside = table.loc[(table["t_end_s"] >= 3.0) & (table["t_start_s"] <= 5.0)]
    np.testing.assert_array_equal(
        data["boxes"]["chirp_id"], inside["chirp_id"].to_numpy()
    )
    np.testing.assert_array_equal(data["boxes"]["score"], inside["score"].to_numpy())
    spec = data["spec_db"]
    assert spec.shape == (len(data["freqs_hz"]), len(data["times_s"]))
    assert data["times_s"][0] >= 3.0 - 0.005
    assert data["times_s"][-1] <= 5.0 + 0.005
    # both tracks of the trial (track 1 starts at 1 s), with their samples
    assert data["track_ids"].tolist() == [0.0, 1.0]
    assert data["track_offsets"][-1] == len(data["track_times_s"])
    meta = json.loads(str(data["meta"]))
    assert meta["t0_s"] == 3.0
    # the same window again gives the same data (recomputed from the raw data)
    again = CliRunner().invoke(
        app,
        ["render", str(cfg.run_dir), name, "3.0", "5.0", "--out", str(tmp_path / "r2")],
    )
    assert again.exit_code == 0, again.output
    np.testing.assert_array_equal(
        np.load(next((tmp_path / "r2").glob("*.npz")))["spec_db"], spec
    )


def test_cli_run_list_and_dry_run(
    tmp_path: Path,
    stub_bundle: Bundle,
    trial_dirs: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "run.toml"
    config_file.write_text(
        f"""
[run]
name = "cli"
out = "{tmp_path / "out"}"
[models]
detector = "{stub_bundle.path}"
[compute]
device = "cpu"
chunk_tiles = 16
click_workers = 1
[validation]
windows = 0
qc = false
candidates_per_cell = 0
"""
    )
    monkeypatch.setattr(run_module, "load_models", lambda cfg: make_models(stub_bundle))
    runner = CliRunner()
    listed = runner.invoke(
        app, ["run", str(config_file), "--dataset", str(trial_dirs[0].parent), "--list"]
    )
    assert listed.exit_code == 0, listed.output
    assert listed.output.count("todo") == 2
    result = runner.invoke(
        app,
        ["run", str(config_file), str(trial_dirs[0]), "--dry-run", "--limit-s", "3"],
    )
    assert result.exit_code == 0, result.output
    rec = tmp_path / "out" / "cli_dryrun" / trial_dirs[0].name
    meta = json.loads((rec / "meta.json").read_text())
    assert meta["t_range_s"] == [0.0, 3.0]
    table, _ = read_chirps(rec / "chirps_rtdetr.parquet")
    assert table["t_center_s"].to_numpy().max() < 3.0 + 3.2


def test_filter_uses_the_pipeline_assignment(
    tmp_path: Path, stub_bundle: Bundle
) -> None:
    """One assignment: the filter sees track_id as emitter_id, flags, keeps rows."""
    from gridtools.io import Tracks

    from chirpdetector.pipeline.assign import assign_detections
    from chirpdetector.pipeline.run import _postprocess, build_table, postprocess_config

    times = np.round(np.arange(0.0, 60.0, 0.1), 6)
    t1 = times[(times < 20) | (times > 25)]  # a 5-s gap: live for max_gap 10 s
    t2 = times
    tracks = Tracks(
        times_s=times,
        freqs_hz=np.r_[np.full(len(t1), 700.0), np.full(len(t2), 760.0)],
        idents=np.r_[np.full(len(t1), 4.0), np.full(len(t2), 9.0)],
        time_indices=np.r_[np.searchsorted(times, t1), np.searchsorted(times, t2)],
    )
    det = pd.DataFrame(
        {
            "recording": "r",
            "t_start_s": [10.0, 10.01, 30.0, 22.0],
            "t_end_s": [10.2, 10.2, 30.2, 22.2],
            "f_low_hz": [690.0, 692.0, 1390.0, 690.0],
            "f_high_hz": [800.0, 795.0, 1500.0, 800.0],
            "score": [0.9, 0.6, 0.8, 0.7],
            "emitter_hz": [701.0, 701.5, 1400.5, 700.5],
            "tile": [1, 1, 2, 3],
            "span_id": [0, 0, 0, 0],
        }
    )
    cfg = _config(tmp_path, stub_bundle, assignment={"max_gap_s": 10.0})
    assert postprocess_config(cfg).max_gap_s == cfg.assignment.max_gap_s
    assignment = assign_detections(det, tracks, cfg.assignment)
    flagged, info = _postprocess(det, assignment, tracks, "r", cfg)
    assert info["status"] == "applied"
    assert info["counts"]["duplicate"] == 1
    assert info["counts"]["harmonic"] == 1
    assert flagged["pp_drop_reason"].tolist() == ["", "duplicate", "harmonic", ""]
    assert flagged["pp_duplicate_of"].tolist()[1] == 0
    assert flagged["pp_harmonic_of_track"].tolist()[2] == 4.0
    # the detection in track 4's 5-s gap is assigned and protected
    assert assignment["track_id"].tolist()[3] == 4.0
    chunks = pd.DataFrame(
        {"span_id": [0], "click_fraction": [0.0], "n_click_runs": [0]}
    )
    table = build_table("r", flagged, assignment, chunks, 0.5)
    assert table["kept"].tolist() == [True, False, True, False]  # sorted by time
    assert table.loc[
        table["pp_drop_reason"] == "duplicate", "pp_duplicate_of"
    ].tolist() == [0]
    _, ids_out = legacy_arrays(table)
    np.testing.assert_array_equal(ids_out, [4.0, 4.0])  # neither copy nor duplicate
    off, off_info = _postprocess(
        det,
        assignment,
        tracks,
        "r",
        cfg.with_overrides(detection={"postprocess": False}),
    )
    assert off_info["status"] == "off"
    assert (off["pp_drop_reason"].to_numpy() == "").all()
