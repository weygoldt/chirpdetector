"""The human benchmark: sampling, blindness of its files, scoring."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gridtools.io import Tracks, save_tracks
from typer.testing import CliRunner

from chirpdetector.cli import app
from chirpdetector.labeling.benchmark import (
    CANDIDATE_COLUMNS,
    BenchmarkConfig,
    build_benchmark,
    score_benchmark,
    write_benchmark,
    write_report,
)

FISH_HZ = (700.0, 760.0)


def _tracks(directory: Path, duration_s: float) -> None:
    times = np.round(np.arange(0.0, duration_s, 0.3277), 6)
    n = len(times)
    save_tracks(
        Tracks(
            times_s=times,
            freqs_hz=np.concatenate([np.full(n, f) for f in FISH_HZ]),
            idents=np.concatenate([np.full(n, float(i)) for i in range(len(FISH_HZ))]),
            time_indices=np.concatenate([np.arange(n)] * len(FISH_HZ)),
        ),
        directory,
    )


def _chirps(centres: list[float], rng: np.random.Generator) -> pd.DataFrame:
    n = len(centres)
    emitter = np.arange(n) % 2
    base = np.asarray(FISH_HZ)[emitter]
    excursion = rng.uniform(20.0, 300.0, n)
    snr = rng.normal(-5.0, 15.0, n)
    t = np.asarray(centres)
    return pd.DataFrame(
        {
            "chirp_id": np.arange(n),
            "emitter_id": emitter,
            "t_onset_s": t - 0.01,
            "t_center_s": t,
            "t_offset_s": t + 0.01,
            "duration_s": 0.02,
            "baseline_hz": base,
            "excursion_hz": excursion,
            "f_low_hz": base,
            "f_high_hz": base + excursion,
            "dip": rng.uniform(0.05, 0.6, n),
            "truncated": False,
            "box_t0_s": t - 0.12,
            "box_t1_s": t + 0.12,
            "box_f0_hz": base - 5.0,
            "box_f1_hz": base + excursion + 5.0,
            "snr_db": snr,
            "visible": snr > -25.0,
            "harmonic_box_t0_s": np.nan,
            "harmonic_box_t1_s": np.nan,
            "harmonic_box_f0_hz": np.nan,
            "harmonic_box_f1_hz": np.nan,
        }
    )


def make_scene(directory: Path, seed: int, duration_s: float = 30.0) -> pd.DataFrame:
    """A scene without audio (chirps every 2 s) and the model's detections."""
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    _tracks(directory, duration_s)
    centres = [float(t) for t in np.arange(1.5, duration_s - 1.0, 2.0)]
    chirps = _chirps(centres, rng)
    chirps.to_parquet(directory / "chirps.parquet", index=False)
    (directory / "scene.json").write_text(json.dumps({"duration_s": duration_s}))
    rows = []
    for c in chirps.to_dict("records"):
        if c["visible"] and rng.uniform() < 0.8:
            rows.append(
                (
                    c["t_center_s"] - 0.1,
                    c["t_center_s"] + 0.1,
                    c["f_low_hz"] - 3,
                    c["f_high_hz"],
                    rng.uniform(0.3, 1.0),
                    c["emitter_id"],
                )
            )
    for t in np.asarray(centres)[:-1] + 1.0:  # false positives between chirps
        if rng.uniform() < 0.5:
            rows.append((t - 0.08, t + 0.08, 690.0, 760.0, rng.uniform(0.05, 0.7), 0))
    return pd.DataFrame(
        rows,
        columns=[
            "t_start_s",
            "t_end_s",
            "f_low_hz",
            "f_high_hz",
            "score",
            "emitter_id",
        ],
    ).assign(recording=directory.name, drop_reason=None)


SMALL = BenchmarkConfig(n_chirps=8, n_ambiguous=2, n_hard=4, n_easy=3, n_invisible=2)


@pytest.fixture
def scenes(tmp_path: Path) -> tuple[list[Path], dict[str, pd.DataFrame]]:
    dirs, dets = [], {}
    for k in range(4):
        d = tmp_path / "scenes" / f"{7700 + k:06d}"
        dets[d.name] = make_scene(d, k)
        dirs.append(d)
    return dirs, dets


def test_build_is_deterministic(scenes) -> None:
    dirs, dets = scenes
    a = build_benchmark(dirs, dets, SMALL)
    b = build_benchmark(list(reversed(dirs)), dets, SMALL)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])
    c = build_benchmark(dirs, dets, BenchmarkConfig(**{**SMALL.__dict__, "seed": 1}))
    assert c[1]["item_id"].tolist() != a[1]["item_id"].tolist()


def test_strata_truth_and_model_decisions(scenes) -> None:
    dirs, dets = scenes
    candidates, truth, summary = build_benchmark(dirs, dets, SMALL)
    assert summary["per_stratum"] == {
        "chirp_ambiguous": 2,
        "chirp_other": 6,
        "negative_hard": 4,
        "negative_easy": 3,
        "invisible": 2,
    }
    assert tuple(candidates.columns) == CANDIDATE_COLUMNS
    assert all(re.fullmatch(r"h[0-9a-f]{10}", c) for c in candidates["candidate_id"])
    assert (
        truth["candidate_id"].to_numpy() == candidates["candidate_id"].to_numpy()
    ).all()
    assert set(truth.loc[truth["stratum"] == "invisible", "scored"]) == {False}
    for row in truth.to_dict("records"):
        scene = Path(row["recording"]).name
        chirps = pd.read_parquet(Path(row["recording"]) / "chirps.parquet")
        det = dets[scene]
        if row["truth"] == "chirp":
            chirp = chirps.loc[chirps["chirp_id"] == row["chirp_id"]].iloc[0]
            assert bool(chirp["visible"])
            assert row["emitter_id"] == chirp["emitter_id"]
            hit = det[(det["t_start_s"] - (chirp["t_center_s"] - 0.1)).abs() < 1e-9]
            assert row["model_score"] == (
                float(hit["score"].iloc[0]) if len(hit) else 0.0
            )
        elif row["stratum"] == "negative_hard":
            assert row["model_score"] >= 0.05
            near = (chirps["box_t1_s"] > row["t_start_s"] - 0.1) & (
                chirps["box_t0_s"] < row["t_end_s"] + 0.1
            )
            assert not near.any()
        elif row["stratum"] == "negative_easy":
            near = (chirps["box_t1_s"] > row["t_start_s"] - 0.5) & (
                chirps["box_t0_s"] < row["t_end_s"] + 0.5
            )
            assert not near.any()
            assert row["fish_id"] in (0, 1)
    is_amb = truth["stratum"].to_numpy() == "chirp_ambiguous"
    assert (truth["excursion_hz"].to_numpy(float)[is_amb] < 60).all()
    snr = truth["snr_db"].to_numpy(float)[is_amb]
    assert (snr < summary["ambiguous_snr_below_db"]).all()
    margin = SMALL.margin_s
    centre = 0.5 * (
        truth["t_start_s"].to_numpy(float) + truth["t_end_s"].to_numpy(float)
    )
    assert ((centre >= margin) & (centre <= 30.0 - margin)).all()


def test_truth_goes_to_another_directory(scenes, tmp_path: Path) -> None:
    dirs, dets = scenes
    candidates, truth, summary = build_benchmark(dirs, dets, SMALL)
    with pytest.raises(ValueError, match="another directory"):
        write_benchmark(
            candidates, truth, summary, tmp_path / "c.parquet", tmp_path / "t.parquet"
        )
    write_benchmark(
        candidates,
        truth,
        summary,
        tmp_path / "c" / "c.parquet",
        tmp_path / "t" / "t.parquet",
    )
    assert json.loads((tmp_path / "t" / "t.json").read_text())["n_items"] == len(truth)
    assert pd.read_parquet(tmp_path / "c" / "c.parquet").columns.tolist() == list(
        CANDIDATE_COLUMNS
    )


def test_labeller_never_reads_the_truth(
    tmp_path: Path, make_synthetic_trial: Callable[[Path], Path]
) -> None:
    """With the truth unreadable, the whole blind labelling path still works."""
    import matplotlib.pyplot as plt

    from chirpdetector.labeling.candidates import load_or_build_candidate_queue
    from chirpdetector.labeling.server import LabelServer
    from chirpdetector.labeling.session import LabelSession
    from chirpdetector.labeling.single import SingleLabeller
    from chirpdetector.labeling.store import LabelStore

    from .test_server import Client

    scene = make_synthetic_trial(tmp_path / "scenes" / "007700")
    chirps = _chirps([4.0], np.random.default_rng(0)).assign(
        visible=True, snr_db=-20.0, box_t0_s=3.93, box_t1_s=4.07, excursion_hz=50.0
    )
    chirps.to_parquet(scene / "chirps.parquet", index=False)
    (scene / "scene.json").write_text(json.dumps({"duration_s": 8.0}))
    det = pd.DataFrame(
        {
            "t_start_s": [3.93, 5.92],
            "t_end_s": [4.07, 6.08],
            "f_low_hz": [690.0, 690.0],
            "f_high_hz": [800.0, 760.0],
            "score": [0.9, 0.3],
            "emitter_id": [0, 0],
        }
    )
    config = BenchmarkConfig(
        n_chirps=1, n_ambiguous=0, n_hard=1, n_easy=1, n_invisible=0
    )
    candidates, truth, summary = build_benchmark([scene], {"007700": det}, config)
    cand_path = tmp_path / "candidates" / "bench.parquet"
    truth_dir = tmp_path / "truth"
    write_benchmark(
        candidates, truth, summary, cand_path, truth_dir / "bench_truth.parquet"
    )
    truth_dir.chmod(0o000)
    try:
        out = tmp_path / "labels"
        queue = load_or_build_candidate_queue(cand_path, None, out)
        assert len(queue) == 3
        with LabelStore(out, "tester", "2") as store:
            session = LabelSession(queue, store, frontend="browser", blind=True)
            server = LabelServer(session, port=0)
            server.start()
            try:
                client = Client(server)
                assert client.json("/api/state")[0] == 200
                for i in range(3):
                    status, view = client.json(f"/api/view?index={i}&window=0.2048")
                    assert status == 200
                    assert view["error"] is None
                    assert client.request(view["panels"]["zoom"]["src"])[0] == 200
                    client.json(
                        "/api/action",
                        {
                            "index": i,
                            "window_s": 0.2048,
                            "action": "not_chirp",
                            "elapsed_s": 1.0,
                        },
                    )
            finally:
                server.shutdown()
        with LabelStore(out, "tester", "2") as store:
            lab = SingleLabeller(
                LabelSession(queue, store, background=False, blind=True),
                figure=plt.figure(),
            )
            lab.fig.canvas.draw()
            plt.close(lab.fig)
    finally:
        truth_dir.chmod(0o755)


def _truth() -> pd.DataFrame:
    rows = [
        # stratum, truth, emitter, model score, model emitter, snr quartile, excursion bin
        ("chirp_ambiguous", "chirp", 0, 0.9, 0, 1, "<60"),
        ("chirp_ambiguous", "chirp", 1, 0.2, None, 1, "<60"),
        ("chirp_other", "chirp", 0, 0.8, 1, 3, ">=150"),
        ("chirp_other", "chirp", 1, 0.7, 1, 4, "60-150"),
        ("negative_hard", "no_chirp", None, 0.6, 0, None, None),
        ("negative_hard", "no_chirp", None, 0.1, 0, None, None),
        ("negative_easy", "no_chirp", None, 0.0, None, None, None),
        ("invisible", "invisible_chirp", 0, 0.55, 0, 1, "<60"),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "stratum",
            "truth",
            "emitter_id",
            "model_score",
            "model_emitter_id",
            "snr_quartile",
            "excursion_bin",
        ],
    )
    df["item_id"] = [f"s#{i}" for i in range(len(df))]
    df["scored"] = df["stratum"] != "invisible"
    df["weight"] = [3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    return df


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["s#0", "s#1", "s#2", "s#3", "s#4", "s#5", "s#6", "s#7"],
            "label": [
                "chirp",
                "chirp_emitter_unclear",
                "not_chirp",
                "chirp",
                "not_chirp",
                "chirp",
                "not_chirp",
                "chirp",
            ],
            "emitter_id": [0, None, None, 0, None, 1, None, 0],
            "labeler": "pw",
            "elapsed_s": [2.0, 4.0, 3.0, 1.0, 2.0, 5.0, 1.0, 3.0],
        }
    )


def test_scoring_on_a_tiny_set(tmp_path: Path) -> None:
    score = score_benchmark(_truth(), _labels(), threshold=0.5, n_boot=200)
    m = score.metrics
    p = m["point"]
    # human: chirps 0, 1 (unclear), 3 -> 3/4; model: 0, 2, 3 -> 3/4
    assert p["recall_human"] == pytest.approx(0.75)
    assert p["recall_model"] == pytest.approx(0.75)
    # negatives: human says chirp on #5; model on #4
    assert p["false_alarm_human"] == pytest.approx(1 / 3)
    assert p["false_alarm_model"] == pytest.approx(1 / 3)
    assert p["false_alarm_hard_human"] == pytest.approx(0.5)
    assert p["false_alarm_easy_model"] == 0.0
    # right emitter: human #0 (#3 wrong, #1 unclear); model #0 and #3
    assert p["emitter_recall_human"] == pytest.approx(0.25)
    assert p["emitter_recall_model"] == pytest.approx(0.5)
    # weighted: ambiguous items count 3x
    assert p["recall_weighted_human"] == pytest.approx((3 + 3 + 0 + 1) / 8)
    assert m["mcnemar"]["chirps"] == {
        "n": 4,
        "only_human": 1,
        "only_model": 1,
        "p": 1.0,
    }
    assert m["emitter_accuracy"]["human_n"] == 2
    assert m["emitter_accuracy"]["human_unclear"] == 1
    assert m["invisible"] == {"n": 1, "human_chirp": 1, "model_chirp": 1}
    assert m["time_s"]["median"] == pytest.approx(2.5)
    low, high = m["ci95"]["recall_human"]
    assert 0.0 <= low <= 0.75 <= high <= 1.0
    by_q = score.tables["by_snr_quartile"].set_index("snr_quartile")
    assert by_q.loc["1", "recall_human"] == 1.0
    assert by_q.loc["1", "recall_model"] == 0.5
    path = write_report(
        score,
        tmp_path / "report",
        summary={"snr_quartile_edges_db": [-10.0, 0.0, 10.0]},
    )
    text = path.read_text()
    assert "recall on true chirps" in text
    for name in (
        "report.png",
        "metrics.json",
        "items.csv",
        "by_stratum.csv",
        "model_curve.csv",
    ):
        assert (tmp_path / "report" / name).exists()


def test_unlabelled_items_are_left_out_and_counted() -> None:
    labels = _labels().iloc[:4]
    score = score_benchmark(_truth(), labels, n_boot=50)
    assert score.metrics["unlabelled"] == 4
    assert score.metrics["n_scored"] == 4
    with pytest.raises(ValueError, match="pass labeler"):
        score_benchmark(
            _truth(), pd.concat([labels, labels.assign(labeler="xx")]), n_boot=10
        )
    assert (
        score_benchmark(
            _truth(),
            pd.concat([labels, labels.assign(labeler="xx")]),
            labeler="pw",
            n_boot=10,
        ).metrics["n_labelled"]
        == 4
    )


def test_cli_build_and_score(scenes, tmp_path: Path) -> None:
    dirs, dets = scenes
    det_dir = tmp_path / "dets"
    det_dir.mkdir()
    for name, det in dets.items():
        det.to_parquet(det_dir / f"{name}.parquet", index=False)
    runner = CliRunner()
    cands, truth = tmp_path / "c" / "bench.parquet", tmp_path / "t" / "truth.parquet"
    result = runner.invoke(
        app,
        [
            "label-benchmark",
            "build",
            "--scenes",
            str(dirs[0].parent),
            "--detections",
            str(det_dir),
            "--candidates",
            str(cands),
            "--truth",
            str(truth),
            "--n-chirps",
            "8",
            "--n-ambiguous",
            "2",
            "--n-hard",
            "4",
            "--n-easy",
            "3",
            "--n-invisible",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    table = pd.read_parquet(truth)
    labels_dir = tmp_path / "labels" / "labels"
    labels_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "category": "chirp",
            "kind": "span",
            "channel": None,
            "t_start_s": table["t_start_s"],
            "t_end_s": table["t_end_s"],
            "f_low_hz": table["f_low_hz"],
            "f_high_hz": table["f_high_hz"],
            "note": "",
            "label": np.where(table["truth"] == "chirp", "chirp", "not_chirp"),
            "emitter_id": table["emitter_id"],
            "labeler": "pw",
            "item_id": table["item_id"],
        }
    ).to_csv(labels_dir / "all.csv", index=False)
    result = runner.invoke(
        app,
        [
            "label-benchmark",
            "score",
            "--truth",
            str(truth),
            "--labels",
            str(tmp_path / "labels"),
            "--out",
            str(tmp_path / "report"),
            "--n-boot",
            "50",
        ],
    )
    assert result.exit_code == 0, result.output
    metrics = json.loads((tmp_path / "report" / "metrics.json").read_text())
    assert metrics["point"]["recall_human"] == 1.0
    assert metrics["point"]["false_alarm_human"] == 0.0
    assert metrics["point"]["emitter_recall_human"] == 1.0
