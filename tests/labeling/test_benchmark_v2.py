"""Benchmark v2: features, model answers, strata, repeats, gaps, scoring."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gridtools.io import Tracks, load_tracks, save_tracks
from typer.testing import CliRunner

from chirpdetector.cli import app
from chirpdetector.labeling.benchmark import CANDIDATE_COLUMNS
from chirpdetector.labeling.benchmark_report import (
    merge_sets,
    score_v2,
    write_report_v2,
)
from chirpdetector.labeling.benchmark_v2 import (
    BuildV2Config,
    build_v2,
    make_gap_scene,
    model_outcomes,
    plan_gaps,
    read_model_dir,
    scene_features,
)

from .test_benchmark import _chirps

NO_CLICKS = np.zeros((0, 2))


def _write_tracks(
    directory: Path, lines: list[tuple[float, float]], duration_s: float
) -> None:
    """Tracks f = a + b t every 0.12 s (simulated tracks are gap-free)."""
    times = np.round(np.arange(0.0, duration_s, 0.12), 6)
    n = len(times)
    save_tracks(
        Tracks(
            times_s=times,
            freqs_hz=np.concatenate([a + b * times for a, b in lines]),
            idents=np.concatenate([np.full(n, float(i)) for i in range(len(lines))]),
            time_indices=np.concatenate([np.arange(n)] * len(lines)),
        ),
        directory,
    )


def _scene(
    directory: Path,
    centres: list[float],
    emitters: list[int],
    lines: list[tuple[float, float]],
    *,
    duration_s: float = 40.0,
    n_electrodes: int = 9,
    seed: int = 0,
    excursion_hz: float | None = None,
) -> pd.DataFrame:
    """A scene without audio: chirps at ``centres`` by ``emitters``."""
    directory.mkdir(parents=True, exist_ok=True)
    _write_tracks(directory, lines, duration_s)
    chirps = _chirps(centres, np.random.default_rng(seed))
    base = np.array(
        [lines[e][0] + lines[e][1] * t for e, t in zip(emitters, centres, strict=True)]
    )
    exc = (
        chirps["excursion_hz"].to_numpy()
        if excursion_hz is None
        else np.full(len(centres), excursion_hz)
    )
    chirps = chirps.assign(
        emitter_id=emitters,
        baseline_hz=base,
        f_low_hz=base,
        f_high_hz=base + exc,
        excursion_hz=exc,
        box_f0_hz=base - 5.0,
        box_f1_hz=base + exc + 5.0,
        visible=True,
        template_id=np.nan,
        family="small",
        harmonic_box_t0_s=np.asarray(centres) - 0.12,
        harmonic_box_t1_s=np.asarray(centres) + 0.12,
        harmonic_box_f0_hz=2 * base - 10.0,
        harmonic_box_f1_hz=2 * (base + exc) + 10.0,
    )
    chirps.to_parquet(directory / "chirps.parquet", index=False)
    (directory / "scene.json").write_text(
        json.dumps({"duration_s": duration_s, "grid": {"n_electrodes": n_electrodes}})
    )
    return chirps


def _det(
    rows: list[tuple[float, float, float, float, float, str, float]],
) -> pd.DataFrame:
    """Detections: (t0, t1, f0, f1, score, status, emitter)."""
    t0, t1, f0, f1, score, status, emitter = (
        (list(v) for v in zip(*rows, strict=True)) if rows else ([],) * 7
    )
    return pd.DataFrame(
        {
            "t_start_s": t0,
            "t_end_s": t1,
            "f_low_hz": f0,
            "f_high_hz": f1,
            "score": score,
            "status": status,
            "emitter_id": emitter,
            "harmonic_f_low_hz": f0,
            "harmonic_f_high_hz": f1,
        }
    )


# ------------------------------------------------------------------ features


def test_features_of_crafted_chirps(tmp_path: Path) -> None:
    # fish 0 at 700 Hz, fish 1 at 703 Hz (dF 3), fish 2 at 351 Hz (fish 0 sits
    # at its 2nd harmonic), fish 3 rising 2 Hz/s
    lines = [(700.0, 0.0), (703.0, 0.0), (351.0, 0.0), (800.0, 2.0)]
    centres = [10.0, 10.5, 11.0, 10.05, 20.0, 30.0]
    emitters = [0, 0, 0, 1, 3, 2]
    _scene(tmp_path / "s", centres, emitters, lines)
    feats, info = scene_features(tmp_path / "s", clicks=np.array([[29.98, 30.0]]))
    assert info["n_electrodes"] == 9
    row = feats.set_index("chirp_id")
    assert row.loc[0, "df_nearest_hz"] == pytest.approx(3.0)
    assert bool(row.loc[0, "near_harmonic"])  # 700 ~ 2 x 351
    assert bool(row.loc[0, "bout"])  # three chirps of fish 0 in 1 s
    assert bool(row.loc[1, "bout"])
    assert bool(row.loc[3, "overlap"])  # fish 1 at 10.05 overlaps fish 0 at 10.0
    assert bool(row.loc[4, "drift"])
    assert row.loc[4, "drift_hz"] == pytest.approx(4.0)
    assert bool(row.loc[5, "click"])
    assert not bool(row.loc[4, "click"])
    assert not np.asarray(feats["few_electrodes"], dtype=bool).any()


def test_gap_scene_breaks_one_track_only(tmp_path: Path) -> None:
    _scene(tmp_path / "s", [10.0, 20.0], [0, 1], [(700.0, 0.0), (760.0, 0.0)])
    target = make_gap_scene(tmp_path / "s", tmp_path / "g", 0, [10.0], 1.5)
    tracks, source = load_tracks(target), load_tracks(tmp_path / "s")
    gap = np.diff(tracks.track(0).times_s).max()
    assert gap == pytest.approx(3.0, abs=0.13)
    np.testing.assert_array_equal(tracks.track(1).freqs_hz, source.track(1).freqs_hz)
    assert (target / "chirps.parquet").is_symlink()


# ------------------------------------------------------------------ models


def test_read_model_dirs(tmp_path: Path) -> None:
    v2 = tmp_path / "v2"
    v2.mkdir()
    pd.DataFrame(
        {
            "t_start_s": [1.0, 2.0, 3.0],
            "t_end_s": [1.1, 2.1, 3.1],
            "f_low_hz": [700.0, 1400.0, 350.0],
            "f_high_hz": [800.0, 1600.0, 450.0],
            "score": [0.9, 0.8, 0.7],
            "emitter_id": [0.0, np.nan, 2.0],
            "drop_reason": ["", "harmonic", ""],
            "reattributed": ["", "", "harmonic"],
            "harmonic_f_low_hz": [np.nan, np.nan, 700.0],
            "harmonic_f_high_hz": [np.nan, np.nan, 900.0],
        }
    ).to_parquet(v2 / "s.parquet")
    (tmp_path / "v1" / "sim").mkdir(parents=True)
    pd.DataFrame(
        {
            "t_start_s": [1.0, 2.0],
            "t_end_s": [1.1, 2.1],
            "f_low_hz": [700.0] * 2,
            "f_high_hz": [800.0] * 2,
            "score": [0.9, 0.6],
            "assignable": [True, False],
        }
    ).to_csv(tmp_path / "v1" / "sim" / "s.csv", index=False)
    a = read_model_dir(v2)["s"]
    assert a["status"].tolist() == ["kept", "harmonic", "reattributed"]
    assert a["harmonic_f_low_hz"].tolist() == [700.0, 1400.0, 700.0]
    b = read_model_dir(tmp_path / "v1", "v1")["s"]
    assert b["status"].tolist() == ["kept", "unassignable"]
    assert np.isnan(b["emitter_id"].to_numpy(dtype=float)).all()


def test_model_outcomes_threshold_first(tmp_path: Path) -> None:
    scene = tmp_path / "s"
    chirps = _scene(
        scene, [10.0, 20.0], [0, 1], [(700.0, 0.0), (760.0, 0.0)], excursion_hz=80.0
    )
    det = _det(
        [
            (9.99, 10.01, 698.0, 780.0, 0.10, "kept", 0.0),  # closer in time, low score
            (9.90, 10.14, 698.0, 780.0, 0.90, "kept", 0.0),  # the real hit
            (
                20.0,
                20.1,
                1520.0,
                1690.0,
                0.70,
                "harmonic",
                np.nan,
            ),  # copy of chirp 1, dropped
            (30.0, 30.1, 700.0, 760.0, 0.60, "kept", 0.0),  # false alarm
            (19.95, 20.05, 760.0, 840.0, 0.30, "kept", 1.0),  # chirp 1, below threshold
        ]
    )
    items = pd.DataFrame(
        [
            {
                "item_id": "a",
                "recording": str(scene),
                "truth": "chirp",
                "stratum": "x",
                "chirp_id": 0,
                "t_start_s": 9.88,
                "t_end_s": 10.12,
                "f_low_hz": 695.0,
                "f_high_hz": 785.0,
            },
            {
                "item_id": "b",
                "recording": str(scene),
                "truth": "chirp",
                "stratum": "x",
                "chirp_id": 1,
                "t_start_s": 19.88,
                "t_end_s": 20.12,
                "f_low_hz": 755.0,
                "f_high_hz": 845.0,
            },
            {
                "item_id": "c",
                "recording": str(scene),
                "truth": "no_chirp",
                "stratum": "negative_harmonic_detected",
                "chirp_id": None,
                **dict(
                    zip(
                        ("t_start_s", "t_end_s", "f_low_hz", "f_high_hz"),
                        chirps[
                            [
                                "harmonic_box_t0_s",
                                "harmonic_box_t1_s",
                                "harmonic_box_f0_hz",
                                "harmonic_box_f1_hz",
                            ]
                        ]
                        .iloc[1]
                        .tolist(),
                        strict=True,
                    )
                ),
            },
            {
                "item_id": "d",
                "recording": str(scene),
                "truth": "no_chirp",
                "stratum": "negative_model_fa",
                "chirp_id": None,
                "t_start_s": 29.95,
                "t_end_s": 30.15,
                "f_low_hz": 690.0,
                "f_high_hz": 770.0,
            },
            {
                "item_id": "e",
                "recording": str(tmp_path / "other"),
                "truth": "no_chirp",
                "stratum": "negative_baseline",
                "chirp_id": None,
                "t_start_s": 1.0,
                "t_end_s": 1.2,
                "f_low_hz": 690.0,
                "f_high_hz": 770.0,
            },
        ]
    )
    out = model_outcomes(items, {"m": {"s": det}}).set_index("item_id")
    assert out.loc["a", "score"] == pytest.approx(
        0.9
    )  # not stolen by the closer weak box
    assert out.loc["b", "score"] == pytest.approx(
        0.3
    )  # below threshold: best admissible
    assert out.loc["c", "status"] == "harmonic"
    assert out.loc["d", "score"] == pytest.approx(0.6)
    assert out.loc["e", "status"] == "missing"


# ------------------------------------------------------------------ build


@pytest.fixture
def scenes(tmp_path: Path) -> dict[str, tuple[Path, pd.DataFrame, dict]]:
    out = {}
    rng = np.random.default_rng(5)
    for k in range(6):
        name = f"{8000 + k:06d}"
        centres = list(np.arange(4.0, 36.0, 1.6) + rng.uniform(-0.1, 0.1, 20))
        emitters = [i % 2 for i in range(len(centres))]
        lines = [(700.0, 0.0), (700.0 + (3.0 if k % 2 else 40.0), 0.0)]
        _scene(
            tmp_path / "scenes" / name,
            centres,
            emitters,
            lines,
            n_electrodes=3 if k == 0 else 9,
            seed=k,
        )
        feats, info = scene_features(
            tmp_path / "scenes" / name,
            clicks=np.array([[20.0, 20.01]]) if k == 1 else NO_CLICKS,
        )
        out[name] = (tmp_path / "scenes" / name, feats, info)
    return out


def _models(scenes: dict) -> dict[str, dict[str, pd.DataFrame]]:
    models: dict[str, dict[str, pd.DataFrame]] = {"dualhead": {}, "v1": {}}
    for name, (_, feats, _) in scenes.items():
        rows = []
        for c in feats.to_dict("records"):
            if c["chirp_id"] % 3:
                rows.append(
                    (
                        c["t_center_s"] - 0.1,
                        c["t_center_s"] + 0.1,
                        c["f_low_hz"] - 2,
                        c["f_high_hz"],
                        0.8,
                        "kept",
                        float(c["emitter_id"]),
                    )
                )
        rows.append((37.0, 37.1, 700.0, 760.0, 0.3, "kept", 0.0))
        models["dualhead"][name] = _det(rows)
        models["v1"][name] = _det(
            [(r[0], r[1], r[2], r[3], r[4] * 0.5, r[5], np.nan) for r in rows]
        )
    return models


SMALL = BuildV2Config(
    n_per_stratum=(
        ("chirp_df_0_5", 4),
        ("chirp_few_electrodes", 3),
        ("chirp_click", 1),
        ("invisible", 0),
        ("negative_harmonic_detected", 2),
        ("negative_harmonic_missed", 2),
        ("negative_baseline", 3),
    ),
    n_core_per_cell=1,
    n_model_fa=(("dualhead", 1),),
    n_repeats=2,
    margin_s=2.6,
    max_attempts=5000,
)


def test_build_v2_strata_weights_and_blindness(scenes) -> None:
    models = _models(scenes)
    repeat = pd.DataFrame(
        [
            {
                "item_id": f"old#{i}",
                "candidate_id": f"h{i}",
                "scene": "007700",
                "recording": "/x/007700",
                "stratum": "chirp_other",
                "truth": "chirp",
                "scored": True,
                "t_start_s": 1.0,
                "t_end_s": 1.2,
                "f_low_hz": 700.0,
                "f_high_hz": 800.0,
            }
            for i in range(5)
        ]
    )
    cands, truth, summary = build_v2(scenes, models, SMALL, repeat_pool=repeat)
    assert tuple(cands.columns) == CANDIDATE_COLUMNS
    assert truth["item_id"].is_unique
    per = summary["per_stratum"]
    assert per["chirp_df_0_5"] == 4
    assert per["chirp_few_electrodes"] == 3  # scene 008000 has 3 electrodes
    assert per["chirp_click"] == 1
    assert per["negative_harmonic_detected"] + per["negative_harmonic_missed"] == 4
    assert per["negative_model_fa"] == 1
    # each stratum's weights add up to its pool
    chirps = truth[(truth["truth"] == "chirp") & truth["retest_of"].isna()]
    for stratum, group in chirps.groupby("stratum"):
        weight = float(np.sum(group["weight"].to_numpy(dtype=float)))
        assert weight == pytest.approx(summary["pools"][stratum])
    # repeats: new opaque ids, the original marked, not scored on their own
    rep = truth[truth["retest_of"].notna()]
    assert len(rep) == 2
    assert set(rep["retest_of"]) <= set(repeat["item_id"])
    assert not np.asarray(rep["scored"], dtype=bool).any()
    assert not set(rep["candidate_id"]) & set(repeat["candidate_id"])
    again = build_v2(scenes, models, SMALL, repeat_pool=repeat)
    pd.testing.assert_frame_equal(truth, again[1])


def test_gap_copies_form_the_gap_stratum(scenes, tmp_path: Path) -> None:
    gaps = plan_gaps(scenes, 2, seed=1, margin_s=2.6, target_root=tmp_path / "gaps")
    assert len(gaps) == 2
    exclude = {(i["gap_of"], i["gap_chirp_id"]) for _, _, i in gaps.values()}
    _, truth, _summary = build_v2(
        scenes, _models(scenes), SMALL, gap_scenes=gaps, exclude=exclude
    )
    gap = truth[truth["stratum"] == "chirp_gap"]
    assert len(gap) == 2
    assert all(Path(r).parent == tmp_path / "gaps" for r in gap["recording"])
    shown = {
        (s, int(c))
        for s, c in zip(truth["scene"], truth["chirp_id"].fillna(-1), strict=True)
    }
    assert not exclude & shown


# ------------------------------------------------------------------ scoring


def _items() -> pd.DataFrame:
    rows = [
        # stratum, truth, emitter, df, near_harmonic, set, retest_of
        ("chirp_df_0_5", "chirp", 0, 3.0, False, "a", None),
        ("chirp_df_0_5", "chirp", 1, 4.0, False, "a", None),
        ("chirp_core", "chirp", 0, 40.0, True, "a", None),
        ("chirp_core", "chirp", 1, 60.0, False, "b", None),
        ("negative_model_fa", "no_chirp", None, None, None, "a", None),
        ("negative_harmonic_missed", "no_chirp", 1, None, None, "b", None),
        ("negative_baseline", "no_chirp", None, None, None, "b", None),
        ("invisible", "invisible_chirp", 0, 10.0, False, "b", None),
        ("repeat:chirp_core", "chirp", 0, 40.0, True, "b", "a#2"),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "stratum",
            "truth",
            "emitter_id",
            "df_nearest_hz",
            "near_harmonic",
            "set",
            "retest_of",
        ],
    )
    df["item_id"] = [f"{s}#{i}" for i, s in enumerate(df["set"])]
    df["scored"] = [
        t in ("chirp", "no_chirp") and pd.isna(r)
        for t, r in zip(df["truth"], df["retest_of"], strict=True)
    ]
    df["weight"] = [10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    df["snr_db"] = [-25.0, -15.0, 5.0, 15.0, np.nan, np.nan, np.nan, -30.0, 5.0]
    df["excursion_hz"] = [40.0, 50.0, 100.0, 200.0, np.nan, np.nan, np.nan, 30.0, 100.0]
    df["recording"] = "/x/s"
    return df


def test_score_v2_on_a_tiny_set(tmp_path: Path) -> None:
    items = _items()
    labels = pd.DataFrame(
        {
            "item_id": items["item_id"],
            "label": [
                "chirp",
                "not_chirp",
                "chirp",
                "chirp",
                "chirp",
                "chirp",
                "not_chirp",
                "not_chirp",
                "chirp",
            ],
            "emitter_id": [0, None, 1, 1, 0, 1, None, None, 0],
            "labeler": "pw",
            "elapsed_s": [1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    sets = [
        (name, pd.DataFrame(items[items["set"] == name]).drop(columns="set"), labels)
        for name in ("a", "b")
    ]
    merged = merge_sets(sets)
    outcomes = pd.DataFrame(
        [
            {"item_id": i, "model": m, "score": s, "status": st, "emitter_id": e}
            for i, m, s, st, e in [
                ("a#0", "dh", 0.9, "kept", 0.0),
                ("a#1", "dh", 0.8, "kept", 0.0),
                ("a#2", "dh", 0.2, "kept", 0.0),
                ("b#3", "dh", 0.7, "kept", 1.0),
                ("a#4", "dh", 0.6, "kept", 0.0),
                ("b#5", "dh", 0.9, "reattributed", 1.0),
                ("b#6", "dh", 0.0, "none", np.nan),
                ("b#7", "dh", 0.0, "none", np.nan),
                ("b#8", "dh", 0.1, "kept", 0.0),
                *[
                    (f"{s}#{i}", "box", 0.6, "kept", np.nan)
                    for i, s in enumerate(items["set"])
                ],
            ]
        ]
    )
    report = score_v2(merged, outcomes, n_boot=100)
    p = report.metrics["point"]
    # human: chirps a#0, a#2, b#3 said chirp (3/4); dh: a#0, a#1, b#3 (3/4)
    assert p["recall_human"] == pytest.approx(0.75)
    assert p["recall_dh"] == pytest.approx(0.75)
    # weighted: a#1 (weight 10) missed by the human only
    assert p["recall_weighted_human"] == pytest.approx(12 / 22)
    # negatives: human says chirp on a#4 (FA) and on the harmonic b#5 with its
    # own fish (re-attributed: not an FA); dh: a#4 FA, b#5 re-attributed to fish 1
    assert p["false_alarm_human"] == pytest.approx(1 / 3)
    assert p["false_alarm_dh"] == pytest.approx(1 / 3)
    assert p["false_alarm_harmonic_missed_human"] == 0.0
    # right emitter: human a#0 (b#3 right too) -> 2/4; dh a#0, b#3 -> 2/4 (a#1 wrong)
    assert p["emitter_recall_human"] == pytest.approx(0.5)
    assert p["emitter_recall_dh"] == pytest.approx(0.5)
    assert math.isnan(p["emitter_recall_box"])  # no emitter head
    harmonic = report.tables["harmonic"].iloc[0]
    assert (harmonic["human_to_its_fish"], harmonic["dh_reattributed_right"]) == (1, 1)
    retest = report.metrics["retest"]
    assert retest["n_pairs"] == 1
    assert retest["decision_agreement"] == 1.0
    invisible = report.tables["invisible"].set_index("decider")
    assert invisible.loc["human", "said_chirp"] == 0
    assert set(report.tables["features"]["feature"]) == {"near_harmonic"}
    path = write_report_v2(report, tmp_path / "r")
    text = path.read_text()
    assert "## Where the production model" not in text  # no 'dualhead' model here
    for name in (
        "fig_recall.png",
        "fig_edge_cases.png",
        "fig_operating_emitter.png",
        "metrics.json",
        "items.csv",
    ):
        assert (tmp_path / "r" / name).exists()


def test_cli_build_v2_and_score_sets(scenes, tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for name, (_, feats, info) in scenes.items():
        feats.to_parquet(feature_dir / f"{name}.parquet", index=False)
        (feature_dir / f"{name}.json").write_text(json.dumps(info))
    models = _models(scenes)
    for model, by_scene in models.items():
        root = tmp_path / "models" / model
        root.mkdir(parents=True)
        for name, det in by_scene.items():
            det.assign(drop_reason="", reattributed="").to_parquet(
                root / f"{name}.parquet", index=False
            )
    runner = CliRunner()
    cands, truth = tmp_path / "c" / "b.parquet", tmp_path / "t" / "t.parquet"
    result = runner.invoke(
        app,
        [
            "label-benchmark",
            "build-v2",
            "--scenes",
            str(tmp_path / "scenes"),
            "--features",
            str(feature_dir),
            "--candidates",
            str(cands),
            "--truth",
            str(truth),
            "--model",
            f"dualhead={tmp_path / 'models' / 'dualhead'}",
            "--model",
            f"v1={tmp_path / 'models' / 'v1'}",
        ],
    )
    assert result.exit_code == 0, result.output
    table = pd.read_parquet(truth)
    labels = tmp_path / "labels" / "labels"
    labels.mkdir(parents=True)
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
    ).to_csv(labels / "all.csv", index=False)
    result = runner.invoke(
        app,
        [
            "label-benchmark",
            "score",
            "--out",
            str(tmp_path / "report"),
            "--set",
            f"v2={truth}:{tmp_path / 'labels'}",
            "--model",
            f"dualhead={tmp_path / 'models' / 'dualhead'}",
            "--features",
            str(feature_dir),
            "--n-boot",
            "50",
        ],
    )
    assert result.exit_code == 0, result.output
    metrics = json.loads((tmp_path / "report" / "metrics.json").read_text())
    assert metrics["point"]["recall_human"] == 1.0
    assert metrics["point"]["false_alarm_human"] == 0.0
    assert 0 < metrics["point"]["recall_dualhead"] < 1


def test_retest_pairs_compare_two_sessions() -> None:
    from chirpdetector.labeling.benchmark_report import retest_pairs

    original = pd.DataFrame(
        {
            "item_id": ["s#a", "s#b", "s#c"],
            "candidate_id": ["a", "b", "c"],
            "truth": ["chirp", "no_chirp", "chirp"],
        }
    )
    repeat = pd.DataFrame(
        {
            "item_id": ["s#x", "s#y", "s#z"],
            "candidate_id": ["x", "y", "z"],
            "truth": ["chirp", "no_chirp", "chirp"],
            "retest_of": ["a", "b", "s#c"],
        }
    )
    first = pd.DataFrame(
        {
            "item_id": ["s#a", "s#b", "s#c"],
            "label": ["chirp", "not_chirp", "chirp"],
            "emitter_id": [1, None, 2],
        }
    )
    second = pd.DataFrame(
        {
            "item_id": ["s#x", "s#y", "s#z"],
            "label": ["chirp", "chirp", "chirp"],
            "emitter_id": [1, 0, 3],
        }
    )
    result = retest_pairs(repeat, second, original, first)
    assert result["n_pairs"] == 3  # by candidate id and by item id
    assert result["decision_agreement"] == pytest.approx(2 / 3)
    assert result["emitter_agreement"] == pytest.approx(0.5)  # a agrees, c does not
    assert result["correct_first"] == 1.0
    assert result["correct_repeat"] == pytest.approx(2 / 3)
