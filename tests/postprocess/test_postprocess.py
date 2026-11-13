"""Post-processing: harmonic copies, duplicates, the pair rule, a simulated scene."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gridtools.io import Tracks, load_tracks

from chirpdetector.postprocess import (
    COLUMNS,
    PostprocessConfig,
    drop_flagged,
    duplicates,
    flag_detections,
    postprocess,
    summarize,
)

from ..data.conftest import scene_config, write_template_library


def constant_tracks(freqs_hz: dict[int, float], t_end_s: float = 20.0) -> Tracks:
    """Tracks at constant frequencies, one detection every 0.1 s."""
    times = np.arange(0.0, t_end_s, 0.1)
    ids = sorted(freqs_hz)
    return Tracks(
        times_s=times,
        freqs_hz=np.concatenate([np.full(len(times), freqs_hz[i]) for i in ids]),
        idents=np.concatenate([np.full(len(times), float(i)) for i in ids]),
        time_indices=np.concatenate([np.arange(len(times)) for _ in ids]),
    )


def box(
    t: float,
    f_low: float,
    f_high: float,
    score: float = 0.9,
    *,
    emitter: float | None = None,
    width: float = 0.22,
    recording: str = "r",
) -> dict:
    row = {
        "recording": recording,
        "t_start_s": t - width / 2,
        "t_end_s": t + width / 2,
        "f_low_hz": f_low,
        "f_high_hz": f_high,
        "score": score,
    }
    if emitter is not None:
        row["emitter_hz"] = emitter
    return row


def reasons(flagged: pd.DataFrame) -> list[str]:
    return flagged["drop_reason"].astype(str).tolist()


# ------------------------------------------------------------ harmonic copies


def test_harmonic_copy_is_dropped_and_fundamentals_kept() -> None:
    tracks = {"r": constant_tracks({1: 640.0, 2: 760.0})}
    table = pd.DataFrame(
        [
            box(5.0, 610, 780, 0.95, emitter=641.0),  # chirp of fish 1
            box(5.0, 1250, 1450, 0.8, emitter=1283.0),  # its second harmonic
            box(8.0, 730, 850, 0.9, emitter=759.0),  # chirp of fish 2
        ]
    )
    flagged = flag_detections(table, tracks)
    assert reasons(flagged) == ["", "harmonic", ""]
    assert flagged["harmonic_of_id"].tolist() == [pd.NA, 1, pd.NA]
    kept = drop_flagged(flagged)
    assert kept["emitter_hz"].tolist() == [641.0, 759.0]
    assert set(COLUMNS) <= set(kept.columns)


def test_fish_at_twice_another_fish_keeps_its_chirps() -> None:
    table = pd.DataFrame(
        [
            box(5.0, 570, 700, 0.9, emitter=601.0),
            box(5.0, 1170, 1300, 0.9, emitter=1201.0),  # a real chirp at 2 x 600 Hz
        ]
    )
    with_fish = flag_detections(table, {"r": constant_tracks({1: 600.0, 2: 1200.0})})
    assert reasons(with_fish) == ["", ""]
    # without the second fish's track, the same box is a harmonic copy
    without = flag_detections(table, {"r": constant_tracks({1: 600.0})})
    assert reasons(without) == ["", "harmonic"]


def test_an_assigned_emitter_is_never_a_harmonic_copy() -> None:
    tracks = {"r": constant_tracks({1: 640.0})}
    table = pd.DataFrame([box(5.0, 1250, 1450, emitter=1281.0)])
    assert reasons(flag_detections(table, tracks)) == ["harmonic"]
    assigned = table.assign(emitter_id=pd.array([7], dtype="Int64"))
    assert reasons(flag_detections(assigned, tracks)) == [""]


def test_dead_tracks_do_not_count() -> None:
    tracks = {"r": constant_tracks({1: 640.0}, t_end_s=4.0)}
    table = pd.DataFrame([box(10.0, 1250, 1450, emitter=1281.0)])
    assert reasons(flag_detections(table, tracks)) == [""]


def test_box_only_models_use_the_lower_edge() -> None:
    tracks = {"r": constant_tracks({1: 645.0, 2: 760.0})}
    table = pd.DataFrame(
        [
            box(5.0, 600, 760, 0.95),  # fundamental: baseline 45 Hz above f_low
            box(5.0, 1258, 1430, 0.8),  # copy: 2 x 645 = 1290 is 32 Hz above
            box(9.0, 700, 820, 0.9),  # fish 2
        ]
    )
    flagged = flag_detections(table, tracks)
    assert reasons(flagged) == ["", "harmonic", ""]
    assert flagged.loc[1, "harmonic_of_id"] == 1
    # a fish whose fundamental lies in the window protects the box
    with_fish = flag_detections(table, {"r": constant_tracks({1: 645.0, 3: 1300.0})})
    assert reasons(with_fish)[1] == ""


# ----------------------------------------------------------------- duplicates


@pytest.mark.parametrize("head", [True, False])
def test_duplicates_keep_the_higher_score(head: bool) -> None:
    def emitter(e: float) -> float | None:
        return e if head else None

    table = pd.DataFrame(
        [
            box(2.0, 770, 960, 0.89, emitter=emitter(790.0)),
            box(2.03, 770.3, 910, 0.93, emitter=emitter(791.5), width=0.18),  # nested
            box(6.0, 700, 820, 0.5, emitter=emitter(740.0)),
            box(6.0, 700.5, 822, 0.8, emitter=emitter(741.0)),  # same box
        ]
    )
    flagged = flag_detections(table)
    assert reasons(flagged) == ["duplicate", "", "duplicate", ""]
    assert flagged["duplicate_of"].tolist() == [1, pd.NA, 3, pd.NA]


def test_simultaneous_chirps_of_two_fish_are_kept() -> None:
    # head: emitters 8 Hz apart; box only: lower edges 10 Hz apart
    head = pd.DataFrame(
        [box(3.0, 700, 830, 0.9, emitter=738.0), box(3.0, 702, 840, 0.8, emitter=746.0)]
    )
    assert reasons(flag_detections(head)) == ["", ""]
    edge = pd.DataFrame([box(3.0, 700, 830, 0.9), box(3.0, 710, 840, 0.8)])
    assert reasons(flag_detections(edge)) == ["", ""]
    # same emitter frequency, but assigned to two tracks: kept
    ids = head.assign(emitter_hz=[740.0, 742.0], emitter_id=pd.array([1, 2], "Int64"))
    assert reasons(flag_detections(ids)) == ["", ""]
    assert (
        reasons(flag_detections(ids.assign(emitter_id=pd.array([1, 1], "Int64"))))[1]
        == "duplicate"
    )


def test_simultaneous_chirps_with_shared_box_edges_are_kept() -> None:
    """Seen on sim_v4: two fish 18-20 Hz apart chirp at once; the small and
    the big chirp's boxes share their lower edge (box only), or two boxes of
    one track's emitter overlap at temporal IoU 0.55-0.63 (emitter head)."""
    edge = pd.DataFrame(
        [
            box(5.06, 640.0, 1053.0, 0.96, width=0.31),
            box(5.075, 641.3, 744.7, 0.90, width=0.25),  # frequency IoU 0.25
        ]
    )
    assert reasons(flag_detections(edge)) == ["", ""]
    head = pd.DataFrame(
        [
            box(48.855, 646.0, 1131.0, 0.94, emitter=678.5, width=0.73),
            box(48.76, 637.0, 1109.0, 0.81, emitter=677.7, width=0.40),
        ]
    ).assign(emitter_id=pd.array([1, 1], dtype="Int64"))
    assert reasons(flag_detections(head)) == ["", ""]


@pytest.mark.parametrize("head", [True, False])
def test_two_quick_chirps_of_one_fish_are_kept(head: bool) -> None:
    # 0.22 s boxes (small chirps + window smear) 0.1 s apart: each centre lies
    # inside the other box, temporal IoU 0.38
    e = 740.0 if head else None
    table = pd.DataFrame(
        [box(4.0, 700, 820, 0.9, emitter=e), box(4.1, 700, 820, 0.85, emitter=e)]
    )
    assert reasons(flag_detections(table)) == ["", ""]


def test_duplicates_respect_recordings_and_inactive_rows() -> None:
    table = pd.DataFrame(
        [box(1.0, 700, 800, 0.9), box(1.0, 700, 800, 0.8, recording="other")]
    )
    assert reasons(flag_detections(table)) == ["", ""]
    same = pd.DataFrame([box(1.0, 700, 800, 0.9), box(1.0, 700, 800, 0.8)])
    np.testing.assert_array_equal(duplicates(same, PostprocessConfig()), [-1, 0])
    np.testing.assert_array_equal(
        duplicates(same, PostprocessConfig(), active=[False, True]), [-1, -1]
    )


def test_postprocessing_commutes_with_the_score_threshold() -> None:
    """Flagging all detections, then thresholding, equals the reverse."""
    rng = np.random.default_rng(3)
    tracks = {"r": constant_tracks({1: 640.0, 2: 700.0, 3: 900.0})}
    rows = []
    for _ in range(300):
        t = rng.uniform(0.5, 19.5)
        f = rng.choice([640.0, 700.0, 900.0, 1280.0, 1400.0]) + rng.normal(0, 3)
        rows.append(
            box(
                t,
                f - rng.uniform(20, 60),
                f + rng.uniform(40, 300),
                float(rng.uniform(0.05, 1)),
                emitter=f,
                width=rng.uniform(0.15, 0.4),
            )
        )
    table = pd.DataFrame(rows)
    for threshold in (0.3, 0.5, 0.8):
        late = postprocess(table, tracks)
        late = pd.DataFrame(late.loc[late["score"] >= threshold]).reset_index(drop=True)
        early = postprocess(
            pd.DataFrame(table.loc[table["score"] >= threshold]), tracks
        )
        pd.testing.assert_frame_equal(
            late.drop(columns=list(COLUMNS)), early.drop(columns=list(COLUMNS))
        )


# ------------------------------------------------------------- without tracks


def test_pair_rule_flags_by_default_and_drops_on_request() -> None:
    table = pd.DataFrame(
        [
            box(5.0, 610, 780, 0.95, emitter=641.0),
            box(5.0, 1250, 1450, 0.8, emitter=1287.0),
            box(9.0, 1250, 1450, 0.8, emitter=1287.0),  # alone: no partner
        ]
    )
    flagged = flag_detections(table)
    assert reasons(flagged) == ["", "", ""]
    assert flagged["harmonic_pair_of"].tolist() == [pd.NA, 0, pd.NA]
    dropped = flag_detections(table, config=PostprocessConfig(trackless="drop"))
    assert reasons(dropped) == ["", "harmonic", ""]
    off = flag_detections(table, config=PostprocessConfig(trackless="off"))
    assert off["harmonic_pair_of"].isna().to_numpy().all()
    # a partner must be at least as confident
    weak = table.assign(score=[0.5, 0.8, 0.8])
    assert flag_detections(weak)["harmonic_pair_of"].isna().to_numpy().all()
    # recordings with tracks use the tracked rule instead
    tracked = flag_detections(table, {"r": constant_tracks({1: 641.0, 2: 1287.0})})
    assert tracked["harmonic_pair_of"].isna().to_numpy().all()
    assert reasons(tracked) == ["", "", ""]


def test_pair_rule_for_box_only_models() -> None:
    # lower edges: fundamental 600 (baseline ~641), copy 1245 (2 x 641 - 37)
    table = pd.DataFrame([box(5.0, 600, 760, 0.95), box(5.0, 1245, 1430, 0.8)])
    assert flag_detections(table)["harmonic_pair_of"].tolist() == [pd.NA, 0]


# ---------------------------------------------------------------- the table


def test_switch_off_and_drop_flagged_renumbers() -> None:
    table = pd.DataFrame(
        [
            box(1.0, 700, 800, 0.9),
            box(1.0, 700, 800, 0.8),  # duplicate of row 0
            box(5.0, 610, 780, 0.95, emitter=641.0),
            box(5.0, 1250, 1450, 0.8, emitter=1287.0),  # pair partner: row 2
        ]
    )
    off = flag_detections(
        table, config=PostprocessConfig(harmonic=False, duplicate=False)
    )
    assert reasons(off) == ["", "", "", ""]
    assert off.loc[:, list(COLUMNS[1:])].isna().to_numpy().all()
    flagged = flag_detections(table)
    assert reasons(flagged) == ["", "duplicate", "", ""]
    assert summarize(flagged) == {
        "n": 4,
        "kept": 3,
        "harmonic": 0,
        "duplicate": 1,
        "harmonic_pair_flagged": 1,
    }
    kept = drop_flagged(flagged)
    assert len(kept) == 3
    assert kept["harmonic_pair_of"].tolist() == [pd.NA, pd.NA, 1]  # was row 2
    assert postprocess(table, keep_dropped=True).equals(flagged)
    # an empty table gets the columns
    empty = flag_detections(table.iloc[:0])
    assert list(empty.columns[-len(COLUMNS) :]) == list(COLUMNS)
    assert len(drop_flagged(empty)) == 0


def test_config_round_trips_and_validates() -> None:
    config = PostprocessConfig(trackless="drop", edge_harmonic_window_hz=(10, 150))
    assert PostprocessConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="unknown"):
        PostprocessConfig.from_dict({"tolerance": 1})
    with pytest.raises(ValueError, match="trackless"):
        PostprocessConfig(trackless="maybe")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lo < hi"):
        PostprocessConfig(edge_fundamental_window_hz=(10, 5))
    with pytest.raises(ValueError, match="pair_min_tiou"):
        PostprocessConfig(pair_min_tiou=0)


# ------------------------------------------------------ a simulated scene


@pytest.fixture(scope="module")
def harmonic_scene(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two fish at 600-700 Hz (second harmonics in band), chirping often."""
    from gridtools.sim.config import ChirpConfig, PopulationConfig
    from gridtools.sim.scene import simulate_scene, write_scene

    root = tmp_path_factory.mktemp("pp_scene")
    library = write_template_library(root / "library")
    config = scene_config(
        library,
        layout=(3, 3),
        population=PopulationConfig(
            n_fish=(2, 2), baseline_uniform_hz=(600.0, 700.0), baseline_uniform_prob=1.0
        ),
        chirps=ChirpConfig(library=str(library), rate_hz=(3.0, 3.0)),
    )
    scene = simulate_scene(replace(config, name="pp"), 11)
    return write_scene(scene, root / "000011", plots=False)


def _true_boxes(scene: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    chirps = pd.read_parquet(scene / "chirps.parquet")
    chirps = chirps[chirps["visible"].astype(bool) & chirps["box_t0_s"].notna()]
    fundamentals = pd.DataFrame(
        {
            "recording": "scene",
            "t_start_s": chirps["box_t0_s"],
            "t_end_s": chirps["box_t1_s"],
            "f_low_hz": chirps["box_f0_hz"],
            "f_high_hz": chirps["box_f1_hz"],
            "score": 0.9,
            "emitter_hz": chirps["baseline_hz"],
            "emitter_id_true": chirps["emitter_id"],
            "t_center_s": chirps["t_center_s"],
        }
    )
    has = chirps["harmonic_box_t0_s"].notna()
    copies = pd.DataFrame(
        {
            "recording": "scene",
            "t_start_s": chirps.loc[has, "harmonic_box_t0_s"],
            "t_end_s": chirps.loc[has, "harmonic_box_t1_s"],
            "f_low_hz": chirps.loc[has, "harmonic_box_f0_hz"],
            "f_high_hz": chirps.loc[has, "harmonic_box_f1_hz"],
            "score": 0.8,
            "emitter_hz": 2 * chirps.loc[has, "baseline_hz"],
            "emitter_id_true": chirps.loc[has, "emitter_id"],
            "t_center_s": chirps.loc[has, "t_center_s"],
        }
    )
    return fundamentals.reset_index(drop=True), copies.reset_index(drop=True)


def test_true_chirps_of_a_simulated_scene_survive(harmonic_scene: Path) -> None:
    fundamentals, copies = _true_boxes(harmonic_scene)
    tracks = {"scene": load_tracks(harmonic_scene)}
    # the scene must contain what is tested: copies and quick successions
    assert len(copies) >= 3
    gaps = [
        np.diff(np.sort(g["t_center_s"].to_numpy()))
        for _, g in fundamentals.groupby("emitter_id_true")
    ]
    assert min(float(g.min()) for g in gaps if len(g)) < 0.3
    table = pd.concat([fundamentals, copies], ignore_index=True)
    table = table.drop(columns="emitter_id_true")
    n = len(fundamentals)
    flagged = flag_detections(table, tracks)
    assert reasons(flagged)[:n] == [""] * n
    assert set(reasons(flagged)[n:]) == {"harmonic"}
    # box-only: no true chirp is dropped either
    edge = flag_detections(table.drop(columns="emitter_hz"), tracks)
    assert reasons(edge)[:n] == [""] * n
