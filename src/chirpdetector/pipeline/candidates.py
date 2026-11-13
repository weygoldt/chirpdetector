"""Validation candidates of a run, for ``chirpdetector label-emitters --candidates``.

A stratified sample of the run's detections, so that the precision and the
assignment of the production output can be audited by hand:

- score x ΔF cells: score bins (`SCORE_EDGES`, from the store threshold up,
  so detections below the operating threshold are audited too) crossed with
  ΔF bins of the nearest track to its closest live neighbour (``single``
  with fewer than two live tracks, ``unknown`` without emitter or tracks);
  ``candidates_per_cell`` per cell;
- detections at or above the operating threshold that the decoder flagged
  (``edge``, ``band``, ``size``, ``uncertain``): 2 x ``candidates_per_cell``
  per flag;
- detections that post-processing flagged (``harmonic``, ``duplicate``):
  3 x ``candidates_per_cell`` per reason;
- harmonic copies re-attributed to their fish: up to 5 x
  ``candidates_per_cell``. Their box is the table's, at the fundamental
  (where the fish's missed chirp should be); the detected box at the
  harmonic is in ``candidates_strata.csv`` (``pp_harmonic_*``).

Within a stratum every recording is equally likely (rows are weighted by
the inverse of their recording's count in the stratum), so dense
recordings do not crowd out the others. Sampling is seeded.

``candidates.csv`` has the labeller's columns only: ``recording`` (path of
the recording directory), ``t_start_s``, ``t_end_s``, ``f_low_hz``,
``f_high_hz``, ``candidate_id`` (``<recording>:<chirp_id>``), ``score``,
``emitter_hz``, ``track_id`` (assigned track, empty if unassigned),
``source`` (``pipeline:<run>:<stratum>``) and ``tracks`` (directory of the
wavetracker arrays). ``candidates_strata.csv`` repeats the candidate ids
with the stratum and the assignment fields, for the evaluation.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chirpdetector.pipeline.config import PipelineConfig
from chirpdetector.pipeline.outputs import read_chirps

__all__ = [
    "CANDIDATE_COLUMNS",
    "DF_EDGES_HZ",
    "SCORE_EDGES",
    "df_bin",
    "write_candidates",
]

log = logging.getLogger(__name__)

CANDIDATE_COLUMNS = (
    "recording",
    "t_start_s",
    "t_end_s",
    "f_low_hz",
    "f_high_hz",
    "candidate_id",
    "score",
    "emitter_hz",
    "track_id",
    "source",
    "tracks",
)

SCORE_EDGES = (0.2, 0.35, 0.5, 0.65, 0.8, 0.9)
"""Inner edges of the score bins (the first starts at the store threshold)."""

DF_EDGES_HZ = (0.0, 10.0, 25.0, 50.0, 100.0)
"""Lower edges of the ΔF bins (the last is open)."""


def df_bin(table: pd.DataFrame) -> np.ndarray:
    """ΔF bin of every detection (module docstring)."""
    dfo = table["track_df_other_hz"].to_numpy(dtype=float)
    n_live = table["n_live_tracks"].to_numpy(dtype=np.int64)
    status = table["assign_status"].to_numpy(dtype=object)
    names = [f"{int(a)}-{int(b)}" for a, b in itertools.pairwise(DF_EDGES_HZ)] + [
        f"{int(DF_EDGES_HZ[-1])}+"
    ]
    idx = np.clip(np.digitize(np.nan_to_num(dfo), DF_EDGES_HZ) - 1, 0, len(names) - 1)
    out = np.asarray(names, dtype=object)[idx]
    out = np.where(n_live < 2, "single", out)
    out = np.where(np.isin(status, ["no_emitter", "no_tracks"]), "unknown", out)
    return out.astype(object)


def _score_bin(score: np.ndarray, store: float) -> np.ndarray:
    edges = (store, *SCORE_EDGES, 1.0)
    names = [f"{a:.2f}-{b:.2f}" for a, b in itertools.pairwise(edges)]
    idx = np.clip(np.digitize(score, edges[1:-1]), 0, len(names) - 1)
    return np.asarray(names, dtype=object)[idx]


def _sample(
    pool: pd.DataFrame, n: int, rng: np.random.Generator, taken: set[str]
) -> pd.DataFrame:
    pool = pool.loc[~pool["candidate_id"].isin(list(taken))]
    if not len(pool) or n <= 0:
        return pool.iloc[:0]
    counts = pool.groupby("recording_name")["recording_name"].transform("size")
    w = 1.0 / counts.to_numpy(dtype=float)
    k = min(n, len(pool))
    idx = rng.choice(len(pool), size=k, replace=False, p=w / w.sum())
    return pool.iloc[np.sort(idx)]


def write_candidates(run_dir: Path, config: PipelineConfig) -> dict[str, Any]:
    """Sample the run's complete recordings into ``candidates.csv``."""
    tables = []
    for meta_path in sorted(run_dir.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("status") != "complete":
            continue
        chirps = meta_path.parent / meta["outputs"]["chirps"]["file"]
        table, _ = read_chirps(chirps)
        if not len(table):
            continue
        table["recording_name"] = meta["recording"]
        table["recording_dir"] = meta["inputs"]["recording_dir"]
        table["tracks_dir"] = (meta.get("tracks") or {}).get("dir") or ""
        tables.append(table)
    if not tables:
        log.warning("no complete recordings: no validation candidates")
        return {"n": 0}
    df = pd.concat(tables, ignore_index=True)
    df["candidate_id"] = df["recording_name"] + ":" + df["chirp_id"].astype(str)
    df["score_bin"] = _score_bin(
        df["score"].to_numpy(dtype=float), config.detection.store_threshold
    )
    df["df_bin"] = df_bin(df)
    rng = np.random.default_rng(config.validation.seed)
    k = config.validation.candidates_per_cell
    taken: set[str] = set()
    parts = []

    def add(pool: pd.DataFrame, n: int, stratum: str) -> None:
        got = _sample(pool, n, rng, taken).assign(stratum=stratum)
        taken.update(got["candidate_id"])
        parts.append(got)

    kept = df["kept"].to_numpy(dtype=bool)
    for (sb, db), cell in df.loc[kept].groupby(["score_bin", "df_bin"], sort=True):
        add(cell, k, f"score={sb}|df={db}")
    operating = df["operating"].to_numpy(dtype=bool)
    if "params_flag" in df.columns:
        flags = df["params_flag"].astype(str)
        for flag in ("edge", "band", "size", "uncertain"):
            add(
                df.loc[operating & kept & flags.str.contains(flag).to_numpy()],
                2 * k,
                f"decoder={flag}",
            )
    reasons = df["pp_drop_reason"].astype(str)
    for reason in ("harmonic", "duplicate"):
        add(df.loc[(reasons == reason).to_numpy()], 3 * k, f"postprocess={reason}")
    if "pp_reattributed" in df.columns:
        again = df["pp_reattributed"].astype(str).to_numpy() == "harmonic"
        add(df.loc[again], 5 * k, "postprocess=reattributed")
    sample = pd.concat(parts, ignore_index=True)
    out = pd.DataFrame(
        {
            "recording": sample["recording_dir"],
            "t_start_s": sample["t_start_s"],
            "t_end_s": sample["t_end_s"],
            "f_low_hz": sample["f_low_hz"],
            "f_high_hz": sample["f_high_hz"],
            "candidate_id": sample["candidate_id"],
            "score": sample["score"],
            "emitter_hz": sample["emitter_hz"],
            "track_id": sample["track_id"],
            "source": f"pipeline:{config.name}:" + sample["stratum"],
            "tracks": sample["tracks_dir"],
        }
    ).loc[:, list(CANDIDATE_COLUMNS)]
    out.to_csv(run_dir / "candidates.csv", index=False)
    strata_cols = [
        "candidate_id",
        "stratum",
        "score_bin",
        "df_bin",
        "operating",
        "kept",
        "pp_drop_reason",
        "assign_status",
        "assign_confidence",
        "nearest_track_id",
        "nearest_df_hz",
        "second_track_id",
        "second_df_hz",
        "track_df_other_hz",
        "n_live_tracks",
        "pp_reattributed",
        "pp_harmonic_of_track",
        "pp_harmonic_f_low_hz",
        "pp_harmonic_f_high_hz",
        "pp_harmonic_emitter_hz",
    ]
    strata_cols = [c for c in strata_cols if c in sample.columns]
    if "params_flag" in sample.columns:
        strata_cols.append("params_flag")
    sample.loc[:, strata_cols].to_csv(run_dir / "candidates_strata.csv", index=False)
    counts = sample["stratum"].str.split("=").str[0].value_counts().to_dict()
    log.info("validation candidates: %d -> %s", len(out), run_dir / "candidates.csv")
    return {
        "n": len(out),
        "file": "candidates.csv",
        "by_kind": {str(a): int(b) for a, b in counts.items()},
    }
