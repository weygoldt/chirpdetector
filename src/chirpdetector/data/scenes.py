"""Detector datasets from simulated scenes (``gridtools sim run``).

Every scene becomes one source and one group (its id, ``<name>-<seed>``),
of kind ``hybrid`` when it has a real background segment (``scene.json``
``background.used``), else ``simulation``. Splits go by seed or by the
directory a scene lies in (`SplitRule`). Its recording is cleaned of clicks exactly as inference will clean it
(`ClickRemoval` when `can_remove_clicks` allows it for the scene's
electrode count and rate, otherwise not at all, so that the detector learns
to reject the clicks it will see), its spectrogram is computed once with
the dataset's front end, and the spectrogram is cut into tiles that cover
the scene end to end (`TileLayout`); each tile is normalised on its own.

Labels
------
Boxes are the chirps' footprints (plan decision 7): the scene's own
``box_*`` columns when its footprint front end equals the dataset's
reference channel (`footprints_match`), otherwise recomputed with
`FrontEndSpectrogram` on the scene's render model. Per tile:

* visible chirps are positives, clipped to the tile, if at least
  ``edge_fraction`` (70 %) of their box's duration and their
  ``t_center_s`` lie in the tile, else ignore regions (``reason="edge"``);
* invisible chirps are ignore regions (``"invisible"``);
* second-harmonic footprints in the band are ignore regions
  (``"harmonic"``).

Every row carries ``emitter_hz`` (the emitter's baseline at the chirp
centre), ``emitter_id`` (its track id), ``live_tracks_hz`` (all fish's
baselines at the chirp centre from the scene's tracks, including the real,
non-chirping fish of a hybrid background, idents >= 1000),
``min_df_live_hz`` (ΔF to the nearest of those other fish), and the
scene's per-chirp SNR, family, excursion, duration, ΔF to the nearest
simulated fish, overlap flag and template id.

Workers compute the tiles of one scene each and return images and box
tables; the parent process alone writes through `DatasetWriter` (which is
not process-safe).
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import resource
import time
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from chirpdetector.data.contract import SPLITS, Dataset, DatasetWriter, Split
from chirpdetector.data.footprint import FrontEndSpectrogram, footprints_match
from chirpdetector.data.tiles import (
    EDGE_FRACTION,
    TileLayout,
    assign_to_tile,
    render_tile,
)
from chirpdetector.preprocessing import ClickRemoval, can_remove_clicks, remove_clicks
from chirpdetector.spectrogram import (
    ImageGeometry,
    Spectrogram,
    SpectrogramConfig,
    compute_spectrogram,
)

__all__ = [
    "DEFAULT_SPLIT",
    "SceneDatasetConfig",
    "SceneTiles",
    "SplitRule",
    "build_scene_dataset",
    "find_scenes",
    "scene_boxes",
    "scene_id",
    "tile_scene",
]

log = logging.getLogger(__name__)

Device = Literal["cpu", "cuda", "auto"]

_SCENE_FILES = ("scene.json", "chirps.csv", "recording.wav", "times.npy", "fund_v.npy")
_EXTRA_COLUMNS = (
    "chirp_id",
    "snr_db",
    "snr_baseline_db",
    "snr_excursion_db",
    "visible_fraction",
    "family",
    "excursion_hz",
    "duration_s",
    "min_df_hz",
    "overlaps_other_chirp",
    "template_id",
    "truncated",
)
_BOX = ("box_t0_s", "box_t1_s", "box_f0_hz", "box_f1_hz")
_HARMONIC_BOX = (
    "harmonic_box_t0_s",
    "harmonic_box_t1_s",
    "harmonic_box_f0_hz",
    "harmonic_box_f1_hz",
)


# ------------------------------------------------------------------ splits


@dataclass(frozen=True)
class SplitRule:
    """Which split each scene goes to.

    Exactly one of ``ranges`` (split -> half-open seed range ``[lo, hi)``;
    every seed must fall in one), ``fractions`` (split -> fraction; seeds
    are shuffled with ``seed`` and cut in the order train, val, test) or
    ``roots`` (split -> directory; a scene belongs to the split whose
    directory contains it, e.g. scenes whose backgrounds come from
    disjoint recordings per split).
    """

    ranges: Mapping[str, tuple[int, int]] | None = None
    fractions: Mapping[str, float] | None = None
    seed: int = 0
    roots: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        given = [x is not None for x in (self.ranges, self.fractions, self.roots)]
        if sum(given) != 1:
            msg = "give exactly one of seed ranges, fractions or split roots"
            raise ValueError(msg)
        names = set(self.ranges or self.fractions or self.roots or {})
        unknown = sorted(names - set(SPLITS))
        if unknown:
            msg = f"unknown splits {unknown}; allowed {SPLITS}"
            raise ValueError(msg)
        if self.ranges is not None:
            spans = sorted((lo, hi) for lo, hi in self.ranges.values())
            if any(lo >= hi for lo, hi in spans):
                msg = f"empty seed range in {dict(self.ranges)}"
                raise ValueError(msg)
            if any(a[1] > b[0] for a, b in itertools.pairwise(spans)):
                msg = f"seed ranges overlap: {dict(self.ranges)}"
                raise ValueError(msg)
        if self.fractions is not None:
            values = list(self.fractions.values())
            if any(not 0 <= v <= 1 for v in values) or not math.isclose(
                sum(values), 1.0, abs_tol=1e-6
            ):
                msg = (
                    f"fractions must lie in [0, 1] and sum to 1: {dict(self.fractions)}"
                )
                raise ValueError(msg)
        if self.roots is not None:
            paths = [Path(p).resolve() for p in self.roots.values()]
            for a, b in itertools.permutations(paths, 2):
                if a.is_relative_to(b):
                    msg = f"split roots must not contain each other: {dict(self.roots)}"
                    raise ValueError(msg)

    @classmethod
    def from_roots(cls, items: Iterable[str]) -> SplitRule:
        """Parse ``split=DIRECTORY`` items (one per split)."""
        roots: dict[str, str] = {}
        for item in items:
            name, _, path = item.partition("=")
            name = name.strip()
            if not path or name in roots:
                msg = f"expected one split=DIRECTORY per split, got {item!r}"
                raise ValueError(msg)
            roots[name] = path.strip()
        return cls(roots=roots)

    @classmethod
    def parse(cls, text: str, *, seed: int = 0) -> SplitRule:
        """Parse ``train=0:1600,val=1600:1800,test=1800:2000`` or ``train=0.8,...``."""
        items = {}
        for part in text.split(","):
            name, _, value = part.strip().partition("=")
            if not value:
                msg = f"expected split=value, got {part!r}"
                raise ValueError(msg)
            items[name.strip()] = value.strip()
        if all(":" in v for v in items.values()):
            ranges = {}
            for name, value in items.items():
                lo, hi = value.split(":")
                ranges[name] = (int(lo), int(hi))
            return cls(ranges=ranges, seed=seed)
        if any(":" in v for v in items.values()):
            msg = f"mix of seed ranges and fractions in {text!r}"
            raise ValueError(msg)
        return cls(fractions={k: float(v) for k, v in items.items()}, seed=seed)

    def assign_scenes(self, seeds: Mapping[Path, int]) -> dict[Path, Split]:
        """Split of every scene directory (scene dir -> seed given)."""
        if self.roots is None:
            by_seed = self.assign(seeds.values())
            return {d: by_seed[int(s)] for d, s in seeds.items()}
        roots = {k: Path(v).resolve() for k, v in self.roots.items()}
        out: dict[Path, Split] = {}
        for d in seeds:
            hits = [
                k for k, root in roots.items() if Path(d).resolve().is_relative_to(root)
            ]
            if len(hits) != 1:
                msg = f"scene {d} lies in no split root {dict(self.roots)}"
                raise ValueError(msg)
            out[d] = cast("Split", hits[0])
        return out

    def assign(self, seeds: Iterable[int]) -> dict[int, Split]:
        """Split of every seed (seed ranges or fractions)."""
        if self.roots is not None:
            msg = "split roots assign scene directories, not seeds; use assign_scenes"
            raise ValueError(msg)
        seeds = sorted(set(int(s) for s in seeds))
        out: dict[int, Split] = {}
        if self.ranges is not None:
            for s in seeds:
                hits = [k for k, (lo, hi) in self.ranges.items() if lo <= s < hi]
                if not hits:
                    msg = f"scene seed {s} lies in no split range {dict(self.ranges)}"
                    raise ValueError(msg)
                out[s] = cast("Split", hits[0])
            return out
        fractions = self.fractions or {}
        order = np.random.default_rng(self.seed).permutation(len(seeds))
        shuffled = [seeds[i] for i in order]
        start = 0
        names = [s for s in SPLITS if fractions.get(s, 0.0) > 0]
        for k, name in enumerate(names):
            if k == len(names) - 1:
                stop = len(shuffled)
            else:
                # at least one seed per split, while there are enough
                left = len(names) - 1 - k
                stop = start + round(fractions[name] * len(shuffled))
                stop = max(start + 1, min(stop, len(shuffled) - left))
                stop = min(stop, len(shuffled))
            for s in shuffled[start:stop]:
                out[s] = cast("Split", name)
            start = stop
        return out

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form (for the provenance)."""
        if self.ranges is not None:
            return {"ranges": {k: list(v) for k, v in self.ranges.items()}}
        if self.roots is not None:
            return {"roots": {k: str(Path(v).resolve()) for k, v in self.roots.items()}}
        return {"fractions": dict(self.fractions or {}), "seed": self.seed}


DEFAULT_SPLIT = SplitRule(
    ranges={"train": (0, 1600), "val": (1600, 1800), "test": (1800, 2000)}
)
"""Seeds 0-1599 train, 1600-1799 val, 1800-1999 test."""


# ------------------------------------------------------------------ config


@dataclass(frozen=True)
class SceneDatasetConfig:
    """How simulated scenes become a detector dataset.

    Parameters
    ----------
    spectrogram
        Front end of the images (default: chirpdetector's default, 3
        windows of 0.2048/0.0512/0.0128 s, hop 5 ms, 500-1500 Hz).
    layout
        Tile duration and overlap (default 3.2 s = 640 frames, 1.0 s).
    click_removal
        Click removal applied where `can_remove_clicks` allows it (as
        inference does); None disables it everywhere.
    edge_fraction
        Fraction of a box's duration that must lie in a tile for a cut
        chirp to stay a positive.
    harmonic_ignores
        Add second-harmonic footprints as ignore regions.
    split
        Group -> split rule.
    workers
        Processes computing scenes; 0 or 1 computes in this process.
    threads_per_worker
        Torch threads of each worker.
    device
        Device of the STFT: ``cpu``, ``cuda`` or ``auto`` (cuda if
        available). On CUDA each worker limits itself to
        ``CHIRP_MAX_VRAM_FRACTION / workers`` of the GPU memory and falls
        back to the CPU for a scene that does not fit.
    """

    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    layout: TileLayout = field(default_factory=TileLayout)
    click_removal: ClickRemoval | None = field(default_factory=ClickRemoval)
    edge_fraction: float = EDGE_FRACTION
    harmonic_ignores: bool = True
    split: SplitRule = DEFAULT_SPLIT
    workers: int = 12
    threads_per_worker: int = 2
    device: Device = "cpu"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form (for the provenance)."""
        return {
            "spectrogram": self.spectrogram.to_dict(),
            "tile_duration_s": self.layout.tile_duration_s,
            "overlap_s": self.layout.overlap_s,
            "click_removal": None
            if self.click_removal is None
            else self.click_removal.to_dict(),
            "edge_fraction": self.edge_fraction,
            "harmonic_ignores": self.harmonic_ignores,
            "split": self.split.to_dict(),
            "workers": self.workers,
            "threads_per_worker": self.threads_per_worker,
            "device": self.device,
        }


# ------------------------------------------------------------------ scenes


def scene_id(info: Mapping[str, Any]) -> str:
    """Source and group id of a scene: ``<name>-<seed, 6 digits>``."""
    return f"{info['name']}-{int(info['seed']):06d}"


def find_scenes(pattern: str | Path) -> list[Path]:
    """Scene directories matching a glob (or a directory of scenes), by name.

    Raises
    ------
    FileNotFoundError
        If nothing matches, or a matching directory is incomplete (no
        ``scene.json``, which ``gridtools sim run`` writes last, or another
        required file).
    """
    path = Path(pattern)
    if path.is_dir() and not (path / "scene.json").exists():
        candidates = sorted(p for p in path.iterdir() if p.is_dir())
    elif path.is_dir():
        candidates = [path]
    else:
        anchor = Path(path.anchor) if path.is_absolute() else Path()
        relative = path.relative_to(anchor) if path.is_absolute() else path
        candidates = sorted(p for p in anchor.glob(str(relative)) if p.is_dir())
    if not candidates:
        msg = f"no scene directories match {pattern}"
        raise FileNotFoundError(msg)
    incomplete = [
        str(p) for p in candidates if not all((p / f).exists() for f in _SCENE_FILES)
    ]
    if incomplete:
        msg = (
            f"{len(incomplete)} incomplete scene directories (missing one of "
            f"{_SCENE_FILES}), e.g. {incomplete[:3]}"
        )
        raise FileNotFoundError(msg)
    return candidates


def _scene_info(scene_dir: Path) -> dict[str, Any]:
    return json.loads((scene_dir / "scene.json").read_text())


def _recompute_footprints(
    scene_dir: Path,
    chirps: pd.DataFrame,
    info: Mapping[str, Any],
    config: SpectrogramConfig,
    recording: NDArray[np.float32],
    device: str,
) -> pd.DataFrame:
    """Footprint columns of ``chirps`` recomputed with the dataset's front end."""
    from gridtools.sim.footprint import scene_footprints

    fp = dict(info.get("footprint") or {})
    footprints = scene_footprints(
        scene_dir,
        FrontEndSpectrogram(config, device=device),
        chirp_ids=chirps["chirp_id"].astype(int).tolist(),
        threshold_db=float(fp.get("threshold_db", 30.0)),
        visible_snr_db=float(fp.get("visible_snr_db", 0.0)),
        margin_s=float(fp.get("margin_s", 0.05)),
        harmonic_band=(config.f_min_hz, config.f_max_hz),
        recording=recording,
        background=str(fp.get("background", "others")),
    )
    out = chirps.copy()
    for name, k in zip(_BOX, range(4), strict=True):
        out[name] = [f.box[k] for f in footprints]
    for name, k in zip(_HARMONIC_BOX, range(4), strict=True):
        out[name] = [f.harmonic_box[k] for f in footprints]
    for name in ("snr_db", "snr_baseline_db", "snr_excursion_db", "visible_fraction"):
        out[name] = [getattr(f, name) for f in footprints]
    out["visible"] = [bool(f.visible) for f in footprints]
    return out


def scene_boxes(
    chirps: pd.DataFrame,
    tracks: Any,
    *,
    harmonic_ignores: bool = True,
) -> pd.DataFrame:
    """Candidate boxes of a scene's chirps (before tiling).

    Parameters
    ----------
    chirps
        The scene's ``chirps.csv`` with footprint columns.
    tracks
        The scene's `gridtools.io.Tracks` (ground-truth baselines).
    harmonic_ignores
        Also return second-harmonic footprints (label ``"harmonic"``).

    Returns
    -------
    pandas.DataFrame
        One row per fundamental footprint (label ``"chirp"`` if visible,
        else ``"invisible"``) and per harmonic footprint, with the box
        edges, ``t_center_s``, ``label``, ``emitter_hz``, ``emitter_id``,
        ``live_tracks_hz`` and the extras. Chirps without a footprint in
        the band (NaN box) are left out.
    """
    columns = [
        "t_start_s",
        "t_end_s",
        "f_low_hz",
        "f_high_hz",
        "t_center_s",
        "label",
        "emitter_hz",
        "emitter_id",
        "live_tracks_hz",
        *_EXTRA_COLUMNS,
        "min_df_live_hz",
    ]
    if len(chirps) == 0:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in columns})
    centres = chirps["t_center_s"].to_numpy(dtype=np.float64)
    track_list = list(tracks)
    per_track = [t.frequency_at(centres) for t in track_list]
    live = np.stack(per_track, axis=1) if per_track else np.zeros((len(centres), 0))
    live_lists = [[float(f) for f in row if np.isfinite(f) and f > 0] for row in live]
    # ΔF to the nearest other live fish (simulated or real) at the centre
    idents = np.array([int(t.ident) for t in track_list], dtype=np.int64)
    emitters = chirps["emitter_id"].to_numpy(dtype=np.int64)
    baselines = chirps["baseline_hz"].to_numpy(dtype=np.float64)
    others = np.where(idents[None, :] != emitters[:, None], live, np.nan)
    df = np.abs(others - baselines[:, None])
    has = np.isfinite(df).any(axis=1)
    min_df_live = np.full(len(centres), np.nan)
    min_df_live[has] = np.nanmin(df[has], axis=1)
    base = {
        "t_center_s": centres,
        "emitter_hz": chirps["baseline_hz"].to_numpy(dtype=np.float64),
        "emitter_id": chirps["emitter_id"].to_numpy(dtype=np.int64),
        "live_tracks_hz": live_lists,
        **{c: chirps[c].to_numpy() for c in _EXTRA_COLUMNS},
        "min_df_live_hz": min_df_live,
    }
    fundamental = pd.DataFrame(
        {
            "t_start_s": chirps["box_t0_s"].to_numpy(dtype=np.float64),
            "t_end_s": chirps["box_t1_s"].to_numpy(dtype=np.float64),
            "f_low_hz": chirps["box_f0_hz"].to_numpy(dtype=np.float64),
            "f_high_hz": chirps["box_f1_hz"].to_numpy(dtype=np.float64),
            "label": np.where(
                chirps["visible"].to_numpy(dtype=bool), "chirp", "invisible"
            ),
            **base,
        }
    )
    parts = [fundamental]
    if harmonic_ignores:
        parts.append(
            pd.DataFrame(
                {
                    "t_start_s": chirps["harmonic_box_t0_s"].to_numpy(dtype=np.float64),
                    "t_end_s": chirps["harmonic_box_t1_s"].to_numpy(dtype=np.float64),
                    "f_low_hz": chirps["harmonic_box_f0_hz"].to_numpy(dtype=np.float64),
                    "f_high_hz": chirps["harmonic_box_f1_hz"].to_numpy(
                        dtype=np.float64
                    ),
                    "label": "harmonic",
                    **base,
                }
            )
        )
    boxes = pd.concat(parts, ignore_index=True)
    edges = boxes[["t_start_s", "t_end_s", "f_low_hz", "f_high_hz"]].to_numpy()
    finite = np.isfinite(edges).all(axis=1)
    boxes = boxes.loc[finite].reset_index(drop=True)
    return boxes.loc[:, columns]


@dataclass
class SceneTiles:
    """Images and boxes of one scene, ready for the writer."""

    source_id: str
    seed: int
    scene_dir: str
    kind: Literal["simulation", "hybrid"]
    samplerate_hz: float
    source_extra: dict[str, Any]
    images: list[tuple[str, NDArray[np.uint8], ImageGeometry, dict[str, Any]]]
    chirps: list[pd.DataFrame]
    ignore: list[pd.DataFrame]
    stats: dict[str, Any]


def _spectrogram(
    samples: NDArray[np.float32], fs: float, config: SpectrogramConfig, device: str
) -> tuple[Spectrogram, str]:
    if device == "cuda":
        import torch

        try:
            return compute_spectrogram(samples, fs, config, device="cuda"), "cuda"
        except torch.OutOfMemoryError:
            log.warning("STFT does not fit on the GPU; computing it on the CPU")
            torch.cuda.empty_cache()
    return compute_spectrogram(samples, fs, config), "cpu"


def tile_scene(
    scene_dir: str | Path,
    config: SceneDatasetConfig | None = None,
    *,
    device: str = "cpu",
) -> SceneTiles:
    """Images, chirp boxes and ignore regions of every tile of one scene."""
    from gridtools.io import load_tracks, open_recording

    cfg = SceneDatasetConfig() if config is None else config
    t_start = time.perf_counter()
    scene_dir = Path(scene_dir).resolve()
    info = _scene_info(scene_dir)
    source = scene_id(info)
    recording = open_recording(scene_dir / "recording.wav")
    fs = float(recording.samplerate_hz)
    raw = recording.read(0, recording.n_samples)
    n_electrodes = raw.shape[1]

    timings: dict[str, float] = {}
    t = time.perf_counter()
    applied = cfg.click_removal is not None and can_remove_clicks(
        n_electrodes, cfg.click_removal, samplerate_hz=fs
    )
    repaired = 0.0
    if applied:
        samples, mask = remove_clicks(raw, fs, cfg.click_removal)
        repaired = float(mask.mean())
    else:
        samples = raw
    timings["clicks_s"] = time.perf_counter() - t

    t = time.perf_counter()
    chirps = pd.read_csv(scene_dir / "chirps.csv")
    match = footprints_match(info.get("footprint") or {}, cfg.spectrogram)
    if not match:
        chirps = _recompute_footprints(
            scene_dir, chirps, info, cfg.spectrogram, raw, device
        )
    tracks = load_tracks(scene_dir)
    boxes = scene_boxes(chirps, tracks, harmonic_ignores=cfg.harmonic_ignores)
    timings["labels_s"] = time.perf_counter() - t
    del raw

    t = time.perf_counter()
    spec, used = _spectrogram(samples, fs, cfg.spectrogram, device)
    del samples
    timings["stft_s"] = time.perf_counter() - t

    t = time.perf_counter()
    tiling = cfg.layout.tiling(cfg.spectrogram)
    starts = cfg.layout.starts(spec.n_times, cfg.spectrogram)
    images, chirp_tables, ignore_tables = [], [], []
    positive_ids: set[int] = set()
    reasons: dict[str, int] = {}
    t0s = boxes["t_start_s"].to_numpy(dtype=np.float64)
    t1s = boxes["t_end_s"].to_numpy(dtype=np.float64)
    for index, start in enumerate(starts):
        image, geometry = render_tile(spec, start, tiling)
        lo, hi = geometry.time_extent_s
        near = boxes.loc[(t0s < hi) & (t1s > lo)]
        tile_chirps, tile_ignore = assign_to_tile(
            near, geometry, edge_fraction=cfg.edge_fraction
        )
        image_id = f"{source}_f{spec.frame_start + start:07d}"
        extra = {"frame_start": int(spec.frame_start + start), "tile": index}
        images.append((image_id, image, geometry, extra))
        chirp_tables.append(tile_chirps)
        ignore_tables.append(tile_ignore)
        positive_ids.update(int(c) for c in tile_chirps["chirp_id"])
        for reason, n in tile_ignore["reason"].value_counts().items():
            reasons[str(reason)] = reasons.get(str(reason), 0) + int(n)
    timings["tiles_s"] = time.perf_counter() - t
    del spec

    visible = chirps["visible"].to_numpy(dtype=bool)
    no_box = ~np.isfinite(chirps[list(_BOX)].to_numpy(dtype=np.float64)).all(axis=1)
    ids = chirps["chirp_id"].to_numpy(dtype=np.int64)
    never = sorted(set(ids[visible & ~no_box].tolist()) - positive_ids)
    timings["total_s"] = time.perf_counter() - t_start
    stats = {
        "n_chirps": len(chirps),
        "n_visible": int(visible.sum()),
        "n_no_footprint": int(no_box.sum()),
        "visible_never_positive": never,
        "n_tiles": len(starts),
        "n_positive_boxes": int(sum(len(c) for c in chirp_tables)),
        "ignore_reasons": reasons,
        "timings": timings,
        "device": used,
        "footprints": "scene" if match else "recomputed",
    }
    background = dict(info.get("background") or {})
    hybrid = bool(background.get("used", False))
    real_fish = list(info.get("real_fish") or [])
    source_extra = {
        "scene_dir": str(scene_dir),
        "seed": int(info["seed"]),
        "duration_s": float(info["duration_s"]),
        "n_electrodes": int(n_electrodes),
        "click_removal_applied": bool(applied),
        "click_repaired_fraction": repaired,
        "n_clicks_simulated": int((info.get("clicks") or {}).get("n_clicks", -1)),
        "footprints": stats["footprints"],
        "n_chirps": len(chirps),
        "n_visible": int(visible.sum()),
        "background": {
            k: background.get(k)
            for k in ("segment_id", "dataset", "recording", "split_group")
        }
        if hybrid
        else None,
        "real_fish_idents": [int(f["ident"]) for f in real_fish],
    }
    return SceneTiles(
        source_id=source,
        seed=int(info["seed"]),
        scene_dir=str(scene_dir),
        kind="hybrid" if hybrid else "simulation",
        samplerate_hz=fs,
        source_extra=source_extra,
        images=images,
        chirps=chirp_tables,
        ignore=ignore_tables,
        stats=stats,
    )


# ------------------------------------------------------------------ build


_WORKER: dict[str, Any] = {}


def _init_worker(threads: int, device: str, vram_fraction: float) -> None:
    import torch

    torch.set_num_threads(max(1, threads))
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(vram_fraction)
    _WORKER["device"] = device


def _work(scene_dir: str, config: SceneDatasetConfig) -> SceneTiles:
    return tile_scene(scene_dir, config, device=_WORKER.get("device", "cpu"))


def _resolve_device(device: Device) -> str:
    import torch

    if device == "cpu":
        return "cpu"
    available = torch.cuda.is_available()
    if device == "cuda" and not available:
        msg = "device='cuda' but no CUDA device is available"
        raise RuntimeError(msg)
    return "cuda" if available else "cpu"


def _git_sha(path: Path) -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def builder_provenance() -> dict[str, Any]:
    """Versions and git SHAs of chirpdetector and gridtools."""
    import platform
    from datetime import UTC, datetime

    import gridtools

    import chirpdetector

    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "chirpdetector": {
            "version": chirpdetector.__version__,
            "git": _git_sha(Path(chirpdetector.__file__).parent),
        },
        "gridtools": {
            "version": getattr(gridtools, "__version__", None),
            "git": _git_sha(Path(gridtools.__file__).parent),
        },
    }


def _max_rss_mb() -> dict[str, float]:
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return {"parent_mb": self_kb / 1024, "largest_worker_mb": child_kb / 1024}


def build_scene_dataset(
    scene_dirs: Sequence[str | Path],
    out: str | Path,
    config: SceneDatasetConfig | None = None,
    *,
    overwrite: bool = False,
    diagnostics: bool = True,
    provenance: Mapping[str, Any] | None = None,
) -> Dataset:
    """Build a detector dataset from simulated scenes.

    Parameters
    ----------
    scene_dirs
        Scene directories (see `find_scenes`).
    out
        Output directory (must not exist or be empty, unless
        ``overwrite``).
    config
        Front end, tiling, labelling, split and parallelism.
    overwrite
        Replace an existing dataset.
    diagnostics
        Write a contact sheet and per-split counts to ``out/diagnostics``.
    provenance
        Extra provenance (e.g. the command line), merged into the
        manifest's.

    Returns
    -------
    Dataset
        The written, validated dataset.
    """
    cfg = SceneDatasetConfig() if config is None else config
    dirs = sorted({Path(d).resolve() for d in scene_dirs})
    if not dirs:
        msg = "no scenes given"
        raise ValueError(msg)
    infos = {d: _scene_info(d) for d in dirs}
    ids = {d: scene_id(info) for d, info in infos.items()}
    if len(set(ids.values())) != len(ids):
        msg = "two scene directories have the same name and seed"
        raise ValueError(msg)
    seeds = {d: int(info["seed"]) for d, info in infos.items()}
    splits = cfg.split.assign_scenes(seeds)
    device = _resolve_device(cfg.device)
    workers = max(1, int(cfg.workers))
    started = time.perf_counter()

    writer = DatasetWriter(
        out,
        spectrogram=cfg.spectrogram,
        tiling=cfg.layout.tiling(cfg.spectrogram),
        preprocessing={
            "click_removal": None
            if cfg.click_removal is None
            else cfg.click_removal.to_dict()
        },
        provenance={
            "builder": "chirpdetector.data.scenes.build_scene_dataset",
            **builder_provenance(),
            "config": cfg.to_dict(),
            "labels": {
                "boxes": "chirp footprints (gridtools.sim.footprint), decision 7",
                "positive": "visible chirps with >= edge_fraction of the box "
                "duration and t_center_s in the tile, clipped to the tile",
                "ignore_reasons": {
                    "invisible": "footprint SNR below the scene's visible_snr_db",
                    "edge": "visible chirp cut by the tile edge (rule above)",
                    "harmonic": "second-harmonic footprint in the band",
                },
                "emitter_hz": "baseline_hz of the emitter at t_center_s",
                "live_tracks_hz": "all ground-truth tracks at t_center_s",
            },
            "scenes": {
                "n": len(dirs),
                "roots": sorted({str(d.parent) for d in dirs}),
                "seeds": [min(seeds.values()), max(seeds.values())],
            },
            **dict(provenance or {}),
        },
        overwrite=overwrite,
    )
    totals: dict[str, Any] = {
        "devices": {},
        "footprints": {},
        "visible_never_positive": {},
        "timings": {},
    }

    def write(result: SceneTiles) -> None:
        writer.add_source(
            result.source_id,
            group=result.source_id,
            kind=result.kind,
            tracks=result.scene_dir,
            samplerate_hz=result.samplerate_hz,
            extra=result.source_extra,
        )
        writer.assign_split(result.source_id, splits[Path(result.scene_dir)])
        for (image_id, image, geometry, extra), chirps, ignore in zip(
            result.images, result.chirps, result.ignore, strict=True
        ):
            writer.add_image(
                image_id,
                image,
                geometry,
                source=result.source_id,
                chirps=chirps,
                ignore=ignore,
                extra=extra,
            )
        stats = result.stats
        for key in ("device", "footprints"):
            bucket = totals["devices" if key == "device" else "footprints"]
            bucket[stats[key]] = bucket.get(stats[key], 0) + 1
        if stats["visible_never_positive"]:
            totals["visible_never_positive"][result.source_id] = stats[
                "visible_never_positive"
            ]
        for name, value in stats["timings"].items():
            totals["timings"][name] = totals["timings"].get(name, 0.0) + value

    done = 0

    def report() -> None:
        elapsed = time.perf_counter() - started
        log.info(
            "%d/%d scenes, %.1f scenes/min, %s",
            done,
            len(dirs),
            60.0 * done / max(elapsed, 1e-9),
            _max_rss_mb(),
        )

    if workers == 1:
        import torch

        if device == "cuda":
            fraction = float(os.environ.get("CHIRP_MAX_VRAM_FRACTION", "0.4"))
            torch.cuda.set_per_process_memory_fraction(min(1.0, fraction))
        for d in dirs:
            write(tile_scene(d, cfg, device=device))
            done += 1
            if done % 20 == 0:
                report()
    else:
        import multiprocessing

        fraction = float(os.environ.get("CHIRP_MAX_VRAM_FRACTION", "0.4")) / workers
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(cfg.threads_per_worker, device, fraction),
        ) as pool:
            queue: deque[Future[SceneTiles]] = deque()
            pending = iter(dirs)
            for d in pending:
                queue.append(pool.submit(_work, str(d), cfg))
                if len(queue) >= 2 * workers:
                    break
            while queue:
                write(queue.popleft().result())
                done += 1
                nxt = next(pending, None)
                if nxt is not None:
                    queue.append(pool.submit(_work, str(nxt), cfg))
                if done % 50 == 0:
                    report()
    report()
    t_write = time.perf_counter()
    dataset = writer.close()
    elapsed = time.perf_counter() - started
    totals["elapsed_s"] = elapsed
    totals["close_s"] = time.perf_counter() - t_write
    totals["scenes_per_min"] = 60.0 * len(dirs) / elapsed
    totals["memory"] = _max_rss_mb()
    log.info(
        "wrote %d images of %d scenes in %.0f s (%.1f scenes/min); peak RSS %s",
        len(dataset.manifest.images),
        len(dirs),
        elapsed,
        totals["scenes_per_min"],
        totals["memory"],
    )
    if totals["visible_never_positive"]:
        n = sum(len(v) for v in totals["visible_never_positive"].values())
        log.warning("%d visible chirps are a positive in no tile", n)
    if diagnostics:
        from chirpdetector.data.diagnostics import write_diagnostics

        write_diagnostics(dataset, build=totals)
    return dataset
