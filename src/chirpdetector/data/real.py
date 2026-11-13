"""The locked real test set: hand-labelled regions of grid recordings.

Each recording is one source and one group, all in split ``test``. Only
its labelled regions (audian category ``labeled_region``) are tiled, with
the same front end, tiling and click removal rule as the simulated
training data (`chirpdetector.data.scenes`): the region's frames (those
centred inside it) are computed from the recording with its real context
on both sides, cleaned of clicks where `can_remove_clicks` allows it, and
cut into tiles normalised on their own.

Labels
------
* Hand-drawn chirp boxes (category ``chirp``) are kept as drawn (they are
  not footprints), clipped to the tile, with the same edge rule as the
  simulated data (`chirpdetector.data.tiles.assign_to_tile`, centre = box
  midpoint); extra ``label_source="hand"``.
* A region is exhaustively labelled only inside its frequency band, so
  the parts of a tile below and above that band become ignore regions
  (``reason="unlabelled"``); each image also records its labelled extent
  (``extra.labelled_*``) for scoring.
* ``emitter_hz`` is not known: it is the baseline of the live track
  nearest to the box's lower edge at the box midpoint where the recording
  has tracks (``emitter_hz_source="nearest_track_to_f_low"``), else
  ``f_low_hz`` (``"f_low_proxy"``, and ``live_tracks_hz = [f_low_hz]``).
  These proxies only satisfy the contract; the real test is scored on
  detection.

Raw recordings are in volts and the simulations in another unit; neither
matters, because click removal (robust noise thresholds), the spectrogram
and the normalisation (fixed dB range around a level measured on the data)
are invariant to a global gain.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chirpdetector.data.contract import Dataset, DatasetWriter, sha256_file
from chirpdetector.data.scenes import builder_provenance
from chirpdetector.data.tiles import (
    EDGE_FRACTION,
    TileLayout,
    assign_to_tile,
    concat_boxes,
    render_tile,
)
from chirpdetector.preprocessing import ClickRemoval, can_remove_clicks, read_clean
from chirpdetector.spectrogram import (
    ImageGeometry,
    SpectrogramConfig,
    compute_spectrogram_span,
    plan_span,
)

__all__ = [
    "LABEL_FILE",
    "RealDatasetConfig",
    "build_real_dataset",
    "hand_boxes",
    "region_tiles",
]

log = logging.getLogger(__name__)

LABEL_FILE = "traces-grid1-editable-labels.csv"
"""audian label file of a recording directory."""

_TRACK_FILES = ("times.npy", "fund_v.npy", "ident_v.npy", "idx_v.npy")
_EDGES = ("t_start_s", "t_end_s", "f_low_hz", "f_high_hz")


@dataclass(frozen=True)
class RealDatasetConfig:
    """Front end, tiling and preprocessing of the real test set.

    Must equal the training set's so that the test images are rendered as
    the detector's inputs are; see `chirpdetector.data.scenes.
    SceneDatasetConfig` for the fields.
    """

    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    layout: TileLayout = field(default_factory=TileLayout)
    click_removal: ClickRemoval | None = field(default_factory=ClickRemoval)
    edge_fraction: float = EDGE_FRACTION
    device: str = "cpu"
    click_workers: int = 4

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
            "device": self.device,
        }


def _has_tracks(directory: Path) -> bool:
    return all((directory / name).exists() for name in _TRACK_FILES)


def hand_boxes(labels: pd.DataFrame, tracks: Any | None) -> pd.DataFrame:
    """Candidate boxes of hand-labelled chirps, with emitter proxies.

    Parameters
    ----------
    labels
        Rows of category ``chirp`` of an audian label file (index = row in
        the file).
    tracks
        The recording's `gridtools.io.Tracks`, or None.

    Returns
    -------
    pandas.DataFrame
        Box edges, ``t_center_s`` (box midpoint), ``label="chirp"``,
        ``emitter_hz``, ``emitter_id``, ``live_tracks_hz``,
        ``emitter_hz_source``, ``label_source="hand"`` and ``label_row``.
    """
    rows = []
    track_list = list(tracks) if tracks is not None else []
    edges = labels.loc[:, list(_EDGES)].to_numpy(dtype=np.float64)
    for index, (t0, t1, f0, f1) in zip(labels.index, edges.tolist(), strict=True):
        if not all(math.isfinite(v) for v in (t0, t1, f0, f1)) or t1 <= t0 or f1 <= f0:
            log.warning("skipping malformed chirp label %s", index)
            continue
        tc = 0.5 * (t0 + t1)
        live: dict[int, float] = {}
        for track in track_list:
            f = float(track.frequency_at(tc)[0])
            if math.isfinite(f) and f > 0:
                live[int(track.ident)] = f
        if live:
            ident = min(live, key=lambda k: abs(live[k] - f0))
            emitter, emitter_id = live[ident], ident
            live_hz = [live[k] for k in sorted(live)]
            source = "nearest_track_to_f_low"
        else:
            emitter, emitter_id, live_hz = f0, None, [f0]
            source = "f_low_proxy"
        rows.append(
            {
                "t_start_s": t0,
                "t_end_s": t1,
                "f_low_hz": f0,
                "f_high_hz": f1,
                "t_center_s": tc,
                "label": "chirp",
                "emitter_hz": emitter,
                "emitter_id": emitter_id,
                "live_tracks_hz": live_hz,
                "emitter_hz_source": source,
                "label_source": "hand",
                "label_row": int(str(index)),
            }
        )
    columns = [
        *_EDGES,
        "t_center_s",
        "label",
        "emitter_hz",
        "emitter_id",
        "live_tracks_hz",
        "emitter_hz_source",
        "label_source",
        "label_row",
    ]
    out = pd.DataFrame(rows, columns=columns)
    out["emitter_id"] = pd.array(out["emitter_id"].tolist(), dtype="Int64")
    return out


def _unlabelled(
    geometry: ImageGeometry, band: tuple[float, float], min_hz: float
) -> pd.DataFrame:
    """Ignore regions for the parts of a tile outside the labelled band."""
    t0, t1 = geometry.time_extent_s
    f0, f1 = geometry.freq_extent_hz
    rows = []
    if band[0] - f0 >= min_hz:
        rows.append((t0, t1, f0, band[0]))
    if f1 - band[1] >= min_hz:
        rows.append((t0, t1, band[1], f1))
    out = pd.DataFrame(rows, columns=list(_EDGES))
    out["t_center_s"] = 0.5 * (out["t_start_s"] + out["t_end_s"])
    out["reason"] = "unlabelled"
    out["label_source"] = "hand"
    return out


def region_tiles(
    recording: Any,
    region: tuple[float, float, float, float],
    boxes: pd.DataFrame,
    config: RealDatasetConfig,
) -> tuple[
    list[tuple[Any, ImageGeometry, int]],
    list[pd.DataFrame],
    list[pd.DataFrame],
    dict[str, Any],
]:
    """Tiles of one labelled region of a recording.

    Returns
    -------
    tiles, chirps, ignore, info
        ``(image, geometry, first frame)`` per tile, the tiles' chirp and
        ignore tables, and click-removal information.
    """
    cfg = config
    fs = float(recording.samplerate_hz)
    layout = cfg.spectrogram.layout(fs)
    hop = layout.hop_samples
    rt0, rt1, rf0, rf1 = region
    k0 = math.ceil(rt0 * fs / hop - 1e-9)
    k1 = math.floor(rt1 * fs / hop + 1e-9)
    start = k0 * hop
    stop = min(k1 * hop + 1, recording.n_samples)
    plan = plan_span(start, stop, recording.n_samples, fs, cfg.spectrogram)
    applied = cfg.click_removal is not None and can_remove_clicks(
        recording.n_channels, cfg.click_removal, samplerate_hz=fs
    )
    repaired = 0.0
    if applied:
        samples, mask = read_clean(
            recording,
            plan.read_start,
            plan.read_stop,
            cfg.click_removal,
            workers=cfg.click_workers,
        )
        repaired = float(mask.mean())
    else:
        samples = recording.read(plan.read_start, plan.read_stop)
    spec = compute_spectrogram_span(samples, plan, cfg.spectrogram, device=cfg.device)
    del samples
    tiling = cfg.layout.tiling(cfg.spectrogram)
    tiles, chirp_tables, ignore_tables = [], [], []
    t0s = boxes["t_start_s"].to_numpy(dtype=np.float64)
    t1s = boxes["t_end_s"].to_numpy(dtype=np.float64)
    for first in cfg.layout.starts(spec.n_times, cfg.spectrogram):
        image, geometry = render_tile(spec, first, tiling)
        lo, hi = geometry.time_extent_s
        near = boxes.loc[(t0s < hi) & (t1s > lo)]
        chirps, ignore = assign_to_tile(near, geometry, edge_fraction=cfg.edge_fraction)
        unlabelled = _unlabelled(geometry, (rf0, rf1), min_hz=geometry.df_hz)
        ignore = concat_boxes([ignore, unlabelled])
        tiles.append((image, geometry, int(spec.frame_start + first)))
        chirp_tables.append(chirps)
        ignore_tables.append(ignore)
    info = {"click_removal_applied": bool(applied), "click_repaired_fraction": repaired}
    return tiles, chirp_tables, ignore_tables, info


def build_real_dataset(
    out: str | Path,
    recordings: Sequence[str | Path],
    config: RealDatasetConfig | None = None,
    *,
    overwrite: bool = False,
    diagnostics: bool = True,
    provenance: Mapping[str, Any] | None = None,
) -> Dataset:
    """Build the real test set from hand-labelled recordings.

    Parameters
    ----------
    out
        Output directory.
    recordings
        Recording directories, each with a raw recording, ``fishgrid.cfg``
        and the audian label file `LABEL_FILE` (tracks optional).
    config
        Front end, tiling, click removal; must match the training set's.
    overwrite
        Replace an existing dataset.
    diagnostics
        Write a contact sheet and counts to ``out/diagnostics``.
    provenance
        Extra provenance merged into the manifest's.

    Returns
    -------
    Dataset
        The written, validated dataset (every group in split ``test``).
    """
    from gridtools.io import load_tracks, open_recording, read_labels

    cfg = RealDatasetConfig() if config is None else config
    started = time.perf_counter()
    dirs = [Path(r).resolve() for r in recordings]
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
            "builder": "chirpdetector.data.real.build_real_dataset",
            **builder_provenance(),
            "config": cfg.to_dict(),
            "recordings": [str(d) for d in dirs],
            "labels": {
                "boxes": "hand-drawn audian boxes, kept as drawn",
                "positive": "chirp boxes with >= edge_fraction of their duration "
                "and their midpoint in the tile, clipped to the tile",
                "ignore_reasons": {
                    "edge": "chirp cut by the tile edge (rule above)",
                    "unlabelled": "tile area outside the labelled region's band",
                },
                "emitter_hz": "nearest live track to f_low_hz at the box midpoint, "
                "else f_low_hz (proxies; the real test is scored on detection)",
            },
            **dict(provenance or {}),
        },
        overwrite=overwrite,
    )
    build: dict[str, Any] = {"recordings": {}}
    for directory in dirs:
        name = directory.name
        label_path = directory / LABEL_FILE
        labels = read_labels(label_path)
        category = labels["category"].astype(str)
        regions = labels.loc[category == "labeled_region"]
        chirp_rows = labels.loc[category == "chirp"]
        recording = open_recording(directory)
        tracks = load_tracks(directory) if _has_tracks(directory) else None
        boxes = hand_boxes(chirp_rows, tracks)
        mid = boxes["t_center_s"].to_numpy(dtype=np.float64)
        in_any = np.zeros(len(boxes), dtype=bool)
        region_list = []
        for _, region in regions.iterrows():
            edges = tuple(float(region[c]) for c in _EDGES)
            region_list.append(list(edges))
            in_any |= (mid >= edges[0]) & (mid <= edges[1])
        if (~in_any).any():
            log.warning(
                "%s: %d chirp labels outside every labelled region (not used)",
                name,
                int((~in_any).sum()),
            )
        stats: dict[str, Any] = {
            "n_regions": len(region_list),
            "n_chirp_labels": len(boxes),
            "n_outside_regions": int((~in_any).sum()),
            "tracks": tracks is not None,
            "n_electrodes": recording.n_channels,
            "samplerate_hz": float(recording.samplerate_hz),
        }
        applied_any = False
        repaired = []
        n_images = 0
        pending = []
        for k, edges in enumerate(region_list):
            t_region = time.perf_counter()
            inside = boxes.loc[(mid >= edges[0]) & (mid <= edges[1])]
            tiles, chirps, ignore, info = region_tiles(
                recording,
                (edges[0], edges[1], edges[2], edges[3]),
                inside.assign(region=k),
                cfg,
            )
            applied_any |= info["click_removal_applied"]
            repaired.append(info["click_repaired_fraction"])
            for (image, geometry, frame), c, i in zip(
                tiles, chirps, ignore, strict=True
            ):
                image_id = f"{name}_r{k:02d}_f{frame:08d}"
                extra = {
                    "region": k,
                    "frame_start": frame,
                    "labelled_t_start_s": edges[0],
                    "labelled_t_end_s": edges[1],
                    "labelled_f_low_hz": edges[2],
                    "labelled_f_high_hz": edges[3],
                }
                pending.append((image_id, image, geometry, c, i, extra))
            n_images += len(tiles)
            log.info(
                "%s region %d (%.0f s): %d tiles in %.1f s",
                name,
                k,
                edges[1] - edges[0],
                len(tiles),
                time.perf_counter() - t_region,
            )
        writer.add_source(
            name,
            group=name,
            kind="real",
            tracks=str(directory) if tracks is not None else None,
            samplerate_hz=float(recording.samplerate_hz),
            extra={
                "recording_dir": str(directory),
                "label_file": str(label_path),
                "label_sha256": sha256_file(label_path),
                "regions": region_list,
                "n_electrodes": recording.n_channels,
                "click_removal_applied": applied_any,
                "click_repaired_fraction": float(np.mean(repaired))
                if repaired
                else 0.0,
                "n_chirp_labels": len(boxes),
                "n_chirp_labels_outside_regions": int((~in_any).sum()),
            },
        )
        writer.assign_split(name, "test")
        for image_id, image, geometry, c, i, extra in pending:
            writer.add_image(
                image_id, image, geometry, source=name, chirps=c, ignore=i, extra=extra
            )
        stats["n_images"] = n_images
        stats["click_removal_applied"] = applied_any
        build["recordings"][name] = stats
    dataset = writer.close()
    build["elapsed_s"] = time.perf_counter() - started
    if diagnostics:
        from chirpdetector.data.diagnostics import write_diagnostics

        write_diagnostics(dataset, build=build)
    return dataset
