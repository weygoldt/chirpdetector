"""Detect chirps in a span of a recording, tile by tile.

Tiling
------
The span's spectrogram (`compute_spectrogram_span`, frames identical to
those of the whole recording) is cut into tiles of the bundle's width
(``tiling.width_px`` frames) that overlap by ``round(tiling.overlap_s /
hop_s)`` frames: tiles start at frame 0 of the span and step by ``width -
overlap``; the last tile is aligned to the last frame, so every frame is
covered and no tile reaches past the span. Each tile is normalised on its
own frames (`normalize`), as the dataset tiles were. A span shorter than a
tile is widened symmetrically to one tile (within the recording).

Duplicates across tiles
-----------------------
A chirp in the overlap of two tiles may be detected in both. Each tile
*owns* the centre times from the middle of its overlap with the previous
tile to the middle of its overlap with the next, and keeps only detections
centred there. With the default 1 s overlap and chirps of at most 0.6 s, a
chirp centred in a tile's owned part lies entirely inside that tile, at
least 0.2 s from its edge, so nothing is lost. Unlike non-maximum
suppression across tiles (v1, IoU 0.2, audit N11), this never merges two
different chirps: within a tile the model is NMS-free, and between tiles
only detections of *different* tiles near a shared boundary are compared:
of two whose centres are within `DUPLICATE_TOLERANCE_S` (50 ms) and whose
frequency ranges overlap, the lower-scoring one is dropped. That catches a
chirp centred right at a boundary that both tiles localise on their own
side of it.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from chirpdetector.bundle import Bundle, load_bundle
from chirpdetector.data.contract import Tiling
from chirpdetector.models import Detector
from chirpdetector.prediction import load_detector, predict_images
from chirpdetector.preprocessing import ClickRemoval, can_remove_clicks, read_clean
from chirpdetector.resources import apply_vram_limit
from chirpdetector.spectrogram import (
    ImageGeometry,
    Spectrogram,
    SpectrogramConfig,
    compute_spectrogram_span,
    normalize,
    plan_span,
    to_image,
    to_three_channels,
)

__all__ = [
    "DUPLICATE_TOLERANCE_S",
    "TiledSpan",
    "merge_tiles",
    "predict_span",
    "render_tiles",
    "tile_starts",
]

log = logging.getLogger(__name__)

DUPLICATE_TOLERANCE_S = 0.05
"""Detections of different tiles closer than this (centre times) may be duplicates."""


def tile_frames(tiling: Tiling, config: SpectrogramConfig) -> tuple[int, int]:
    """``(width, overlap)`` of a tile in frames."""
    if tiling.pixels_per_frame != 1 or tiling.pixels_per_bin != 1:
        msg = "tiles with upsampled pixels are not supported"
        raise ValueError(msg)
    width = tiling.width_px
    overlap = round(tiling.overlap_s / config.hop_s)
    if not 0 <= overlap < width:
        msg = f"overlap of {overlap} frames does not fit tiles of {width}"
        raise ValueError(msg)
    return width, overlap


def tile_starts(n_frames: int, width: int, overlap: int) -> list[int]:
    """First frame of every tile (module docstring).

    Raises
    ------
    ValueError
        If ``n_frames < width``.
    """
    if n_frames < width:
        msg = f"{n_frames} frames are fewer than one tile ({width} frames)"
        raise ValueError(msg)
    last = n_frames - width
    starts = list(range(0, last + 1, width - overlap))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _slice(spec: Spectrogram, start: int, stop: int) -> Spectrogram:
    return Spectrogram(
        power=spec.power[:, :, start:stop],
        freqs_hz=spec.freqs_hz,
        times_s=spec.times_s[start:stop],
        frame_start=spec.frame_start + start,
        complete=spec.complete[start:stop],
        config=spec.config,
        layout=spec.layout,
    )


@dataclass(frozen=True)
class TiledSpan:
    """Tiles of a span: images, geometries, first frames, owned centre times."""

    images: list[NDArray[np.uint8]]
    geometries: list[ImageGeometry]
    starts: list[int]
    owned_s: list[tuple[float, float]]


def render_tiles(spec: Spectrogram, tiling: Tiling) -> TiledSpan:
    """Cut a spectrogram into normalised tile images (module docstring)."""
    width, overlap = tile_frames(tiling, spec.config)
    starts = tile_starts(spec.n_times, width, overlap)
    images, geometries = [], []
    for start in starts:
        tile = _slice(spec, start, start + width)
        image = to_image(normalize(tile, tile.config))
        if image.shape[2] != 3:
            if tiling.channel_fill is None:
                msg = f"{image.shape[2]}-window images need tiling.channel_fill"
                raise ValueError(msg)
            image = to_three_channels(image, fill=tiling.channel_fill)
        images.append(image)
        geometries.append(ImageGeometry.from_spectrogram(tile))
    times = spec.times_s
    bounds = [
        0.5 * (times[nxt] + times[cur + width - 1])
        for cur, nxt in itertools.pairwise(starts)
    ]
    edges = [-math.inf, *bounds, math.inf]
    owned = list(itertools.pairwise(edges))
    return TiledSpan(images=images, geometries=geometries, starts=starts, owned_s=owned)


def merge_tiles(
    predictions: pd.DataFrame,
    owned_s: Sequence[tuple[float, float]],
    tolerance_s: float = DUPLICATE_TOLERANCE_S,
) -> pd.DataFrame:
    """Keep each tile's own detections and drop cross-tile duplicates.

    Parameters
    ----------
    predictions
        `predict_images` output for the tiles of one span (``image_index`` =
        tile).
    owned_s
        Owned centre-time interval ``[lo, hi)`` of each tile.
    tolerance_s
        Centre-time tolerance of the cross-tile duplicate check.

    Returns
    -------
    DataFrame
        The kept detections, sorted by centre time; ``image_index`` is
        renamed ``tile``.
    """
    df = predictions.reset_index(drop=True)
    centre = 0.5 * (df["t_start_s"].to_numpy() + df["t_end_s"].to_numpy())
    tile = df["image_index"].to_numpy(dtype=np.int64)
    lo = np.array([owned_s[k][0] for k in tile]) if len(tile) else np.zeros(0)
    hi = np.array([owned_s[k][1] for k in tile]) if len(tile) else np.zeros(0)
    keep = (centre >= lo) & (centre < hi)
    order = np.argsort(-df["score"].to_numpy(), kind="stable")
    f_low, f_high = df["f_low_hz"].to_numpy(), df["f_high_hz"].to_numpy()
    for i in order:
        if not keep[i]:
            continue
        twins = (
            keep
            & (tile != tile[i])
            & (np.abs(centre - centre[i]) <= tolerance_s)
            & (f_low <= f_high[i])
            & (f_high >= f_low[i])
        )
        keep &= ~twins
    out = df.loc[keep].rename(columns={"image_index": "tile"})
    out = out.assign(_centre=centre[keep]).sort_values("_centre", kind="stable")
    return out.drop(columns="_centre").reset_index(drop=True)


class _Channels:
    """A recording restricted to some channels (`SampleSource`)."""

    def __init__(self, recording: object, channels: Sequence[int] | None) -> None:
        self._recording = recording
        self._channels = None if channels is None else list(channels)

    @property
    def samplerate_hz(self) -> float:
        return float(self._recording.samplerate_hz)  # type: ignore[attr-defined]

    @property
    def n_samples(self) -> int:
        return int(self._recording.n_samples)  # type: ignore[attr-defined]

    @property
    def n_channels(self) -> int:
        if self._channels is not None:
            return len(self._channels)
        return int(self._recording.n_channels)  # type: ignore[attr-defined]

    def read(self, start: int, stop: int) -> NDArray[np.float32]:
        return self._recording.read(start, stop, self._channels)  # type: ignore[attr-defined]


def predict_span(
    bundle: Bundle | str | Path,
    recording: object,
    t_start_s: float,
    t_end_s: float,
    *,
    name: str,
    channels: Sequence[int] | None = None,
    score_threshold: float | None = None,
    detector: Detector | None = None,
    device: str | None = None,
    tolerance_s: float = DUPLICATE_TOLERANCE_S,
) -> pd.DataFrame:
    """Detect chirps in ``[t_start_s, t_end_s)`` of a recording.

    Parameters
    ----------
    bundle
        Model bundle (or its directory).
    recording
        A `gridtools.io.Recording` (``samplerate_hz``, ``n_samples``,
        ``n_channels``, ``read(start, stop, channels)``); time 0 is its
        first sample.
    t_start_s, t_end_s
        The span: the frames centred in it (inclusive) are tiled, from its
        first frame on; widened to one tile if shorter.
    name
        Value of the ``recording`` column.
    channels
        Electrodes to use (all if None), as the dataset did.
    score_threshold
        Default: the bundle's.
    detector
        A loaded model of the bundle (`chirpdetector.prediction.load_detector`).
    device
        Torch device for the spectrogram and the model.
    tolerance_s
        See `merge_tiles`.

    Returns
    -------
    DataFrame
        Detections in absolute time and frequency (`predict_images`
        columns, ``tile`` instead of ``image_index``), de-duplicated.
    """
    bundle = bundle if isinstance(bundle, Bundle) else load_bundle(bundle)
    source = _Channels(recording, channels)
    config, tiling = bundle.spectrogram, bundle.tiling
    fs, total = source.samplerate_hz, source.n_samples
    width, _ = tile_frames(tiling, config)
    hop = config.layout(fs).hop_samples
    # frames centred in [t_start_s, t_end_s], as the real test set was cut
    first = max(0, math.ceil(t_start_s * fs / hop - 1e-9))
    last = math.floor(t_end_s * fs / hop + 1e-9)
    start = first * hop
    stop = min(total, last * hop + 1)
    needed = width * hop
    if stop - start < needed:
        centre = (start + stop) // 2
        start = max(0, min(centre - needed // 2, total - needed))
        stop = min(total, start + needed)
    plan = plan_span(start, stop, total, fs, config)
    click = bundle.preprocessing.get("click_removal")
    if click is not None and can_remove_clicks(
        source.n_channels, ClickRemoval.from_dict(dict(click)), samplerate_hz=fs
    ):
        samples, _ = read_clean(
            source, plan.read_start, plan.read_stop, ClickRemoval.from_dict(dict(click))
        )
    else:
        if click is not None:
            log.warning("%s: too few electrodes for click removal; skipped", name)
        samples = source.read(plan.read_start, plan.read_stop)
    if device is not None and device != "cpu":
        apply_vram_limit(None, device)
    spec = compute_spectrogram_span(samples, plan, config, device=device)
    tiles = render_tiles(spec, tiling)
    detector = detector or load_detector(bundle, device=device)
    predictions = predict_images(
        bundle,
        tiles.images,
        tiles.geometries,
        recordings=[name] * len(tiles.images),
        score_threshold=score_threshold,
        detector=detector,
    )
    return merge_tiles(predictions, tiles.owned_s, tolerance_s)
