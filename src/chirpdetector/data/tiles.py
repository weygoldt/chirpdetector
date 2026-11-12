"""Tiles of a spectrogram and the boxes that fall into them.

Shared by the dataset builders (`chirpdetector.data.scenes`,
`chirpdetector.data.real`). A recording's (or a labelled region's)
spectrogram is computed once and cut into tiles of a fixed number of frames
(`TileLayout`); each tile is normalised on its own (`normalize` on the
tile's frames, as inference does) and turned into an image whose
`ImageGeometry` maps pixels to absolute time and frequency exactly
(`render_tile`). Inference cuts recordings the same way: `TileLayout.
from_tiling` rebuilds the layout from the `Tiling` stored in a dataset or
model bundle, `TileLayout.starts` gives the tiles' first frames and
`render_tile` the images, with the channel fill of that `Tiling`.

Boxes are assigned to tiles by `assign_to_tile`: a box that lies entirely in
a tile is kept; a box cut by the tile edge stays a positive (clipped to
the tile) if at least ``edge_fraction`` of its duration and its centre
time lie inside, and becomes an ignore region (``reason="edge"``)
otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from chirpdetector.data.contract import Tiling
from chirpdetector.spectrogram import (
    ImageGeometry,
    Spectrogram,
    SpectrogramConfig,
    normalize,
    to_image,
)

__all__ = [
    "EDGE_FRACTION",
    "TileLayout",
    "assign_to_tile",
    "concat_boxes",
    "render_tile",
    "slice_spectrogram",
]

EDGE_FRACTION = 0.7
"""Fraction of a box's duration that must lie in a tile to stay a positive."""

_WIDTH_MULTIPLE = 32
_EDGES = ("t_start_s", "t_end_s", "f_low_hz", "f_high_hz")


@dataclass(frozen=True, slots=True)
class TileLayout:
    """How a spectrogram is cut into tiles.

    Parameters
    ----------
    tile_duration_s
        Nominal duration of a tile; ``round(tile_duration_s / hop_s)``
        frames, which must be a multiple of 32 (detector strides).
    overlap_s
        Nominal overlap of consecutive tiles; ``round(overlap_s / hop_s)``
        frames.
    channel_fill
        How images of a front end with fewer than 3 windows are filled up
        to 3 channels (`chirpdetector.spectrogram.to_three_channels`);
        unused for 3 windows.
    """

    tile_duration_s: float = 3.2
    overlap_s: float = 1.0
    channel_fill: Literal["zeros", "reference"] = "zeros"

    @classmethod
    def from_tiling(cls, tiling: Tiling) -> TileLayout:
        """The layout that cut images with this `Tiling` (dataset or bundle)."""
        if tiling.pixels_per_frame != 1 or tiling.pixels_per_bin != 1:
            msg = "upsampled tilings (pixels_per_frame/bin > 1) are not supported"
            raise ValueError(msg)
        return cls(
            tile_duration_s=tiling.tile_duration_s,
            overlap_s=tiling.overlap_s,
            channel_fill=tiling.channel_fill or "zeros",
        )

    def width_frames(self, config: SpectrogramConfig) -> int:
        """Frames per tile."""
        width = round(self.tile_duration_s / config.hop_s)
        if width < _WIDTH_MULTIPLE or width % _WIDTH_MULTIPLE:
            msg = (
                f"a {self.tile_duration_s:g} s tile is {width} frames of "
                f"{config.hop_s:g} s; it must be a positive multiple of "
                f"{_WIDTH_MULTIPLE}"
            )
            raise ValueError(msg)
        return width

    def overlap_frames(self, config: SpectrogramConfig) -> int:
        """Frames shared by consecutive tiles."""
        overlap = round(self.overlap_s / config.hop_s)
        if not 0 <= overlap < self.width_frames(config):
            msg = f"overlap_s ({self.overlap_s:g} s) must lie in [0, tile duration)"
            raise ValueError(msg)
        return overlap

    def starts(self, n_frames: int, config: SpectrogramConfig) -> list[int]:
        """First frame of every tile over ``n_frames`` frames.

        Tiles step by ``width - overlap`` frames from frame 0; the last tile
        is aligned to the last frame, so the tiles cover every frame and
        none reaches beyond. Raises if ``n_frames`` is shorter than a tile.
        """
        width = self.width_frames(config)
        stride = width - self.overlap_frames(config)
        if n_frames < width:
            msg = f"{n_frames} frames are fewer than one tile ({width} frames)"
            raise ValueError(msg)
        last = n_frames - width
        starts = list(range(0, last + 1, stride))
        if starts[-1] != last:
            starts.append(last)
        return starts

    def tiling(self, config: SpectrogramConfig) -> Tiling:
        """The contract's `Tiling` of images cut with this layout."""
        fill = None if config.n_windows == 3 else self.channel_fill
        return Tiling(
            tile_duration_s=self.width_frames(config) * config.hop_s,
            overlap_s=self.overlap_frames(config) * config.hop_s,
            width_px=self.width_frames(config),
            height_px=config.n_freqs,
            channel_fill=fill,
        )


def slice_spectrogram(spec: Spectrogram, start: int, stop: int) -> Spectrogram:
    """Frames ``[start, stop)`` (indices into ``spec``) as a spectrogram."""
    if not 0 <= start < stop <= spec.n_times:
        msg = f"frames [{start}, {stop}) outside the {spec.n_times} frames"
        raise ValueError(msg)
    return Spectrogram(
        power=spec.power[:, :, start:stop],
        freqs_hz=spec.freqs_hz,
        times_s=spec.times_s[start:stop],
        frame_start=spec.frame_start + start,
        complete=spec.complete[start:stop],
        config=spec.config,
        layout=spec.layout,
    )


def render_tile(
    spec: Spectrogram, start: int, tiling: Tiling
) -> tuple[NDArray[np.uint8], ImageGeometry]:
    """Image and geometry of one tile of a spectrogram.

    Parameters
    ----------
    spec
        Spectrogram of a recording or a span of one.
    start
        Index (into ``spec``'s frames) of the tile's first frame, e.g. from
        `TileLayout.starts`.
    tiling
        Image layout (a dataset's or a bundle's): the tile has
        ``tiling.width_px`` frames, and a front end with fewer than 3
        windows is filled up with ``tiling.channel_fill``.

    Returns
    -------
    image, geometry
        ``(H, W, 3)`` uint8 image and its geometry. The tile is normalised
        on its own frames (`normalize`), as inference normalises the tiles
        it cuts.
    """
    from chirpdetector.spectrogram import to_three_channels

    if tiling.pixels_per_frame != 1 or tiling.pixels_per_bin != 1:
        msg = "upsampled tilings (pixels_per_frame/bin > 1) are not supported"
        raise ValueError(msg)
    if tiling.height_px != spec.n_freqs:
        msg = (
            f"the tiling has {tiling.height_px} rows, the spectrogram "
            f"{spec.n_freqs} frequency bins"
        )
        raise ValueError(msg)
    tile = slice_spectrogram(spec, start, start + tiling.width_px)
    image = to_image(normalize(tile, tile.config))
    if image.shape[2] != 3:
        if tiling.channel_fill is None:
            msg = f"{image.shape[2]} channels but the tiling has no channel_fill"
            raise ValueError(msg)
        image = to_three_channels(image, fill=tiling.channel_fill)
    return image, ImageGeometry.from_spectrogram(tile)


def _clip(
    edges: NDArray[np.float64], extent: tuple[float, float, float, float]
) -> NDArray[np.float64]:
    t0, t1, f0, f1 = extent
    out = edges.copy()
    out[:, 0] = np.clip(out[:, 0], t0, t1)
    out[:, 1] = np.clip(out[:, 1], t0, t1)
    out[:, 2] = np.clip(out[:, 2], f0, f1)
    out[:, 3] = np.clip(out[:, 3], f0, f1)
    return out


def assign_to_tile(
    boxes: pd.DataFrame,
    geometry: ImageGeometry,
    *,
    edge_fraction: float = EDGE_FRACTION,
    min_size_px: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split candidate boxes into a tile's chirps and ignore regions.

    Parameters
    ----------
    boxes
        One row per box with ``t_start_s, t_end_s, f_low_hz, f_high_hz``,
        ``t_center_s`` (the chirp's centre time), ``label`` (``"chirp"`` for
        a positive candidate, else the ignore reason, e.g. ``"invisible"``
        or ``"harmonic"``), and any further columns (kept).
    geometry
        The tile.
    edge_fraction
        A ``chirp`` box cut by the tile edge stays a chirp if at least this
        fraction of its duration and its ``t_center_s`` lie in the tile;
        otherwise it becomes an ignore region with reason ``"edge"``.
    min_size_px
        Boxes whose part inside the tile is narrower or lower than this
        (in pixels) are dropped: a sliver at the edge is no region.

    Returns
    -------
    chirps, ignore
        Boxes clipped to the tile. Chirps keep their ``t_center_s``; ignore
        regions get the midpoint of their clipped box (the chirp centre may
        lie outside the tile) and a ``reason`` column. ``label`` is dropped.
    """
    t0, t1 = geometry.time_extent_s
    f0, f1 = geometry.freq_extent_hz
    if len(boxes) == 0:
        empty = boxes.iloc[:0].drop(columns="label")
        return empty.copy(), empty.assign(reason=pd.Series(dtype=object))
    edges = boxes.loc[:, list(_EDGES)].to_numpy(dtype=np.float64)
    clipped = _clip(edges, (t0, t1, f0, f1))
    width_px = (clipped[:, 1] - clipped[:, 0]) / geometry.dt_s
    height_px = (clipped[:, 3] - clipped[:, 2]) / geometry.df_hz
    inside = (width_px >= min_size_px) & (height_px >= min_size_px)
    duration = edges[:, 1] - edges[:, 0]
    fraction = np.divide(
        clipped[:, 1] - clipped[:, 0],
        duration,
        out=np.zeros_like(duration),
        where=duration > 0,
    )
    centre = boxes["t_center_s"].to_numpy(dtype=np.float64)
    centre_in = (centre >= t0) & (centre <= t1)
    label = boxes["label"].astype(str).to_numpy()
    positive = label == "chirp"
    keep_positive = positive & inside & (fraction >= edge_fraction - 1e-12) & centre_in
    edge = positive & inside & ~keep_positive
    ignore = (~positive & inside) | edge

    out = boxes.copy()
    out[list(_EDGES)] = clipped
    chirps = out.loc[keep_positive].drop(columns="label")
    chirps["t_center_s"] = np.clip(
        chirps["t_center_s"].to_numpy(dtype=np.float64),
        chirps["t_start_s"].to_numpy(dtype=np.float64),
        chirps["t_end_s"].to_numpy(dtype=np.float64),
    )
    reasons = np.where(edge, "edge", label)
    ignored = out.loc[ignore].drop(columns="label")
    ignored["reason"] = reasons[ignore]
    ignored["t_center_s"] = 0.5 * (ignored["t_start_s"] + ignored["t_end_s"])
    return chirps.reset_index(drop=True), ignored.reset_index(drop=True)


def overlaps(
    edges: Sequence[float] | NDArray[np.float64], geometry: ImageGeometry
) -> bool:
    """Whether a ``(t0, t1, f0, f1)`` box overlaps the tile."""
    t0, t1 = geometry.time_extent_s
    f0, f1 = geometry.freq_extent_hz
    a, b, c, d = (float(v) for v in edges)
    if not all(math.isfinite(v) for v in (a, b, c, d)):
        return False
    return a < t1 and b > t0 and c < f1 and d > f0


def concat_boxes(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate box tables; rows without ``live_tracks_hz`` get None.

    Plain `pandas.concat` fills a list column missing from one table with
    NaN, which the dataset writer would read as a list.
    """
    used = [df for df in frames if len(df)]
    if not used:
        return frames[0].iloc[:0].copy() if frames else pd.DataFrame()
    out = pd.concat(used, ignore_index=True)
    if "live_tracks_hz" in out.columns:
        out["live_tracks_hz"] = pd.Series(
            [
                v if isinstance(v, list | tuple | np.ndarray) else None
                for v in out["live_tracks_hz"]
            ],
            dtype=object,
        )
    return out
