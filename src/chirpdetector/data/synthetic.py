"""Tiny synthetic detector datasets: bright boxes on noise, for tests.

The images are not spectrograms of anything; only the contract is real
(geometry from a real front-end config, boxes in time and frequency,
emitter frequencies, ignore regions, group-wise splits). Chirps are drawn
as bright rectangles in all three channels, ignore regions as rectangles
of intermediate brightness, so that a detector can learn the task in a
few epochs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from chirpdetector.data.contract import Dataset, DatasetWriter, Split, Tiling
from chirpdetector.spectrogram import (
    ImageGeometry,
    SpectrogramConfig,
    pixel_box_to_timefreq,
)

__all__ = ["SMALL_SPECTROGRAM", "make_synthetic_dataset"]

SMALL_SPECTROGRAM = SpectrogramConfig(f_min_hz=500.0, f_max_hz=810.0)
"""Front end of the synthetic datasets: 63 frequency rows."""

_DEFAULT_SPLITS: tuple[Split, ...] = ("train", "train", "train", "train", "val", "test")
_CHIRP_VALUE = 230
_IGNORE_VALUE = 150


def _place(
    rng: np.random.Generator,
    taken: np.ndarray,
    width: int,
    height: int,
    size_w: tuple[int, int],
    size_h: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """A free axis-aligned box (integer pixel edges), or None after 50 tries."""
    for _ in range(50):
        w = int(rng.integers(*size_w))
        h = int(rng.integers(*size_h))
        x0 = int(rng.integers(0, width - w + 1))
        y0 = int(rng.integers(0, height - h + 1))
        if not taken[max(0, y0 - 2) : y0 + h + 2, max(0, x0 - 2) : x0 + w + 2].any():
            taken[y0 : y0 + h, x0 : x0 + w] = True
            return x0, y0, x0 + w, y0 + h
    return None


def make_synthetic_dataset(
    directory: str | Path,
    *,
    splits: Sequence[Split] = _DEFAULT_SPLITS,
    images_per_group: int = 2,
    width_px: int = 128,
    spectrogram: SpectrogramConfig = SMALL_SPECTROGRAM,
    samplerate_hz: float = 20_000.0,
    chirps_per_image: tuple[int, int] = (1, 3),
    ignore_per_image: int = 1,
    seed: int = 0,
    overwrite: bool = False,
) -> Dataset:
    """Write a small valid detector dataset of boxes drawn on noise.

    Parameters
    ----------
    directory
        Output directory (must not exist or be empty unless ``overwrite``).
    splits
        Split of each group; one source per group (``scene-<k>``).
    images_per_group
        Consecutive, non-overlapping tiles per source.
    width_px
        Image width; the height is the front end's number of bins.
    spectrogram
        Front end the geometry is taken from (at ``samplerate_hz``).
    samplerate_hz
        Sample rate of the imaginary recordings.
    chirps_per_image
        Inclusive range of the number of chirp boxes per image.
    ignore_per_image
        Ignore regions per image.
    seed
        Seed of all random draws.
    overwrite
        Replace an existing directory.

    Returns
    -------
    Dataset
        The validated dataset.
    """
    rng = np.random.default_rng(seed)
    layout = spectrogram.layout(samplerate_hz)
    height_px = spectrogram.n_freqs
    tiling = Tiling(
        tile_duration_s=width_px * spectrogram.hop_s,
        overlap_s=0.0,
        width_px=width_px,
        height_px=height_px,
        channel_fill=None if spectrogram.n_windows == 3 else "reference",
    )
    writer = DatasetWriter(
        directory,
        spectrogram=spectrogram,
        tiling=tiling,
        provenance={"builder": "chirpdetector.data.synthetic", "seed": seed},
        overwrite=overwrite,
    )
    for k, split in enumerate(splits):
        source = f"scene-{k:03d}"
        writer.add_source(
            source,
            group=source,
            kind="simulation",
            samplerate_hz=samplerate_hz,
            extra={"seed": seed},
        )
        writer.assign_split(source, split)
        for j in range(images_per_group):
            geometry = ImageGeometry(
                width_px=width_px,
                height_px=height_px,
                t_first_s=j * width_px * layout.hop_s,
                dt_s=layout.hop_s,
                f_lowest_hz=float(layout.freqs_hz[0]),
                df_hz=layout.df_hz,
            )
            noise = rng.normal(40.0, 12.0, size=(height_px, width_px, 3))
            image = np.clip(noise, 0, 255).astype(np.uint8)
            taken = np.zeros((height_px, width_px), dtype=bool)
            chirps, ignore = [], []
            n_chirps = int(rng.integers(chirps_per_image[0], chirps_per_image[1] + 1))
            for kind, n, value in (
                ("chirp", n_chirps, _CHIRP_VALUE),
                ("ignore", ignore_per_image, _IGNORE_VALUE),
            ):
                for _ in range(n):
                    box = _place(rng, taken, width_px, height_px, (8, 24), (8, 20))
                    if box is None:
                        continue
                    x0, y0, x1, y1 = box
                    image[y0:y1, x0:x1, :] = value
                    t0, t1, f0, f1 = (
                        float(v)
                        for v in pixel_box_to_timefreq(x0, y0, x1, y1, geometry)
                    )
                    row = {
                        "t_start_s": t0,
                        "t_end_s": t1,
                        "f_low_hz": f0,
                        "f_high_hz": f1,
                    }
                    if kind == "chirp":
                        emitter = f0 + 0.5 * layout.df_hz
                        row |= {
                            "emitter_hz": emitter,
                            "emitter_id": len(chirps) % 2,
                            "live_tracks_hz": [emitter, emitter + 37.0],
                            "snr_db": float(rng.uniform(0, 20)),
                        }
                        chirps.append(row)
                    else:
                        ignore.append(row | {"reason": "invisible"})
            writer.add_image(
                f"{source}_t{j:03d}",
                image,
                geometry,
                source=source,
                chirps=pd.DataFrame(chirps) if chirps else None,
                ignore=pd.DataFrame(ignore) if ignore else None,
            )
    return writer.close()
