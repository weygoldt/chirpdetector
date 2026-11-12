"""Image layout of the spectrogram and exact pixel <-> time-frequency maps.

Orientation
-----------
``to_image`` returns ``(H, W, C)`` uint8 with ``H = n_freqs``, ``W = n_times``
and ``C = n_windows`` (channel ``c`` = window ``c``, channel 0 = reference).
Row 0 is the HIGHEST frequency; column 0 is the earliest frame. This is the
only orientation; nothing downstream flips images.

Pixel coordinates
-----------------
Pixel coordinates are continuous ``(x, y)`` with ``x`` along columns (time)
and ``y`` along rows (frequency, increasing downwards). Pixel ``(row i,
column j)`` covers ``[j, j + 1) x [i, i + 1)``; its CENTRE ``(j + 0.5, i +
0.5)`` corresponds to the centre time of frame ``j`` and the centre
frequency of the bin shown in row ``i``. The image spans ``x in [0, W]``,
``y in [0, H]``: ``x = 0`` is half a hop before the first frame centre,
``y = 0`` half a bin above the highest bin centre. Box edges are therefore
ordinary continuous coordinates, as detectors (e.g. YOLO ``xywh`` scaled by
the image size) expect.

Mapping (``t_first`` = centre time of column 0, ``f_lowest`` = centre
frequency of the bottom row ``H - 1``)::

    x = (t - t_first) / dt + 0.5          t = t_first + (x - 0.5) * dt
    y = H - 0.5 - (f - f_lowest) / df     f = f_lowest + (H - 0.5 - y) * df
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from chirpdetector.spectrogram.compute import Spectrogram

Float64Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ImageGeometry:
    """Placement of a spectrogram image in time and frequency.

    Attributes
    ----------
    width_px, height_px
        Image size: number of frames and of frequency bins.
    t_first_s
        Centre time of column 0.
    dt_s
        Time per column: the actual hop, ``hop_samples / samplerate_hz``.
    f_lowest_hz
        Centre frequency of the bottom row (``height_px - 1``).
    df_hz
        Frequency per row: the actual bin spacing, ``samplerate_hz /
        n_fft``.
    """

    width_px: int
    height_px: int
    t_first_s: float
    dt_s: float
    f_lowest_hz: float
    df_hz: float

    @classmethod
    def from_spectrogram(cls, spec: Spectrogram) -> ImageGeometry:
        """Geometry of ``to_image(normalize(spec, ...))``.

        Uses the actual hop and bin spacing at the recording's rate
        (``spec.layout``), not the config's nominal values.
        """
        return cls(
            width_px=spec.n_times,
            height_px=spec.n_freqs,
            t_first_s=float(spec.times_s[0]),
            dt_s=spec.layout.hop_s,
            f_lowest_hz=float(spec.freqs_hz[0]),
            df_hz=spec.layout.df_hz,
        )

    @property
    def time_extent_s(self) -> tuple[float, float]:
        """Times of the left (``x = 0``) and right (``x = W``) image edges."""
        start = self.t_first_s - 0.5 * self.dt_s
        return start, start + self.width_px * self.dt_s

    @property
    def freq_extent_hz(self) -> tuple[float, float]:
        """Frequencies of the bottom (``y = H``) and top (``y = 0``) edges."""
        low = self.f_lowest_hz - 0.5 * self.df_hz
        return low, low + self.height_px * self.df_hz


def timefreq_to_pixel(
    t_s: ArrayLike, f_hz: ArrayLike, geometry: ImageGeometry
) -> tuple[Float64Array, Float64Array]:
    """Map times and frequencies to continuous pixel coordinates.

    Parameters
    ----------
    t_s, f_hz
        Times (absolute, same clock as ``Spectrogram.times_s``) and
        frequencies; broadcastable.
    geometry
        Image geometry.

    Returns
    -------
    x, y
        Column and row coordinates (see module docstring); values outside
        ``[0, W]`` / ``[0, H]`` lie outside the image.
    """
    t = np.asarray(t_s, dtype=np.float64)
    f = np.asarray(f_hz, dtype=np.float64)
    x = (t - geometry.t_first_s) / geometry.dt_s + 0.5
    y = geometry.height_px - 0.5 - (f - geometry.f_lowest_hz) / geometry.df_hz
    return x, y


def pixel_to_timefreq(
    x: ArrayLike, y: ArrayLike, geometry: ImageGeometry
) -> tuple[Float64Array, Float64Array]:
    """Map continuous pixel coordinates to times and frequencies.

    Exact inverse of `timefreq_to_pixel`.

    Parameters
    ----------
    x, y
        Column and row coordinates; broadcastable.
    geometry
        Image geometry.

    Returns
    -------
    t_s, f_hz
        Times and frequencies.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    t = geometry.t_first_s + (xa - 0.5) * geometry.dt_s
    f = geometry.f_lowest_hz + (geometry.height_px - 0.5 - ya) * geometry.df_hz
    return t, f


def timefreq_box_to_pixel(
    t_start_s: ArrayLike,
    t_stop_s: ArrayLike,
    f_low_hz: ArrayLike,
    f_high_hz: ArrayLike,
    geometry: ImageGeometry,
) -> tuple[Float64Array, Float64Array, Float64Array, Float64Array]:
    """Map time-frequency boxes to pixel boxes.

    Parameters
    ----------
    t_start_s, t_stop_s, f_low_hz, f_high_hz
        Box edges; broadcastable.
    geometry
        Image geometry.

    Returns
    -------
    x_min, y_min, x_max, y_max
        Pixel box edges; ``y_min`` (top) comes from ``f_high_hz`` because
        row 0 is the highest frequency.
    """
    x_min, y_min = timefreq_to_pixel(t_start_s, f_high_hz, geometry)
    x_max, y_max = timefreq_to_pixel(t_stop_s, f_low_hz, geometry)
    return x_min, y_min, x_max, y_max


def pixel_box_to_timefreq(
    x_min: ArrayLike,
    y_min: ArrayLike,
    x_max: ArrayLike,
    y_max: ArrayLike,
    geometry: ImageGeometry,
) -> tuple[Float64Array, Float64Array, Float64Array, Float64Array]:
    """Map pixel boxes to time-frequency boxes (inverse of the above).

    Returns
    -------
    t_start_s, t_stop_s, f_low_hz, f_high_hz
        Box edges in time and frequency.
    """
    t_start, f_high = pixel_to_timefreq(x_min, y_min, geometry)
    t_stop, f_low = pixel_to_timefreq(x_max, y_max, geometry)
    return t_start, t_stop, f_low, f_high


def to_image(normalized: torch.Tensor | NDArray[np.floating]) -> NDArray[np.uint8]:
    """Convert a normalised spectrogram to an image.

    Parameters
    ----------
    normalized
        Output of `normalize`, shape ``(n_windows, n_freqs, n_times)``,
        values in [0, 1] (clipped if outside).

    Returns
    -------
    NDArray[np.uint8]
        ``(H, W, C) = (n_freqs, n_times, n_windows)``, row 0 = highest
        frequency, ``round(255 * value)``.
    """
    if isinstance(normalized, torch.Tensor):
        array = normalized.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        array = np.asarray(normalized, dtype=np.float32)
    if array.ndim != 3:
        msg = f"expected (n_windows, n_freqs, n_times), got shape {array.shape}"
        raise ValueError(msg)
    if np.isnan(array).any():
        msg = "normalized spectrogram contains NaN"
        raise ValueError(msg)
    pixels = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(pixels[:, ::-1, :].transpose(1, 2, 0))


def to_three_channels(
    image: NDArray[np.uint8], *, fill: Literal["zeros", "reference"]
) -> NDArray[np.uint8]:
    """Adapt an image to exactly three channels for RGB-only detectors.

    Parameters
    ----------
    image
        ``(H, W, C)`` from `to_image`, ``C <= 3``.
    fill
        What the missing channels contain: ``"zeros"``, or ``"reference"``
        (copies of channel 0, the reference window). Required, so that
        replicating channels is always a visible decision.

    Returns
    -------
    NDArray[np.uint8]
        ``(H, W, 3)``; the input channels come first, in order. A
        three-channel input is returned unchanged.

    Raises
    ------
    ValueError
        If the image has more than three channels; dropping a channel is
        not this adapter's decision.
    """
    if image.ndim != 3:
        msg = f"expected (H, W, C), got shape {image.shape}"
        raise ValueError(msg)
    n_channels = image.shape[2]
    if n_channels == 3:
        return image
    if n_channels > 3:
        msg = f"cannot adapt {n_channels} channels to 3 without dropping some"
        raise ValueError(msg)
    if fill == "zeros":
        extra = np.zeros((*image.shape[:2], 3 - n_channels), dtype=image.dtype)
    elif fill == "reference":
        extra = np.repeat(image[:, :, :1], 3 - n_channels, axis=2)
    else:
        msg = f"fill must be 'zeros' or 'reference', got {fill!r}"
        raise ValueError(msg)
    return np.concatenate([image, extra], axis=2)
