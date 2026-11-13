"""The decoder's input: a crop of the detector's front end around a box.

`decoder_inputs` is the one function that turns a spectrogram and
detection boxes into what the chirp-parameter decoder sees. Training,
evaluation and inference all call it, on spectrograms computed with the
detector's front end (`chirpdetector.spectrogram`, after the detector's
preprocessing), so a crop is the same wherever it is made.

Definition
----------
For a box ``(t0, t1, f0, f1)`` in absolute time and frequency:

1. **Canvas.** The ``canvas_frames`` frames (``CropConfig.canvas_s``, by
   default 3.2 s = one detector tile of 640 frames at 5 ms) centred on the
   frame nearest the box's centre time, shifted to lie inside the frames
   of the spectrogram where it would reach past them (at a recording's
   start and end). The canvas is normalised exactly as a detector tile is
   (`chirpdetector.spectrogram.normalize`, reference level measured on the
   canvas's own frames).
2. **Crop window.** The box widened by ``margin_rel`` (15 %) of its
   duration and height on every side.
3. **Resampling.** The normalised canvas is sampled at ``size = (H, W)``
   points spread evenly over the crop window (end points included: rows
   from ``f0 - margin`` to ``f1 + margin``, **row 0 the lowest frequency**;
   columns from ``t0 - margin`` to ``t1 + margin``), by bilinear
   interpolation between frames and frequency bins. Sample points outside
   the canvas's frames or outside the front end's frequency band read 0.
4. **Quantisation.** ``rint(255 * value)`` as uint8, like a detector image
   (`chirpdetector.spectrogram.to_image`).

Two scalars accompany each crop: the log box duration and the log box
height (floored at ``min_box_duration_s`` and ``min_box_height_hz``); the
crop alone does not say how long or how high the box is.

Why a box-centred canvas and not the detection tile
---------------------------------------------------
The canvas has the detector tile's width and normalisation rule, so the
crop shows the chirp at the contrast the detector saw. Centring it on the
box makes the input independent of where tile boundaries happened to fall
and removes the special case of chirps that cross a tile edge (a 1.2 s
chirp plus margins always fits a 3.2 s canvas). Frames do not depend on how
a recording is chunked (`chirpdetector.spectrogram.framing`), so a crop made
from a span's spectrogram equals the crop made from the whole recording's,
provided the span reaches `CropConfig.context_s` beyond the box centre
(inference widens its spans by that much; see
`chirpdetector.inference.predict_span`).

Domain flags
------------
`DecoderInputs` reports the fraction of the crop's columns outside the
canvas (``outside_time``: the crop window crosses the recording's start or
end, or a box longer than the canvas) and of its rows outside the band
(``outside_freq``: the margin or the box reaches past 500 or 1500 Hz), and
whether the canvas had to be shifted (``canvas_shifted``);
`chirpdetector.params.inference` turns them into domain flags.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from chirpdetector.data.tiles import slice_spectrogram
from chirpdetector.spectrogram import Spectrogram, normalize, reference_level_db

__all__ = [
    "SCALAR_NAMES",
    "CropConfig",
    "DecoderInputs",
    "box_scalars",
    "canvas_range",
    "crop_window",
    "decoder_inputs",
]

SCALAR_NAMES: tuple[str, ...] = ("log_box_duration_s", "log_box_height_hz")
"""The scalar inputs of the decoder, in order."""

FloatArray = npt.NDArray[np.float64]
_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class CropConfig:
    """Geometry of the decoder's input (module docstring).

    Parameters
    ----------
    size
        ``(H, W)``: rows (frequency) and columns (time) of the crop.
    margin_rel
        Margin added on every side, as a fraction of the box's duration
        (left, right) and height (bottom, top).
    canvas_s
        Duration of the normalisation canvas; ``round(canvas_s / hop)``
        frames. The default is the detector's tile (3.2 s).
    min_box_duration_s, min_box_height_hz
        Floors of the scalar inputs (a degenerate box must not give
        ``log(0)``).
    """

    size: tuple[int, int] = (96, 96)
    margin_rel: float = 0.15
    canvas_s: float = 3.2
    min_box_duration_s: float = 0.005
    min_box_height_hz: float = 5.0

    def __post_init__(self) -> None:
        size = tuple(int(v) for v in self.size)
        object.__setattr__(self, "size", size)
        if len(size) != 2 or min(size) < 2:
            msg = f"size must be two integers >= 2, got {self.size}"
            raise ValueError(msg)
        checks = {
            "margin_rel": self.margin_rel >= 0,
            "canvas_s": self.canvas_s > 0,
            "min_box_duration_s": self.min_box_duration_s > 0,
            "min_box_height_hz": self.min_box_height_hz > 0,
        }
        for name, ok in checks.items():
            value = getattr(self, name)
            if not (math.isfinite(value) and ok):
                msg = f"{name} is out of range: {value!r}"
                raise ValueError(msg)

    def canvas_frames(self, hop_s: float) -> int:
        """Frames of the canvas at a frame hop of ``hop_s``."""
        return max(2, round(self.canvas_s / hop_s))

    @property
    def context_s(self) -> float:
        """Spectrogram needed on each side of a box centre for a centred canvas."""
        return 0.5 * self.canvas_s + 0.01

    def to_dict(self) -> dict[str, Any]:
        """JSON form."""
        out = asdict(self)
        out["size"] = list(self.size)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CropConfig:
        """Inverse of `to_dict`; unknown keys raise."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            msg = f"unknown crop keys {unknown}; known: {sorted(known)}"
            raise ValueError(msg)
        values = dict(data)
        if "size" in values:
            values["size"] = tuple(values["size"])
        return cls(**values)


@dataclass(frozen=True)
class DecoderInputs:
    """Decoder inputs of a set of boxes (module docstring).

    Attributes
    ----------
    crops
        ``(n, n_windows, H, W)`` uint8; row 0 is the lowest frequency.
    scalars
        ``(n, 2)`` float32: `SCALAR_NAMES`.
    outside_time, outside_freq
        Fraction of the crop's columns outside the canvas and of its rows
        outside the frequency band.
    canvas_shifted
        The canvas could not be centred on the box (spectrogram edge).
    boxes
        ``(n, 4)`` the boxes, ``t_start, t_end, f_low, f_high``.
    """

    crops: npt.NDArray[np.uint8]
    scalars: npt.NDArray[np.float32]
    outside_time: FloatArray
    outside_freq: FloatArray
    canvas_shifted: npt.NDArray[np.bool_]
    boxes: FloatArray

    def __len__(self) -> int:
        return int(self.crops.shape[0])


def _as_boxes(boxes: npt.ArrayLike) -> FloatArray:
    array = np.asarray(boxes, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 4))
    array = array.reshape(-1, 4)
    bad = ~np.isfinite(array).all(axis=1) | (array[:, 1] <= array[:, 0])
    bad |= array[:, 3] <= array[:, 2]
    if bad.any():
        msg = (
            f"{int(bad.sum())} boxes are not finite with t_end > t_start and "
            f"f_high > f_low, e.g. {array[np.flatnonzero(bad)[0]].tolist()}"
        )
        raise ValueError(msg)
    return array


def box_scalars(boxes: npt.ArrayLike, config: CropConfig) -> npt.NDArray[np.float32]:
    """`SCALAR_NAMES` of ``(n, 4)`` boxes ``t_start, t_end, f_low, f_high``."""
    array = _as_boxes(boxes)
    if len(array) == 0:
        return np.zeros((0, len(SCALAR_NAMES)), dtype=np.float32)
    duration = np.maximum(array[:, 1] - array[:, 0], config.min_box_duration_s)
    height = np.maximum(array[:, 3] - array[:, 2], config.min_box_height_hz)
    return np.stack([np.log(duration), np.log(height)], axis=1).astype(np.float32)


def crop_window(
    box: tuple[float, float, float, float], margin_rel: float
) -> tuple[float, float, float, float]:
    """The box widened by ``margin_rel`` of its duration and height per side."""
    t0, t1, f0, f1 = box
    mt, mf = margin_rel * (t1 - t0), margin_rel * (f1 - f0)
    return t0 - mt, t1 + mt, f0 - mf, f1 + mf


def canvas_range(n_frames: int, centre: int, width: int) -> tuple[int, int, bool]:
    """Frames ``[lo, hi)`` of a canvas of ``width`` frames centred on ``centre``.

    The canvas is shifted to lie inside ``[0, n_frames)``; with fewer than
    ``width`` frames it is all of them. Returns ``(lo, hi, shifted)``.
    """
    if n_frames <= width:
        return 0, n_frames, n_frames < width
    lo = centre - width // 2
    clamped = min(max(lo, 0), n_frames - width)
    return clamped, clamped + width, clamped != lo


def _interpolation_matrix(
    positions: FloatArray, n: int
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Linear interpolation weights ``(len(positions), n)``; rows outside are 0."""
    valid = (positions >= -_EPS) & (positions <= n - 1 + _EPS)
    matrix = np.zeros((positions.size, n))
    if n == 1:
        matrix[valid, 0] = 1.0
        return matrix, valid
    pos = np.clip(positions, 0.0, n - 1.0)
    lo = np.minimum(np.floor(pos).astype(np.int64), n - 2)
    w = pos - lo
    rows = np.flatnonzero(valid)
    matrix[rows, lo[rows]] = 1.0 - w[rows]
    matrix[rows, lo[rows] + 1] += w[rows]
    return matrix, valid


def decoder_inputs(
    spec: Spectrogram, boxes: npt.ArrayLike, config: CropConfig | None = None
) -> DecoderInputs:
    """Crops and scalars of boxes on a spectrogram (module docstring).

    Parameters
    ----------
    spec
        Spectrogram of a recording or of a span of one, computed with the
        decoder's front end (the detector's) after its preprocessing.
    boxes
        ``(n, 4)``: ``t_start_s, t_end_s, f_low_hz, f_high_hz`` (absolute).
    config
        Crop geometry (default `CropConfig()`).

    Returns
    -------
    DecoderInputs

    Raises
    ------
    ValueError
        If a box is not finite or has no extent.
    """
    config = config or CropConfig()
    array = _as_boxes(boxes)
    n = len(array)
    height, width = config.size
    crops = np.zeros((n, spec.n_windows, height, width), dtype=np.uint8)
    outside_t = np.zeros(n)
    outside_f = np.zeros(n)
    shifted = np.zeros(n, dtype=bool)
    hop = spec.layout.hop_s
    df = spec.layout.df_hz
    n_canvas = config.canvas_frames(hop)
    t_first = float(spec.times_s[0])
    f_first = float(spec.freqs_hz[0])
    device = spec.power.device
    for i, box in enumerate(array):
        t0, t1, f0, f1 = crop_window(
            (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            config.margin_rel,
        )
        centre = round((0.5 * (box[0] + box[1]) - t_first) / hop)
        lo, hi, shifted[i] = canvas_range(spec.n_times, centre, n_canvas)
        pos_t = (np.linspace(t0, t1, width) - t_first) / hop - lo
        pos_f = (np.linspace(f0, f1, height) - f_first) / df
        w_t, valid_t = _interpolation_matrix(pos_t, hi - lo)
        w_f, valid_f = _interpolation_matrix(pos_f, spec.n_freqs)
        outside_t[i] = 1.0 - valid_t.mean()
        outside_f[i] = 1.0 - valid_f.mean()
        used_t = np.flatnonzero(w_t.any(axis=0))
        used_f = np.flatnonzero(w_f.any(axis=0))
        if used_t.size == 0 or used_f.size == 0:
            continue
        reference = reference_level_db(slice_spectrogram(spec, lo, hi))
        if reference is None:  # digital silence: the detector sees zeros too
            continue
        a, b = lo + int(used_t[0]), lo + int(used_t[-1]) + 1
        block = normalize(
            slice_spectrogram(spec, a, b), spec.config, reference_db=reference
        )
        block = block[:, int(used_f[0]) : int(used_f[-1]) + 1, :]
        wt = torch.as_tensor(
            w_t[:, used_t[0] : used_t[-1] + 1], dtype=torch.float32, device=device
        )
        wf = torch.as_tensor(
            w_f[:, used_f[0] : used_f[-1] + 1], dtype=torch.float32, device=device
        )
        values = torch.einsum("hf,cft,wt->chw", wf, block, wt)
        crops[i] = np.rint(
            np.clip(values.detach().cpu().numpy(), 0.0, 1.0) * 255.0
        ).astype(np.uint8)
    return DecoderInputs(
        crops=crops,
        scalars=box_scalars(array, config),
        outside_time=outside_t,
        outside_freq=outside_f,
        canvas_shifted=shifted,
        boxes=array,
    )
