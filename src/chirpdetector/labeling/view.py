"""What the single-candidate frontends show, and how a click becomes a track.

Both frontends (the matplotlib window and the browser) draw a
:class:`CandidateView` and map clicks with the functions here, so their
geometry and snapping are the same code.

A view has two panels sharing the time axis:

``overview``
    ``span_s`` (2 s) centred on the box in time and ``span_hz`` (500 Hz) in
    frequency, centred on the box and the baselines near it (both grow if
    the box plus ``pad_s``/``pad_hz`` and those baselines need more), with
    the selected window (``w`` toggles 204.8 / 51.2 ms).
``zoom``
    The baselines near the box only: from ``zoom_pad_hz`` below the lowest
    to ``zoom_pad_hz`` above the highest live track whose baseline is
    within ``near_below_hz`` below the box bottom or inside the box (all
    live tracks if none is), at least ``zoom_min_span_hz`` wide, always
    with the long window ``zoom_window_s`` (a short window cannot separate
    baselines a few Hz apart). Spectra are zero-padded to ``zoom_bin_hz``
    bins (sinc interpolation; the resolution is still that of the window),
    so a few Hz span many pixels.

Colours run from the ``floor_percentile`` (5th) percentile of a panel's
dB values to its maximum, so the noise floor stays visible as texture
instead of clipping to black.

A track is live if its detections around the box centre are at most
``max_gap_s`` apart. wavetracker tracks often have gaps of a few seconds
(around chirps, too), so the view uses 10 s, more than the 1 s of the
queue's and the evaluation's ΔF bins; :attr:`TrackView.gap_s` tells how far
apart the detections around the box are. Live tracks are numbered 1..n by
their baseline (frequency at the box start), lowest first; ties by track
id. The numbers depend only on the tracks, never on a model's prediction.

A click at ``(t, f)`` snaps to the track whose frequency *at t* is nearest
to f, if that distance is at most the tolerance; otherwise it is rejected.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from gridtools.io import Recording, Tracks

from chirpdetector.labeling.queue import QueueItem
from chirpdetector.labeling.render import band_spectrogram

__all__ = [
    "PANELS",
    "CandidateView",
    "Panel",
    "Prediction",
    "Snap",
    "TrackView",
    "ViewConfig",
    "colour_lut",
    "colourize",
    "data_to_pixel",
    "pixel_to_data",
    "prediction_of",
    "render_view",
    "snap",
    "track_views",
    "view_windows",
]

PANELS = ("overview", "zoom")
"""Panel names, top to bottom."""


@dataclass(frozen=True)
class ViewConfig:
    """How to render a single-candidate view.

    Parameters
    ----------
    span_s
        Time span of both panels, centred on the box (2 s).
    span_hz
        Frequency span of the overview, centred on the box and the
        baselines near it (500 Hz).
    pad_s
        Least time shown before the box start and after the box end (the
        span grows for chirps longer than ``span_s``).
    pad_hz
        Overview: least frequency shown below the lowest and above the
        highest of the box and the baselines near it.
    window_s
        Overview spectrogram window (0.2048 s: 4.9 Hz resolution, separates
        close baselines; 0.0512 s shows the chirp's shape).
    zoom_window_s
        Window of the baseline zoom (whatever the overview shows).
    hop_s
        Hop between spectra.
    floor_percentile
        Lower colour limit of both panels: this percentile of the panel's
        dB values (the noise floor stays visible); the upper is the maximum.
    overview_fft_pad
        Zero-padding factor of the overview's spectra.
    zoom_bin_hz
        Largest frequency bin of the zoom panel (zero padding).
    zoom_pad_hz
        Zoom: frequency shown beyond the outermost near tracks.
    zoom_min_span_hz
        Smallest frequency span of the zoom panel.
    near_below_hz
        Tracks up to this far below the box bottom (and inside the box) are
        near the box and set the zoom panel's range.
    max_gap_s
        A track is live (shown, numbered, snapped to) if its detections
        around the box centre are at most this far apart.
    track_points
        Points of each track's polyline across the window.
    """

    span_s: float = 2.0
    span_hz: float = 500.0
    pad_s: float = 0.25
    pad_hz: float = 40.0
    window_s: float = 0.2048
    zoom_window_s: float = 0.2048
    hop_s: float = 0.0016
    floor_percentile: float = 5.0
    overview_fft_pad: int = 4
    zoom_bin_hz: float = 0.25
    zoom_pad_hz: float = 8.0
    zoom_min_span_hz: float = 24.0
    near_below_hz: float = 25.0
    max_gap_s: float = 10.0
    track_points: int = 64

    def __post_init__(self) -> None:
        if not (self.window_s > 0 and self.zoom_window_s > 0 and self.hop_s > 0):
            msg = "window_s, zoom_window_s and hop_s must be > 0 (seconds)"
            raise ValueError(msg)
        if max(self.window_s, self.zoom_window_s) > 10.0:
            msg = f"window_s is {self.window_s:g} s; is this a sample count?"
            raise ValueError(msg)
        if self.max_gap_s <= 0:
            msg = "max_gap_s must be > 0"
            raise ValueError(msg)
        if min(self.pad_s, self.pad_hz, self.zoom_pad_hz, self.near_below_hz) < 0:
            msg = "paddings must be >= 0"
            raise ValueError(msg)
        if min(self.span_s, self.span_hz, self.zoom_bin_hz) <= 0:
            msg = "span_s, span_hz and zoom_bin_hz must be > 0"
            raise ValueError(msg)
        if not 0 <= self.floor_percentile < 100:
            msg = "floor_percentile must lie in [0, 100)"
            raise ValueError(msg)
        if self.zoom_min_span_hz <= 0:
            msg = "zoom_min_span_hz must be > 0"
            raise ValueError(msg)
        if self.overview_fft_pad < 1 or self.track_points < 2:
            msg = "overview_fft_pad must be >= 1 and track_points >= 2"
            raise ValueError(msg)


@dataclass(frozen=True)
class Panel:
    """One spectrogram panel.

    Parameters
    ----------
    name
        ``overview`` or ``zoom``.
    db
        Power in dB, shape ``(n_freq, n_time)``, row 0 the lowest frequency.
    extent
        ``(t0_s, t1_s, f0_hz, f1_hz)`` of the image's outer pixel edges.
        Drawn into a rectangle, the left edge is ``t0``, the right ``t1``,
        the bottom ``f0`` and the top ``f1`` (see :func:`pixel_to_data`).
    clim
        Colour limits in dB.
    """

    name: str
    db: npt.NDArray[np.float32]
    extent: tuple[float, float, float, float]
    clim: tuple[float, float]


@dataclass(frozen=True)
class TrackView:
    """A live track as drawn and snapped to.

    Parameters
    ----------
    key
        Its number on screen (1..n, by ``baseline_hz``).
    track_id
        wavetracker identity.
    baseline_hz
        Frequency at the box start (the baseline before the chirp),
        linearly interpolated from the track; at the box centre if the
        track is not live at the start.
    times_s, freqs_hz
        Polyline of the track inside the view's time window (where live).
    gap_s
        Time between the track's detections before and after the box
        centre (0 if one is at the centre); above ~1 s the shown track is
        interpolated across a gap.
    """

    key: int
    track_id: int
    baseline_hz: float
    times_s: npt.NDArray[np.float64]
    freqs_hz: npt.NDArray[np.float64]
    gap_s: float = 0.0

    def frequency_at(self, t_s: float) -> float:
        """Frequency of the polyline at ``t_s`` (held constant beyond its ends)."""
        return float(np.interp(t_s, self.times_s, self.freqs_hz))


@dataclass(frozen=True)
class Prediction:
    """A model's predicted emitter of a candidate (from its candidates row)."""

    emitter_hz: float | None
    track_id: int | None
    score: float | None
    source: str


@dataclass(frozen=True)
class CandidateView:
    """Everything a frontend draws for one candidate."""

    item: QueueItem
    window_s: float
    overview: Panel
    zoom: Panel
    tracks: tuple[TrackView, ...]
    render_ms: float

    def panel(self, name: str) -> Panel:
        """The panel of this name.

        Raises
        ------
        KeyError
            If the name is not one of :data:`PANELS`.
        """
        if name == "overview":
            return self.overview
        if name == "zoom":
            return self.zoom
        raise KeyError(name)

    def track(self, key: int) -> TrackView | None:
        """The track numbered ``key`` on screen, if any."""
        return next((t for t in self.tracks if t.key == key), None)

    @property
    def candidates(self) -> tuple[tuple[int, float], ...]:
        """``(track_id, baseline_hz)`` of the live tracks, in key order."""
        return tuple((t.track_id, t.baseline_hz) for t in self.tracks)


@dataclass(frozen=True)
class Snap:
    """A click snapped to a track."""

    track: TrackView
    track_hz: float
    offset_hz: float


def prediction_of(item: QueueItem) -> Prediction | None:
    """The model's predicted emitter of an item (None if it has none)."""
    c = item.candidate_columns
    emitter = c.get("candidate_emitter_hz")
    track = c.get("candidate_track_id")
    if emitter is None and track is None:
        return None
    score = c.get("candidate_score")
    return Prediction(
        emitter_hz=None if emitter is None else float(emitter),
        track_id=None if track is None else int(track),
        score=None if score is None else float(score),
        source=str(c.get("candidate_source") or ""),
    )


def time_window(item: QueueItem, config: ViewConfig) -> tuple[float, float]:
    """Both panels' time span: ``span_s`` centred on the box, or the box
    padded by ``pad_s`` if that is longer."""
    centre = 0.5 * (item.t_start_s + item.t_end_s)
    half = max(
        0.5 * config.span_s, 0.5 * (item.t_end_s - item.t_start_s) + config.pad_s
    )
    return centre - half, centre + half


def track_views(
    tracks: Tracks, item: QueueItem, config: ViewConfig
) -> tuple[TrackView, ...]:
    """The tracks live at the box centre, numbered by baseline.

    Polylines span the panels' time window (:func:`time_window`).
    """
    t0, t1 = time_window(item, config)
    grid = np.linspace(t0, t1, config.track_points)
    found: list[
        tuple[float, int, npt.NDArray[np.float64], npt.NDArray[np.float64], float]
    ] = []
    centre = item.t_center_s
    for ident, f_centre in tracks.frequencies_at(centre, config.max_gap_s).items():
        track = tracks.track(int(ident))
        baseline = float(track.frequency_at(item.t_start_s, config.max_gap_s)[0])
        if not math.isfinite(baseline):
            baseline = float(f_centre)
        f = track.frequency_at(grid, config.max_gap_s)
        live = np.isfinite(f)
        if live.any():
            times, freqs = grid[live], f[live]
        else:  # live at the centre only between grid points
            times, freqs = np.array([centre]), np.array([f_centre])
        k = int(np.searchsorted(track.times_s, centre))
        before = track.times_s[max(k - 1, 0)]
        after = track.times_s[min(k, len(track.times_s) - 1)]
        gap = 0.0 if after == centre or before == centre else float(after - before)
        found.append((baseline, int(ident), times.copy(), freqs.copy(), gap))
    found.sort(key=lambda x: (x[0], x[1]))
    return tuple(
        TrackView(key=k, track_id=ident, baseline_hz=b, times_s=t, freqs_hz=f, gap_s=g)
        for k, (b, ident, t, f, g) in enumerate(found, start=1)
    )


def view_windows(
    item: QueueItem,
    tracks: tuple[TrackView, ...],
    config: ViewConfig,
    samplerate_hz: float,
) -> dict[str, tuple[float, float, float, float]]:
    """Requested ``(t0, t1, f0, f1)`` of the overview and zoom panels."""
    nyquist = 0.5 * samplerate_hz
    t0, t1 = time_window(item, config)
    close = [
        t
        for t in tracks
        if item.f_low_hz - config.near_below_hz <= t.baseline_hz <= item.f_high_hz
    ]
    lo = min([item.f_low_hz, *(float(t.freqs_hz.min()) for t in close)])
    hi = max([item.f_high_hz, *(float(t.freqs_hz.max()) for t in close)])
    lo, hi = lo - config.pad_hz, hi + config.pad_hz
    if hi - lo < config.span_hz:
        centre = 0.5 * (lo + hi)
        lo, hi = centre - 0.5 * config.span_hz, centre + 0.5 * config.span_hz
    overview = (t0, t1, max(0.0, lo), min(nyquist, hi))
    near = close or list(tracks)
    if near:
        lo = min(float(t.freqs_hz.min()) for t in near) - config.zoom_pad_hz
        hi = max(float(t.freqs_hz.max()) for t in near) + config.zoom_pad_hz
    else:
        lo, hi = item.f_low_hz - config.zoom_pad_hz, item.f_low_hz + config.zoom_pad_hz
    if hi - lo < config.zoom_min_span_hz:
        centre = 0.5 * (lo + hi)
        lo, hi = (
            centre - config.zoom_min_span_hz / 2,
            centre + config.zoom_min_span_hz / 2,
        )
    zoom = (t0, t1, max(0.0, lo), min(nyquist, hi))
    return {"overview": overview, "zoom": zoom}


def _panel(
    name: str,
    rec: Recording,
    window: tuple[float, float, float, float],
    config: ViewConfig,
    window_s: float,
    fft_pad: int,
) -> Panel:
    spec = band_spectrogram(
        rec, window, window_s=window_s, hop_s=config.hop_s, fft_pad=fft_pad
    )
    vmax = float(spec.db.max())
    floor = float(np.percentile(spec.db, config.floor_percentile))
    return Panel(
        name=name, db=spec.db, extent=spec.extent, clim=(min(floor, vmax - 1.0), vmax)
    )


def render_view(
    rec: Recording,
    tracks: Tracks,
    item: QueueItem,
    config: ViewConfig,
    window_s: float | None = None,
) -> CandidateView:
    """Render both panels of one candidate.

    Parameters
    ----------
    rec, tracks
        The candidate's recording and tracks.
    item
        The candidate.
    config
        Rendering parameters.
    window_s
        Overview window; ``config.window_s`` if None. The zoom always uses
        ``config.zoom_window_s``.
    """
    tic = time.perf_counter()
    window_s = config.window_s if window_s is None else window_s
    views = track_views(tracks, item, config)
    windows = view_windows(item, views, config, rec.samplerate_hz)
    zoom_window = config.zoom_window_s
    zoom_pad = min(64, max(1, math.ceil(1.0 / (zoom_window * config.zoom_bin_hz))))
    return CandidateView(
        item=item,
        window_s=window_s,
        overview=_panel(
            "overview",
            rec,
            windows["overview"],
            config,
            window_s,
            config.overview_fft_pad,
        ),
        zoom=_panel("zoom", rec, windows["zoom"], config, zoom_window, zoom_pad),
        tracks=views,
        render_ms=1e3 * (time.perf_counter() - tic),
    )


def snap(
    tracks: tuple[TrackView, ...], t_s: float, f_hz: float, tolerance_hz: float
) -> Snap | None:
    """The track nearest to a click, if within ``tolerance_hz``.

    Distances are measured to each track's frequency at the click time
    (:meth:`TrackView.frequency_at`); ties go to the lower key.

    Returns
    -------
    The snapped track, its frequency at ``t_s`` and the signed offset
    ``f_hz - track_hz``; None if no track is within the tolerance.
    """
    best: Snap | None = None
    for track in tracks:
        f = track.frequency_at(t_s)
        if best is None or abs(f_hz - f) < abs(best.offset_hz):
            best = Snap(track=track, track_hz=f, offset_hz=f_hz - f)
    if best is None or abs(best.offset_hz) > tolerance_hz:
        return None
    return best


def pixel_to_data(
    x_px: float,
    y_px: float,
    width_px: float,
    height_px: float,
    extent: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Time and frequency of a point in a panel drawn into a rectangle.

    The image's outer pixel edges are the rectangle's edges: ``x = 0`` is
    ``extent[0]``, ``x = width`` is ``extent[1]``, ``y = 0`` (top) is
    ``extent[3]`` and ``y = height`` (bottom) is ``extent[2]``.

    Raises
    ------
    ValueError
        If the rectangle is empty or the point lies outside it.
    """
    if not (width_px > 0 and height_px > 0):
        msg = f"empty panel rectangle {width_px} x {height_px}"
        raise ValueError(msg)
    if not (0 <= x_px <= width_px and 0 <= y_px <= height_px):
        msg = f"point ({x_px}, {y_px}) outside the panel ({width_px} x {height_px})"
        raise ValueError(msg)
    t0, t1, f0, f1 = extent
    return t0 + x_px / width_px * (t1 - t0), f1 - y_px / height_px * (f1 - f0)


def data_to_pixel(
    t_s: float,
    f_hz: float,
    width_px: float,
    height_px: float,
    extent: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Inverse of :func:`pixel_to_data` (not range checked)."""
    t0, t1, f0, f1 = extent
    return (t_s - t0) / (t1 - t0) * width_px, (f1 - f_hz) / (f1 - f0) * height_px


def colour_lut(name: str = "magma") -> npt.NDArray[np.uint8]:
    """256 x 3 RGB lookup table of a matplotlib colormap."""
    import matplotlib

    cmap = matplotlib.colormaps[name]
    return np.asarray(cmap(np.linspace(0.0, 1.0, 256), bytes=True), dtype=np.uint8)[
        :, :3
    ]


def colourize(
    panel: Panel, lut: npt.NDArray[np.uint8], *, top_down: bool = False
) -> npt.NDArray[np.uint8]:
    """RGB image of a panel, shape ``(n_freq, n_time, 3)``.

    Row 0 is the lowest frequency, or the highest with ``top_down`` (for
    image files and canvases, whose row 0 is the top).
    """
    lo, hi = panel.clim
    scaled = np.clip((panel.db - lo) / (hi - lo), 0.0, 1.0)
    index = np.round(scaled * (len(lut) - 1)).astype(np.intp)
    rgb = lut[index]
    return rgb[::-1] if top_down else rgb
