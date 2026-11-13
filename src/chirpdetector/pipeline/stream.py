"""Detect chirps in a whole recording, chunk by chunk, as `predict_span` would.

Reference
---------
The reference result for a recording is ``predict_span(bundle, recording,
0, duration, params=decoder)``: one tiling of all the recording's frames
(tiles start at frame 0 and step by ``width - overlap``; the last tile is
aligned to the last frame), every tile normalised on its own frames, each
tile owning the centre times between the middles of its overlaps with its
neighbours, and cross-tile duplicates merged (`merge_tiles`). Holding a
6-h spectrogram in memory is not possible, so `detect_recording` computes
the same thing in chunks.

Chunks
------
A chunk is ``chunk_tiles`` consecutive tiles of that global tiling (a
multiple of the detector's batch of 16, so that the detector sees the same
batches as in one pass). For each chunk:

1. the samples of its frames, widened by the decoder's context (half a
   crop canvas), are read and cleaned of clicks (`read_clean`; cleaning is
   defined in absolute blocks, so it does not depend on the chunking);
2. the spectrogram of those frames is computed (`compute_spectrogram_span`;
   frames are those of the whole recording);
3. the chunk's own frames are tiled (`render_tiles`; its tiles are the
   global ones) and run through the detector (`predict_images`), down to
   ``store_threshold``;
4. detections whose centre lies outside their tile's owned interval
   (global) are dropped, as `merge_tiles` drops them;
5. the decoder estimates the parameters of the remaining boxes on the
   widened spectrogram (the crop each box gets in the whole recording).

Reading and cleaning (CPU, the HDD) run ahead of the GPU in a thread pool
(``prefetch`` chunks); reads are serialised in chunk order (`_Turnstile`)
so the disk sees one sequential stream. With ``clean_processes > 0`` the
click removal runs in worker processes on shared memory, so that it does
not compete with the GPU loop for the GIL (on 15 electrodes, cleaning a
chunk takes about 3x as long as detecting in it). After the last chunk, cross-tile duplicates are
merged with `merge_tiles` itself, on groups of detections separated by more
than its tolerance (the twin relation never links detections further
apart, so the result equals one global call).

On the CPU the spectrogram of a chunk is bit-identical to the same frames of
the whole recording, and so are the detections; on CUDA the spectrogram
agrees within 1e-6 relative (`compute_spectrogram_span`), which can flip a
pixel's rounding (see ``scripts/pipeline_equivalence.py``).
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import math
import multiprocessing
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from chirpdetector.bundle import Bundle
from chirpdetector.inference.spans import (
    DUPLICATE_TOLERANCE_S,
    _slice,  # pyright: ignore[reportPrivateUsage]
    merge_tiles,
    render_tiles,
    tile_frames,
    tile_starts,
)
from chirpdetector.models import Detector
from chirpdetector.prediction import predict_images
from chirpdetector.preprocessing import ClickRemoval, can_remove_clicks
from chirpdetector.preprocessing.clicks import CleanPlan, clean_span, plan_clean
from chirpdetector.spectrogram import (
    Spectrogram,
    compute_spectrogram_span,
    plan_span,
)
from chirpdetector.spectrogram.framing import frame_times_s

if TYPE_CHECKING:
    from chirpdetector.params.inference import ParamsEstimator

__all__ = [
    "Chunk",
    "ChunkStats",
    "RecordingSlice",
    "SampleRecording",
    "StreamResult",
    "TileGrid",
    "detect_recording",
    "merge_global",
    "tile_grid",
]

log = logging.getLogger(__name__)


class SampleRecording(Protocol):
    """What the pipeline needs of a recording (`gridtools.io.Recording`)."""

    @property
    def samplerate_hz(self) -> float: ...

    @property
    def n_samples(self) -> int: ...

    @property
    def n_channels(self) -> int: ...

    def read(
        self, start: int, stop: int, channels: Sequence[int] | None = None
    ) -> NDArray[np.float32]: ...


@dataclass(frozen=True)
class RecordingSlice:
    """Samples ``[start, stop)`` of a recording, seen as a recording of its own.

    Time 0 is sample ``start`` of the parent. Used to compare the pipeline
    with `predict_span` on a part of a real recording.
    """

    parent: SampleRecording
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.stop <= self.parent.n_samples:
            msg = f"slice [{self.start}, {self.stop}) outside the recording"
            raise ValueError(msg)

    @property
    def samplerate_hz(self) -> float:
        return float(self.parent.samplerate_hz)

    @property
    def n_samples(self) -> int:
        return self.stop - self.start

    @property
    def n_channels(self) -> int:
        return int(self.parent.n_channels)

    def read(
        self, start: int, stop: int, channels: Sequence[int] | None = None
    ) -> NDArray[np.float32]:
        if not 0 <= start <= stop <= self.n_samples:
            msg = f"span [{start}, {stop}) outside slice of {self.n_samples} samples"
            raise IndexError(msg)
        return self.parent.read(self.start + start, self.start + stop, channels)


@dataclass(frozen=True)
class TileGrid:
    """The global tiling of a recording's frames.

    Attributes
    ----------
    starts
        First frame of every tile.
    width
        Tile width in frames.
    owned_s
        Owned centre-time interval ``[lo, hi)`` of every tile.
    hop
        Hop in samples.
    n_frames
        Frames of the recording.
    """

    starts: NDArray[np.int64]
    width: int
    owned_s: list[tuple[float, float]]
    hop: int
    n_frames: int

    @property
    def n_tiles(self) -> int:
        return len(self.starts)


def tile_grid(bundle: Bundle, samplerate_hz: float, n_samples: int) -> TileGrid:
    """Tiles of the whole recording (as `render_tiles` on all its frames)."""
    config = bundle.spectrogram
    layout = config.layout(samplerate_hz)
    hop = layout.hop_samples
    n_frames = -(-n_samples // hop)
    width, overlap = tile_frames(bundle.tiling, config)
    starts = np.asarray(tile_starts(n_frames, width, overlap), dtype=np.int64)
    # times of the frames that bound the overlaps, exactly as render_tiles
    # computes them from the spectrogram's times_s
    nxt = frame_times_s(starts[1:], samplerate_hz, config)
    last = frame_times_s(starts[:-1] + width - 1, samplerate_hz, config)
    bounds = [0.5 * (float(a) + float(b)) for a, b in zip(nxt, last, strict=True)]
    edges = [-math.inf, *bounds, math.inf]
    owned = list(itertools.pairwise(edges))
    return TileGrid(
        starts=starts, width=width, owned_s=owned, hop=hop, n_frames=n_frames
    )


@dataclass(frozen=True)
class Chunk:
    """Consecutive tiles ``[tile_start, tile_stop)`` of the global grid.

    ``frame_start``/``frame_stop``: the chunk's own frames (its tiles);
    ``read_start``/``read_stop``: recording samples whose frames are
    computed (own frames widened by the decoder context).
    """

    index: int
    tile_start: int
    tile_stop: int
    frame_start: int
    frame_stop: int
    span_start: int
    span_stop: int


def plan_chunks(
    grid: TileGrid,
    n_samples: int,
    chunk_tiles: int,
    context_samples: int,
    tile_range: tuple[int, int] | None = None,
) -> list[Chunk]:
    """Chunks of ``chunk_tiles`` tiles (the last may be shorter).

    ``tile_range`` restricts to tiles ``[a, b)`` (``a`` is rounded down to a
    chunk boundary, so the chunks are those of the full run).
    """
    first, last = (0, grid.n_tiles) if tile_range is None else tile_range
    first = (max(0, first) // chunk_tiles) * chunk_tiles
    last = min(grid.n_tiles, last)
    chunks = []
    for t0 in range(first, last, chunk_tiles):
        t1 = min(t0 + chunk_tiles, last)
        f0 = int(grid.starts[t0])
        f1 = int(grid.starts[t1 - 1]) + grid.width
        start = f0 * grid.hop
        stop = min(n_samples, (f1 - 1) * grid.hop + 1)
        chunks.append(
            Chunk(
                index=t0 // chunk_tiles,
                tile_start=t0,
                tile_stop=t1,
                frame_start=f0,
                frame_stop=f1,
                span_start=max(0, start - context_samples),
                span_stop=min(n_samples, stop + context_samples),
            )
        )
    return chunks


@dataclass
class ChunkStats:
    """Per-chunk statistics (one row of the chunk table)."""

    span_id: int
    t_start_s: float
    t_end_s: float
    n_tiles: int
    n_raw: int
    n_owned: int
    click_removal: bool
    click_fraction: float
    n_click_runs: int
    read_s: float
    clean_s: float
    gpu_s: float


@dataclass
class _Prepared:
    chunk: Chunk
    samples: NDArray[np.float32]
    any_click: NDArray[np.bool_] | None  # per sample of [read_start, read_stop)
    plan_read_start: int
    read_s: float
    clean_s: float
    click_removal: bool
    shm: Any = None  # shared memory holding ``samples`` (process mode)

    def release(self) -> None:
        """Free the samples (and their shared memory)."""
        self.samples = np.zeros((0, 0), dtype=np.float32)
        if self.shm is not None:
            shm, self.shm = self.shm, None
            with contextlib.suppress(BufferError):  # a view still alive (traceback)
                shm.close()
            shm.unlink()


@dataclass
class StreamResult:
    """Output of `detect_recording`.

    ``detections``: merged detections, sorted by centre time, with
    ``tile`` (global tile index), ``span_id`` (chunk), the decoder columns
    and ``click_fraction_box`` (fraction of the box's samples where a click
    was repaired on any electrode). ``chunks``: one `ChunkStats` row per
    chunk. ``n_owned``: detections before the cross-tile merge.
    """

    detections: pd.DataFrame
    chunks: pd.DataFrame
    n_owned: int
    click_removal: bool
    timings: dict[str, float] = field(default_factory=dict)


def _click_config(bundle: Bundle) -> ClickRemoval | None:
    click = bundle.preprocessing.get("click_removal")
    return None if click is None else ClickRemoval.from_dict(dict(click))


def _prepare(
    recording: SampleRecording,
    bundle: Bundle,
    chunk: Chunk,
    click: ClickRemoval | None,
    workers: int,
    order: _Turnstile,
) -> _Prepared:
    """Read a chunk's samples (in chunk order) and clean them of clicks.

    The same reads and arithmetic as `span_spectrogram` / `read_clean`.
    """
    fs, total = float(recording.samplerate_hz), int(recording.n_samples)
    order.wait(chunk.index)
    try:
        plan = plan_span(
            chunk.span_start, chunk.span_stop, total, fs, bundle.spectrogram
        )
        apply = click is not None and can_remove_clicks(
            recording.n_channels, click, samplerate_hz=fs
        )
        cplan = None
        t = time.perf_counter()
        if apply:
            assert click is not None
            cplan = plan_clean(plan.read_start, plan.read_stop, total, fs, click)
            raw = recording.read(cplan.read_start, cplan.read_stop)
        else:
            raw = recording.read(plan.read_start, plan.read_stop)
        read_s = time.perf_counter() - t
    finally:
        order.done(chunk.index)
    t = time.perf_counter()
    any_click = None
    if cplan is not None:
        samples, mask = clean_span(raw, cplan, fs, click, workers=workers)
        any_click = mask.any(axis=1)
    else:
        samples = raw
    return _Prepared(
        chunk=chunk,
        samples=samples,
        any_click=any_click,
        plan_read_start=plan.read_start,
        read_s=read_s,
        clean_s=time.perf_counter() - t,
        click_removal=apply,
    )


def _clean_in_process(
    shm_name: str,
    shape: tuple[int, int],
    cplan: CleanPlan,
    samplerate_hz: float,
    click: dict[str, Any],
    workers: int,
) -> tuple[NDArray[np.bool_], float]:
    """Clean the raw samples in shared memory in a worker process.

    The cleaned samples ``[cplan.start, cplan.stop)`` are written to the
    start of the same buffer; returns the any-electrode click mask and the
    cleaning time.
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        raw = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
        t = time.perf_counter()
        out, mask = clean_span(
            raw, cplan, samplerate_hz, ClickRemoval.from_dict(click), workers=workers
        )
        raw[: out.shape[0]] = out
        del raw
        return mask.any(axis=1), time.perf_counter() - t
    finally:
        shm.close()


def _prepare_in_process(
    recording: SampleRecording,
    bundle: Bundle,
    chunk: Chunk,
    click: ClickRemoval,
    workers: int,
    order: _Turnstile,
    processes: ProcessPoolExecutor,
) -> _Prepared:
    """`_prepare` with the cleaning in a worker process (no GIL contention).

    The parent reads the samples (in chunk order) into shared memory; a
    worker cleans them there. Same arithmetic as `_prepare`.
    """
    fs, total = float(recording.samplerate_hz), int(recording.n_samples)
    order.wait(chunk.index)
    try:
        plan = plan_span(
            chunk.span_start, chunk.span_stop, total, fs, bundle.spectrogram
        )
        cplan = plan_clean(plan.read_start, plan.read_stop, total, fs, click)
        shape = (cplan.n_read, int(recording.n_channels))
        shm = shared_memory.SharedMemory(
            create=True, size=max(1, 4 * shape[0] * shape[1])
        )
        t = time.perf_counter()
        try:
            buf = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
            buf[:] = recording.read(cplan.read_start, cplan.read_stop)
            del buf
        except BaseException:
            shm.close()
            shm.unlink()
            raise
        read_s = time.perf_counter() - t
    finally:
        order.done(chunk.index)
    try:
        any_click, clean_s = processes.submit(
            _clean_in_process, shm.name, shape, cplan, fs, click.to_dict(), workers
        ).result()
    except BaseException:
        shm.close()
        shm.unlink()
        raise
    samples = np.ndarray((cplan.n_samples, shape[1]), dtype=np.float32, buffer=shm.buf)
    return _Prepared(
        chunk=chunk,
        samples=samples,
        any_click=any_click,
        plan_read_start=plan.read_start,
        read_s=read_s,
        clean_s=clean_s,
        click_removal=True,
        shm=shm,
    )


class _Turnstile:
    """Lets chunk reads start in chunk order (one sequential disk stream)."""

    def __init__(self, first: int) -> None:
        self._next = first
        self._cond = threading.Condition()

    def wait(self, index: int) -> None:
        with self._cond:
            self._cond.wait_for(lambda: self._next >= index)

    def done(self, index: int) -> None:
        with self._cond:
            self._next = max(self._next, index + 1)
            self._cond.notify_all()


def _owned(
    df: pd.DataFrame, owned_s: Sequence[tuple[float, float]]
) -> NDArray[np.bool_]:
    """The ownership rule of `merge_tiles` (same arithmetic)."""
    if not len(df):
        return np.zeros(0, dtype=bool)
    centre = 0.5 * (df["t_start_s"].to_numpy() + df["t_end_s"].to_numpy())
    tile = df["image_index"].to_numpy(dtype=np.int64)
    lo = np.array([owned_s[k][0] for k in tile])
    hi = np.array([owned_s[k][1] for k in tile])
    return (centre >= lo) & (centre < hi)


def _box_click_fraction(
    df: pd.DataFrame, any_click: NDArray[np.bool_] | None, first_sample: int, fs: float
) -> NDArray[np.float64]:
    if any_click is None or not len(df):
        return np.full(len(df), np.nan if any_click is None else 0.0)
    cum = np.concatenate([[0], np.cumsum(any_click, dtype=np.int64)])
    n = len(any_click)
    a = np.clip(
        np.ceil(df["t_start_s"].to_numpy() * fs).astype(np.int64) - first_sample, 0, n
    )
    b = np.clip(
        np.floor(df["t_end_s"].to_numpy() * fs).astype(np.int64) + 1 - first_sample,
        0,
        n,
    )
    width = np.maximum(b - a, 1)
    return (cum[b] - cum[a]) / width


def merge_global(
    owned: pd.DataFrame,
    owned_s: Sequence[tuple[float, float]],
    tolerance_s: float = DUPLICATE_TOLERANCE_S,
    group_rows: int = 500,
) -> pd.DataFrame:
    """`merge_tiles` on all owned detections of a recording, in groups.

    Rows are grouped by centre time where consecutive centres are more than
    ``tolerance_s`` apart; the twin relation of `merge_tiles` never links
    detections in different groups, so merging group by group (each in the
    rows' original order) equals one call on everything.

    Parameters
    ----------
    owned
        Detections (``image_index`` = global tile), in tile order.
    owned_s
        Owned interval of every global tile.
    """
    if not len(owned):
        return merge_tiles(owned, owned_s, tolerance_s)
    df = owned.reset_index(drop=True)
    centre = 0.5 * (df["t_start_s"].to_numpy() + df["t_end_s"].to_numpy())
    order = np.argsort(centre, kind="stable")
    sorted_c = centre[order]
    breaks = np.flatnonzero(np.diff(sorted_c) > tolerance_s) + 1
    groups = np.split(order, breaks)
    batch: list[NDArray[np.intp]] = []
    parts = []
    size = 0

    def flush() -> None:
        rows = np.sort(np.concatenate(batch))  # original order within the batch
        parts.append(merge_tiles(df.iloc[rows], owned_s, tolerance_s))

    for g in groups:
        batch.append(g)
        size += len(g)
        if size >= group_rows:
            flush()
            batch, size = [], 0
    if batch:
        flush()
    out = pd.concat(parts, ignore_index=True)
    c = 0.5 * (out["t_start_s"].to_numpy() + out["t_end_s"].to_numpy())
    return out.iloc[np.argsort(c, kind="stable")].reset_index(drop=True)


ProgressFn = Callable[[int, int, float], None]
"""``progress(chunks_done, chunks_total, recording_seconds_done)``."""


def detect_recording(
    bundle: Bundle,
    recording: SampleRecording,
    *,
    name: str,
    detector: Detector,
    estimator: ParamsEstimator | None = None,
    store_threshold: float = 0.05,
    chunk_tiles: int = 48,
    click_workers: int = 4,
    prefetch: int = 3,
    clean_processes: int = 0,
    device: str | None = None,
    t_range_s: tuple[float, float] | None = None,
    progress: ProgressFn | None = None,
    tolerance_s: float = DUPLICATE_TOLERANCE_S,
) -> StreamResult:
    """Detect (and decode) chirps in a whole recording, chunk by chunk.

    Parameters
    ----------
    bundle
        Detector bundle.
    recording
        The recording; time 0 is its first sample.
    name
        Value of the ``recording`` column.
    detector
        The bundle's loaded model.
    estimator
        Optional chirp-parameter decoder (checked against the detector by
        the caller).
    store_threshold
        Lowest score kept.
    chunk_tiles
        Tiles per chunk.
    click_workers, prefetch
        Threads per chunk for click removal; chunks prepared ahead.
    clean_processes
        Clean in this many worker processes (0: in threads of this
        process, where they compete with the GPU loop for the GIL). The
        samples go through shared memory; results are identical.
    device
        Torch device of the spectrogram.
    t_range_s
        Only the chunks covering ``[t0, t1]`` (a dry run); results inside
        equal those of the full run, except for duplicates merged across
        the range's ends.
    progress
        Called after every chunk.
    tolerance_s
        See `merge_tiles`.

    Returns
    -------
    StreamResult
    """
    fs, total = float(recording.samplerate_hz), int(recording.n_samples)
    grid = tile_grid(bundle, fs, total)
    context = 0
    if estimator is not None:
        context = math.ceil(estimator.context_s * fs / grid.hop) * grid.hop
    tile_range = None
    if t_range_s is not None:
        t0, t1 = t_range_s
        frame0 = math.floor(t0 * fs / grid.hop)
        frame1 = math.ceil(t1 * fs / grid.hop)
        a = int(np.searchsorted(grid.starts + grid.width, frame0, side="right"))
        b = int(np.searchsorted(grid.starts, frame1, side="right"))
        tile_range = (a, max(a + 1, b))
    chunks = plan_chunks(grid, total, chunk_tiles, context, tile_range)
    click = _click_config(bundle)
    if click is not None and not can_remove_clicks(
        recording.n_channels, click, samplerate_hz=fs
    ):
        log.warning(
            "%s: %d electrodes are too few for guarded click removal; skipped",
            name,
            recording.n_channels,
        )
    order = _Turnstile(chunks[0].index if chunks else 0)
    owned_parts: list[pd.DataFrame] = []
    stats: list[ChunkStats] = []
    removal = False
    t_all = time.perf_counter()
    gpu_total = 0.0
    use_processes = (
        clean_processes > 0
        and click is not None
        and can_remove_clicks(recording.n_channels, click, samplerate_hz=fs)
    )
    with ExitStack() as stack:
        pool = stack.enter_context(
            ThreadPoolExecutor(max_workers=prefetch, thread_name_prefix="prefetch")
        )
        procs = (
            stack.enter_context(
                ProcessPoolExecutor(
                    max_workers=clean_processes,
                    mp_context=multiprocessing.get_context("spawn"),
                )
            )
            if use_processes
            else None
        )
        pending: list[Future[_Prepared]] = []
        it: Iterator[Chunk] = iter(chunks)

        def submit() -> None:
            nxt = next(it, None)
            if nxt is None:
                return
            if procs is not None:
                assert click is not None
                future = pool.submit(
                    _prepare_in_process,
                    recording,
                    bundle,
                    nxt,
                    click,
                    click_workers,
                    order,
                    procs,
                )
            else:
                future = pool.submit(
                    _prepare, recording, bundle, nxt, click, click_workers, order
                )
            pending.append(future)

        for _ in range(prefetch + 1):
            submit()
        done = 0
        try:
            while pending:
                prepared = pending.pop(0).result()
                submit()
                t = time.perf_counter()
                try:
                    part, stat = _detect_chunk(
                        bundle,
                        recording,
                        prepared,
                        grid,
                        name=name,
                        detector=detector,
                        estimator=estimator,
                        store_threshold=store_threshold,
                        device=device,
                    )
                finally:
                    prepared.release()
                stat.gpu_s = time.perf_counter() - t
                gpu_total += stat.gpu_s
                removal |= prepared.click_removal
                owned_parts.append(part)
                stats.append(stat)
                done += 1
                if progress is not None:
                    progress(done, len(chunks), stat.t_end_s)
        finally:
            # free the shared memory of chunks prepared but not processed
            for future in pending:
                future.cancel()
            for future in pending:
                if not future.cancelled():
                    with contextlib.suppress(BaseException):
                        future.result().release()
    # empty parts would turn every column into object dtype
    nonempty = [p for p in owned_parts if len(p)]
    if nonempty:
        owned = pd.concat(nonempty, ignore_index=True)
    elif owned_parts:
        owned = owned_parts[0]
    else:
        owned = pd.DataFrame()
    t = time.perf_counter()
    if len(owned):
        merged = merge_global(owned, grid.owned_s, tolerance_s)
    else:
        merged = owned.rename(columns={"image_index": "tile"})
    merge_s = time.perf_counter() - t
    chunk_table = pd.DataFrame([s.__dict__ for s in stats])
    timings = {
        "wall_s": time.perf_counter() - t_all,
        "gpu_s": gpu_total,
        "read_s": float(chunk_table["read_s"].to_numpy().sum())
        if len(chunk_table)
        else 0.0,
        "clean_s": float(chunk_table["clean_s"].to_numpy().sum())
        if len(chunk_table)
        else 0.0,
        "merge_s": merge_s,
    }
    return StreamResult(
        detections=merged,
        chunks=chunk_table,
        n_owned=len(owned),
        click_removal=removal,
        timings=timings,
    )


def _detect_chunk(
    bundle: Bundle,
    recording: SampleRecording,
    prepared: _Prepared,
    grid: TileGrid,
    *,
    name: str,
    detector: Detector,
    estimator: ParamsEstimator | None,
    store_threshold: float,
    device: str | None,
) -> tuple[pd.DataFrame, ChunkStats]:
    chunk = prepared.chunk
    fs, total = float(recording.samplerate_hz), int(recording.n_samples)
    config = bundle.spectrogram
    plan = plan_span(chunk.span_start, chunk.span_stop, total, fs, config)
    wide = compute_spectrogram_span(prepared.samples, plan, config, device=device)
    first = chunk.frame_start - wide.frame_start
    last = chunk.frame_stop - wide.frame_start
    own: Spectrogram = _slice(wide, first, last)
    tiles = render_tiles(own, bundle.tiling)
    expected = (
        grid.starts[chunk.tile_start : chunk.tile_stop] - chunk.frame_start
    ).tolist()
    if tiles.starts != expected:  # pragma: no cover - guards the chunk arithmetic
        msg = f"chunk {chunk.index}: tiles {tiles.starts[:3]}... differ from the global grid"
        raise RuntimeError(msg)
    raw = predict_images(
        bundle,
        tiles.images,
        tiles.geometries,
        recordings=[name] * len(tiles.images),
        score_threshold=store_threshold,
        detector=detector,
    )
    raw["image_index"] = raw["image_index"].to_numpy(dtype=np.int64) + chunk.tile_start
    keep = _owned(raw, grid.owned_s)
    owned = raw.loc[keep].reset_index(drop=True)
    if estimator is not None:
        values = estimator.estimate(wide, owned) if len(owned) else _empty_params()
        values.index = owned.index
        owned = pd.concat([owned, values], axis=1)
    owned["span_id"] = np.full(len(owned), chunk.index, dtype=np.int64)
    owned["click_fraction_box"] = _box_click_fraction(
        owned, prepared.any_click, prepared.plan_read_start, fs
    )
    # click statistics of the chunk's own samples (no double counting)
    own_a, own_b = chunk.frame_start * grid.hop, min(total, chunk.frame_stop * grid.hop)
    frac, runs = float("nan"), 0
    if prepared.any_click is not None:
        lo = max(0, own_a - prepared.plan_read_start)
        hi = max(lo, min(len(prepared.any_click), own_b - prepared.plan_read_start))
        seg = prepared.any_click[lo:hi]
        frac = float(seg.mean()) if len(seg) else 0.0
        runs = int(
            np.count_nonzero(np.diff(seg.astype(np.int8)) == 1) + int(seg[:1].any())
        )
    stat = ChunkStats(
        span_id=chunk.index,
        t_start_s=own_a / fs,
        t_end_s=own_b / fs,
        n_tiles=chunk.tile_stop - chunk.tile_start,
        n_raw=len(raw),
        n_owned=len(owned),
        click_removal=prepared.click_removal,
        click_fraction=frac,
        n_click_runs=runs,
        read_s=prepared.read_s,
        clean_s=prepared.clean_s,
        gpu_s=0.0,
    )
    return owned, stat


def _empty_params() -> pd.DataFrame:
    from chirpdetector.params.inference import ESTIMATE_COLUMNS

    data: dict[str, Any] = {c: pd.Series(dtype=float) for c in ESTIMATE_COLUMNS}
    data["params_ok"] = pd.Series(dtype=bool)
    data["params_flag"] = pd.Series(dtype=object)
    return pd.DataFrame(data)
