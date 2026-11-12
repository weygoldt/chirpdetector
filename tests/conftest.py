"""Shared test setup: torch threads, synthetic trials, the ``data`` marker."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import torch
from gridtools.io import Tracks, save_tracks
from scipy.io import wavfile

# Test workloads are tiny; many intra-op threads only add contention on a
# shared machine (a 1 s FFT test took 9 s with 16 threads under full load).
torch.set_num_threads(min(4, torch.get_num_threads()))

DATA_TRIAL = Path("/mnt/data2/2019_tube_competition/2019-12-09-10_00")

SAMPLERATE_HZ = 20_000.0
FISH_HZ = (700.0, 760.0)
GAINS = (np.array([1.0, 0.5, 0.2, 0.05]), np.array([0.05, 0.2, 0.5, 1.0]))
CHIRP_T_S = 4.0
CHIRP_BOX = (3.93, 4.07, 680.0, 820.0)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests marked ``data`` when the recordings are not mounted."""
    if DATA_TRIAL.exists():
        return
    skip = pytest.mark.skip(reason=f"{DATA_TRIAL} not available")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)


def write_detections(
    path: Path,
    boxes: Sequence[tuple[float, float, float, float]],
    track_ids: Sequence[float],
    emitter_hz: Sequence[float],
    scores: Sequence[float] | np.ndarray | None = None,
) -> None:
    """Write a v1 ``chirpdetector_bboxes.csv`` with the given boxes."""
    n = len(boxes)
    t1, t2, f1, f2 = (np.array([b[i] for b in boxes]) for i in range(4))
    pd.DataFrame(
        {
            "recording": path.parent.name,
            "batch": 0.0,
            "window": 0.0,
            "spec": 0.0,
            "box_ident": np.arange(n),
            "raw_indices": "(0, 0)",
            "freq_range": "(500, 1500)",
            "x1": 0.0,
            "y1": 0.0,
            "x2": 1.0,
            "y2": 1.0,
            "t1": t1,
            "f1": f1,
            "t2": t2,
            "f2": f2,
            "score": np.full(n, 0.9) if scores is None else np.asarray(scores),
            "emitter_eodf": np.asarray(emitter_hz, dtype=float),
            "track_id": np.asarray(track_ids, dtype=float),
        }
    ).to_csv(path, index=False)


def write_tracks(
    directory: Path,
    tracks: Sequence[tuple[np.ndarray, np.ndarray]],
    gains: Sequence[np.ndarray] | None = None,
    dt_s: float = 0.05,
) -> None:
    """Write wavetracker arrays: one ``(times_s, freqs_hz)`` per track id."""
    times = np.round(np.arange(0.0, 60.0, dt_s), 6)
    fund, ident, idx, power = [], [], [], []
    for i, (t, f) in enumerate(tracks):
        frames = np.searchsorted(times, np.round(t, 6))
        fund.append(f)
        ident.append(np.full(len(t), float(i)))
        idx.append(frames)
        g = np.ones(4) if gains is None else gains[i]
        power.append(np.tile(20 * np.log10(g), (len(t), 1)))
    save_tracks(
        Tracks(
            times_s=times,
            freqs_hz=np.concatenate(fund),
            idents=np.concatenate(ident),
            time_indices=np.concatenate(idx),
            powers_db=np.concatenate(power),
        ),
        directory,
    )


def _synthetic_trial(directory: Path) -> Path:
    """Two fish (700, 760 Hz) on 4 electrodes; fish 0 chirps at 4 s.

    Fish 1's track starts at 1 s, so the detection at 0.5 s has a single
    live track. v1 assigns the chirp to track 0.
    """
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    duration_s = 8.0
    t = np.arange(int(duration_s * SAMPLERATE_HZ)) / SAMPLERATE_HZ
    excursion = 100.0 * np.exp(-0.5 * ((t - CHIRP_T_S) / 0.0085) ** 2)  # 20 ms FWHM
    signal = np.zeros((len(t), 4))
    for k, (f, gains) in enumerate(zip(FISH_HZ, GAINS, strict=True)):
        freq = f + (excursion if k == 0 else np.zeros_like(t))
        phase = 2 * np.pi * np.cumsum(freq) / SAMPLERATE_HZ
        signal += np.sin(phase)[:, None] * gains[None, :]
    signal += 0.01 * rng.standard_normal(signal.shape)
    wavfile.write(
        directory / "recording.wav", int(SAMPLERATE_HZ), signal.astype(np.float32)
    )

    track_t = np.round(np.arange(0.0, duration_s, 0.05), 6)
    late = track_t[track_t >= 1.0]
    write_tracks(
        directory,
        [
            (track_t, np.full(len(track_t), FISH_HZ[0])),
            (late, np.full(len(late), FISH_HZ[1])),
        ],
        gains=GAINS,
    )
    boxes = [
        (c - 0.07, c + 0.07, 690.0, 800.0) for c in (0.5, 1.5, 2.5, 3.2, 5.0, 6.0, 7.0)
    ]
    boxes.insert(4, CHIRP_BOX)
    write_detections(
        directory / "chirpdetector_bboxes.csv",
        boxes,
        track_ids=[0.0] * len(boxes),
        emitter_hz=[FISH_HZ[0]] * len(boxes),
    )
    return directory


@pytest.fixture(scope="session")
def synthetic_trial(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A short synthetic trial with a recording, tracks and v1 detections."""
    return _synthetic_trial(tmp_path_factory.mktemp("trials") / "2000-01-01-10_00")


@pytest.fixture
def make_synthetic_trial() -> Callable[[Path], Path]:
    """Factory writing the synthetic trial into a given directory."""
    return _synthetic_trial


def _ramp_trial(directory: Path) -> Path:
    """Tracks and detections (no recording) with ΔF = 2 + 20 t Hz.

    Track 1 starts at 0.2 s. Detection centres at 0.03 + 0.1 k s
    (k = 0..99) fall into the bins single: 2, 0-10: 2, 10-25: 8,
    25-50: 12, 50-100: 25, 100+: 51.
    """
    directory.mkdir(parents=True, exist_ok=True)
    t = np.round(np.arange(0.0, 10.05, 0.05), 6)
    late = t[t >= 0.2]
    write_tracks(
        directory,
        [(t, np.full(len(t), 700.0)), (late, 702.0 + 20.0 * late)],
    )
    centres = 0.03 + 0.1 * np.arange(100)
    boxes = [(c - 0.03, c + 0.03, 690.0, 800.0) for c in centres]
    scores = np.random.default_rng(0).uniform(0.5, 1.0, len(boxes))
    write_detections(
        directory / "chirpdetector_bboxes.csv",
        boxes,
        track_ids=[1.0] * len(boxes),
        emitter_hz=[702.0] * len(boxes),
        scores=scores,
    )
    return directory


@pytest.fixture
def make_ramp_trial() -> Callable[[Path], Path]:
    """Factory writing a ramp trial (see :func:`_ramp_trial`)."""
    return _ramp_trial
