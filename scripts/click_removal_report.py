"""Validate click removal: clicks removed, chirps intact.

Writes results (CSV/JSON) and figures under ``--out`` (default
``/mnt/data1/chirp_v2/preprocessing/clicks``), one subcommand per section:

- ``scan``: clicks in whole recordings, as an event table (``scans/``).
- ``synthetic``: two fish with chirps, coloured noise and injected clicks.
- ``real``: the four hand-labelled recordings.
- ``v1``: v1 detections that coincide with a click (needs ``scan``).
- ``throughput``: seconds of 15-channel recording cleaned per second.
- ``tables``: markdown tables of all results (``tables.md``).

Run from the chirpdetector checkout, e.g.::

    uv run python scripts/click_removal_report.py scan --config default
    uv run python scripts/click_removal_report.py synthetic
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gridtools.io import Recording, open_recording, read_labels, read_v1_detections
from gridtools.sim import (
    APTERONOTUS_LEPTORHYNCHUS,
    ColoredNoiseParams,
    Harmonics,
    colored_noise,
    synthesize_eod,
)
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, sosfiltfilt

from chirpdetector.preprocessing import (
    ClickRemoval,
    clean_span,
    click_spans,
    plan_clean,
    read_clean,
    remove_clicks,
)
from chirpdetector.spectrogram import SpectrogramConfig, compute_spectrogram

log = logging.getLogger("click_removal_report")

OUT = Path("/mnt/data1/chirp_v2/preprocessing/clicks")
RECORDINGS = {
    "2019-12-09-10_00": Path("/mnt/data2/2019_tube_competition/2019-12-09-10_00"),
    "2020-06-03-10_10": Path("/mnt/data2/2019_tube_competition/2020-06-03-10_10"),
    "2022-05-12-10_00": Path("/mnt/data2/2022_tube_competition/raw/2022-05-12-10_00"),
    "2022-06-01-10_00": Path("/mnt/data2/2022_tube_competition/raw/2022-06-01-10_00"),
}
V1_RECORDINGS = ("2019-12-09-10_00", "2020-06-03-10_10")
LABELS = "traces-grid1-editable-labels.csv"

#: Settings compared throughout. "default" is audian's / wavetracker's.
CONFIGS: dict[str, ClickRemoval] = {
    "default": ClickRemoval(),
    "band0.40": ClickRemoval(high_band=0.40),
    "thr24": ClickRemoval(threshold=24.0, extent_threshold=16.0),
    "thr48": ClickRemoval(threshold=48.0, extent_threshold=32.0),
    "band0.40thr24": ClickRemoval(
        high_band=0.40, threshold=24.0, extent_threshold=16.0
    ),
    "guard": ClickRemoval(coincidence=0.5),
}

#: Band of the chirpdetector spectrogram (SpectrogramConfig default).
BAND_HZ = (500.0, 1500.0)
#: Analysis windows of the box powers: 205 ms and 51 ms at 20 kHz.
NFFTS = (4096, 1024)
#: Data around a box its local spectrogram is computed from [s]: half the
#: longest window so that no frame in the box sees the excerpt's edge, plus
#: chirp-free frames on both sides for the background spectrum.
BOX_PAD_S = 0.6


def git_describe() -> str:
    """Short commit of the chirpdetector checkout, with ``-dirty`` if so."""
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=10"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _json_default(value: object) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    msg = f"not JSON serialisable: {type(value)}"
    raise TypeError(msg)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=_json_default))
    tmp.replace(path)


def meta() -> dict[str, Any]:
    return {
        "commit": git_describe(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "loadavg": os.getloadavg(),
        "configs": {k: v.to_dict() for k, v in CONFIGS.items()},
    }


# ------------------------------------------------------------- spectra


def band_footprint_db(
    a: NDArray[np.floating[Any]],
    b: NDArray[np.floating[Any]],
    centre: int,
    samplerate_hz: float,
    nfft: int = 1024,
) -> float:
    """Median over `BAND_HZ` of ``10 log10(P_a / P_b)`` around ``centre``.

    One Hann window of ``nfft`` samples, power summed over electrodes. The
    median over bins is the broadband lift a click puts under the fish
    lines; NaN where the window does not fit.
    """
    lo = centre - nfft // 2
    hi = lo + nfft
    if lo < 0 or hi > len(a):
        return float("nan")
    w = np.hanning(nfft + 1)[:-1, None]
    f = np.fft.rfftfreq(nfft, 1.0 / samplerate_hz)
    sel = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1])
    p_a = (np.abs(np.fft.rfft(w * a[lo:hi], axis=0)) ** 2).sum(1)[sel]
    p_b = (np.abs(np.fft.rfft(w * b[lo:hi], axis=0)) ** 2).sum(1)[sel]
    return float(
        np.median(10 * np.log10(np.maximum(p_a, 1e-30) / np.maximum(p_b, 1e-30)))
    )


def spec_config(nfft: int) -> SpectrogramConfig:
    """Single-window, electrode-summed spectrogram for box powers."""
    return SpectrogramConfig(
        window_lengths_samples=(nfft,),
        hop_samples=50,
        f_min_hz=300.0,
        f_max_hz=2400.0,
        aggregation="sum",
    )


def spectrogram(
    samples: NDArray[np.floating[Any]], nfft: int, t0_s: float = 0.0
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``(times_s, freqs_hz, power)`` of the electrode-summed spectrogram."""
    spec = compute_spectrogram(
        torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)),
        spec_config(nfft),
        t0_s=t0_s,
    )
    return spec.times_s, spec.freqs_hz, spec.power[0].numpy().astype(np.float64)


def box_power(
    times: NDArray[np.float64],
    freqs: NDArray[np.float64],
    power: NDArray[np.float64],
    box: Sequence[float],
) -> float:
    """Summed power inside ``(t0, t1, f0, f1)``."""
    t0, t1, f0, f1 = box
    ti = (times >= t0) & (times <= t1)
    fi = (freqs >= f0) & (freqs <= f1)
    return float(power[np.ix_(fi, ti)].sum())


def db(x: Any) -> Any:
    return 10 * np.log10(np.maximum(x, 1e-30))


def off_line_bins(
    freqs: NDArray[np.float64], background: NDArray[np.float64], box: Sequence[float]
) -> NDArray[np.bool_]:
    """Bins of the box away from the stationary lines (EODs, harmonics).

    A line is a bin whose background power exceeds 4x the median of the
    box's bins; it and two bins on either side are left out, so that the
    beats of the baseline EODs do not count as chirp.
    """
    fi = (freqs >= box[2]) & (freqs <= box[3])
    level = background[fi]
    line = level > 4 * np.median(level)
    line = maximum_filter1d(line, 5, mode="constant", cval=False)
    keep = np.zeros(len(freqs), dtype=bool)
    keep[np.flatnonzero(fi)[~line]] = True
    return keep


def excursion(
    times: NDArray[np.float64],
    power: NDArray[np.float64],
    box: Sequence[float],
    background: NDArray[np.float64],
    bins: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Per frame of the box, the power above the background off the lines."""
    ti = (times >= box[0]) & (times <= box[1])
    excess = power[np.ix_(bins, ti)] - background[bins, None]
    return np.clip(excess, 0.0, None).sum(axis=0)


def box_change(
    reference: NDArray[np.floating[Any]],
    changed: NDArray[np.floating[Any]],
    box: Sequence[float],
    t0_s: float,
) -> dict[str, float]:
    """A box of ``changed`` against the same box of ``reference``.

    Both are local excerpts starting at ``t0_s``. For each window ``N`` in
    `NFFTS`:

    - ``delta_db_N``: plain box power, changed over reference. Dominated by
      the baseline EOD lines, so insensitive to the chirp itself;
    - ``error_db_N``: power of the difference over the reference's box
      power (-inf when nothing in the box changed);
    - ``chirp_db_N``: the chirp's power, changed over reference: in the
      box, off the stationary lines (`off_line_bins`), the power above the
      background spectrum (per-bin median of the reference's frames whose
      window does not reach the box), summed;
    - ``chirp_peak_db_N``: the same in the frame where the reference's
      chirp power peaks: what a cut through the chirp's peak costs;
    - ``chirp_error_db_N``: power of the difference off the lines over the
      reference's chirp power.
    """
    out: dict[str, float] = {}
    diff = changed.astype(np.float64) - reference.astype(np.float64)
    t_stop = t0_s + len(reference) / SYN_RATE
    for nfft in NFFTS:
        times, freqs, p_ref = spectrogram(reference, nfft, t0_s)
        p_new = spectrogram(changed, nfft, t0_s)[2]
        ref = box_power(times, freqs, p_ref, box)
        out[f"delta_db_{nfft}"] = float(db(box_power(times, freqs, p_new, box) / ref))
        half = nfft / 2 / SYN_RATE
        quiet = (
            (times >= t0_s + half)
            & (times <= t_stop - half)
            & ((times < box[0] - half) | (times > box[1] + half))
        )
        exc_total = float("nan")
        out[f"chirp_db_{nfft}"] = float("nan")
        out[f"chirp_peak_db_{nfft}"] = float("nan")
        bins = np.zeros(len(freqs), dtype=bool)
        if quiet.sum() >= 3:
            background = np.median(p_ref[:, quiet], axis=1)
            bins = off_line_bins(freqs, background, box)
            e_ref = excursion(times, p_ref, box, background, bins)
            e_new = excursion(times, p_new, box, background, bins)
            exc_total = float(e_ref.sum())
            if exc_total > 0:
                out[f"chirp_db_{nfft}"] = float(db(e_new.sum() / exc_total))
                peak = int(np.argmax(e_ref))
                out[f"chirp_peak_db_{nfft}"] = float(db(e_new[peak] / e_ref[peak]))
        if np.any(diff):
            p_err = spectrogram(diff, nfft, t0_s)[2]
            out[f"error_db_{nfft}"] = float(
                db(box_power(times, freqs, p_err, box) / ref)
            )
            ti = (times >= box[0]) & (times <= box[1])
            err_off = float(p_err[np.ix_(bins, ti)].sum())
            out[f"chirp_error_db_{nfft}"] = (
                float(db(err_off / exc_total)) if exc_total > 0 else float("nan")
            )
        else:
            out[f"error_db_{nfft}"] = -np.inf
            out[f"chirp_error_db_{nfft}"] = -np.inf
    return out


def unchanged_box() -> dict[str, float]:
    out: dict[str, float] = {}
    for nfft in NFFTS:
        out[f"delta_db_{nfft}"] = 0.0
        out[f"error_db_{nfft}"] = -np.inf
        out[f"chirp_db_{nfft}"] = 0.0
        out[f"chirp_peak_db_{nfft}"] = 0.0
        out[f"chirp_error_db_{nfft}"] = -np.inf
    return out


# ---------------------------------------------------------------- events


@dataclass
class Events:
    """Click events: runs of repaired samples on any electrode."""

    start: NDArray[np.int64]
    stop: NDArray[np.int64]
    channels: NDArray[np.int64]  # bit mask of the electrodes repaired
    peak_change: NDArray[np.float64]  # max |raw - clean| over the event
    band_db: NDArray[np.float64]  # band_footprint_db(raw, clean) at the peak

    @property
    def n_channels(self) -> NDArray[np.int64]:
        return np.array([int(c).bit_count() for c in self.channels], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.start)

    def save(self, path: Path, **info: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            start=self.start,
            stop=self.stop,
            channels=self.channels,
            peak_change=self.peak_change,
            band_db=self.band_db,
            meta=json.dumps(info, default=_json_default),
        )

    @classmethod
    def load(cls, path: Path) -> tuple[Events, dict[str, Any]]:
        with np.load(path) as z:
            band = z["band_db"] if "band_db" in z else z["band_median_db"]
            events = cls(z["start"], z["stop"], z["channels"], z["peak_change"], band)
            info = json.loads(str(z["meta"]))
        return events, info

    @classmethod
    def empty(cls) -> Events:
        e = np.zeros(0, dtype=np.int64)
        return cls(e, e, e, np.zeros(0), np.zeros(0))


def events_from_mask(
    raw: NDArray[np.floating[Any]],
    clean: NDArray[np.floating[Any]],
    mask: NDArray[np.bool_],
    offset: int,
    samplerate_hz: float,
) -> Events:
    """Event table of one chunk; ``raw[0]`` is recording sample ``offset``."""
    spans = click_spans(mask)
    if len(spans) == 0:
        return Events.empty()
    weights = 1 << np.arange(mask.shape[1], dtype=np.int64)
    channels, peak, band = [], [], []
    for s, e in spans:
        channels.append(int((weights * mask[s:e].any(0)).sum()))
        diff = np.abs(raw[s:e].astype(np.float64) - clean[s:e])
        peak.append(float(diff.max()))
        centre = int(s + np.argmax(diff.max(1)))
        band.append(band_footprint_db(raw, clean, centre, samplerate_hz))
    return Events(
        spans[:, 0] + offset,
        spans[:, 1] + offset,
        np.array(channels, dtype=np.int64),
        np.array(peak),
        np.array(band),
    )


def concat_events(parts: list[Events]) -> Events:
    """Concatenate chunk tables, merging events that touch at a seam."""
    parts = [p for p in parts if len(p)]
    if not parts:
        return Events.empty()
    start = np.concatenate([p.start for p in parts])
    stop = np.concatenate([p.stop for p in parts])
    channels = np.concatenate([p.channels for p in parts])
    peak = np.concatenate([p.peak_change for p in parts])
    band = np.concatenate([p.band_db for p in parts])
    keep = np.ones(len(start), dtype=bool)
    last = 0
    for i in range(1, len(start)):
        if start[i] <= stop[last]:
            stop[last] = max(stop[last], stop[i])
            channels[last] |= channels[i]
            peak[last] = max(peak[last], peak[i])
            band[last] = np.nanmax([band[last], band[i]])
            keep[i] = False
        else:
            last = i
    return Events(start[keep], stop[keep], channels[keep], peak[keep], band[keep])


def clean_range(
    rec: Recording,
    start: int,
    stop: int,
    config: ClickRemoval,
    *,
    chunk_s: float = 60.0,
    workers: int = 8,
    keep: bool = False,
    progress: Callable[[int], None] | None = None,
) -> tuple[Events, int, NDArray[np.float32] | None, NDArray[np.bool_] | None]:
    """Clean recording samples ``[start, stop)`` chunk by chunk.

    Returns the events, the number of repaired samples (all electrodes),
    and with ``keep`` the cleaned samples and their mask.
    """
    rate = rec.samplerate_hz
    chunk = config.block_samples(rate) * max(1, round(chunk_s / config.block_s))
    parts: list[Events] = []
    cleaned: list[NDArray[np.float32]] = []
    masks: list[NDArray[np.bool_]] = []
    repaired = 0
    a = start
    while a < stop:
        b = min(stop, a + chunk)
        clean, mask = read_clean(rec, a, b, config, workers=workers)
        raw = rec.read(a, b)
        repaired += int(mask.sum())
        parts.append(events_from_mask(raw, clean, mask, a, rate))
        if keep:
            cleaned.append(clean)
            masks.append(mask)
        if progress is not None:
            progress(b)
        a = b
    if keep:
        return (
            concat_events(parts),
            repaired,
            np.concatenate(cleaned),
            np.concatenate(masks),
        )
    return concat_events(parts), repaired, None, None


def scan_path(out: Path, name: str, config: str) -> Path:
    return out / "scans" / f"scan_{name}_{config}.npz"


def cmd_scan(args: argparse.Namespace) -> None:
    for config_name in args.config:
        config = CONFIGS[config_name]
        for name in args.recordings or list(V1_RECORDINGS):
            rec = open_recording(RECORDINGS[name])
            path = scan_path(args.out, name, config_name)
            if path.exists() and not args.force:
                log.info("%s exists, skipping", path)
                continue
            t0 = time.perf_counter()
            last = [t0]

            def progress(
                b: int, rec: Recording = rec, t0: float = t0, last: list[float] = last
            ) -> None:
                now = time.perf_counter()
                if now - last[0] > 60:
                    last[0] = now
                    log.info(
                        "%s: %.0f / %.0f s (%.1fx realtime)",
                        rec.path.parent.name,
                        b / rec.samplerate_hz,
                        rec.duration_s,
                        b / rec.samplerate_hz / (now - t0),
                    )

            events, repaired, _, _ = clean_range(
                rec, 0, rec.n_samples, config, workers=args.workers, progress=progress
            )
            elapsed = time.perf_counter() - t0
            events.save(
                path,
                recording=name,
                config=config_name,
                config_dict=config.to_dict(),
                n_samples=rec.n_samples,
                n_channels=rec.n_channels,
                samplerate_hz=rec.samplerate_hz,
                repaired_samples=repaired,
                elapsed_s=elapsed,
                workers=args.workers,
                loadavg=os.getloadavg(),
                commit=git_describe(),
            )
            log.info(
                "%s %s: %d events, %.4f%% repaired, %.0f s",
                name,
                config_name,
                len(events),
                100 * repaired / (rec.n_samples * rec.n_channels),
                elapsed,
            )


# ------------------------------------------------------------- synthetic

SYN_RATE = 20_000.0
SYN_COLUMNS, SYN_ROWS, SYN_SPACING_M = 3, 5, 0.5
#: name: (excursion Hz, FWHM s, amplitude dip)
CHIRP_TYPES = {
    "small": (60.0, 0.010, 0.15),
    "big": (300.0, 0.030, 0.55),
    "huge": (600.0, 0.100, 0.6),
}
#: EOD waveforms: gridtools' A. leptorhynchus (from thunderfish), and the
#: relative harmonic amplitudes measured on 2022-05-12-10_00 electrode 10
#: (642 Hz fish next to the electrode; 4 s at 10300 s; phases set to 0).
HARMONICS = {
    "model": APTERONOTUS_LEPTORHYNCHUS,
    "measured": Harmonics(
        amplitudes=(1.0, 0.685, 0.333, 0.180, 0.078, 0.050, 0.026, 0.0055),
        phases_rad=(0.0,) * 8,
    ),
}
LOUDNESS = (3.0, 10.0, 30.0, 100.0, 300.0)
F0_PAIRS = ((650.0, 800.0), (900.0, 750.0), (1100.0, 950.0))


@dataclass
class Scene:
    """A synthetic grid recording with its ground truth."""

    signal: NDArray[np.float32]  # with clicks
    truth: NDArray[np.float32]  # without clicks
    clicks: pd.DataFrame
    chirps: pd.DataFrame
    gains: NDArray[np.float64]  # (n_fish, n_electrodes), fish amplitude included
    high_sigma: float  # robust sd of the noise above 0.2 fs (noise sd is 1)


def electrode_positions() -> NDArray[np.float64]:
    xs, ys = np.meshgrid(
        np.arange(SYN_COLUMNS) * SYN_SPACING_M, np.arange(SYN_ROWS) * SYN_SPACING_M
    )
    return np.stack([xs.ravel(), ys.ravel()], axis=1)


def fish_gains(position: NDArray[np.float64], near_m: float = 0.1) -> NDArray:
    """Falloff ``a^2 / (r^2 + a^2)`` with distance: 1 at an electrode."""
    r2 = ((electrode_positions() - position) ** 2).sum(1)
    return near_m**2 / (r2 + near_m**2)


def high_band_sigma(noise: NDArray[np.floating[Any]], rate: float) -> float:
    """Robust sd of the noise above 0.2 fs (the default click search band)."""
    sos = butter(6, 0.2 * rate, "highpass", fs=rate, output="sos")
    high = sosfiltfilt(sos, np.asarray(noise, dtype=np.float64), axis=0)
    return float(np.median(np.median(np.abs(high), axis=0) / 0.6745))


def make_click(
    rng: np.random.Generator, duration_s: float, kind: str, rate: float
) -> NDArray[np.float64]:
    """Unit-peak broadband click lasting ``duration_s``.

    ``impulse``: sharp onset, exponential decay to 1% over the duration.
    ``burst``: white noise under a Hann window.
    """
    n = max(2, round(duration_s * rate))
    t = np.arange(n) / rate
    if kind == "impulse":
        x = np.exp(-t * np.log(100.0) / duration_s) * rng.choice([-1.0, 1.0])
    else:
        x = rng.standard_normal(n) * np.hanning(n + 2)[1:-1]
    return x / np.abs(x).max()


def synth_scene(
    rng: np.random.Generator,
    *,
    loudness: float,
    f0_hz: tuple[float, float] = (900.0, 750.0),
    harmonics: Harmonics = APTERONOTUS_LEPTORHYNCHUS,
    duration_s: float = 30.0,
    clicks_per_s: float = 2.0,
    noise_exponent: float = 1.0,
) -> Scene:
    """Two chirping fish, coloured noise and clicks on a 3 x 5 grid.

    Fish 0 sits 5 cm from electrode 7, fish 1 between electrodes 3, 4, 6
    and 7. ``loudness`` is fish 0's EOD amplitude on electrode 7 in units
    of the robust sd of the noise above 0.2 fs, the band clicks are
    searched in; fish 1 peaks 4x lower. The noise has sd 1 and spectrum
    ``1/f**noise_exponent``. Both fish chirp every 1.3 s (small, big, huge
    in turn). Clicks (Poisson, ``clicks_per_s``): 0.1-5 ms (log-uniform),
    impulse or noise burst, peak 10^2-10^4 x the noise sd (log-uniform),
    on one electrode or on 2-15 with gains 0.1-1. Two more clicks sit at
    the peak of the first big chirp of each fish: on electrodes 7, 3, 4, 6
    (fish 0's loudest) and on electrode 6 only.
    """
    rate = SYN_RATE
    n = round(duration_s * rate)
    n_el = SYN_COLUMNS * SYN_ROWS
    t = np.arange(n) / rate
    noise = colored_noise(
        rng, n, n_el, rate, ColoredNoiseParams(sd=1.0, exponent=noise_exponent)
    ).astype(np.float64)
    sigma_high = high_band_sigma(noise, rate)
    pos = electrode_positions()
    positions = [pos[7] + [0.05, 0.0], (pos[3] + pos[4] + pos[6] + pos[7]) / 4]
    peaks = [loudness * sigma_high, loudness * sigma_high / 4]
    gains = np.stack(
        [
            fish_gains(p) / fish_gains(p).max() * a
            for p, a in zip(positions, peaks, strict=True)
        ]
    )
    # the fundamental's amplitude is what `loudness` sets
    scale = 1.0 / harmonics.amplitudes[0]

    truth = noise.copy()
    rows = []
    types = list(CHIRP_TYPES)
    for fish, f0 in enumerate(f0_hz):
        freq = np.full(n, f0)
        env = np.ones(n)
        for k, tc in enumerate(np.arange(1.5 + 0.35 * fish, duration_s - 1.5, 1.3)):
            name = types[k % len(types)]
            df, fwhm, dip = CHIRP_TYPES[name]
            g = np.exp(-0.5 * ((t - tc) / (fwhm / 2.3548)) ** 2)
            freq += df * g
            env -= dip * g
            rows.append(
                {
                    "fish": fish,
                    "t_s": float(tc),
                    "type": name,
                    "f0_hz": f0,
                    "df_hz": df,
                    "fwhm_s": fwhm,
                    "t0": tc - 1.5 * fwhm - 0.01,
                    "t1": tc + 1.5 * fwhm + 0.01,
                    "f_low": f0 - 15.0,
                    "f_high": f0 + df + 25.0,
                }
            )
        eod = synthesize_eod(
            freq,
            env * scale,
            rate,
            harmonics,
            phase0_rad=float(rng.uniform(0, 2 * np.pi)),
        )
        truth += eod[:, None].astype(np.float64) * gains[fish][None, :]
    chirps = pd.DataFrame(rows).sort_values("t_s").reset_index(drop=True)

    times = list(
        rng.uniform(0.2, duration_s - 0.2, rng.poisson(clicks_per_s * duration_s))
    )
    forced = [
        float(chirps[(chirps["fish"] == f) & (chirps["type"] == "big")]["t_s"].iloc[0])
        for f in (0, 1)
    ]
    signal = truth.copy()
    click_rows = []
    for i, tc in enumerate(times + forced):
        at_chirp = i >= len(times)
        duration = float(np.exp(rng.uniform(np.log(1e-4), np.log(5e-3))))
        kind = str(rng.choice(["impulse", "burst"]))
        amplitude = float(10 ** rng.uniform(2.0, 4.0))
        if at_chirp:
            electrodes = np.array([7, 3, 4, 6]) if i == len(times) else np.array([6])
            el_gain = np.ones(len(electrodes))
        elif rng.random() < 0.5:
            electrodes = rng.choice(n_el, 1)
            el_gain = np.ones(1)
        else:
            k = int(rng.integers(2, n_el + 1))
            electrodes = np.sort(rng.choice(n_el, k, replace=False))
            el_gain = rng.uniform(0.1, 1.0, k)
            el_gain /= el_gain.max()
        shape = make_click(rng, duration, kind, rate)
        s = round(tc * rate)
        e = min(n, s + len(shape))
        for el, g in zip(electrodes, el_gain, strict=True):
            signal[s:e, el] += amplitude * g * shape[: e - s]
        click_rows.append(
            {
                "start": s,
                "stop": e,
                "duration_s": duration,
                "kind": kind,
                "amplitude": amplitude,
                "electrodes": electrodes.tolist(),
                "gains": el_gain.tolist(),
                "at_chirp": at_chirp,
            }
        )
    clicks = pd.DataFrame(click_rows).sort_values("start").reset_index(drop=True)
    return Scene(
        signal.astype(np.float32),
        truth.astype(np.float32),
        clicks,
        chirps,
        gains,
        sigma_high,
    )


def window_changed(
    a: NDArray[np.floating[Any]], b: NDArray[np.floating[Any]], t0: float, t1: float
) -> bool:
    """Whether ``a`` and ``b`` differ where a box's spectrogram frames look."""
    half = max(NFFTS) // 2
    lo = max(0, round(t0 * SYN_RATE) - half)
    hi = min(len(a), round(t1 * SYN_RATE) + half)
    return not np.array_equal(a[lo:hi], b[lo:hi])


def scene_chirps(
    scene: Scene,
    clean: NDArray[np.float32],
    reference: NDArray[np.float32],
    mask: NDArray[np.bool_],
) -> list[dict[str, Any]]:
    """Per chirp: box power of ``clean`` against ``reference``.

    Also ``max_dev``, the largest |clean - reference| inside the chirp's
    support (+-2 FWHM), relative to the fish's peak EOD amplitude, and
    ``touched``, the fraction of the support repaired on any electrode.
    """
    rows = []
    pad = round(BOX_PAD_S * SYN_RATE)
    for c in scene.chirps.itertuples():
        row: dict[str, Any] = {
            "fish": c.fish,
            "t_s": c.t_s,
            "type": c.type,
            "f0_hz": c.f0_hz,
        }
        a = max(0, round((c.t_s - 2 * c.fwhm_s) * SYN_RATE))
        b = min(len(clean), round((c.t_s + 2 * c.fwhm_s) * SYN_RATE))
        dev = np.abs(clean[a:b].astype(np.float64) - reference[a:b]).max()
        row["max_dev"] = float(dev / scene.gains[c.fish].max())
        row["touched"] = float(mask[a:b].any(1).mean())
        if window_changed(clean, reference, c.t0, c.t1):
            lo = max(0, round(c.t0 * SYN_RATE) - pad)
            hi = min(len(clean), round(c.t1 * SYN_RATE) + pad)
            box = (c.t0, c.t1, c.f_low, c.f_high)
            row.update(box_change(reference[lo:hi], clean[lo:hi], box, lo / SYN_RATE))
        else:
            row.update(unchanged_box())
        rows.append(row)
    return rows


def scene_clicks(
    scene: Scene,
    clean: NDArray[np.float32],
    mask: NDArray[np.bool_],
    guard: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Per click and electrode: detected, residual, band footprint.

    Returns the rows, the number of false-positive events (runs of repaired
    samples on any electrode that touch no click +- ``guard``), and the
    number of all events.
    """
    n = len(mask)
    covered = np.zeros(n, dtype=bool)
    rows = []
    for c in scene.clicks.itertuples():
        a, b = int(c.start), int(c.stop)
        covered[max(0, a - guard) : min(n, b + guard)] = True
        centre = a + (b - a) // 2
        before = band_footprint_db(scene.signal, scene.truth, centre, SYN_RATE)
        after = band_footprint_db(clean, scene.truth, centre, SYN_RATE)
        for el, g in zip(c.electrodes, c.gains, strict=True):
            amp = c.amplitude * g
            err = np.abs(clean[a:b, el].astype(np.float64) - scene.truth[a:b, el]).max()
            rows.append(
                {
                    "click": c.Index,
                    "electrode": el,
                    "n_electrodes": len(c.electrodes),
                    "amplitude": amp,
                    "peak_amplitude": c.amplitude,
                    "duration_s": c.duration_s,
                    "kind": c.kind,
                    "at_chirp": c.at_chirp,
                    "detected": bool(mask[a:b, el].any()),
                    "residual": float(err / amp),
                    "band_before_db": before,
                    "band_after_db": after,
                }
            )
    spans = click_spans(mask)
    fp = sum(1 for s, e in spans if not covered[s:e].any())
    return rows, fp, len(spans)


def cmd_synthetic(args: argparse.Namespace) -> None:
    out = args.out / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.workers)
    click_rows: list[dict[str, Any]] = []
    chirp_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    seed = 0
    for harm_name, harmonics in HARMONICS.items():
        for f0 in F0_PAIRS:
            for loud in LOUDNESS:
                seed += 1
                scene = synth_scene(
                    np.random.default_rng(seed),
                    loudness=loud,
                    f0_hz=f0,
                    harmonics=harmonics,
                    duration_s=args.duration,
                )
                tag = {
                    "harmonics": harm_name,
                    "f0_hz": f0[0],
                    "loudness": loud,
                    "seed": seed,
                }
                for name, cfg in CONFIGS.items():
                    t0 = time.perf_counter()
                    clean, mask = remove_clicks(
                        scene.signal, SYN_RATE, cfg, workers=args.workers
                    )
                    elapsed = time.perf_counter() - t0
                    tclean, tmask = remove_clicks(
                        scene.truth, SYN_RATE, cfg, workers=args.workers
                    )
                    guard = round(cfg.guard_s * SYN_RATE) + 2 * round(
                        cfg.resolution_s * SYN_RATE
                    )
                    rows, fp, n_events = scene_clicks(scene, clean, mask, guard)
                    for r in rows:
                        click_rows.append({**tag, "config": name, **r})
                    for run, (c, ref, m) in {
                        "clean_only": (tclean, scene.truth, tmask),
                        "with_clicks": (clean, scene.truth, mask),
                        "raw": (scene.signal, scene.truth, np.zeros_like(mask)),
                    }.items():
                        if run == "raw" and name != "default":
                            continue
                        for r in scene_chirps(scene, c, ref, m):
                            chirp_rows.append({**tag, "config": name, "run": run, **r})
                    scene_rows.append(
                        {
                            **tag,
                            "config": name,
                            "events": n_events,
                            "fp_events": fp,
                            "fp_events_clean_only": len(click_spans(tmask)),
                            "repaired_fraction": float(mask.mean()),
                            "repaired_fraction_clean_only": float(tmask.mean()),
                            "n_clicks": len(scene.clicks),
                            "duration_s": args.duration,
                            "elapsed_s": elapsed,
                        }
                    )
                log.info(
                    "scene %s f0=%s loud=%s done (%d clicks)",
                    harm_name,
                    f0,
                    loud,
                    len(scene.clicks),
                )
                if harm_name == "measured" and f0[0] == 900.0 and loud == 100.0:
                    figure_synthetic_example(scene, out.parent / "figures")
    pd.DataFrame(click_rows).to_csv(out / "clicks.csv", index=False)
    pd.DataFrame(chirp_rows).to_csv(out / "chirps.csv", index=False)
    pd.DataFrame(scene_rows).to_csv(out / "scenes.csv", index=False)
    write_json(out / "meta.json", {**meta(), "duration_s": args.duration})
    figure_synthetic_summary(out, out.parent / "figures")


# ---------------------------------------------------------------- figures


def show_spec(
    ax: Any,
    samples: NDArray[np.floating[Any]],
    nfft: int,
    t0_s: float,
    f_range: tuple[float, float],
    vmax: float | None = None,
    db_range: float = 60.0,
) -> float:
    """Electrode-summed spectrogram in dB; returns the colour maximum."""
    times, freqs, power = spectrogram(samples, nfft, t0_s)
    sel = (freqs >= f_range[0]) & (freqs <= f_range[1])
    level = db(power[sel])
    top = float(level.max()) if vmax is None else vmax
    ax.pcolormesh(
        times,
        freqs[sel],
        level,
        vmin=top - db_range,
        vmax=top,
        shading="nearest",
        cmap="magma",
        rasterized=True,
    )
    return top


def figure_synthetic_example(scene: Scene, figdir: Path) -> None:
    """Truth, with clicks, cleaned (default and thr48), around a click at a chirp."""
    figdir.mkdir(parents=True, exist_ok=True)
    forced = scene.clicks[scene.clicks["at_chirp"]].iloc[0]
    tc = forced["start"] / SYN_RATE
    a = max(0, round((tc - 1.2) * SYN_RATE))
    b = min(len(scene.signal), round((tc + 1.2) * SYN_RATE))
    cleaned = {
        name: remove_clicks(scene.signal[a:b], SYN_RATE, CONFIGS[name], start_sample=a)[
            0
        ]
        for name in ("default", "guard")
    }
    panels = [
        ("truth (no clicks)", scene.truth[a:b]),
        ("with clicks", scene.signal[a:b]),
    ]
    panels += [(f"cleaned, {k}", v) for k, v in cleaned.items()]
    fig, axs = plt.subplots(2, len(panels), figsize=(4 * len(panels), 7), sharex=True)
    for row, nfft in enumerate(NFFTS):
        top = None
        for col, (title, x) in enumerate(panels):
            top = show_spec(axs[row, col], x, nfft, a / SYN_RATE, (550, 1600), top)
            axs[row, col].set_title(f"{title}\nnfft {nfft}", fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel("time [s]")
    for ax in axs[:, 0]:
        ax.set_ylabel("frequency [Hz]")
    fig.suptitle(
        "Synthetic: measured harmonics, fish 0 at 900 Hz, loudness 100; "
        "click on electrodes 7, 3, 4, 6 at the peak of a big chirp",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(figdir / "syn_example.png", dpi=110)
    plt.close(fig)


def figure_synthetic_summary(data: Path, figdir: Path) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    chirps = pd.read_csv(data / "chirps.csv")
    clicks = pd.read_csv(data / "clicks.csv")
    clean_only = chirps[chirps["run"] == "clean_only"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for k, harm in enumerate(HARMONICS):
        sub = clean_only[clean_only["harmonics"] == harm]
        for name in CONFIGS:
            s = (
                sub[sub["config"] == name]
                .groupby("loudness")["touched"]
                .apply(lambda x: float((x > 0).mean()))
            )
            axs[k].plot(s.index, 100 * s.values, "o-", label=name)
        axs[k].set_xscale("log")
        axs[k].set_xlabel("loudness (EOD amplitude / high-band noise sd)")
        axs[k].set_ylabel("chirps touched without any click [%]")
        axs[k].set_title(f"harmonics: {harm}")
    axs[0].legend(fontsize=8)
    per = clicks[~clicks["at_chirp"]]
    bins = np.logspace(0, 4, 13)
    centres = np.sqrt(bins[1:] * bins[:-1])
    for name in CONFIGS:
        s = per[per["config"] == name]
        idx = np.digitize(s["amplitude"], bins) - 1
        rec = [
            s["detected"][idx == i].mean() if (idx == i).any() else np.nan
            for i in range(len(centres))
        ]
        axs[2].plot(centres, 100 * np.array(rec), "o-", label=name)
    axs[2].set_xscale("log")
    axs[2].set_xlabel("click amplitude on the electrode (x noise sd)")
    axs[2].set_ylabel("detected [%] (per click and electrode)")
    axs[2].set_title("click recall")
    fig.tight_layout()
    fig.savefig(figdir / "syn_summary.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------- real


def labelled(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Labelled regions and chirp boxes of a recording."""
    labels = read_labels(RECORDINGS[name] / LABELS)
    regions = labels[labels["category"] == "labeled_region"].reset_index(drop=True)
    chirps = labels[labels["category"] == "chirp"].reset_index(drop=True)
    return regions, chirps


def in_any(
    t: NDArray[np.float64], spans: Iterable[tuple[float, float]]
) -> NDArray[np.bool_]:
    out = np.zeros(len(t), dtype=bool)
    for a, b in spans:
        out |= (t >= a) & (t <= b)
    return out


def cmd_real(args: argparse.Namespace) -> None:
    out = args.out / "real"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.workers)
    region_rows: list[dict[str, Any]] = []
    chirp_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for name, path in RECORDINGS.items():
        rec = open_recording(path)
        rate = rec.samplerate_hz
        regions, chirps = labelled(name)
        pad = round(BOX_PAD_S * rate)
        for region in regions.itertuples():
            a = rec.time_to_sample(region.t_start_s)
            b = rec.time_to_sample(region.t_end_s)
            raw = rec.read(a, b)
            inside = chirps[
                (chirps["t_start_s"] >= region.t_start_s)
                & (chirps["t_end_s"] <= region.t_end_s)
            ]
            for cname, cfg in CONFIGS.items():
                t0 = time.perf_counter()
                events, repaired, clean, mask = clean_range(
                    rec, a, b, cfg, workers=args.workers, keep=True
                )
                elapsed = time.perf_counter() - t0
                assert clean is not None
                assert mask is not None
                region_rows.append(
                    {
                        "recording": name,
                        "region": region.Index,
                        "t_start_s": region.t_start_s,
                        "duration_s": (b - a) / rate,
                        "n_channels": rec.n_channels,
                        "config": cname,
                        "events": len(events),
                        "events_visible": int((events.band_db >= 1.0).sum()),
                        "repaired_fraction": repaired / mask.size,
                        "chirps": len(inside),
                        "elapsed_s": elapsed,
                    }
                )
                for s, e, ch, pk, bd in zip(
                    events.start,
                    events.stop,
                    events.channels,
                    events.peak_change,
                    events.band_db,
                    strict=True,
                ):
                    event_rows.append(
                        {
                            "recording": name,
                            "config": cname,
                            "t_s": s / rate,
                            "duration_ms": 1000 * (e - s) / rate,
                            "n_channels": int(ch).bit_count(),
                            "peak_change": pk,
                            "band_db": bd,
                        }
                    )
                for c in inside.itertuples():
                    i = rec.time_to_sample(c.t_start_s) - a
                    j = rec.time_to_sample(c.t_end_s) - a
                    row: dict[str, Any] = {
                        "recording": name,
                        "config": cname,
                        "t_start_s": c.t_start_s,
                        "t_end_s": c.t_end_s,
                        "f_low_hz": c.f_low_hz,
                        "f_high_hz": c.f_high_hz,
                        "touched": float(mask[i:j].any(1).mean()),
                        "touched_max_electrode": float(mask[i:j].mean(0).max()),
                        "electrodes_touched": int(mask[i:j].any(0).sum()),
                    }
                    half = max(NFFTS) // 2
                    lo, hi = max(0, i - pad), min(len(raw), j + pad)
                    if mask[max(0, i - half) : j + half].any():
                        box = (c.t_start_s, c.t_end_s, c.f_low_hz, c.f_high_hz)
                        row.update(
                            box_change(raw[lo:hi], clean[lo:hi], box, (a + lo) / rate)
                        )
                    else:
                        row.update(unchanged_box())
                    chirp_rows.append(row)
                log.info(
                    "%s region %d %s: %d events, %.3f%% repaired",
                    name,
                    region.Index,
                    cname,
                    len(events),
                    100 * repaired / mask.size,
                )
    pd.DataFrame(region_rows).to_csv(out / "regions.csv", index=False)
    pd.DataFrame(chirp_rows).to_csv(out / "chirps.csv", index=False)
    pd.DataFrame(event_rows).to_csv(out / "events.csv", index=False)
    write_json(out / "meta.json", meta())
    figure_real(out, args.out / "figures")


def cutout(
    rec: Recording,
    t0: float,
    t1: float,
    config: ClickRemoval,
    pad_s: float = 0.3,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], float]:
    a = max(0, rec.time_to_sample(t0 - pad_s))
    b = min(rec.n_samples, rec.time_to_sample(t1 + pad_s))
    clean, mask = read_clean(rec, a, b, config, workers=8)
    return rec.read(a, b), clean, mask, a / rec.samplerate_hz


def before_after_grid(
    items: list[
        tuple[str, Recording, float, float, tuple[float, float], Sequence[float] | None]
    ],
    configs: Sequence[str],
    path: Path,
    nfft: int = 1024,
) -> None:
    """Rows: items; columns: raw, then cleaned with each config, then the mask."""
    if not items:
        return
    ncol = 1 + len(configs)
    fig, axs = plt.subplots(
        len(items), ncol, figsize=(3.4 * ncol, 2.6 * len(items)), squeeze=False
    )
    for r, (title, rec, t0, t1, frange, box) in enumerate(items):
        top = None
        raw, _, _, ts = cutout(rec, t0, t1, CONFIGS[configs[0]])
        top = show_spec(axs[r, 0], raw, nfft, ts, frange, top)
        axs[r, 0].set_title(f"{title}\nraw", fontsize=8)
        for k, cname in enumerate(configs, start=1):
            _, clean, mask, ts = cutout(rec, t0, t1, CONFIGS[cname])
            show_spec(axs[r, k], clean, nfft, ts, frange, top)
            hit = mask.any(1)
            frac = 100 * hit.mean()
            axs[r, k].set_title(f"cleaned, {cname} ({frac:.1f}% repaired)", fontsize=8)
            if hit.any():
                tt = ts + np.flatnonzero(hit) / rec.samplerate_hz
                axs[r, k].plot(
                    tt,
                    np.full(len(tt), frange[0] + 0.03 * (frange[1] - frange[0])),
                    "|",
                    color="cyan",
                    ms=6,
                )
        for ax in axs[r]:
            if box is not None:
                bt0, bt1, bf0, bf1 = box
                ax.add_patch(
                    plt.Rectangle(
                        (bt0, bf0),
                        bt1 - bt0,
                        bf1 - bf0,
                        fill=False,
                        ec="w",
                        lw=0.8,
                        ls="--",
                    )
                )
            ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def figure_real(data: Path, figdir: Path) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    chirps = pd.read_csv(data / "chirps.csv")
    events = pd.read_csv(data / "events.csv")
    recs = {name: open_recording(p) for name, p in RECORDINGS.items()}
    default = chirps[chirps["config"] == "default"].sort_values("chirp_db_1024")
    items = []
    for c in default.head(6).itertuples():
        items.append(
            (
                f"{c.recording} chirp {c.t_start_s:.2f} s\ndefault: chirp "
                f"{c.chirp_db_1024:+.1f} dB, peak {c.chirp_peak_db_1024:+.1f} dB",
                recs[c.recording],
                c.t_start_s,
                c.t_end_s,
                (c.f_low_hz - 100, c.f_high_hz + 150),
                (c.t_start_s, c.t_end_s, c.f_low_hz, c.f_high_hz),
            )
        )
    before_after_grid(
        items, ["default", "band0.40", "guard"], figdir / "real_chirps_damaged.png"
    )
    ev = events[events["config"] == "default"].sort_values("band_db", ascending=False)
    items = []
    for _, top in ev.groupby("recording", sort=True):
        for e in top.head(2).itertuples():
            items.append(
                (
                    f"{e.recording} click {e.t_s:.3f} s ({e.n_channels} el., {e.band_db:+.1f} dB)",
                    recs[str(e.recording)],
                    e.t_s - 0.1,
                    e.t_s + 0.1,
                    (500.0, 1500.0),
                    None,
                )
            )
    before_after_grid(
        items, ["default", "band0.40", "guard"], figdir / "real_clicks.png"
    )
    figure_click_timecourse(data, figdir)


def band_timecourse(
    samples: NDArray[np.floating[Any]], nfft: int, t0_s: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Median over `BAND_HZ` of the electrode-summed power, per frame [dB]."""
    times, freqs, power = spectrogram(samples, nfft, t0_s)
    sel = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    return times, db(np.median(power[sel], axis=0))


def figure_click_timecourse(data: Path, figdir: Path, n: int = 8) -> None:
    """Broadband level in the chirpdetector band around the loudest clicks.

    The median over the band's bins sits between the fish lines, so it
    shows a click's stripe (up) and a repaired hole's notch (down).
    """
    events = pd.read_csv(data / "events.csv")
    ev = events[events["config"] == "default"].sort_values("band_db", ascending=False)
    ev = ev.groupby("recording", sort=True).head(3).head(n)
    if ev.empty:
        return
    figdir.mkdir(parents=True, exist_ok=True)
    recs = {name: open_recording(p) for name, p in RECORDINGS.items()}
    cols = 4
    rows = -(-len(ev) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for ax, e in zip(axs.ravel(), ev.itertuples(), strict=False):
        rec = recs[str(e.recording)]
        raw, _, _, ts = cutout(
            rec, e.t_s - 0.3, e.t_s + 0.3, CONFIGS["default"], pad_s=0.0
        )
        t, level = band_timecourse(raw, 1024, ts)
        ax.plot(t - e.t_s, level, color="k", lw=1.2, label="raw")
        for cname, color in (("default", "C3"), ("band0.40", "C0"), ("guard", "C2")):
            _, clean, _, ts = cutout(
                rec, e.t_s - 0.3, e.t_s + 0.3, CONFIGS[cname], pad_s=0.0
            )
            t, level = band_timecourse(clean, 1024, ts)
            ax.plot(t - e.t_s, level, color=color, lw=0.9, label=cname)
        ax.set_xlim(-0.2, 0.2)
        ax.set_title(f"{e.recording} {e.t_s:.3f} s ({e.n_channels} el.)", fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0, 0].legend(fontsize=7)
    for ax in axs[:, 0]:
        ax.set_ylabel("median power 500-1500 Hz [dB]", fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("time from click [s]", fontsize=8)
    for ax in axs.ravel()[len(ev) :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(figdir / "real_click_timecourse.png", dpi=100)
    plt.close(fig)


def cmd_figures(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.workers)
    figdir = args.out / "figures"
    if (args.out / "real" / "chirps.csv").exists():
        figure_real(args.out / "real", figdir)
    if (args.out / "synthetic" / "chirps.csv").exists():
        figure_synthetic_summary(args.out / "synthetic", figdir)


# --------------------------------------------------------------------- v1


def overlaps(
    t0: NDArray[np.float64],
    t1: NDArray[np.float64],
    e0: NDArray[np.float64],
    e1: NDArray[np.float64],
    tol: float,
) -> NDArray[np.bool_]:
    """For each [t0, t1]: does any event [e0, e1] lie within ``tol`` of it."""
    order = np.argsort(e0)
    e0, e1 = e0[order], e1[order]
    out = np.zeros(len(t0), dtype=bool)
    if len(e0) == 0:
        return out
    # events starting before t1 + tol; of those the latest-ending must reach t0 - tol
    reach = np.maximum.accumulate(e1)
    idx = np.searchsorted(e0, t1 + tol, side="right") - 1
    ok = idx >= 0
    out[ok] = reach[idx[ok]] >= t0[ok] - tol
    return out


def cmd_v1(args: argparse.Namespace) -> None:
    out = args.out / "v1"
    out.mkdir(parents=True, exist_ok=True)
    tol = 0.05
    rows = []
    examples = []
    rng = np.random.default_rng(0)
    for name in V1_RECORDINGS:
        det = read_v1_detections(RECORDINGS[name])
        regions, chirps = labelled(name)
        spans = list(zip(regions["t_start_s"], regions["t_end_s"], strict=True))
        mid = 0.5 * (det["t_start_s"].to_numpy() + det["t_end_s"].to_numpy())
        inside = in_any(mid, spans)
        # a v1 detection matches a labelled chirp if their time spans overlap
        matched = overlaps(
            det["t_start_s"].to_numpy(),
            det["t_end_s"].to_numpy(),
            chirps["t_start_s"].to_numpy(),
            chirps["t_end_s"].to_numpy(),
            0.0,
        )
        for cname in args.config:
            path = scan_path(args.out, name, cname)
            if not path.exists():
                log.warning("%s missing; run scan --config %s first", path, cname)
                continue
            events, info = Events.load(path)
            rate = info["samplerate_hz"]
            duration = info["n_samples"] / rate
            for label, sel in (
                ("all", np.ones(len(events), dtype=bool)),
                ("visible", events.band_db >= 1.0),
            ):
                e0 = events.start[sel] / rate
                e1 = events.stop[sel] / rate
                hit = overlaps(
                    det["t_start_s"].to_numpy(), det["t_end_s"].to_numpy(), e0, e1, tol
                )
                # chance: the same events shifted by a random 5-60 s, circularly
                chance = []
                for _ in range(200):
                    shift = rng.uniform(5, 60)
                    s0 = (e0 + shift) % duration
                    s1 = s0 + (e1 - e0)
                    chance.append(
                        overlaps(
                            det["t_start_s"].to_numpy(),
                            det["t_end_s"].to_numpy(),
                            s0,
                            s1,
                            tol,
                        ).sum()
                    )
                rows.append(
                    {
                        "recording": name,
                        "config": cname,
                        "events": label,
                        "n_events": int(sel.sum()),
                        "event_rate_per_min": 60 * sel.sum() / duration,
                        "detections": len(det),
                        "detections_inside": int(inside.sum()),
                        "overlap": int(hit.sum()),
                        "overlap_chance_mean": float(np.mean(chance)),
                        "overlap_chance_p95": float(np.percentile(chance, 95)),
                        "overlap_inside": int((hit & inside).sum()),
                        "overlap_inside_matched": int((hit & inside & matched).sum()),
                        "overlap_inside_unmatched": int(
                            (hit & inside & ~matched).sum()
                        ),
                        "unmatched_inside": int((inside & ~matched).sum()),
                        "overlap_outside": int((hit & ~inside).sum()),
                    }
                )
                if cname == "default" and label == "visible":
                    pd.DataFrame(
                        {
                            **{
                                k: det[k]
                                for k in (
                                    "t_start_s",
                                    "t_end_s",
                                    "f_low_hz",
                                    "f_high_hz",
                                    "score",
                                )
                            },
                            "inside": inside,
                            "matched": matched,
                            "overlap_click": hit,
                        }
                    ).to_csv(out / f"detections_{name}.csv", index=False)
                    for k in np.flatnonzero(hit & inside & ~matched)[:3]:
                        examples.append((name, det.iloc[k], "inside, unmatched"))
                    for k in np.flatnonzero(hit & ~inside)[:3]:
                        examples.append((name, det.iloc[k], "outside"))
    table = pd.DataFrame(rows)
    table.to_csv(out / "overlap.csv", index=False)
    write_json(out / "meta.json", {**meta(), "tolerance_s": tol})
    figdir = args.out / "figures"
    items = []
    for name, d, where in examples:
        rec = open_recording(RECORDINGS[name])
        items.append(
            (
                f"{name} v1 {d.t_start_s:.2f} s ({where}, score {d.score:.2f})",
                rec,
                d.t_start_s,
                d.t_end_s,
                (500.0, 1500.0),
                (d.t_start_s, d.t_end_s, d.f_low_hz, d.f_high_hz),
            )
        )
    before_after_grid(items, ["default", "guard"], figdir / "v1_examples.png")


# ------------------------------------------------------------- throughput


def cmd_throughput(args: argparse.Namespace) -> None:
    out = args.out / "throughput"
    rec = open_recording(RECORDINGS["2020-06-03-10_10"])
    rate = rec.samplerate_hz
    seconds = args.seconds
    a = rec.time_to_sample(3000.0)
    b = a + round(seconds * rate)
    rows = []
    read_s = float("nan")
    n_read = 0
    for name in ("default", "guard"):
        cfg = CONFIGS[name]
        plan = plan_clean(a, b, rec.n_samples, rate, cfg)
        t0 = time.perf_counter()
        samples = rec.read(plan.read_start, plan.read_stop)
        read_s = time.perf_counter() - t0
        n_read = plan.n_read
        for workers in (1, 4, 8, 16):
            load = os.getloadavg()
            t0 = time.perf_counter()
            _, mask = clean_span(samples, plan, rate, cfg, workers=workers)
            elapsed = time.perf_counter() - t0
            rows.append(
                {
                    "config": name,
                    "workers": workers,
                    "seconds": seconds,
                    "elapsed_s": elapsed,
                    "realtime_factor": seconds / elapsed,
                    "loadavg_1min": load[0],
                    "repaired_fraction": float(mask.mean()),
                }
            )
            log.info(
                "%s, workers %d: %.1fx realtime (load %.1f)",
                name,
                workers,
                seconds / elapsed,
                load[0],
            )
    write_json(
        out / "results.json",
        {
            **meta(),
            "recording": "2020-06-03-10_10",
            "n_channels": rec.n_channels,
            "samplerate_hz": rate,
            "read_s": read_s,
            "read_samples": n_read,
            "cpu_count": os.cpu_count(),
            "rows": rows,
        },
    )


# ---------------------------------------------------------------- tables


def md(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float | np.floating):
            return format(v, floatfmt)
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = [
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([head, sep, *body])


def cmd_tables(args: argparse.Namespace) -> None:
    parts: list[str] = []
    syn = args.out / "synthetic"
    if (syn / "chirps.csv").exists():
        parts += synthetic_tables(syn)
    if (args.out / "real" / "chirps.csv").exists():
        parts += real_tables(args.out / "real")
    v1 = args.out / "v1" / "overlap.csv"
    if v1.exists():
        parts += ["## v1 detections within 50 ms of a click", md(pd.read_csv(v1))]
    parts += scan_tables(args.out)
    tp = args.out / "throughput" / "results.json"
    if tp.exists():
        res = json.loads(tp.read_text())
        parts += [
            f"## Throughput ({res['n_channels']} channels, {res['samplerate_hz']:.0f} Hz, "
            f"{res['cpu_count']} CPUs; reading {res['read_samples'] / res['samplerate_hz']:.0f} s "
            f"took {res['read_s']:.2f} s)",
            md(pd.DataFrame(res["rows"])),
        ]
    (args.out / "tables.md").write_text("\n\n".join(parts) + "\n")
    print("\n\n".join(parts))


def _count_below(threshold: float) -> Callable[[pd.Series], int]:
    return lambda x: int((x < threshold).sum())


def synthetic_tables(syn: Path) -> list[str]:
    clicks = pd.read_csv(syn / "clicks.csv")
    chirps = pd.read_csv(syn / "chirps.csv")
    scenes = pd.read_csv(syn / "scenes.csv")
    free = clicks[~clicks["at_chirp"]]
    keys = ["config", "harmonics", "f0_hz", "loudness", "seed", "click"]
    per_click = (
        free.groupby(keys)
        .agg(
            detected=("detected", "any"),
            band_before=("band_before_db", "first"),
            band_after=("band_after_db", "first"),
        )
        .reset_index()
    )
    t = per_click.groupby("config").agg(
        clicks=("detected", "size"),
        recall=("detected", "mean"),
        band_before_db_median=("band_before", "median"),
        band_after_db_median=("band_after", "median"),
    )
    visible = per_click[per_click["band_before"] >= 1.0]
    t = t.join(
        visible.groupby("config").agg(
            visible_clicks=("detected", "size"),
            visible_recall=("detected", "mean"),
            visible_after_db_max=("band_after", "max"),
            visible_after_ge_1db=("band_after", lambda x: int((x >= 1.0).sum())),
        )
    )
    t = t.join(
        free.groupby("config").agg(
            electrode_recall=("detected", "mean"),
            residual_median=("residual", "median"),
        )
    )
    bins = pd.cut(free["amplitude"], [0, 30, 100, 300, 1000, 1e5])
    by_amp = free.groupby(["config", bins], observed=True)["detected"].mean().unstack()
    by_amp.columns = [f"el. recall {c}" for c in by_amp.columns]
    fp = scenes.groupby("config").agg(
        minutes=("duration_s", lambda x: x.sum() / 60),
        fp_events=("fp_events", "sum"),
        fp_clean_only=("fp_events_clean_only", "sum"),
    )
    fp["fp_per_min"] = fp["fp_events"] / fp["minutes"]
    fp["fp_clean_only_per_min"] = fp["fp_clean_only"] / fp["minutes"]
    out = [
        "## Synthetic: clicks",
        md(t.join(fp).reset_index()),
        md(by_amp.reset_index()),
    ]

    co = chirps[chirps["run"] == "clean_only"]
    g = (
        co.groupby(["harmonics", "loudness", "config"])
        .agg(
            chirps=("touched", "size"),
            touched_pct=("touched", lambda x: 100 * float((x > 0).mean())),
            worst_box_db_1024=("delta_db_1024", "min"),
            worst_chirp_db_4096=("chirp_db_4096", "min"),
            worst_chirp_db_1024=("chirp_db_1024", "min"),
            worst_peak_db_1024=("chirp_peak_db_1024", "min"),
            n_chirp_below_m0p5db=("chirp_db_1024", _count_below(-0.5)),
            n_peak_below_m1db=("chirp_peak_db_1024", _count_below(-1.0)),
            worst_chirp_error_db_1024=("chirp_error_db_1024", "max"),
            max_dev=("max_dev", "max"),
        )
        .reset_index()
    )
    out += ["## Synthetic: chirps without any click (every change is damage)", md(g)]
    by_type = (
        co.groupby(["config", "type"])
        .agg(
            chirps=("touched", "size"),
            touched_pct=("touched", lambda x: 100 * float((x > 0).mean())),
            worst_chirp_db_1024=("chirp_db_1024", "min"),
            worst_chirp_db_4096=("chirp_db_4096", "min"),
            worst_peak_db_1024=("chirp_peak_db_1024", "min"),
        )
        .reset_index()
    )
    out += ["### By chirp type (all loudness levels)", md(by_type)]

    wc = chirps[(chirps["type"] == "big") & chirps["run"].isin(["with_clicks", "raw"])]
    first = wc.groupby(["config", "run", "seed", "fish"])["t_s"].transform("min")
    at = (
        wc[first == wc["t_s"]]
        .groupby(["config", "run", "fish"])
        .agg(
            n=("t_s", "size"),
            box_db_1024_median=("delta_db_1024", "median"),
            chirp_db_4096_median=("chirp_db_4096", "median"),
            chirp_db_1024_median=("chirp_db_1024", "median"),
            chirp_db_1024_min=("chirp_db_1024", "min"),
            chirp_db_1024_max=("chirp_db_1024", "max"),
            peak_db_1024_median=("chirp_peak_db_1024", "median"),
        )
        .reset_index()
    )
    out += [
        "## Synthetic: the first big chirp of each fish, with a click at its peak "
        "(fish 0: electrodes 7, 3, 4, 6; fish 1: electrode 6), against the truth",
        md(at),
    ]
    return out


def real_tables(real: Path) -> list[str]:
    regions = pd.read_csv(real / "regions.csv")
    rc = pd.read_csv(real / "chirps.csv")
    regions["repaired_samples"] = (
        regions["repaired_fraction"] * regions["duration_s"] * regions["n_channels"]
    )
    regions["all_samples"] = regions["duration_s"] * regions["n_channels"]
    rg = regions.groupby(["recording", "config"]).agg(
        duration_s=("duration_s", "sum"),
        events=("events", "sum"),
        events_visible=("events_visible", "sum"),
        repaired=("repaired_samples", "sum"),
        total=("all_samples", "sum"),
    )
    rg["events_per_min"] = 60 * rg["events"] / rg["duration_s"]
    rg["repaired_pct"] = 100 * rg["repaired"] / rg["total"]
    rg = rg.drop(columns=["repaired", "total"]).reset_index()
    out = ["## Real: clicks in the labelled regions", md(rg)]
    agg = {
        "chirps": ("touched", "size"),
        "touched": ("touched", lambda x: int((x > 0).sum())),
        "touched_max_pct": ("touched", lambda x: 100 * float(x.max())),
        "box_db_1024_min": ("delta_db_1024", "min"),
        "chirp_db_4096_min": ("chirp_db_4096", "min"),
        "chirp_db_1024_min": ("chirp_db_1024", "min"),
        "chirp_db_1024_max": ("chirp_db_1024", "max"),
        "peak_db_1024_min": ("chirp_peak_db_1024", "min"),
        "n_peak_below_m1db": ("chirp_peak_db_1024", _count_below(-1.0)),
        "n_peak_below_m3db": ("chirp_peak_db_1024", _count_below(-3.0)),
        "n_chirp_below_m0p5db": ("chirp_db_1024", _count_below(-0.5)),
        "n_chirp_below_m1db": ("chirp_db_1024", _count_below(-1.0)),
        "chirp_error_db_1024_max": ("chirp_error_db_1024", "max"),
    }
    ch = rc.groupby(["recording", "config"]).agg(**agg).reset_index()
    out += ["## Real: labelled chirps, cleaned against raw", md(ch)]
    allc = rc.groupby("config").agg(**agg).reset_index()
    out += ["### All four recordings", md(allc)]
    return out


def scan_tables(out_dir: Path) -> list[str]:
    scans = sorted((out_dir / "scans").glob("scan_*.npz"))
    if not scans:
        return []
    rows = []
    for p in scans:
        ev, info = Events.load(p)
        dur = info["n_samples"] / info["samplerate_hz"]
        visible = int((ev.band_db >= 1).sum())
        rows.append(
            {
                "recording": info["recording"],
                "config": info["config"],
                "hours": dur / 3600,
                "events": len(ev),
                "per_min": 60 * len(ev) / dur,
                "visible (>=1 dB)": visible,
                "visible_per_min": 60 * visible / dur,
                "repaired_pct": 100
                * info["repaired_samples"]
                / (info["n_samples"] * info["n_channels"]),
                "realtime_x": dur / info["elapsed_s"],
                "loadavg_1min": info["loadavg"][0],
            }
        )
    return ["## Whole-recording scans", md(pd.DataFrame(rows))]


# ------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate click removal.")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=8)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="clicks in whole recordings")
    scan.add_argument(
        "--config", nargs="+", default=["default"], choices=sorted(CONFIGS)
    )
    scan.add_argument("--recordings", nargs="*", choices=sorted(RECORDINGS))
    scan.add_argument("--force", action="store_true")
    scan.set_defaults(func=cmd_scan)
    syn = sub.add_parser("synthetic", help="synthetic recordings with known clicks")
    syn.add_argument("--duration", type=float, default=30.0)
    syn.set_defaults(func=cmd_synthetic)
    real = sub.add_parser("real", help="hand-labelled recordings")
    real.set_defaults(func=cmd_real)
    v1 = sub.add_parser("v1", help="v1 detections that coincide with clicks")
    v1.add_argument(
        "--config",
        nargs="+",
        default=["default", "thr48", "band0.40", "guard"],
        choices=sorted(CONFIGS),
    )
    v1.set_defaults(func=cmd_v1)
    tp = sub.add_parser("throughput", help="seconds cleaned per second")
    tp.add_argument("--seconds", type=float, default=300.0)
    tp.set_defaults(func=cmd_throughput)
    figs = sub.add_parser("figures", help="redraw the figures from the results")
    figs.set_defaults(func=cmd_figures)
    tables = sub.add_parser("tables", help="markdown tables of all results")
    tables.set_defaults(func=cmd_tables)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
