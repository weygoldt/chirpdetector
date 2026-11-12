"""Extract per-chirp ground truth and decoder inputs from simulated scenes.

Per chirp (train_v2 hybrid scenes):
* ground truth on the per-EOD-period frequency (`chirpdetector.params.truth`)
  from the renderer's exact phase, plus the template-level values of
  ``chirps.csv`` for comparison;
* covariates: SNR, family, overlap, visibility, nearest other fish (any
  track, including the real background fish) at the chirp centre;
* inputs: A = detector view (3 x 96 x 96), B = rectified crop on the
  electrode sum, C = rectified crop on the emitter's 3 strongest electrodes,
  K = rectified crop of the emitter's clean render alone on its strongest
  electrode (information ceiling). Rectified crops: 4 x 192 x 96, uint8.

Training splits use boxes jittered by N(0, 10 ms) per time edge and
N(0, 5 Hz) per frequency edge (stand-in for detector box error); val/test
use the exact footprint boxes.

Usage: extract.py SPLIT OUT_DIR [--max-scenes N]
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gridtools.io import load_tracks, open_recording
from gridtools.sim.scene import load_scene_model, render_fish

from chirpdetector.params.crops import (
    RectifiedConfig,
    crop_span,
    detector_view,
    rectified_crop,
)
from chirpdetector.params.truth import measure_period_params, per_period_trace
from chirpdetector.spectrogram import SpectrogramConfig, compute_spectrogram, normalize

SCENES = Path("/mnt/data1/chirp_v2/scenes/train_v2")
SPLIT_DIRS = {"train": "train", "val": "val", "test": "test"}
TRUTH_PAD_S = 0.02
JITTER_T_S = 0.010
JITTER_F_HZ = 5.0
A_SIZE = (96, 96)
TOP_K = 3
RCFG = RectifiedConfig()
DCFG = SpectrogramConfig()  # detector front end: 0.2048/0.0512/0.0128 s, 5 ms


def _truth(model, row) -> dict:
    fs = model.samplerate_hz
    fish = int(row.emitter_id)
    s0 = max(0, int(round((row.t_support_start_s - TRUTH_PAD_S) * fs)))
    s1 = min(model.n_samples, int(round((row.t_support_stop_s + TRUTH_PAD_S) * fs)))
    phase = model.phase(fish, s0, s1, chirps="all")
    base = model.base_frequency(fish, s0, s1)
    tau, delta = per_period_trace(phase, base, fs, s0 / fs)
    p = measure_period_params(tau, delta)
    return {
        "height_hz": p.height_hz,
        "fwhm_s": p.fwhm_s,
        "width10_s": p.width10_s,
        "undershoot_hz": p.undershoot_hz,
        "truth_edge_clipped": p.edge_clipped,
        "t_peak_period_s": float(tau[int(np.argmax(delta))]) if delta.size else np.nan,
        "local_baseline_hz": float(np.mean(base)),
    }


def _neighbours(tracks, emitter: int, t: float, baseline: float) -> dict:
    dfs = []
    for ident in tracks.ids:
        if int(ident) == emitter:
            continue
        f = tracks.track(int(ident)).frequency_at([t])[0]
        if np.isfinite(f):
            dfs.append(f - baseline)
    dfs = np.asarray(dfs)
    if dfs.size == 0:
        return {"min_df_all_hz": np.nan, "n_other_fish": 0, "fish_in_band": False}
    return {
        "min_df_all_hz": float(np.abs(dfs).min()),
        "n_other_fish": int(dfs.size),
        "fish_in_band": False,  # filled once the height is known
        "_dfs": dfs,
    }


def _load_scene(directory: Path, split: str, rng: np.random.Generator):
    chirps = pd.read_csv(directory / "chirps.csv")
    chirps = chirps[~chirps["truncated"]].reset_index(drop=True)
    model = load_scene_model(directory)
    rec = open_recording(directory / "recording.wav")
    tracks = load_tracks(directory)
    signal = rec.read(0, rec.n_samples)  # (n, n_el) float32
    rows, segments, cleans = [], [], []
    half = RCFG.fft_s / 2 + 0.005
    for row in chirps.itertuples():
        info = {"seed": int(directory.name), "split": split}
        info.update(row._asdict())
        info.pop("Index")
        info.update(_truth(model, row))
        nb = _neighbours(tracks, int(row.emitter_id), row.t_center_s, row.baseline_hz)
        dfs = nb.pop("_dfs", np.empty(0))
        nb["fish_in_band"] = bool(
            np.any((dfs > -30.0) & (dfs < info["height_hz"] + 30.0))
        )
        info.update(nb)
        box = [row.box_t0_s, row.box_t1_s, row.box_f0_hz, row.box_f1_hz]
        if split == "train":
            box[0] += rng.normal(0, JITTER_T_S)
            box[1] += rng.normal(0, JITTER_T_S)
            box[2] += rng.normal(0, JITTER_F_HZ)
            box[3] += rng.normal(0, JITTER_F_HZ)
            if box[1] - box[0] < 0.05:
                box[0:2] = [row.box_t0_s, row.box_t1_s]
        info.update(
            {"in_t0_s": box[0], "in_t1_s": box[1], "in_f0_hz": box[2], "in_f1_hz": box[3]}
        )
        span = crop_span(box[0], box[1], RCFG)
        info["span_t0_s"], info["span_t1_s"] = span
        powers = tracks.track(int(row.emitter_id)).powers_at([row.t_center_s])[0]
        order = np.argsort(powers)[::-1]
        top = [int(i) for i in order[:TOP_K]]
        info["electrodes_top"] = ",".join(map(str, top))
        fs = rec.samplerate_hz
        a = max(0, int(np.floor((span[0] - half) * fs)))
        b = min(rec.n_samples, int(np.ceil((span[1] + half) * fs)))
        info["seg_t0_s"] = a / fs
        segments.append(signal[a:b].T.copy())
        clean = render_fish(model, int(row.emitter_id), a / fs, b / fs, electrodes=[top[0]])
        cleans.append(np.asarray(clean, dtype=np.float32).T.copy())
        rows.append(info)
    return rows, segments, cleans, signal, rec.samplerate_hz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=list(SPLIT_DIRS))
    parser.add_argument("out", type=Path)
    parser.add_argument("--max-scenes", type=int, default=None)
    args = parser.parse_args()
    frac = float(os.environ.get("CHIRP_MAX_VRAM_FRACTION", "0.1"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(frac)
    torch.set_num_threads(4)
    dirs = sorted(
        p for p in (SCENES / SPLIT_DIRS[args.split]).iterdir() if p.name.isdigit()
    )[: args.max_scenes]
    counts = [
        int((~pd.read_csv(d / "chirps.csv", usecols=["truncated"])["truncated"]).sum())
        for d in dirs
    ]
    n = int(sum(counts))
    out = args.out / args.split
    out.mkdir(parents=True, exist_ok=True)
    shp_r = (n, len(RCFG.windows_s), RCFG.n_freqs, RCFG.n_times)
    mm = {
        k: np.lib.format.open_memmap(out / f"{k}.npy", "w+", np.uint8, shp_r)
        for k in ("B", "C", "K")
    }
    mm["A"] = np.lib.format.open_memmap(
        out / "A.npy", "w+", np.uint8, (n, DCFG.n_windows, *A_SIZE)
    )
    rng = np.random.default_rng(12345)
    metas: list[dict] = []
    k = 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_load_scene, d, args.split, rng) for d in dirs[:3]]
        for i, _d in enumerate(dirs):
            rows, segments, cleans, signal, fs = futures[i].result()
            if i + 3 < len(dirs):
                futures.append(pool.submit(_load_scene, dirs[i + 3], args.split, rng))
            futures[i] = None  # free memory
            with torch.no_grad():
                spec = compute_spectrogram(signal, fs, DCFG, device=device)
                image = normalize(spec, DCFG)
                del signal
                for row, seg, cl in zip(rows, segments, cleans, strict=True):
                    st = torch.as_tensor(seg, device=device)
                    top = [int(e) for e in row["electrodes_top"].split(",")]
                    crop_b, crop_c = rectified_crop(
                        st, row["seg_t0_s"], fs, row["baseline_hz"],
                        (row["span_t0_s"], row["span_t1_s"]), RCFG,
                        electrode_sets=(None, top),
                    )
                    (crop_k,) = rectified_crop(
                        torch.as_tensor(cl, device=device), row["seg_t0_s"], fs,
                        row["baseline_hz"], (row["span_t0_s"], row["span_t1_s"]), RCFG,
                    )
                    crop_a = detector_view(
                        image, spec.freqs_hz, spec.times_s,
                        (row["in_t0_s"], row["in_t1_s"], row["in_f0_hz"], row["in_f1_hz"]),
                        size=A_SIZE,
                    )
                    for key, c in (("A", crop_a), ("B", crop_b), ("C", crop_c), ("K", crop_k)):
                        mm[key][k] = (c * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                    row["row"] = k
                    metas.append(row)
                    k += 1
            if i % 20 == 0:
                el = time.time() - t_start
                print(f"{args.split}: scene {i + 1}/{len(dirs)}, {k}/{n} chirps, {el:.0f} s", flush=True)
    assert k == n, (k, n)
    for m in mm.values():
        m.flush()
    pd.DataFrame(metas).to_parquet(out / "meta.parquet")
    print(f"done {args.split}: {n} chirps in {time.time() - t_start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
