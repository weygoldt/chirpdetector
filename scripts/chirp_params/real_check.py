"""Real-data check on the harvested long chirps of the competition recordings.

Reference: the per-period parameters of each chirp's extracted frequency
trace (template library v4, ``source == "competition"``), an independent
estimate from the raw recording, not ground truth. Inputs are cut around
the v1 detector's box for that chirp (as a production decoder would see
them), optionally mapped to the footprint-box convention (``--box v1cal``) with the trace's baseline for rectification; the trained decoders
(``RUNS/<name>/model.pt``) predict on them.

Usage: real_check.py RUNS OUT [--box v1|v1cal]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gridtools.io import load_tracks, open_recording

from chirpdetector.params.crops import RectifiedConfig, crop_span, detector_view, rectified_crop
from chirpdetector.params.model import ParamNet
from chirpdetector.params.truth import measure_period_params, per_period_trace
from chirpdetector.spectrogram import SpectrogramConfig, compute_spectrogram, normalize

LIB = Path("/mnt/data1/chirp_v2/templates/v4")
ROOT = Path("/mnt/data2/2019_tube_competition")
RCFG = RectifiedConfig()
DCFG = SpectrogramConfig()
PARAMS = ("height_hz", "fwhm_s", "width10_s", "undershoot_hz")
# v1 boxes -> footprint-box convention (population medians, no per-chirp truth):
# v1 starts 44 ms and ends 3 ms outside the support, footprint boxes ~80 ms on
# both sides; v1 tops sit 45 Hz above baseline + excursion, footprints 11 Hz.
BOX_SHIFTS = {"v1": (0.0, 0.0, 0.0, 0.0), "v1cal": (0.036, 0.078, 0.0, 34.0)}
BOX_SHIFT = BOX_SHIFTS["v1"]


def reference_params(trace: np.ndarray, baseline: float, fs: float) -> dict:
    freq = baseline + trace.astype(np.float64)
    cycles = np.zeros(freq.size)
    cycles[1:] = np.cumsum(0.5 * (freq[:-1] + freq[1:])) / fs
    tau, delta = per_period_trace(2 * np.pi * cycles, np.full(freq.size, baseline), fs)
    p = measure_period_params(tau, delta)
    return {"height_hz": p.height_hz, "fwhm_s": p.fwhm_s, "width10_s": p.width10_s, "undershoot_hz": p.undershoot_hz}


def extract(device: torch.device) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    lib = pd.read_parquet(LIB / "templates.parquet")
    comp = lib[lib["source"] == "competition"].reset_index(drop=True)
    traces = np.load(LIB / "delta_f_hz.npy", mmap_mode="r")
    fs_lib = 20000.0
    rows, crops = [], {"A": [], "B": [], "C": []}
    for rec_name, group in comp.groupby("recording"):
        rec = open_recording(ROOT / rec_name)
        tracks = load_tracks(ROOT / rec_name)
        fs = rec.samplerate_hz
        for t in group.itertuples():
            info = {"template_id": int(t.template_id), "recording": rec_name, "baseline_hz": float(t.baseline_hz)}
            info.update(reference_params(np.asarray(traces[int(t.template_id)]), float(t.baseline_hz), fs_lib))
            info.update({"lib_excursion_hz": t.excursion_hz, "lib_duration_50_s": t.duration_50_s, "lib_duration_10_s": t.duration_10_s, "lib_undershoot_hz": t.undershoot_hz, "band_snr_db": t.band_snr_db})
            box = (
                float(t.box_t_start_s) - BOX_SHIFT[0],
                float(t.box_t_end_s) + BOX_SHIFT[1],
                float(t.box_f_low_hz) - BOX_SHIFT[2],
                float(t.box_f_high_hz) - BOX_SHIFT[3],
            )
            info.update({"in_t0_s": box[0], "in_t1_s": box[1], "in_f0_hz": box[2], "in_f1_hz": box[3]})
            span = crop_span(box[0], box[1], RCFG)
            info["span_t0_s"], info["span_t1_s"] = span
            try:
                powers = tracks.track(int(t.emitter_track_id)).powers_at([float(t.align_abs_s)])[0]
                top = [int(i) for i in np.argsort(powers)[::-1][:3]]
            except (KeyError, ValueError):
                top = [int(t.best_channel)]
            info["electrodes_top"] = ",".join(map(str, top))
            half = RCFG.fft_s / 2 + 0.005
            a = max(0, int(np.floor((span[0] - half) * fs)))
            b = min(rec.n_samples, int(np.ceil((span[1] + half) * fs)))
            seg = torch.as_tensor(rec.read(a, b).T.copy(), device=device)
            cb, cc = rectified_crop(seg, a / fs, fs, float(t.baseline_hz), span, RCFG, electrode_sets=(None, top))
            # detector view: spectrogram of +-3 s, normalised on that stretch
            a2 = max(0, int((box[0] - 3.0) * fs))
            b2 = min(rec.n_samples, int((box[1] + 3.0) * fs))
            spec = compute_spectrogram(rec.read(a2, b2), fs, DCFG, t0_s=a2 / fs, device=device)
            ca = detector_view(normalize(spec, DCFG), spec.freqs_hz, spec.times_s, box)
            for k, c in (("A", ca), ("B", cb), ("C", cc)):
                crops[k].append((c * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy())
            rows.append(info)
        print(rec_name, len(group), flush=True)
    return {k: np.stack(v) for k, v in crops.items()}, pd.DataFrame(rows)


@torch.no_grad()
def predict(run: Path, x: np.ndarray, meta: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    ck = torch.load(run / "model.pt", weights_only=False)
    variant = ck["args"]["input"]
    if ck["args"].get("channels"):
        x = np.ascontiguousarray(x[:, ck["args"]["channels"]])
    if variant == "A":
        s = np.stack([np.log(meta.in_t1_s - meta.in_t0_s), np.log((meta.in_f1_hz - meta.in_f0_hz).clip(lower=5))], 1)
    else:
        s = np.log(meta.span_t1_s - meta.span_t0_s).to_numpy()[:, None]
    s = ((s - ck["mu_s"]) / ck["sd_s"]).astype(np.float32)
    model = ParamNet(x.shape[1], tuple(x.shape[2:]), s.shape[1], 4).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    m, lv = model(torch.from_numpy(x).float().div(255).to(device), torch.from_numpy(s).to(device))
    m, lv = m.cpu().numpy(), lv.cpu().numpy()
    out = {}
    for j, p in enumerate(PARAMS):
        val = np.exp(m[:, j] * ck["sd_y"][j] + ck["mu_y"][j])
        out[f"pred_{p}"] = np.clip(val - 1.0, 0, None) if p == "undershoot_hz" else val
        out[f"sigma_log_{p}"] = np.exp(0.5 * lv[:, j]) * ck["sd_y"][j]
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--box", choices=list(BOX_SHIFTS), default="v1")
    args = ap.parse_args()
    global BOX_SHIFT
    BOX_SHIFT = BOX_SHIFTS[args.box]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("CHIRP_MAX_VRAM_FRACTION", "0.1")))
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "crops.npz"
    if cache.exists():
        z = np.load(cache)
        crops = {k: z[k] for k in "ABC"}
        meta = pd.read_parquet(args.out / "meta.parquet")
    else:
        crops, meta = extract(device)
        np.savez(cache, **crops)
        meta.to_parquet(args.out / "meta.parquet")
    preds = []
    for run in sorted(p for p in args.runs.iterdir() if (p / "model.pt").exists()):
        variant = torch.load(run / "model.pt", weights_only=False)["args"]["input"]
        if variant not in crops:
            continue
        df = predict(run, crops[variant], meta, device)
        df["run"] = run.name
        df["template_id"] = meta["template_id"].to_numpy()
        preds.append(df)
    pd.concat(preds, ignore_index=True).to_parquet(args.out / "predictions.parquet")
    print("done", len(meta))


if __name__ == "__main__":
    main()
