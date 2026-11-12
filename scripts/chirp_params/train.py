"""Train a chirp-parameter decoder on one input variant and predict the test set.

Targets (standardised on train): log height, log FWHM, log 10 % width,
log(undershoot + 1 Hz), all from the per-period ground truth. Loss: beta-NLL
(beta 0.5). Rectified inputs (B, C, K) can be trained with a random
baseline error (frequency shift, ``--baseline-sd-hz``); test predictions are
written for the true baseline and for errors drawn from N(0, sd) for each sd
in ``--test-baseline-sd``.

Usage: train.py CACHE RUN_DIR --input B [--baseline-sd-hz 2.4] [--epochs 20]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chirpdetector.params.crops import RectifiedConfig, shift_frequency
from chirpdetector.params.model import ParamNet, beta_nll

TARGETS = ("height_hz", "fwhm_s", "width10_s", "undershoot_hz")
UNDERSHOOT_OFFSET_HZ = 1.0
RCFG = RectifiedConfig()


def targets(meta: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [
            np.log(meta["height_hz"]),
            np.log(meta["fwhm_s"]),
            np.log(meta["width10_s"]),
            np.log(meta["undershoot_hz"] + UNDERSHOOT_OFFSET_HZ),
        ],
        axis=1,
    ).astype(np.float32)


def scalars(meta: pd.DataFrame, variant: str) -> np.ndarray:
    if variant == "A":
        cols = [
            np.log(meta["in_t1_s"] - meta["in_t0_s"]),
            np.log((meta["in_f1_hz"] - meta["in_f0_hz"]).clip(lower=5.0)),
        ]
    else:
        cols = [np.log(meta["span_t1_s"] - meta["span_t0_s"])]
    return np.stack(cols, axis=1).astype(np.float32)


class Rows:
    """Row subset of a memory-mapped uint8 array (keeps RAM use to the page cache)."""

    def __init__(self, x: np.ndarray, rows: np.ndarray, channels: list[int] | None = None) -> None:
        self.x, self.rows = x, rows
        self.channels = channels
        n_ch = x.shape[1] if channels is None else len(channels)
        self.shape = (len(rows), n_ch, *x.shape[2:])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: np.ndarray) -> np.ndarray:
        out = np.asarray(self.x[self.rows[idx]])
        return out if self.channels is None else np.ascontiguousarray(out[:, self.channels])


def load(cache: Path, split: str, variant: str, channels: list[int] | None = None) -> tuple[Rows, pd.DataFrame]:
    meta = pd.read_parquet(cache / split / "meta.parquet")
    x = np.load(cache / split / f"{variant}.npy", mmap_mode="r")
    ok = np.isfinite(targets(meta)).all(axis=1) & (meta["height_hz"] > 0).to_numpy()
    rows = meta["row"].to_numpy()[ok]
    return Rows(x, rows, channels), meta[ok].reset_index(drop=True)


def batches(n: int, size: int, rng: np.random.Generator | None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, size):
        yield np.sort(order[i : i + size])


def to_device(x: np.ndarray, idx: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x[idx]).to(device, non_blocking=True).float().div_(255.0)


@torch.no_grad()
def predict(model, x, s, device, shift_hz: np.ndarray | None = None, batch=512):
    model.eval()
    means, logv = [], []
    for idx in batches(len(x), batch, None):
        xb = to_device(x, idx, device)
        if shift_hz is not None:
            xb = shift_frequency(xb, torch.as_tensor(shift_hz[idx] / RCFG.df_step_hz, device=device, dtype=xb.dtype))
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            m, lv = model(xb, torch.from_numpy(s[idx]).to(device))
        means.append(m.float().cpu().numpy())
        logv.append(lv.float().cpu().numpy())
    return np.concatenate(means), np.concatenate(logv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path)
    ap.add_argument("run", type=Path)
    ap.add_argument("--input", required=True, choices=["A", "B", "C", "K"])
    ap.add_argument("--baseline-sd-hz", type=float, default=0.0)
    ap.add_argument("--test-baseline-sd", type=float, nargs="*", default=[2.4])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--channels", type=int, nargs="*", default=None, help="input channels to keep")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("CHIRP_MAX_VRAM_FRACTION", "0.1")))
    torch.set_num_threads(6)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    args.run.mkdir(parents=True, exist_ok=True)
    rectified = args.input != "A"

    xtr, mtr = load(args.cache, "train", args.input, args.channels)
    if args.max_train and len(mtr) > args.max_train:
        keep = np.sort(rng.choice(len(mtr), args.max_train, replace=False))
        xtr, mtr = Rows(xtr.x, xtr.rows[keep], args.channels), mtr.iloc[keep].reset_index(drop=True)
    xva, mva = load(args.cache, "val", args.input, args.channels)
    xte, mte = load(args.cache, "test", args.input, args.channels)
    ytr = targets(mtr)
    mu_y, sd_y = ytr.mean(0), ytr.std(0)
    str_ = scalars(mtr, args.input)
    mu_s, sd_s = str_.mean(0), str_.std(0)

    def prep(meta):
        return (scalars(meta, args.input) - mu_s) / sd_s, (targets(meta) - mu_y) / sd_y

    s_tr, y_tr = prep(mtr)
    s_va, y_va = prep(mva)
    s_te, _ = prep(mte)
    model = ParamNet(xtr.shape[1], tuple(xtr.shape[2:]), s_tr.shape[1], len(TARGETS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(xtr) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    log = []
    best, best_state = np.inf, None
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for idx in batches(len(xtr), args.batch, rng):
            xb = to_device(xtr, idx, device)
            if rectified and args.baseline_sd_hz > 0:
                err = torch.randn(len(idx), device=device) * args.baseline_sd_hz
                xb = shift_frequency(xb, err / RCFG.df_step_hz)
            xb = (xb + (torch.rand(len(idx), 1, 1, 1, device=device) - 0.5) * 0.16).clamp_(0, 1)
            yb = torch.from_numpy(y_tr[idx]).to(device)
            sb = torch.from_numpy(s_tr[idx]).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                m, lv = model(xb, sb)
            loss = beta_nll(m.float(), lv.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            nb += 1
        m, lv = predict(model, xva, s_va, device)
        vis = mva["visible"].to_numpy()
        val_mae = np.abs(m - y_va)[vis].mean(0) * sd_y
        val_nll = float(np.mean(0.5 * (lv + (m - y_va) ** 2 / np.exp(lv))))
        score = float(val_mae.mean())
        log.append({"epoch": epoch, "train_loss": tot / nb, "val_nll": val_nll, "val_mae_log": val_mae.tolist(), "s": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
        if score < best:
            best = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save({"state": best_state, "mu_y": mu_y, "sd_y": sd_y, "mu_s": mu_s, "sd_s": sd_s, "args": vars(args) | {"cache": str(args.cache), "run": str(args.run)}}, args.run / "model.pt")
    (args.run / "log.json").write_text(json.dumps(log, indent=1))

    def frame(m, lv, tag):
        out = {}
        for j, name in enumerate(TARGETS):
            mean_log = m[:, j] * sd_y[j] + mu_y[j]
            sig = np.exp(0.5 * lv[:, j]) * sd_y[j]
            val = np.exp(mean_log)
            if name == "undershoot_hz":
                val = np.clip(val - UNDERSHOOT_OFFSET_HZ, 0, None)
            out[f"pred_{name}"] = val
            out[f"sigma_log_{name}"] = sig
        df = pd.DataFrame(out)
        df["chirp_row"] = mte["row"].to_numpy()
        df["condition"] = tag
        return df

    preds = [frame(*predict(model, xte, s_te, device), "true_baseline")]
    if rectified:
        trng = np.random.default_rng(7)
        for sd in args.test_baseline_sd:
            err = trng.normal(0, sd, len(xte))
            m, lv = predict(model, xte, s_te, device, shift_hz=err)
            df = frame(m, lv, f"baseline_sd_{sd:g}")
            df["baseline_err_hz"] = err
            preds.append(df)
    jdir = args.cache / "test_jbox"
    if (jdir / "meta.parquet").exists():  # same chirps, jittered boxes
        xj, mj = load(args.cache, "test_jbox", args.input, args.channels)
        assert (mj["row"].to_numpy() == mte["row"].to_numpy()).all()
        sj = (scalars(mj, args.input) - mu_s) / sd_s
        preds.append(frame(*predict(model, xj, sj.astype(np.float32), device), "jittered_box"))
    pd.concat(preds, ignore_index=True).to_parquet(args.run / "test_predictions.parquet")
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
