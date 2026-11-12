"""Predict a cached split (e.g. a test set with other box jitter) with trained runs.

Writes ``RUN/pred_<split>.parquet`` (same columns as test_predictions, with
``condition = <split>``) for every run whose input variant the split holds.

Usage: predict.py CACHE SPLIT RUN [RUN ...]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import TARGETS, UNDERSHOOT_OFFSET_HZ, load, predict, scalars  # noqa: E402

from chirpdetector.params.model import ParamNet  # noqa: E402


def main() -> None:
    cache, split = Path(sys.argv[1]), sys.argv[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("CHIRP_MAX_VRAM_FRACTION", "0.1")))
    for run in map(Path, sys.argv[3:]):
        ck = torch.load(run / "model.pt", weights_only=False)
        a = ck["args"]
        x, meta = load(cache, split, a["input"], a.get("channels"))
        s = ((scalars(meta, a["input"]) - ck["mu_s"]) / ck["sd_s"]).astype(np.float32)
        model = ParamNet(x.shape[1], tuple(x.shape[2:]), s.shape[1], len(TARGETS)).to(device)
        model.load_state_dict(ck["state"])
        m, lv = predict(model, x, s, device)
        out = {}
        for j, name in enumerate(TARGETS):
            val = np.exp(m[:, j] * ck["sd_y"][j] + ck["mu_y"][j])
            if name == "undershoot_hz":
                val = np.clip(val - UNDERSHOOT_OFFSET_HZ, 0, None)
            out[f"pred_{name}"] = val
            out[f"sigma_log_{name}"] = np.exp(0.5 * lv[:, j]) * ck["sd_y"][j]
        df = pd.DataFrame(out)
        df["chirp_row"] = meta["row"].to_numpy()
        df["condition"] = split
        df.to_parquet(run / f"pred_{split}.parquet")
        print(run.name, split, len(df), flush=True)


if __name__ == "__main__":
    main()
