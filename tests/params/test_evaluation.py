"""Evaluation statistics on synthetic predictions: coupling, template reuse."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chirpdetector.params.evaluation import coupling, template_reuse
from chirpdetector.params.model import to_model_space
from chirpdetector.params.truth import PARAMETERS


def _predictions(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    """Chirps whose error grows for small heights at low SNR only."""
    rng = np.random.default_rng(seed)
    true = np.stack(
        [
            rng.uniform(30, 500, n),  # height
            rng.uniform(0.005, 0.2, n),  # fwhm
            rng.uniform(0.01, 0.3, n),  # width10
            rng.uniform(0, 30, n),  # undershoot
        ],
        axis=1,
    )
    snr = rng.uniform(-20, 20, n)
    log_true = to_model_space(true)
    spread = np.full_like(log_true, 0.05)
    small = (log_true[:, 0] - np.log(30)) / np.log(500 / 30)  # 0 small .. 1 big
    spread[:, 0] = np.where(snr <= -10, 0.05 + 0.3 * (1 - small), 0.05)
    bias = np.zeros_like(log_true)
    bias[:, 0] = np.where(snr <= -10, 0.2 * (1 - small), 0.0)  # overestimated
    est = log_true + bias + spread * rng.normal(size=log_true.shape)
    out = pd.DataFrame({"snr_db": snr})
    for j, p in enumerate(PARAMETERS):
        out[f"true_{p}"] = true[:, j]
        out[f"mean_log_{p}"] = est[:, j]
        out[f"cal_log_{p}"] = est[:, j]
    return out


def test_coupling_finds_size_dependent_error_within_an_snr_stratum() -> None:
    bins, ranks = coupling(_predictions())
    h = ranks.loc[
        (ranks["error_of"] == "height_hz")
        & (ranks["true_of"] == "height_hz")
        & (ranks["estimate"] == "cal")
    ].set_index("stratum")
    assert h.loc["SNR <= -10 dB", "rho_abs"] < -0.3
    assert h.loc["SNR <= -10 dB", "rho_signed"] < -0.3
    assert abs(h.loc["SNR > 10 dB", "rho_abs"]) < 0.1
    low = bins.loc[
        (bins["parameter"] == "height_hz")
        & (bins["stratum"] == "SNR <= -10 dB")
        & (bins["estimate"] == "cal")
    ].sort_values("bin")
    assert len(low) == 5
    assert low["bias"].iloc[0] > 0.1 > low["bias"].iloc[-1]
    assert low["abs_error"].iloc[0] > 2 * low["abs_error"].iloc[-1]
    assert low["true_hi"].iloc[0] <= low["true_lo"].iloc[1] + 1e-9
    # the other parameters carry no coupling
    fwhm = ranks.loc[
        (ranks["error_of"] == "fwhm_s")
        & (ranks["estimate"] == "cal")
        & (ranks["stratum"] == "all")
    ]
    assert (fwhm["rho_abs"].abs() < 0.1).all()


def test_template_reuse_counts_templates_and_individuals(
    params_library: Path,
) -> None:
    from gridtools.templates.library import individual_labels, load_library

    meta = load_library(params_library, mmap=True).metadata
    labels = individual_labels(meta)
    ids = meta["template_id"].to_numpy(np.int64)
    train = ids[:3]
    test = ids[:6]
    seen, group_seen, info = template_reuse(test, train, params_library)
    np.testing.assert_array_equal(seen, [True] * 3 + [False] * 3)
    expected_groups = np.isin(labels[:6], labels[:3])
    np.testing.assert_array_equal(group_seen, expected_groups)
    assert info["fraction_template_seen"] == pytest.approx(0.5)
    assert info["n_train_templates"] == 3
