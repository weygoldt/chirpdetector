"""Apply a params bundle: calibrated chirp parameters for detection boxes.

`ParamsEstimator` loads a params bundle and estimates, for boxes on a
spectrogram computed with the bundle's front end, the columns
`PARAM_COLUMNS`:

``height_hz, fwhm_s, width10_s, undershoot_hz``
    Calibrated estimates (`chirpdetector.params.calibration`): the
    decoder's model-space mean mapped through the inverse bias curve, back
    to natural units.
``*_sd_*``
    Calibrated standard deviations in natural units (first order:
    ``(value + offset) * sigma``, `chirpdetector.params.model.natural_sd`);
    ``value +- k95 * sd`` (``bundle.calibration.k95``) covered 95 % of the
    validation chirps.
``params_ok``, ``params_flag``
    Whether the box is inside the decoder's validated domain, and if not,
    why (comma-separated): ``edge`` (the crop window crosses the start or
    end of the recording, or of the spectrogram given), ``band`` (the crop
    window reaches past the front end's frequency band), ``size`` (box
    duration or height outside the range seen in training), ``uncertain``
    (a calibrated sigma above the bundle's threshold, the
    ``calibration.flag_quantile`` quantile of the validation sigmas).

The detector knows nothing of this module; `chirpdetector.inference.
predict_span` calls `ParamsEstimator.estimate` on its detections when
given a params bundle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from chirpdetector.bundle import Bundle
from chirpdetector.params.bundle import ParamsBundle, load_params_bundle
from chirpdetector.params.crops import SCALAR_NAMES, DecoderInputs, decoder_inputs
from chirpdetector.params.model import ParamNet, from_model_space, natural_sd
from chirpdetector.params.truth import PARAMETERS
from chirpdetector.resources import apply_vram_limit
from chirpdetector.spectrogram import Spectrogram

__all__ = [
    "BOX_COLUMNS",
    "FLAG_REASONS",
    "PARAM_COLUMNS",
    "ParamsEstimator",
    "load_decoder",
]

BOX_COLUMNS: tuple[str, ...] = ("t_start_s", "t_end_s", "f_low_hz", "f_high_hz")
"""Box columns read from a detection table."""

_SD_NAMES = {
    "height_hz": "height_sd_hz",
    "fwhm_s": "fwhm_sd_s",
    "width10_s": "width10_sd_s",
    "undershoot_hz": "undershoot_sd_hz",
}

PARAM_COLUMNS: tuple[str, ...] = (
    *(c for p in PARAMETERS for c in (p, _SD_NAMES[p])),
    "params_ok",
    "params_flag",
)
"""Columns added by `ParamsEstimator.estimate`."""

FLAG_REASONS: tuple[str, ...] = ("edge", "band", "size", "uncertain")
"""Possible entries of ``params_flag``."""

FloatArray = npt.NDArray[np.float64]


def load_decoder(
    bundle: ParamsBundle, *, device: str | torch.device = "cpu"
) -> ParamNet:
    """The bundle's network with its weights, in eval mode on ``device``."""
    model = ParamNet(
        int(bundle.model["in_channels"]),
        (int(bundle.model["input_hw"][0]), int(bundle.model["input_hw"][1])),
        int(bundle.model["n_scalars"]),
        len(PARAMETERS),
        bundle.model_config,
    )
    state = torch.load(bundle.weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def _device(device: str | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class ParamsEstimator:
    """A params bundle, loaded for inference.

    Parameters
    ----------
    bundle
        Params bundle (or its directory).
    device
        Torch device of the network (default: CUDA if available).
    batch_size
        Crops per forward pass.
    """

    def __init__(
        self,
        bundle: ParamsBundle | str | Path,
        *,
        device: str | None = None,
        batch_size: int = 512,
    ) -> None:
        self.bundle = (
            bundle if isinstance(bundle, ParamsBundle) else load_params_bundle(bundle)
        )
        self.device = _device(device)
        if self.device.type == "cuda":
            apply_vram_limit(None, self.device)
        self.model = load_decoder(self.bundle, device=self.device)
        self.batch_size = int(batch_size)

    @property
    def context_s(self) -> float:
        """Spectrogram needed beyond a box centre (`CropConfig.context_s`)."""
        return self.bundle.crop.context_s

    def check_detector(self, detector: Bundle, *, allow_mismatch: bool = False) -> None:
        """Refuse a detector other than the one the decoder was trained with.

        The front end and preprocessing must equal the detector's always;
        the detector's weights must equal the recorded ones unless
        ``allow_mismatch`` (e.g. footprint-trained decoders, or tests).

        Raises
        ------
        ValueError
            On a mismatch.
        """
        if detector.spectrogram != self.bundle.spectrogram:
            msg = "the params bundle's front end differs from the detector's"
            raise ValueError(msg)
        if dict(detector.preprocessing) != dict(self.bundle.preprocessing):
            msg = "the params bundle's preprocessing differs from the detector's"
            raise ValueError(msg)
        recorded = (self.bundle.detector or {}).get("weights_sha256")
        if recorded != detector.weights_sha256 and not allow_mismatch:
            msg = (
                f"the params bundle was trained on the boxes of detector {recorded}, "
                f"not {detector.weights_sha256}; its parameters are only valid with "
                "that detector (pass allow_detector_mismatch=True to override)"
            )
            raise ValueError(msg)

    @torch.no_grad()
    def raw(self, inputs: DecoderInputs) -> tuple[FloatArray, FloatArray]:
        """Uncalibrated mean and sigma, model space, ``(n, 4)`` each."""
        n = len(inputs)
        if n == 0:
            empty = np.zeros((0, len(PARAMETERS)))
            return empty, empty.copy()
        scalars = self.bundle.standardization.scalars(inputs.scalars)
        means, log_vars = [], []
        for start in range(0, n, self.batch_size):
            stop = min(n, start + self.batch_size)
            x = torch.from_numpy(np.ascontiguousarray(inputs.crops[start:stop]))
            x = x.to(self.device).float().div_(255.0)
            s = torch.from_numpy(scalars[start:stop]).to(self.device)
            mean, log_var = self.model(x, s)
            means.append(mean.float().cpu().numpy())
            log_vars.append(log_var.float().cpu().numpy())
        return self.bundle.standardization.unscale(
            np.concatenate(means), np.concatenate(log_vars)
        )

    def flags(
        self, inputs: DecoderInputs, sigma_log: FloatArray
    ) -> tuple[npt.NDArray[np.bool_], list[str]]:
        """``params_ok`` and ``params_flag`` (module docstring)."""
        n = len(inputs)
        domain = self.bundle.domain
        reasons: list[list[str]] = [[] for _ in range(n)]
        edge = inputs.outside_time > 0
        band = inputs.outside_freq > 0
        size = np.zeros(n, dtype=bool)
        for k, name in enumerate(SCALAR_NAMES):
            lo, hi = domain.get("scalar_range", {}).get(name, (-np.inf, np.inf))
            size |= (inputs.scalars[:, k] < lo) | (inputs.scalars[:, k] > hi)
        uncertain = np.zeros(n, dtype=bool)
        thresholds = domain.get("sigma_threshold", {})
        for j, name in enumerate(PARAMETERS):
            if name in thresholds:
                uncertain |= ~(sigma_log[:, j] <= float(thresholds[name]))
        for mask, reason in zip(
            (edge, band, size, uncertain), FLAG_REASONS, strict=True
        ):
            for i in np.flatnonzero(mask):
                reasons[i].append(reason)
        ok = np.array([not r for r in reasons], dtype=bool)
        return ok, [",".join(r) for r in reasons]

    def estimate_inputs(self, inputs: DecoderInputs) -> pd.DataFrame:
        """`PARAM_COLUMNS` for decoder inputs (one row per crop)."""
        mean, sigma = self.raw(inputs)
        cal_mean, cal_sigma = self.bundle.calibration.apply(mean, sigma)
        values = from_model_space(cal_mean)
        sds = natural_sd(values, cal_sigma)
        ok, flag = self.flags(inputs, cal_sigma)
        data: dict[str, Any] = {}
        for j, name in enumerate(PARAMETERS):
            data[name] = values[:, j]
            data[_SD_NAMES[name]] = sds[:, j]
        data["params_ok"] = ok
        data["params_flag"] = pd.Series(flag, dtype=object)
        return pd.DataFrame(data).loc[:, list(PARAM_COLUMNS)]

    def estimate(self, spec: Spectrogram, boxes: pd.DataFrame) -> pd.DataFrame:
        """`PARAM_COLUMNS` for the boxes of a detection table.

        Parameters
        ----------
        spec
            Spectrogram with the bundle's front end and preprocessing that
            covers the boxes (and `context_s` around their centres, so that
            the canvases are those of the whole recording).
        boxes
            Table with `BOX_COLUMNS`.

        Returns
        -------
        DataFrame
            One row per box, index of ``boxes``.
        """
        if spec.config != self.bundle.spectrogram:
            msg = "the spectrogram's front end differs from the params bundle's"
            raise ValueError(msg)
        array = boxes.loc[:, list(BOX_COLUMNS)].to_numpy(dtype=np.float64)
        inputs = decoder_inputs(spec, array, self.bundle.crop)
        out = self.estimate_inputs(inputs)
        out.index = boxes.index
        return out
