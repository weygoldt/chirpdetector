"""Colletction of chirp assignment algorithms."""

from abc import ABC, abstractmethod
from typing import List, Self

import numpy as np
import pandas as pd
import torch
from gridtools.datasets.models import Dataset
from PIL import Image
from scipy.signal import find_peaks
from torch import nn

from chirpdetector.models.utils import get_device


class AbstractBoxAssigner(ABC):
    """Default wrapper around different box assignment methods."""

    def __init__(  # noqa
        self: Self,
        model: nn.Module,
    ) -> None:
        """Initialize the BoxAssigner."""
        self.model = model

    @abstractmethod
    def assign(  # noqa
        self: Self,
        batch_specs: List,
        batch_times: List,
        batch_freqs: List,
        batch_detections: pd.DataFrame,
        data: Dataset,
    ) -> pd.DataFrame:
        """Assign boxes to tracks."""
        pass


class SpectrogramPowerTroughBoxAssignerMLP(AbstractBoxAssigner):
    """Assign boxes to tracks based on power troughs on spectrograms.

    ... but using a multi layer perceptron to do the assignment.
    """

    def assign(  # noqa
        self: Self,
        batch_specs: List,
        batch_times: List,
        batch_freqs: List,
        batch_detections: pd.DataFrame,
        data: Dataset,
    ) -> pd.DataFrame:
        """Extract trougths in power and assign boxes to tracks using a MLP.

        Parameters
        ----------
        - `batch_specs` : `List`
            List of spectrograms.
        - `batch_times` : `List`
            List of time vectors.
        - `batch_freqs` : `List`
            List of frequency vectors.
        - `batch_detections` : `pd.DataFrame`
            DataFrame with the detections.
        - `data` : `Dataset`
            Dataset object.

        Returns
        -------
        - `pd.DataFrame`
            DataFrame with predicted emitter EODfs.
        """
        # TODO: This function takes torch.tensors, processes them on cpu and
        # with numpy and then puts them back to the device. This is not
        # efficient and should be optimized.

        # load model
        model = self.model

        boxes = batch_detections[["t1", "f1", "t2", "f2"]].to_numpy()
        boxes_spec_idx = batch_detections["spec"].to_numpy().astype(int)

        if len(boxes) == 0:
            return batch_detections

        # print(f"max spec idx: {np.max(boxes_spec_idx)}")
        # print(f"min spec idx: {np.min(boxes_spec_idx)}")

        assigned_eodfs = []
        for idx, box in zip(boxes_spec_idx, boxes):
            upper_y_cutoff = 0.66 * (box[3] - box[1]) + box[1]

            # get the spectrogram window in the box
            window_spec = batch_specs[idx][
                0, :, (batch_times[idx] > box[0]) & (batch_times[idx] < box[2])
            ]
            window_spec = window_spec[
                (batch_freqs[idx] > box[1])
                & (batch_freqs[idx] < upper_y_cutoff)
            ]
            window_freqs = batch_freqs[idx][
                (batch_freqs[idx] > box[1])
                & (batch_freqs[idx] < upper_y_cutoff)
            ]
            window_times = batch_times[idx][
                (batch_times[idx] > box[0]) & (batch_times[idx] < box[2])
            ]

            # interpolate the window to 100x100
            res = 100
            window_spec = np.array(
                Image.fromarray(window_spec.cpu().numpy()).resize((res, res))
            )
            window_freqs = np.linspace(box[1], upper_y_cutoff, res)
            window_times = np.linspace(box[0], box[2], res)

            # get the candidate baseline EODfs by finding peaks on the
            # power spectrum
            target_prom = np.mean(window_spec) * res * 0.5
            target_prom = (np.max(window_spec) - np.min(window_spec)) * 0.1
            power = np.mean(window_spec, axis=1)
            peaks = find_peaks(power, prominence=target_prom)[0]

            if len(peaks) == 0:
                assigned_eodfs.append(np.nan)
                # print("no peaks found")
                continue

            # window width to 10 Hz in frequency
            window_radius = 8  # Hz

            # get the start and end of the peak
            starts = []
            ends = []

            # TODO: This is probably super slow and should be optimized
            for idx2 in peaks:
                # get the start of the peak
                start = idx2
                while True:
                    if start == 0:
                        break
                    if (
                        window_freqs[idx2] - window_freqs[start]
                        > window_radius
                    ):
                        break
                    if np.sign(power[idx2] - power[start]) == -1:
                        break
                    start -= 1

                # get the end of the peak
                end = idx2
                while True:
                    if end > len(power) - 1:
                        break
                    if window_freqs[end] - window_freqs[idx2] > window_radius:
                        break
                    if np.sign(power[idx2] - power[end]) == -1:
                        break
                    end += 1

                starts.append(start)
                ends.append(end)

            # get the average power in the window over time
            powers = []
            eodfs = []
            for start, end in zip(starts, ends):
                mu = np.mean(window_spec[start:end, :], axis=0)
                eodf = np.mean(window_freqs[start:end])
                powers.append(mu)
                eodfs.append(eodf)

            powers = np.array(powers)
            eodfs = np.array(eodfs)

            # shift each power to its own mean
            powers = powers - np.mean(powers, axis=1)[:, np.newaxis]

            # make tensor and put to device
            powers = torch.tensor(powers, dtype=torch.float32).unsqueeze(0)
            powers = powers.to(get_device())

            with torch.no_grad():
                pred = model(powers)

            if pred.is_cuda:
                pred = pred.cpu()
            pred = pred.numpy()

            # turn into 1d array
            while len(pred.shape) > 1:
                pred = pred.flatten()

            emitter = np.argmax(pred).item()

            thresh = 0.0  # should be thresh at f1 score
            if pred[emitter] < thresh:
                # print("no emitter found")
                assigned_eodfs.append(np.nan)
                continue

            emitter_eodf = eodfs[emitter]

            # print(f"output: {pred}")
            # print(f"emitter: {emitter}")
            # print(f"emitter eodf: {emitter_eodf}")

            assigned_eodfs.append(emitter_eodf)

            # fig, ax = plt.subplots(1,3)
            # ax[0].imshow(window_spec, aspect="auto", extent=[window_times[0], window_times[-1], window_freqs[0], window_freqs[-1]])
            # ax[0].set_title("Spectrogram")
            # ax[0].axhline(emitter_eodf, c="r")
            # ax[0].set_ylim([window_freqs[0], window_freqs[-1]])
            # for eodf in eodfs:
            #     ax[0].axhline(eodf, c="gray")
            #
            # ax[1].plot(power, window_freqs)
            # ax[1].scatter(power[peaks], window_freqs[peaks], c="r")
            # ax[1].axvline(target_prom, c="r")
            # for s,e in zip(starts, ends):
            #     ax[1].axhline(window_freqs[s], c="r")
            #     ax[1].axhline(window_freqs[e], c="r")
            # ax[1].set_ylim([window_freqs[0], window_freqs[-1]])
            #
            # for i, p in enumerate(powers):
            #     if i == emitter:
            #         ax[2].plot(window_times, p, c="r")
            #     else:
            #         ax[2].plot(window_times, p, c="gray")
            #
            # plt.show()
            #
            #
            # # plot the full spectrogram
            # fig, ax = plt.subplots(1, 1)
            # ax.imshow(batch_specs[idx][0, :, :].cpu().numpy(), aspect="auto", extent=[batch_times[idx][0], batch_times[idx][-1], batch_freqs[idx][0], batch_freqs[idx][-1]])
            # ax.set_title("Spectrogram")
            # ax.axhline(emitter_eodf, c="r")
            # ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor="r"))
            # ax.set_ylim([batch_freqs[idx][0], batch_freqs[idx][-1]])
            # plt.show()

        batch_detections.loc[:, "emitter_eodf"] = assigned_eodfs

        # drop all boxes that were not assigned
        batch_detections = batch_detections.dropna()

        return batch_detections
