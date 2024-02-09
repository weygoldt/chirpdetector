"""Colletction of chirp assignment algorithms."""

from abc import ABC, abstractmethod
from typing import List, Self

import numpy as np
import pandas as pd
import torch
from gridtools.datasets.models import Dataset
from PIL import Image
from scipy.signal import find_peaks

from chirpdetector.config import Config
from chirpdetector.models.mlp_assigner import load_trained_mlp
from chirpdetector.models.utils import get_device
from torch import nn


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


# class SpectrogramPowerTroughBoxAssigner(AbstractBoxAssigner):
#     """Assign boxes to tracks based on power troughs on spectrograms.
#
#     The idea is to assign boxes to tracks by checking which of the tracks
#     has a trough in spectrogram power in the spectrogram bbox.
#
#     This is done by combining the information included the chirp detection
#     spectrogram, which has a fine temporal resolution, and the approximation
#     of the fishes fundamental frequencies, which have fine frequency
#     resolution.
#
#     To do this, I take the chirp detection spectrogram and extract the powers
#     that lie below a frequency track of a fish for each bounding box. During
#     chirps, there usually is a trough in power. If a fish did not chirp but the
#     chirp of another fish crosses its frequency band, there should be a peak
#     in power as the signal of the chirper and the signal of the non-chirper
#     add up. In short: Chirping fish have a trough in power, non-chirping fish
#     have a peak in power (in the ideal world).
#
#     This method uses this notion and just assigns chirps by peak detection.
#     """
#
#     def assign(  # noqa
#         self: Self,
#         batch_specs: List,
#         batch_times: List,
#         batch_freqs: List,
#         batch_detections: pd.DataFrame,
#         data: Dataset,
#     ) -> pd.DataFrame:
#         """Assign boxes to tracks by troughts in power.
#
#         Assignment by checking which of the tracks has a trough in spectrogram
#         power in the spectrogram bbox.
#         """
#         padding = 0.05  # seconds before and after bbox bounds to ad
#
#         # retrieve frequency and time for each fish id
#         track_ids = np.unique(data.track.ids)
#         track_freqs = [
#             data.track.freqs[data.track.idents == ident] for ident in track_ids
#         ]
#         track_times = [
#             data.track.times[data.track.indices[data.track.idents == ident]]
#             for ident in track_ids
#         ]
#         assigned_ids = []
#         assigned_eodfs = []
#         for i in range(len(batch_detections)):
#             # get the current box
#             box = batch_detections.iloc[i]
#
#             # get the time and frequency indices for the box
#             t1 = box["t1"]
#             f1 = box["f1"]
#             t2 = box["t2"]
#             f2 = box["f2"]
#             spec_idx = box["spec"].astype(int)
#
#             # get the power in the box for each track
#             box_powers = []
#             box_power_times = []
#             box_power_ids = []
#             box_power_ffreqs = []
#             for track_id, track_freq, track_time in zip(
#                 track_ids, track_freqs, track_times
#             ):
#                 # get the time indices for the track
#                 # as the dataset is interpolated, time and freq indices
#                 # the same
#                 track_t1_idx = np.argmin(np.abs(track_time - (t1 - padding)))
#                 track_t2_idx = np.argmin(np.abs(track_time - (t2 + padding)))
#
#                 # get the track snippet in the current bbox
#                 track_freq_snippet = track_freq[track_t1_idx:track_t2_idx]
#
#                 # Check if the frequency values of the snippet are
#                 # inside the bbox
#                 if (np.min(track_freq_snippet) > f2) or (
#                     np.max(track_freq_snippet) < f1
#                 ):
#                     # the track does not lie in the box
#                     continue
#
#                 # append the ffreqs
#                 box_power_ffreqs.append(np.mean(track_freq_snippet))
#
#                 # Now get the power on spec underneath the track
#                 # and plot it
#                 spec_powers = batch_specs[spec_idx].cpu().numpy()[0]
#                 spec_times = batch_times[spec_idx]
#                 spec_freqs = batch_freqs[spec_idx]
#
#                 spec_t1_idx = np.argmin(np.abs(spec_times - (t1 - padding)))
#                 spec_t2_idx = np.argmin(np.abs(spec_times - (t2 + padding)))
#
#                 spec_powers = spec_powers[:, spec_t1_idx:spec_t2_idx]
#                 spec_times = spec_times[spec_t1_idx:spec_t2_idx]
#
#                 spec_f_indices = [
#                     np.argmin(np.abs(spec_freqs - freq))
#                     for freq in track_freq_snippet
#                 ]
#
#                 spec_powers = [
#                     spec_powers[f_idx, t_idx]
#                     for f_idx, t_idx in zip(
#                         spec_f_indices, range(len(spec_times))
#                     )
#                 ]
#
#                 # store the powers
#                 box_powers.append(spec_powers)
#                 box_power_times.append(spec_times)
#                 box_power_ids.append(track_id)
#
#             # shift the track power baseline to same level
#             starts = [power[0] for power in box_powers]
#             box_powers = [
#                 power - start for power, start in zip(box_powers, starts)
#             ]
#
#             # detect peaks in the power
#             ids = []
#             costs = []
#             for power, time, track_id in zip(
#                 box_powers, box_power_times, box_power_ids
#             ):
#                 peaks, props = find_peaks(-power, prominence=0)
#                 proms = props["prominences"]
#                 if len(proms) == 0:
#                     # no peaks found
#                     continue
#
#                 # takes the highest peak
#                 peak = peaks[np.argmax(proms)]
#                 prom = proms[np.argmax(proms)]
#
#                 # Compute peak distance to box center
#                 box_center = (t1 + t2) / 2
#                 peak_dist = np.abs(box_center - time[peak])
#
#                 # cost is high when peak prominence is low and peak is far away
#                 # from box center
#                 cost = (1 / prom) * peak_dist
#
#                 ids.append(track_id)
#                 costs.append(cost)
#
#             # assign the box to the track with the lowest cost
#             if len(costs) != 0:
#                 best_id = ids[np.argmin(costs)]
#                 assigned_ids.append(best_id)
#                 assigned_eodf = box_power_ffreqs[box_power_ids == best_id]
#                 assigned_eodfs.append(assigned_eodf)
#             else:
#                 best_id = np.nan
#                 assigned_ids.append(best_id)
#                 assigned_eodfs.append(np.nan)
#
#         # batch_detections.loc[:, "track_id"] = assigned_ids
#         batch_detections.loc[:, "emitter_eodf"] = assigned_eodfs
#
#         # drop all boxes that were not assigned
#         batch_detections = batch_detections.dropna()
#
#         return batch_detections
#

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
        """Extract trougths in power and assign boxes to tracks using a MLP."""
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
            # print(f"box {idx}: {box}")
            # print(box)

            upper_y_cutoff = 0.66 * (box[3] - box[1]) + box[1]

            # print(f"Upper bbox border: {box[3]}")
            # print(f"Lower bbox border: {box[1]}")
            # print(f"Upper y cutoff: {upper_y_cutoff}")

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

            # print(f"min freq: {window_freqs[0]}")
            # print(f"max freq: {window_freqs[-1]}")

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
            # print(f"len powers: {len(powers)}")
            # print(f"len eodfs: {len(eodfs)}")

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
