"""Colletction of chirp assignment algorithms."""

from abc import ABC, abstractmethod
from typing import List, Self

import numpy as np
import pandas as pd
from gridtools.datasets.models import Dataset
from scipy.signal import find_peaks

from chirpdetector.config import Config


class AbstractBoxAssigner(ABC):
    """Default wrapper around different box assignment methods."""

    def __init__( # noqa
        self: Self,
        cfg: Config,
    ) -> None:
        """Initialize the BoxAssigner."""
        self.cfg = cfg

    @abstractmethod
    def assign( # noqa
        self: Self,
        batch_specs: List,
        batch_times: List,
        batch_freqs: List,
        batch_detections: pd.DataFrame,
        data: Dataset,
    ) -> pd.DataFrame:
        """Assign boxes to tracks."""
        pass


class SpectrogramPowerTroughBoxAssigner(AbstractBoxAssigner):
    """Assign boxes to tracks based on power troughs on spectrograms.

    The idea is to assign boxes to tracks by checking which of the tracks
    has a trough in spectrogram power in the spectrogram bbox.

    This is done by combining the information included the chirp detection
    spectrogram, which has a fine temporal resolution, and the approximation
    of the fishes fundamental frequencies, which have fine frequency
    resolution.

    To do this, I take the chirp detection spectrogram and extract the powers
    that lie below a frequency track of a fish for each bounding box. During
    chirps, there usually is a trough in power. If a fish did not chirp but the
    chirp of another fish crosses its frequency band, there should be a peak
    in power as the signal of the chirper and the signal of the non-chirper
    add up. In short: Chirping fish have a trough in power, non-chirping fish
    have a peak in power (in the ideal world).

    This method uses this notion and just assigns chirps by peak detection.
    """

    def assign( #noqa
        self: Self,
        batch_specs: List,
        batch_times: List,
        batch_freqs: List,
        batch_detections: pd.DataFrame,
        data: Dataset,
    ) -> pd.DataFrame:
        """Assign boxes to tracks by troughts in power.

        Assignment by checking which of the tracks has a trough in spectrogram
        power in the spectrogram bbox.
        """
        padding = 0.05 # seconds before and after bbox bounds to ad

        # retrieve frequency and time for each fish id
        track_ids = np.unique(data.track.ids)
        track_freqs = [
            data.track.freqs[data.track.idents == ident]
            for ident in track_ids
        ]
        track_times = [
            data.track.times[data.track.indices[
            data.track.idents == ident
        ]]
            for ident in track_ids
        ]
        assigned_ids = []
        for i in range(len(batch_detections)):

            # get the current box
            box = batch_detections.iloc[i]

            # get the time and frequency indices for the box
            t1 = box["t1"]
            f1 = box["f1"]
            t2 = box["t2"]
            f2 = box["f2"]
            spec_idx = box["spec"].astype(int)

            # get the power in the box for each track
            box_powers = []
            box_power_times = []
            box_power_ids = []
            for track_id, track_freq, track_time in zip(
                track_ids, track_freqs, track_times
            ):
                # get the time indices for the track
                # as the dataset is interpolated, time and freq indices
                # the same
                track_t1_idx = np.argmin(np.abs(track_time - (t1 - padding)))
                track_t2_idx = np.argmin(np.abs(track_time - (t2 + padding)))

                # get the track snippet in the current bbox
                track_freq_snippet = track_freq[track_t1_idx:track_t2_idx]

                # Check if the frequency values of the snippet are
                # inside the bbox
                if (np.min(track_freq_snippet) > f2) or \
                    (np.max(track_freq_snippet) < f1):
                    # the track does not lie in the box
                    continue

                # Now get the power on spec underneath the track
                # and plot it
                spec_powers = batch_specs[spec_idx].cpu().numpy()[0]
                spec_times = batch_times[spec_idx]
                spec_freqs = batch_freqs[spec_idx]

                spec_t1_idx = np.argmin(np.abs(spec_times - (t1 - padding)))
                spec_t2_idx = np.argmin(np.abs(spec_times - (t2 + padding)))

                spec_powers = spec_powers[:, spec_t1_idx:spec_t2_idx]
                spec_times = spec_times[spec_t1_idx:spec_t2_idx]

                spec_f_indices = [
                    np.argmin(np.abs(spec_freqs - freq))
                    for freq in track_freq_snippet
                ]

                spec_powers = [
                    spec_powers[f_idx, t_idx] for f_idx, t_idx in zip(
                        spec_f_indices, range(len(spec_times))
                    )
                ]

                # store the powers
                box_powers.append(spec_powers)
                box_power_times.append(spec_times)
                box_power_ids.append(track_id)

            # shift the track power baseline to same level
            starts = [power[0] for power in box_powers]
            box_powers = [
                power - start for power, start in zip(box_powers, starts)
            ]

            # detect peaks in the power
            ids = []
            costs = []
            for power, time, track_id in zip(
                box_powers, box_power_times, box_power_ids
            ):
                peaks, props = find_peaks(-power, prominence=0)
                proms = props["prominences"]
                if len(proms) == 0:
                    # no peaks found
                    continue

                # takes the highest peak
                peak = peaks[np.argmax(proms)]
                prom = proms[np.argmax(proms)]

                # Compute peak distance to box center
                box_center = (t1 + t2) / 2
                peak_dist = np.abs(box_center - time[peak])

                # cost is high when peak prominence is low and peak is far away
                # from box center
                cost = (1 / prom) * peak_dist

                ids.append(track_id)
                costs.append(cost)

            # assign the box to the track with the lowest cost
            if len(costs) != 0:
                best_id = ids[np.argmin(costs)]
                assigned_ids.append(best_id)
            else:
                best_id = np.nan
                assigned_ids.append(best_id)

        batch_detections.loc[:, "track_id"] = assigned_ids

        # drop all boxes that were not assigned
        batch_detections = batch_detections.dropna()

        return batch_detections
