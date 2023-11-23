#!/usr/bin/env python3

"""Assign chirps detected on a spectrogram to wavetracker tracks."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gridtools.datasets import Dataset, load
from IPython import embed
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)
from scipy.signal import find_peaks

from .utils.configfiles import Config, load_config
from .utils.logging import make_logger
from .utils.filters import bandpass_filter, envelope

# initialize the progress bar
prog = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
)


def non_max_suppression_fast(chirp_df, overlapThresh):
    # slightly modified version of
    # https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    # convert boxes to list of tuples and then to numpy array
    boxes = chirp_df[["t1", "f1", "t2", "f2"]].values.tolist()
    boxes = np.array(boxes)

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # compute the ratio of overlap (intersection over union)
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )
        # return the indicies of the picked boxes
    return pick


def track_filter(
    chirp_df: pd.DataFrame, minf: float, maxf: float
) -> pd.DataFrame:
    # remove all chirp bboxes that have no overlap with the range spanned by
    # minf and maxf

    # first build a box that spans the entire range
    range_box = np.array([0, minf, np.max(chirp_df.t2), maxf])

    # now compute the intersection between the range box and each chirp bboxes
    # and keep only those that have an intersection area > 0
    chirp_df_tf = chirp_df.copy()
    intersection = chirp_df_tf.apply(
        lambda row: (
            max(0, min(row["t2"], range_box[2]) - max(row["t1"], range_box[0]))
            * max(
                0, min(row["f2"], range_box[3]) - max(row["f1"], range_box[1])
            )
        ),
        axis=1,
    )
    chirp_df_tf = chirp_df_tf.loc[intersection > 0, :]
    return chirp_df_tf


def clean_bboxes(data: Dataset, chirp_df: pd.DataFrame) -> pd.DataFrame:
    # non-max suppression: remove all chirp bboxes that overlap with
    # another more than threshold
    pick_indices = non_max_suppression_fast(chirp_df, 0.5)
    chirp_df_nms = chirp_df.loc[pick_indices, :]

    # track filter: remove all chirp bboxes that do not overlap with
    # the range spanned by the min and max of the wavetracker frequency tracks
    minf = np.min(data.track.freqs)
    maxf = np.max(data.track.freqs)
    chirp_df_tf = track_filter(chirp_df_nms, minf, maxf)

    # maybe add some more cleaning here, such
    # as removing chirps that are too short or too long

    # import matplotlib
    # from matplotlib.patches import Rectangle
    # matplotlib.use("TkAgg")
    # fig, ax = plt.subplots(figsize=(10, 10))
    # for fish_id in data.track.ids:
    #     time = data.track.times[data.track.indices[data.track.idents == fish_id]]
    #     freq = data.track.freqs[data.track.idents == fish_id]
    #     plt.plot(time, freq, color="black", zorder = 1000)
    # for i, row in chirp_df.iterrows():
    #     ax.add_patch(
    #         Rectangle(
    #             (row["t1"], row["f1"]),
    #             row["t2"] - row["t1"],
    #             row["f2"] - row["f1"],
    #             fill=False,
    #             edgecolor="red",
    #             linewidth=1,
    #         )
    #     )
    # for i, row in chirp_df_tf.iterrows():
    #     ax.add_patch(
    #         Rectangle(
    #             (row["t1"], row["f1"]),
    #             row["t2"] - row["t1"],
    #             row["f2"] - row["f1"],
    #             fill=False,
    #             edgecolor="green",
    #             linewidth=1,
    #             linestyle="dashed",
    #         )
    #     )
    # ax.set_xlim(chirp_df.t1.min(), chirp_df.t2.max())
    # ax.set_ylim(chirp_df.f1.min(), chirp_df.f2.max())
    # plt.show()

    return chirp_df_tf


def bbox_to_chirptimes(chirp_df: pd.DataFrame) -> pd.DataFrame:
    chirp_df["chirp_times"] = np.mean(chirp_df[["t1", "t2"]], axis=1)
    return chirp_df


def assign_chirps(data: Dataset, chirp_df: pd.DataFrame) -> None:
    # first clean the bboxes
    chirp_df = clean_bboxes(data, chirp_df)

    # sort chirps in df by time, i.e. t1
    chirp_df = chirp_df.sort_values(by="t1", ascending=True)

    # compute chirp times, i.e. center of the bbox x axis
    chirp_df = bbox_to_chirptimes(chirp_df)

    # now loop over all tracks and assign chirps to tracks
    assigned_chirps = []  # index to chirp in df
    assigned_chirp_ids = []  # track id for each chirp

    for fish_id in data.track.ids:
        # get chirps, times and freqs and powers for this track
        chirps = np.array(chirp_df.chirp_times.values)
        time = data.track.times[
            data.track.indices[data.track.idents == fish_id]
        ]
        freq = data.track.freqs[data.track.idents == fish_id]
        powers = data.track.powers[data.track.idents == fish_id, :]

        iter = 0

        for idx, chirp in enumerate(chirps):
            # find the closest time, freq and power to the chirp time
            closest_idx = np.argmin(np.abs(time - chirp))
            best_electrode = np.argmax(powers[closest_idx, :])
            second_best_electrode = np.argsort(powers[closest_idx, :])[-2]
            best_freq = freq[closest_idx]

            # determine start and stop index of time window on raw data
            # using bounding box start and stop times of chirp detection
            diffr = chirp_df.t2.values[idx] - chirp_df.t1.values[idx]
            t1 = chirp_df.t1.values[idx] - 0.2 * diffr
            t2 = chirp_df.t2.values[idx] + 0.2 * diffr
            start_idx = int(np.round(t1 * data.grid.samplerate))
            stop_idx = int(np.round(t2 * data.grid.samplerate))
            center_idx = int(np.round(chirp * data.grid.samplerate))

            # determine bandpass cutoffs above and below baseline frequency from track
            lower_f = best_freq - 15
            upper_f = best_freq + 15

            # get the raw signal on the 2 best electrodes and make differential
            raw1 = data.grid.rec[start_idx:stop_idx, best_electrode]
            raw2 = data.grid.rec[start_idx:stop_idx, second_best_electrode]
            raw = raw1 - raw2

            # bandpass filter the raw signal
            raw_filtered = bandpass_filter(
                raw, data.grid.samplerate, lower_f, upper_f
            )

            # compute the envelope of the filtered signal
            env = envelope(
                signal=raw_filtered,
                samplerate=data.grid.samplerate,
                cutoff_frequency=50,
            )
            env_unf = envelope(
                signal=raw,
                samplerate=data.grid.samplerate,
                cutoff_frequency=50,
            )

            import matplotlib

            matplotlib.use("TkAgg")
            fig, ax = plt.subplots(3, 1, figsize=(10, 10))
            ax[0].plot(raw)
            ax[0].plot(raw_filtered)
            ax[0].axvline(center_idx - start_idx, color="black")
            ax[1].plot(env)
            ax[1].axvline(center_idx - start_idx, color="black")
            ax[2].plot(env_unf)
            plt.show()

            iter += 1

            if iter > 10:
                exit()

        # NEXTUP: For each candidate track, compute trough prominence and distance to chirp
        # make a cost function and choose the track with the highest trough prominence and
        # lowest distance to chirp


def assign_cli(path: pathlib.Path):
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory")

    if not (path / "chirpdetector.toml").is_file():
        raise ValueError(
            f"{path} does not contain a chirpdetector.toml file"
            "Make sure you are in the correct directory"
        )

    logger = make_logger(__name__, path / "chirpdetector.log")
    config = load_config(path / "chirpdetector.toml")
    recs = list(path.iterdir())
    recs = [r for r in recs if r.is_dir()]

    msg = f"Found {len(recs)} recordings in {path}, starting assignment"
    print(msg)
    logger.info(msg)

    for rec in recs[1:]:
        logger.info(f"Assigning chirps in {rec}")
        print(rec)
        data = load(rec, grid=True)
        chirp_df = pd.read_csv(rec / "chirpdetector_bboxes.csv")
        assign_chirps(data, chirp_df)
