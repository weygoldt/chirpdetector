"""Assign chirps detected on a spectrogram to wavetracker tracks."""

import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
from gridtools.datasets import Dataset, load
from pydantic import BaseModel, ConfigDict
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)
from scipy.signal import find_peaks

from .utils.filters import bandpass_filter, envelope
from .utils.logging import make_logger

# initialize the progress bar
prog = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
)

# TODO: Update docstrings in this module

class ChirpAssignmentData(BaseModel):
    """Data needed for chirp assignment."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    bbox_index: np.ndarray
    env_trough_times: np.ndarray
    env_trough_prominences: np.ndarray
    env_trough_distances: np.ndarray
    env_trough_indices: np.ndarray
    track_ids: np.ndarray
    envs: np.ndarray


def non_max_suppression_fast(
    chirp_df: pd.DataFrame,
    overlapthresh: float,
) -> list:
    """Faster implementation of non-maximum suppression.

    To remove overlapping bounding boxes.
    Is a slightly modified version of
    https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    .

    Parameters
    ----------
    - `chirp_df`: `pd.dataframe`
        Dataframe containing the chirp bboxes
    - `overlapthresh`: `float`
        Threshold for overlap between bboxes

    Returns
    -------
    - `pick`: `list`
        List of indices of bboxes to keep
    """
    # convert boxes to list of tuples and then to numpy array
    boxes = chirp_df[["t1", "f1", "t2", "f2"]].to_numpy()

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
            idxs,
            np.concatenate(([last], np.where(overlap > overlapthresh)[0])),
        )
        # return the indicies of the picked boxes
    return pick


def remove_bboxes_outside_range(
    chirp_dataframe: pd.DataFrame,
    min_frequency: float,
    max_frequency: float,
) -> pd.DataFrame:
    """Remove chirp bboxes that do not overlap with frequency tracks.

    Parameters
    ----------
    - `chirp_dataframe`: `pd.dataframe`
        Dataframe containing the chirp bboxes
    - `min_frequency`: `float`
        Minimum frequency of the range
    - `max_frequency`: `float`
        Maximum frequency of the range

    Returns
    -------
    - `pd.dataframe`
        Dataframe containing the chirp bboxes that overlap with the range
    """
    # remove all chirp bboxes that have no overlap with the range spanned by
    # minf and maxf

    # first build a box that spans the entire range
    range_box = np.array([
        0, min_frequency, np.max(chirp_dataframe.t2), max_frequency
    ])

    # now compute the intersection between the range box and each chirp bboxes
    # and keep only those that have an intersection area > 0
    chirp_df_tf = chirp_dataframe.copy()
    intersection = chirp_df_tf.apply(
        lambda row: (
            max(0, min(row["t2"], range_box[2]) - max(row["t1"], range_box[0]))
            * max(
                0,
                min(row["f2"], range_box[3]) - max(row["f1"], range_box[1]),
            )
        ),
        axis=1,
    )
    return chirp_df_tf.loc[intersection > 0, :]


def clean_bboxes(data: Dataset, chirp_df: pd.DataFrame) -> pd.DataFrame:
    """Clean up the chirp bboxes.

    This is a collection of filters that remove bboxes that
    either overlap, are out of range or otherwise do not make sense.

    Parameters
    ----------
    - `data`: `gridtools.datasets.Dataset`
        Dataset object containing the data
    - `chirp_df`: `pd.dataframe`
        Dataframe containing the chirp bboxes

    Returns
    -------
    - `chirp_df_tf`: `pd.dataframe`
        Dataframe containing the chirp bboxes that overlap with the range
    """
    # non-max suppression: remove all chirp bboxes that overlap with
    # another more than threshold
    pick_indices = non_max_suppression_fast(chirp_df, 0.5)
    chirp_df_nms = chirp_df.loc[pick_indices, :]

    # track filter: remove all chirp bboxes that do not overlap with
    # the range spanned by the min and max of the wavetracker frequency tracks
    minf = np.min(data.track.freqs).astype(float)
    maxf = np.max(data.track.freqs).astype(float)
    chirp_df = remove_bboxes_outside_range(chirp_df_nms, minf, maxf)

    # sort chirps in df by time, i.e. t1
    chirp_df = chirp_df.sort_values(by="t1", ascending=True)

    # compute chirp times, i.e. center of the bbox x axis
    chirp_df["chirp_times"] = np.mean(chirp_df[["t1", "t2"]], axis=1)

    return chirp_df


def make_chirp_indices_on_raw_data(
    chirp_df: pd.DataFrame, data: Dataset, idx: int, chirp: float
) -> Tuple[int, int, int]:
    """Make indices for the chirp window.

    Parameters
    ----------
    - `chirp_df`: `pd.dataframe`
        Dataframe containing the chirp bboxes
    - `data`: `gridtools.datasets.Dataset`
        Dataset object containing the data
    - `idx`: `int`
        Index of the chirp in the chirp_df
    - `chirp`: `float`
        Chirp time

    Returns
    -------
    - `start_idx`: `int`
        Start index of the chirp window
    - `stop_idx`: `int`
        Stop index of the chirp window
    - `center_idx`: `int`
        Center index of the chirp window
    """
    # determine start and stop index of time window on raw data
    # using bounding box start and stop times of chirp detection
    diffr = chirp_df.t2.to_numpy()[idx] - chirp_df.t1.to_numpy()[idx]
    t1 = chirp_df.t1.to_numpy()[idx] - 0.5 * diffr
    t2 = chirp_df.t2.to_numpy()[idx] + 0.5 * diffr

    start_idx = int(np.round(t1 * data.grid.samplerate))
    stop_idx = int(np.round(t2 * data.grid.samplerate))
    center_idx = int(np.round(chirp * data.grid.samplerate))

    return start_idx, stop_idx, center_idx


def extract_envelope_trough(
    data: Dataset,
    best_electrode: int,
    second_best_electrode: int,
    best_freq: float,
    indices: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract envelope troughs.

    Extracts a snippet from the raw data around the chirp time and computes
    the envelope of the bandpass filtered signal. Then finds the troughs in
    the envelope and computes their prominences.

    Parameters
    ----------
    - `data`: `gridtools.datasets.Dataset`
        Dataset object containing the data
    - `best_electrode`: `int`
        Index of the best electrode
    - `second_best_electrode`: `int`
        Index of the second best electrode
    - `best_freq`: `float`
        Frequency of the chirp
    - `indices`: `Tuple[int, int, int]`
        Tuple containing the start, center, stop indices of the chirp

    Returns
    -------
    - `peaks`: `np.ndarray`
        Indices of the envelope troughs
    - `proms`: `np.ndarray`
        Prominences of the envelope troughs
    - `env`: `np.ndarray`
        Envelope of the filtered signal
    """
    start_idx, stop_idx, _= indices

    # determine bandpass cutoffs above and below baseline frequency
    lower_f = best_freq - 15
    upper_f = best_freq + 15

    # get the raw signal on the 2 best electrodes and make differential
    raw1 = data.grid.rec[start_idx:stop_idx, best_electrode]
    raw2 = data.grid.rec[start_idx:stop_idx, second_best_electrode]
    raw = raw1 - raw2

    # bandpass filter the raw signal
    raw_filtered = bandpass_filter(
        raw,
        data.grid.samplerate,
        lower_f,
        upper_f,
    )

    # compute the envelope of the filtered signal
    env = envelope(
        signal=raw_filtered,
        samplerate=data.grid.samplerate,
        cutoff_frequency=50,
    )

    # normalize the envelope using the amplitude of the raw signal
    env = env / np.max(np.abs(raw))

    # cut of the first and last 20% of the envelope
    env[: int(0.25 * len(env))] = np.nan
    env[int(0.75 * len(env)) :] = np.nan

    # find troughs in the envelope and compute trough prominences
    peaks, params = find_peaks(-env, prominence=1e-3)
    proms = params["prominences"]
    return peaks, proms, env


def extract_assignment_data(
    data: Dataset, chirp_df: pd.DataFrame
) -> Tuple[ChirpAssignmentData, pd.DataFrame, Dataset]:
    """Get envelope troughs to determine chirp assignment.

    This algorigthm assigns chirps to wavetracker tracks by a series of steps:

    1. clean the chirp bboxes
    2. for each fish track, filter the signal on the best electrode
    3. find troughs in the envelope of the filtered signal
    4. compute the prominence of the trough and the distance to the chirp
    center
    5. compute a cost function that is high when the trough prominence is high
    and the distance to the chirp center is low
    6. compare the value of the cost function for each track and choose the
    track with the highest cost function value

    Parameters
    ----------
    - `data`: `dataset`
        Dataset object containing the data
    - `chirp_df`: `pd.dataframe`
        Dataframe containing the chirp bboxes
    """
    # clean the chirp bboxes
    chirp_df = clean_bboxes(data, chirp_df)

    array_len = len(chirp_df) * len(data.track.ids)
    track_ids = np.concatenate(
        [np.full(len(chirp_df), fish_id) for fish_id in data.track.ids]
    )
    bbox_index = np.concatenate(
        [np.arange(len(chirp_df)) for _ in range(len(data.track.ids))]
    )

    ad = ChirpAssignmentData(
        bbox_index=bbox_index,
        track_ids=track_ids,
        env_trough_times=np.full(array_len, np.nan),
        env_trough_prominences=np.full(array_len, np.nan),
        env_trough_distances=np.full(array_len, np.nan),
        env_trough_indices=np.full(array_len, np.nan),
        envs = np.full((array_len, 20001), np.nan)
    )

    for outer_idx, fish_id in enumerate(data.track.ids):
        # get chirps, times and freqs and powers for this track
        chirps = chirp_df.chirp_times.to_numpy()
        time = data.track.times[
            data.track.indices[data.track.idents == fish_id]
        ]
        freq = data.track.freqs[data.track.idents == fish_id]
        powers = data.track.powers[data.track.idents == fish_id, :]

        if len(time) == 0:
            continue # skip if no track is found

        for inner_idx, chirp in enumerate(chirps):
            # find the closest time, freq and power to the chirp time
            closest_idx = np.argmin(np.abs(time - chirp))
            best_electrode = np.argmax(powers[closest_idx, :]).astype(int)
            second_best_electrode = np.argsort(powers[closest_idx, :])[-2]
            best_freq = freq[closest_idx]

            # check if chirp overlaps with track
            f1 = chirp_df.f1.to_numpy()[inner_idx]
            f2 = chirp_df.f2.to_numpy()[inner_idx]
            f2 = f1 + (f2 - f1) * 0.5 # range is the lower half of the bbox

            # if chirp does not overlap with track, skip this chirp
            if (f1 > best_freq) or (f2 < best_freq):
                continue

            # determine start and stop index of time window on raw data
            # using bounding box start and stop times of chirp detection
            chirp_indices_on_raw_data = make_chirp_indices_on_raw_data(
                chirp_df, data, inner_idx, chirp
            )

            # extract envelope troughs
            troughs, proms, env = extract_envelope_trough(
                data,
                best_electrode,
                second_best_electrode,
                best_freq,
                chirp_indices_on_raw_data,
            )

            # if no envelope troughs are found, skip this chirp
            # append nan to chirp id and envelope trough time
            if len(troughs) == 0:
                continue

            # compute index to closest peak to chirp center
            distances = np.abs(troughs - (
                    chirp_indices_on_raw_data[2] - chirp_indices_on_raw_data[0]
            ))
            closest_trough_idx = np.argmin(distances)
            trough_time = (
                (chirp_indices_on_raw_data[0] + troughs[closest_trough_idx])
                / data.grid.samplerate
            )

            # store data in assignment data object
            tot_idx = outer_idx * len(chirps) + inner_idx
            ad.env_trough_times[tot_idx] = trough_time
            ad.env_trough_prominences[tot_idx] = proms[closest_trough_idx]
            ad.env_trough_distances[tot_idx] = distances[closest_trough_idx] + 1
            ad.env_trough_indices[tot_idx] = troughs[closest_trough_idx]
            center_env_start= len(ad.envs[0,:]) // 2 - len(env) // 2
            center_env_stop = len(ad.envs[0,:]) // 2 + len(env) // 2 + 1

            # if the envelope is even, add a nan to the end
            # otherwise it cant be centered in the array
            if len(env) % 2 == 0:
                env = np.concatenate((env, np.array([np.nan])))

            # center the env in the array and cut off the ends
            # if it is larger
            if len(env) > len(ad.envs[0,:]):
                env = env[len(env) // 2 - len(ad.envs[0,:]) // 2:
                          len(env) // 2 + len(ad.envs[0,:]) // 2 + 1]

            if len(env) != center_env_stop - center_env_start:
                print("env", len(env))
                print("start", center_env_start)
                print("stop", center_env_stop)
                print("diff", center_env_stop - center_env_start)

            ad.envs[
                tot_idx,
                center_env_start:center_env_stop,
            ] = env

    return (
        ad,
        chirp_df,
        data,
    )


def assign_chirps(
    ad: ChirpAssignmentData,
    chirp_df: pd.DataFrame,
    data: Dataset,
) -> None:
    """Assign chirps to wavetracker tracks.

    This function uses the extracted envelope troughs to assign chirps to
    tracks. It computes a cost function that is high when the trough prominence
    is high and the distance to the chirp center is low. For each chirp, the
    track with the highest cost function value is chosen.

    Parameters
    ----------
    - `assign_data`: `dict`
        Dictionary containing the data needed for assignment
    - `chirp_df`: `pd.dataframe`
        Dataframe containing the chirp bboxes
    - `data`: `gridtools.datasets.Dataset`
        Dataset object containing the data
    """
    # compute cost function.
    # this function is high when the trough prominence is high
    # (-> chirp with high contrast)
    # and when the trough is close to the chirp center as detected by the
    # r-cnn (-> detected chirp is close to the actual chirp)
    cost = (
        ad.env_trough_prominences / ad.env_trough_distances**2
    )

    # set cost to zero for cases where no peak was found
    cost[np.isnan(cost)] = 0

    # for each chirp, choose the track where the cost is highest
    # TODO: to avoid confusion make a cost function where high is good and low
    # is bad. this is more like a "gain function"
    chosen_tracks = [] # the assigned ids
    chosen_env_times= [] # the times of the envelope troughs
    chosen_chirp_envs = [] # here go the full envelopes of the chosen chirps
    non_chosen_chirp_envs = [] # here go the full envelopes of nonchosen chirps
    for idx in np.unique(ad.bbox_index):
        candidate_tracks = ad.track_ids[ad.bbox_index == idx]
        candidate_costs = cost[ad.bbox_index == idx]
        candidate_times = ad.env_trough_times[ad.bbox_index == idx]
        candidate_envs = ad.envs[ad.bbox_index == idx, :]

        if np.all(np.isnan(candidate_times)):
            chosen_tracks.append(np.nan)
            chosen_env_times.append(np.nan)
            continue

        chosen_index = np.argmax(candidate_costs)
        non_chosen_indices = np.arange(len(candidate_costs)) != chosen_index

        env_time = candidate_times[chosen_index]
        chosen_env_times.append(env_time)
        if np.isnan(env_time):
            chosen_tracks.append(np.nan)
            print(f"candidate costs: {candidate_costs}")
            print(f"candidate times: {candidate_times}")
        else:
            chosen_tracks.append(candidate_tracks[chosen_index])

        cenv = candidate_envs[chosen_index, :]
        ncenv = candidate_envs[non_chosen_indices, :]

        chosen_chirp_envs.append(cenv)
        for env in ncenv:
            if np.all(np.isnan(env)):
                continue
            non_chosen_chirp_envs.append(env)

    # print('Finished assigning chirps')
    #
    # chosen_env = np.array(chosen_chirp_envs)
    # print(chosen_env.shape)
    #
    # non_chosen_env = np.array(non_chosen_chirp_envs)
    # print(non_chosen_env.shape)
    #
    # # TODO: Save envs do disk for plotting
    # import matplotlib.pyplot as plt
    # import matplotlib as mpl
    # mpl.use("TkAgg")
    #
    # fig, ax = plt.subplots(1,2)
    # for env in chosen_env:
    #     ax[0].plot(env, alpha=0.05, c="k")
    #
    # for env in non_chosen_env:
    #     ax[1].plot(env, c="k", alpha=0.05)
    # plt.show()
    #
    # store chosen tracks in chirp_df
    chirp_df["assigned_track"] = chosen_tracks

    # store chirp time estimated from envelope trough in chirp_df
    chirp_df["envelope_trough_time"] = chosen_env_times

    # save chirp_df
    chirp_df.to_csv(data.path / "chirpdetector_bboxes.csv", index=False)

    # save old format:
    chosen_tracks = np.array(chosen_tracks)
    chosen_env_times = np.array(chosen_env_times)
    chosen_tracks = chosen_tracks[~np.isnan(chosen_tracks)].astype(int)
    chosen_env_times = chosen_env_times[~np.isnan(chosen_env_times)].astype(
        float
    )
    np.save(data.path / "chirp_ids_rcnn.npy", chosen_tracks)
    np.save(data.path / "chirp_times_rcnn.npy", chosen_env_times)


def assign_cli(path: pathlib.Path) -> None:
    """Assign chirps to wavetracker tracks.

    this is the command line interface for the assign_chirps function.

    Parameters
    ----------
    - `path`: `pathlib.path`
        path to the directory containing the chirpdetector.toml file
    """
    if not path.is_dir():
        msg = f"{path} is not a directory"
        raise ValueError(msg)

    if not (path / "chirpdetector.toml").is_file():
        msg = f"{path} does not contain a chirpdetector.toml file"
        raise ValueError(msg)

    logger = make_logger(__name__, path / "chirpdetector.log")
    recs = list(path.iterdir())
    recs = [r for r in recs if r.is_dir()]

    msg = f"found {len(recs)} recordings in {path}, starting assignment"
    prog.console.log(msg)
    logger.info(msg)

    prog.console.rule("starting assignment")
    with prog:
        task = prog.add_task("assigning chirps", total=len(recs))
        for rec in recs:
            msg = f"assigning chirps in {rec}"
            logger.info(msg)
            prog.console.log(msg)

            data = load(rec)
            chirp_df = pd.read_csv(rec / "chirpdetector_bboxes.csv")
            assign_data, chirp_df, data = extract_assignment_data(
                data, chirp_df
            )
            assign_chirps(assign_data, chirp_df, data)
            prog.update(task, advance=1)
