"""Assign chirps detected on a spectrogram to wavetracker tracks."""

import pathlib
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from gridtools.datasets import Dataset, load
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


def non_max_suppression_fast(
    chirp_df: pd.DataFrame,
    overlapthresh: float,
) -> list:
    """Raster implementation of non-maximum suppression.

    To remove overlapping bounding boxes.

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
    # slightly modified version of
    # https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

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


def track_filter(
    chirp_df: pd.DataFrame,
    minf: float,
    maxf: float,
) -> pd.DataFrame:
    """Remove chirp bboxes that do not overlap with tracks.

    Parameters
    ----------
    - `chirp_df`: `pd.dataframe`
        Dataframe containing the chirp bboxes
    - `minf`: `float`
        Minimum frequency of the range
    - `maxf`: `float`
        Maximum frequency of the range

    Returns
    -------
    - `chirp_df_tf`: `pd.dataframe`
        Dataframe containing the chirp bboxes that overlap with the range
    """
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
                0,
                min(row["f2"], range_box[3]) - max(row["f1"], range_box[1]),
            )
        ),
        axis=1,
    )
    return chirp_df_tf.loc[intersection > 0, :]


def clean_bboxes(data: Dataset, chirp_df: pd.DataFrame) -> pd.DataFrame:
    """Clean the chirp bboxes.

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
    minf = np.min(data.track.freqs)
    maxf = np.max(data.track.freqs)
    # maybe add some more cleaning here, such
    # as removing chirps that are too short or too long
    return track_filter(chirp_df_nms, minf, maxf)


def cleanup(chirp_df: pd.DataFrame, data: Dataset) -> pd.DataFrame:
    """Clean the chirp bboxes.

    This is a collection of filters that remove bboxes that
    either overlap, are out of range or otherwise do not make sense.

    Parameters
    ----------
    - `chirp_df`: `pd.dataframe`
        Dataframe containing the chirp bboxes
    - `data`: `gridtools.datasets.Dataset`
        Dataset object containing the data

    Returns
    -------
    - `chirp_df`: `pd.dataframe`
        Dataframe containing the chirp bboxes that overlap with the range
    """
    # first clean the bboxes
    chirp_df = clean_bboxes(data, chirp_df)
    # sort chirps in df by time, i.e. t1
    chirp_df = chirp_df.sort_values(by="t1", ascending=True)
    # compute chirp times, i.e. center of the bbox x axis
    return bbox_to_chirptimes(chirp_df)


def bbox_to_chirptimes(chirp_df: pd.DataFrame) -> pd.DataFrame:
    """Convert chirp bboxes to chirp times.

    Parameters
    ----------
    - `chirp_df`: `pd.dataframe`
        dataframe containing the chirp bboxes

    Returns
    -------
    - `chirp_df`: `pd.dataframe`
        dataframe containing the chirp bboxes with chirp times.
    """
    chirp_df["chirp_times"] = np.mean(chirp_df[["t1", "t2"]], axis=1)

    return chirp_df


def make_indices(
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
    t1 = chirp_df.t1.to_numpy()[idx] - 0.2 * diffr
    t2 = chirp_df.t2.to_numpy()[idx] + 0.2 * diffr
    start_idx = int(np.round(t1 * data.grid.samplerate))
    stop_idx = int(np.round(t2 * data.grid.samplerate))
    center_idx = int(np.round(chirp * data.grid.samplerate))
    return start_idx, stop_idx, center_idx


def get_env_trough(
    env: np.ndarray,
    raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the envelope troughs and their prominences.

    Parameters
    ----------
    - `env`: `np.ndarray`
        Envelope of the filtered signal
    - `raw`: `np.ndarray`
        Raw signal

    Returns
    -------
    - `peaks`: `np.ndarray`
        Indices of the envelope troughs
    - `proms`: `np.ndarray`
        Prominences of the envelope troughs
    """
    # normalize the envelope using the amplitude of the raw signal
    # to preserve the amplitude of the envelope
    env = env / np.max(np.abs(raw))

    # cut of the first and last 20% of the envelope
    env[: int(0.2 * len(env))] = np.nan
    env[int(0.8 * len(env)) :] = np.nan

    # find troughs in the envelope and compute trough prominences
    peaks, params = find_peaks(-env, prominence=1e-3)
    proms = params["prominences"]
    return peaks, proms


def extract_envelope_trough(
    data: Dataset,
    best_electrode: int,
    second_best_electrode: int,
    best_freq: float,
    indices: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
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
    """
    (
        start_idx,
        _,
        stop_idx,
    ) = indices

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
    peaks, proms = get_env_trough(env, raw_filtered)
    return peaks, proms


def extract_assignment_data(
    data: Dataset, chirp_df: pd.DataFrame
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dataset]:
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
    chirp_df = cleanup(chirp_df, data)

    # now loop over all tracks and assign chirps to tracks
    chirp_indices = []  # index of chirp in chirp_df
    track_ids = []  # id of track / fish
    peak_prominences = []  # prominence of trough in envelope
    peak_distances = []  # distance of trough to chirp center
    peak_times = []  # time of trough in envelope, should be close to chirp

    for fish_id in data.track.ids:
        # get chirps, times and freqs and powers for this track
        chirps = np.array(chirp_df.chirp_times.values)
        time = data.track.times[
            data.track.indices[data.track.idents == fish_id]
        ]
        freq = data.track.freqs[data.track.idents == fish_id]
        powers = data.track.powers[data.track.idents == fish_id, :]

        for idx, chirp in enumerate(chirps):
            # find the closest time, freq and power to the chirp time
            closest_idx = np.argmin(np.abs(time - chirp))
            best_electrode = np.argmax(powers[closest_idx, :]).astype(int)
            second_best_electrode = np.argsort(powers[closest_idx, :])[-2]
            best_freq = freq[closest_idx]

            # check if chirp overlaps with track
            f1 = chirp_df.f1.to_numpy()[idx]
            f2 = chirp_df.f2.to_numpy()[idx]
            if (f1 > best_freq) or (f2 < best_freq):
                peak_distances.append(np.nan)
                peak_prominences.append(np.nan)
                peak_times.append(np.nan)
                chirp_indices.append(idx)
                track_ids.append(fish_id)
                continue

            # determine start and stop index of time window on raw data
            # using bounding box start and stop times of chirp detection
            start_idx, stop_idx, center_idx = make_indices(
                chirp_df, data, idx, chirp
            )

            indices = (start_idx, stop_idx, center_idx)

            peaks, proms = extract_envelope_trough(
                data,
                best_electrode,
                second_best_electrode,
                best_freq,
                indices,
            )

            # if no peaks are found, skip this chirp
            if len(peaks) == 0:
                peak_distances.append(np.nan)
                peak_prominences.append(np.nan)
                peak_times.append(np.nan)
                chirp_indices.append(idx)
                track_ids.append(fish_id)
                continue

            # compute index to closest peak to chirp center
            distances = np.abs(peaks - (center_idx - start_idx))
            closest_peak_idx = np.argmin(distances)

            # store peak prominence and distance to chirp center
            peak_distances.append(distances[closest_peak_idx])
            peak_prominences.append(proms[closest_peak_idx])
            peak_times.append(
                (start_idx + peaks[closest_peak_idx]) / data.grid.samplerate,
            )
            chirp_indices.append(idx)
            track_ids.append(fish_id)

    peak_prominences = np.array(peak_prominences)
    peak_distances = (
        np.array(peak_distances) + 1
    )  # add 1 to avoid division by zero
    peak_times = np.array(peak_times)
    chirp_indices = np.array(chirp_indices)
    track_ids = np.array(track_ids)

    assignment_data = {
        "proms": peak_prominences,
        "peaks": peak_distances,
        "ptimes": peak_times,
        "cindices": chirp_indices,
        "track_ids": track_ids,
    }
    return (
        assignment_data,
        chirp_df,
        data,
    )


def assign_chirps(
    assign_data: Dict[str, np.ndarray],
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
    # extract data from assign_data
    peak_prominences = assign_data["proms"]
    peak_distances = assign_data["peaks"]
    peak_times = assign_data["ptimes"]
    chirp_indices = assign_data["cindices"]
    track_ids = assign_data["track_ids"]

    # compute cost function.
    # this function is high when the trough prominence is high
    # (-> chirp with high contrast)
    # and when the trough is close to the chirp center as detected by the
    # r-cnn (-> detected chirp is close to the actual chirp)
    cost = peak_prominences / peak_distances**2

    # set cost to zero for cases where no peak was found
    cost[np.isnan(cost)] = 0

    # for each chirp, choose the track where the cost is highest
    # TODO: to avoid confusion make a cost function where high is good and low
    # is bad. this is more like a "gain function"
    chosen_tracks = []
    chosen_track_times = []
    for idx in np.unique(chirp_indices):
        candidate_tracks = track_ids[chirp_indices == idx]
        candidate_costs = cost[chirp_indices == idx]
        candidate_times = peak_times[chirp_indices == idx]
        chosen_tracks.append(candidate_tracks[np.argmax(candidate_costs)])
        chosen_track_times.append(candidate_times[np.argmax(candidate_costs)])

    # store chosen tracks in chirp_df
    chirp_df["assigned_track"] = chosen_tracks

    # store chirp time estimated from envelope trough in chirp_df
    chirp_df["envelope_trough_time"] = chosen_track_times

    # save chirp_df
    chirp_df.to_csv(data.path / "chirpdetector_bboxes.csv", index=False)

    # save old format:
    np.save(data.path / "chirp_ids_rcnn.npy", chosen_tracks)
    np.save(data.path / "chirp_times_rcnn.npy", chosen_track_times)


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
    # config = load_config(path / "chirpdetector.toml")
    recs = list(path.iterdir())
    recs = [r for r in recs if r.is_dir()]
    # recs = [path / "subset_2020-03-18-10_34_t0_9320.0_t1_9920.0"]

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

            data = load(rec, grid=True)
            chirp_df = pd.read_csv(rec / "chirpdetector_bboxes.csv")
            assign_data, chirp_df, data = extract_assignment_data(
                data, chirp_df
            )
            assign_chirps(assign_data, chirp_df, data)
            prog.update(task, advance=1)
