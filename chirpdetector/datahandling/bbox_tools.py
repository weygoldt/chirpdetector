"""Tools to work with bounding boxes."""

from typing import List

import numpy as np
import pandas as pd
import torch
from numba import jit
from torchvision.ops import nms

from chirpdetector.models.utils import get_device


@jit(nopython=True, parallel=True)
def float_index_interpolation(
    values: np.ndarray,
    index_arr: np.ndarray,
    data_arr: np.ndarray,
) -> np.ndarray:
    """Convert float indices to values by linear interpolation.

    Interpolates a set of float indices within the given index
    array to obtain corresponding values from the data
    array using linear interpolation.

    Given a set of float indices (`values`), this function determines
    the corresponding values in the `data_arr` by linearly interpolating
    between adjacent indices in the `index_arr`. Linear interpolation
    involves calculating weighted averages based on the fractional
    parts of the float indices.

    This function is used to transform predicted bounding boxes on images
    to the actual corresponding axes of the images (e.g. time and frequency
    axes of spectrograms). The predicted bounding boxes are floats so that
    a bounding box is not restricted to the pixel grid of the image.

    Parameters
    ----------
    - `values` : `np.ndarray`
        The index value as a float that should be interpolated.
    - `index_arr` : `numpy.ndarray`
        The array of indices on the data array.
    - `data_arr` : `numpy.ndarray`
        The array of data.

    Returns
    -------
    - `numpy.ndarray`
        The interpolated value.

    Raises
    ------
    - `ValueError`
        If any of the input float indices (`values`) are outside
        the range of the provided `index_arr`.

    Examples
    --------
    >>> values = np.array([2.5, 3.2, 4.8])
    >>> index_arr = np.array([2, 3, 4, 5])
    >>> data_arr = np.array([10, 15, 20, 25])
    >>> result = float_index_interpolation(values, index_arr, data_arr)
    >>> print(result)
    array([12.5, 16. , 22.5])
    """
    # Check if the values are within the range of the index array
    if np.any(values < (np.min(index_arr) - 1)) or np.any(
        values > (np.max(index_arr) + 1),
    ):
        msg = (
            "Values outside the range of index array\n"
            f"Target values: {values}\n"
            f"Index array: {index_arr}\n"
            f"Data array: {data_arr}"
        )
        raise ValueError(msg)

    # Find the indices corresponding to the values
    lower_indices = np.floor(values).astype(np.int_)
    upper_indices = np.ceil(values).astype(np.int_)

    # Ensure upper indices are within the array bounds
    upper_indices = np.minimum(upper_indices, len(index_arr) - 1)
    lower_indices = np.minimum(lower_indices, len(index_arr) - 1)

    # TODO: Something is wrong with the weights

    # Calculate the interpolation weights
    weights = values - lower_indices

    # Linear interpolation
    return (1 - weights) * data_arr[lower_indices] + weights * data_arr[
        upper_indices
    ]


@jit(nopython=True, parallel=True)
def reverse_float_index_interpolation(
    values: np.ndarray,
    data_arr: np.ndarray,
    index_arr: np.ndarray,
) -> np.ndarray:
    """Convert float data points not on the data array to float indices.

    Interpolates a set of float data points within the given data
    array to obtain corresponding values from the index
    array using linear interpolation.

    This function is used to transform simulated bounding boxes on images
    from frequency and time to pixel coordinates.

    Parameters
    ----------
    - `values` : `np.ndarray`
        The index value as a float that should be interpolated.
    - `data_arr` : `numpy.ndarray`
        The array of data.
    - `index_arr` : `numpy.ndarray`
        The array of indices on the data array.

    Returns
    -------
    - `numpy.ndarray`
        The interpolated value.

    Raises
    ------
    - `ValueError`
        If any of the input float indices (`values`) are outside
        the range of the provided `index_arr`.
    """
    # Check if the values are within the range of the index array
    if np.any(values < (np.min(data_arr) - 1)) or np.any(
        values > (np.max(data_arr) + 1),
    ):
        msg = (
            "Values outside the range of data array\n"
            f"Target values: {values}\n"
            f"Data array: {data_arr}\n"
            f"Index array: {index_arr}"
        )
        raise ValueError(msg)

    # Find the indices corresponding to the values
    lower_indices = np.searchsorted(data_arr, values, side="left")
    upper_indices = lower_indices + 1
    # print(data_arr)
    # print(values[0])
    # print(f"Lower indices: {lower_indices[0]}")
    # print(f"Upper indices: {upper_indices[0]}")

    # Ensure upper indices are within the array bounds
    maxlen = np.ones(len(lower_indices), dtype=np.int_) * (len(index_arr) - 1)
    upper_indices = np.minimum(upper_indices, maxlen)
    lower_indices = np.minimum(lower_indices, maxlen)

    # if np.any(upper_indices == lower_indices):
    #     print("Some upper and lower indices are equal")
    #     print(f"Upper indices: {upper_indices}")
    #     print(f"Lower indices: {lower_indices}")

    # print(f"Maxlen: {maxlen[0]}")
    # print(f"Max upper indices: {np.max(upper_indices)}")
    # print(f"Max lower indices: {np.max(lower_indices)}")

    # Calculate the interpolation weights
    lower_diff = np.abs(values - data_arr[lower_indices])
    upper_diff = np.abs(data_arr[upper_indices] - values)

    # print(f"Lower diff: {lower_diff[-1]}")
    # print(f"Upper diff: {upper_diff[-1]}")

    # print(f"Lower diff: {lower_diff[0]}")
    # print(f"Upper diff: {upper_diff[0]}")

    # compute the weigth for the upper and lower indices
    lower_weights = upper_diff / (lower_diff + upper_diff)
    upper_weights = lower_diff / (lower_diff + upper_diff)

    # fix the weights where the diff is zero
    lower_weights[lower_diff == 0] = 1
    upper_weights[upper_diff == 0] = 1

    # fix the weights where upper and lower indices are equal
    lower_weights[upper_indices == lower_indices] = 0.5
    upper_weights[upper_indices == lower_indices] = 0.5

    # print(f"Lower weights: {np.array(lower_weights)[-1]}")
    # print(f"Upper weights: {upper_weights[-1]}")

    # Linear interpolation
    return lower_indices * lower_weights + upper_indices * upper_weights


def pixel_box_to_timefreq(
    boxes: np.ndarray, time: np.ndarray, freq: np.ndarray
) -> np.ndarray:
    """Convert the pixel coordinates of a box to time and frequency.

    Parameters
    ----------
    boxes : np.ndarray
        The boxes to convert.
    time : np.ndarray
        The time axis of the spectrogram.
    freq : np.ndarray
        The frequency axis of the spectrogram.

    Returns
    -------
    boxes_timefreq : np.ndarray
        The converted boxes.
    """
    freq_indices = np.arange(len(freq))
    time_indices = np.arange(len(time))

    # convert the pixel coordinates to time and frequency
    t1 = float_index_interpolation(boxes[:, 0], time_indices, time)
    f1 = float_index_interpolation(boxes[:, 1], freq_indices, freq)
    t2 = float_index_interpolation(boxes[:, 2], time_indices, time)
    f2 = float_index_interpolation(boxes[:, 3], freq_indices, freq)

    # turn into same shape as input boxes
    t1 = np.expand_dims(t1, axis=1)
    f1 = np.expand_dims(f1, axis=1)
    t2 = np.expand_dims(t2, axis=1)
    f2 = np.expand_dims(f2, axis=1)

    return np.concatenate([t1, f1, t2, f2], axis=1)


def dataframe_nms(
    chirp_df: pd.DataFrame,
    overlapthresh: float,
) -> List:
    """Non maximum suppression with the torchvision nms implementation.

    ...but with a pandas dataframe as input.

    Parameters
    ----------
    chirp_df : pd.DataFrame
        The dataframe with the detections.
    overlapthresh : float
        The overlap threshold for non-maximum suppression.

    Returns
    -------
    indices : List
        The indices of the boxes to keep after non-maximum suppression.
    """
    # convert the dataframe to a list of boxes
    boxes = chirp_df[["t1", "f1", "t2", "f2"]].to_numpy()

    # convert the boxes to the format expected by torchvision
    boxes = torch.tensor(boxes, dtype=torch.float32).to(get_device())

    # convert the scores to the format expected by torchvision
    scores = torch.tensor(
        chirp_df["score"].to_numpy(), dtype=torch.float32
    ).to(get_device())

    # perform non-maximum suppression
    indices = nms(boxes, scores, overlapthresh)

    # retrieve the indices from the gpu if necessary
    if indices.is_cuda:
        indices = indices.cpu()

    return indices.tolist()
