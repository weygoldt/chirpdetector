"""Detect chirps on a spectrogram."""

import logging
import pathlib
import shutil
import time
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gridtools.datasets import Dataset, load, subset
from gridtools.utils.spectrograms import (
    freqres_to_nfft,
    overlap_to_hoplen,
)
from matplotlib.patches import Rectangle
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)

from .models.utils import get_device, load_fasterrcnn
from .utils.configfiles import Config, load_config
from .utils.logging import make_logger
from .utils.signal_processing import (
    compute_sum_spectrogam,
    make_chunk_indices,
    make_spectrogram_axes,
)

# Use non-gui backend for matplotlib to
# avoid memory leaks
mpl.use("Agg")

# initialize the progress bar
prog = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
)


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

    This function is useful to transform float coordinates on a spectrogram
    matrix to the corresponding time and frequency values. The reason for
    this is, that the model outputs bounding boxes in float coordinates,
    i.e. it does not care about the exact pixel location of the bounding
    box.

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
    lower_indices = np.floor(values).astype(int)
    upper_indices = np.ceil(values).astype(int)

    # Ensure upper indices are within the array bounds
    upper_indices = np.minimum(upper_indices, len(index_arr) - 1)
    lower_indices = np.minimum(lower_indices, len(index_arr) - 1)

    # Calculate the interpolation weights
    weights = values - lower_indices

    # Linear interpolation
    return (1 - weights) * data_arr[lower_indices] + weights * data_arr[
        upper_indices
    ]


def coords_to_mpl_rectangle(boxes: np.ndarray) -> np.ndarray:
    """Convert normal bounding box to matplotlib.pathes.Rectangle format.

    Convert box defined by corner coordinates (x1, y1, x2, y2)
    to box defined by lower left, width and height (x1, y1, w, h).

    The corner coordinates are the model output, but the center coordinates
    are needed by the `matplotlib.patches.Rectangle` object for plotting.

    Parameters
    ----------
    - `boxes` : `numpy.ndarray`
        The boxes to be converted.

    Returns
    -------
    - `numpy.ndarray`
        The converted boxes.
    """
    boxes_dims = 2
    if len(boxes.shape) != boxes_dims:
        msg = (
            "The boxes array must be 2-dimensional.\n"
            f"Shape of boxes: {boxes.shape}"
        )
        raise ValueError(msg)
    boxes_cols = 4
    if boxes.shape[1] != boxes_cols:
        msg = (
            "The boxes array must have 4 columns.\n"
            f"Shape of boxes: {boxes.shape}"
        )
        raise ValueError(msg)

    new_boxes = np.zeros_like(boxes)
    new_boxes[:, 0] = boxes[:, 0]
    new_boxes[:, 1] = boxes[:, 1]
    new_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    new_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

    return new_boxes


def plot_detections(
    img_tensor: torch.Tensor,
    output: torch.Tensor,
    threshold: float,
    save_path: pathlib.Path,
    conf: Config,
) -> None:
    """Plot the detections on the spectrogram.

    Parameters
    ----------
    - `img_tensor` : `torch.Tensor`
        The spectrogram.
    - `output` : `torch.Tensor`
        The output of the model.
    - `threshold` : `float`
        The threshold for the detections.
    - `save_path` : `pathlib.Path`
        The path to save the plot to.
    - `conf` : `Config`
        The configuration object.

    Returns
    -------
    - `None`
    """
    # retrieve all the data from the output and convert
    # spectrogram to numpy array
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)[..., 0]
    boxes = output["boxes"].detach().cpu().numpy()
    boxes = coords_to_mpl_rectangle(boxes)
    scores = output["scores"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()
    labels = [conf.hyper.classes[i] for i in labels]

    _, ax = plt.subplots(figsize=(20, 10))

    ax.pcolormesh(img, cmap="magma")

    for i, box in enumerate(boxes):
        if scores[i] > threshold:
            ax.scatter(
                box[0],
                box[1],
            )
            ax.add_patch(
                Rectangle(
                    box[:2],
                    box[2],
                    box[3],
                    fill=False,
                    color="white",
                    linewidth=1,
                ),
            )
            ax.text(
                box[0],
                box[1],
                f"{scores[i]:.2f}",
                color="black",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 1},
            )
    plt.axis("off")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def spec_to_image(spec: torch.Tensor) -> torch.Tensor:
    """Convert a spectrogram to an image.

    Add 3 color channels, normalize to 0-1, etc.

    Parameters
    ----------
    - `spec` : `torch.Tensor`

    Returns
    -------
    - `torch.Tensor`
    """
    # make sure the spectrogram is a tensor
    if not isinstance(spec, torch.Tensor):
        msg = (
            "The spectrogram must be a torch.Tensor.\n"
            f"Type of spectrogram: {type(spec)}"
        )
        raise TypeError(msg)

    # make sure the spectrogram is 2-dimensional
    spec_dims = 2
    if len(spec.size()) != spec_dims:
        msg = (
            "The spectrogram must be a 2-dimensional matrix.\n"
            f"Shape of spectrogram: {spec.size()}"
        )
        raise ValueError(msg)

    # make sure the spectrogram contains some data
    if (
        np.max(spec.detach().cpu().numpy())
        - np.min(spec.detach().cpu().numpy())
        == 0
    ):
        msg = (
            "The spectrogram must contain some data.\n"
            f"Max value: {np.max(spec.detach().cpu().numpy())}\n"
            f"Min value: {np.min(spec.detach().cpu().numpy())}"
        )
        raise ValueError(msg)

    # Get the dimensions of the original matrix
    original_shape = spec.size()

    # Calculate the number of rows and columns in the matrix
    num_rows, num_cols = original_shape

    # duplicate the matrix 3 times
    spec = spec.repeat(3, 1, 1)

    # Reshape the matrix to the desired shape (3, num_rows, num_cols)
    desired_shape = (3, num_rows, num_cols)
    reshaped_tensor = spec.view(desired_shape)

    # normalize the spectrogram to be between 0 and 1
    normalized_tensor = (reshaped_tensor - reshaped_tensor.min()) / (
        reshaped_tensor.max() - reshaped_tensor.min()
    )

    # make sure image is float32
    return normalized_tensor.float()


def pixel_bbox_to_time_frequency(
    bbox_df: pd.DataFrame,
    spec_times: np.ndarray,
    spec_freqs: np.ndarray,
) -> pd.DataFrame:
    """Convert pixel coordinates to time and frequency.

    Parameters
    ----------
    - `bbox_df` : `pandas.DataFrame`
        The dataframe containing the bounding boxes.
    - `spec_times` : `numpy.ndarray`
        The time axis of the spectrogram.
    - `spec_freqs` : `numpy.ndarray`
        The frequency axis of the spectrogram.

    Returns
    -------
    - `pandas.DataFrame`
        The dataframe with the converted bounding boxes.
    """
    # convert x values to time on spec_times
    spec_times_index = np.arange(0, len(spec_times))
    bbox_df["t1"] = float_index_interpolation(
        bbox_df["x1"].to_numpy(),
        spec_times_index,
        spec_times,
    )
    bbox_df["t2"] = float_index_interpolation(
        bbox_df["x2"].to_numpy(),
        spec_times_index,
        spec_times,
    )
    # convert y values to frequency on spec_freqs
    spec_freqs_index = np.arange(len(spec_freqs))
    bbox_df["f1"] = float_index_interpolation(
        bbox_df["y1"].to_numpy(),
        spec_freqs_index,
        spec_freqs,
    )
    bbox_df["f2"] = float_index_interpolation(
        bbox_df["y2"].to_numpy(),
        spec_freqs_index,
        spec_freqs,
    )
    return bbox_df


def cut_spec_to_frequency_limits(
    spec: torch.Tensor,
    spec_freqs: np.ndarray,
    flims: tuple[float, float],
) -> Tuple[torch.Tensor, np.ndarray]:
    """Cut off everything outside the frequency limits.

    Parameters
    ----------
    - `spec` : `torch.Tensor`
        The spectrogram.
    - `spec_freqs` : `numpy.ndarray`
        The frequency axis of the spectrogram.
    - `flims` : `tuple[float, float]`
        The frequency limits.

    Returns
    -------
    - `torch.Tensor`
        The cut spectrogram.
    - `numpy.ndarray`
        The cut frequency axis.
    """
    # cut off everything outside the frequency limits
    spec = spec[(spec_freqs >= flims[0]) & (spec_freqs <= flims[1]), :]
    spec_freqs = spec_freqs[
        (spec_freqs >= flims[0]) & (spec_freqs <= flims[1])
    ]
    return spec, spec_freqs


def convert_model_output_to_df(
    outputs: torch.Tensor, threshold: float
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Convert the model output to a dataframe.

    Parameters
    ----------
    - `outputs` : `torch.Tensor`
        The output of the model.
    - `threshold` : `float`
        The threshold for the detections.

    Returns
    -------
    - `pandas.DataFrame`
        The dataframe containing the bounding boxes.
    - `numpy.ndarray`
        The scores of the detections.
    """
    # put the boxes, scores and labels into the dataset
    bboxes = outputs[0]["boxes"].detach().cpu().numpy()
    scores = outputs[0]["scores"].detach().cpu().numpy()
    labels = outputs[0]["labels"].detach().cpu().numpy()

    # remove all boxes with a score below the threshold
    bboxes = bboxes[scores > threshold]
    labels = labels[scores > threshold]
    scores = scores[scores > threshold]

    # save the bboxes to a dataframe
    bbox_df = pd.DataFrame(
        data=bboxes,
        columns=["x1", "y1", "x2", "y2"],
    )
    bbox_df["score"] = scores
    bbox_df["label"] = labels
    return bbox_df, scores


def detect_chirps(conf: Config, data: Dataset) -> None:  # noqa
    """Detect chirps on a spectrogram.

    Parameters
    ----------
    - `conf` : `Config`
        The configuration object.
    - `data` : `Dataset`
        The gridtools dataset to detect chirps on.

    Returns
    -------
    - `None`
    """
    # load the model and the checkpoint, and set it to evaluation mode
    logger = logging.getLogger(__name__)
    device = get_device()
    model = load_fasterrcnn(num_classes=len(conf.hyper.classes))
    checkpoint = torch.load(
        f"{conf.hyper.modelpath}/model.pt",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # make spec config
    nfft = freqres_to_nfft(conf.spec.freq_res, data.grid.samplerate)  # samples
    hop_len = overlap_to_hoplen(conf.spec.overlap_frac, nfft)  # samples
    chunksize = int(conf.spec.time_window * data.grid.samplerate)  # samples
    nchunks = np.ceil(data.grid.rec.shape[0] / chunksize).astype(int)
    window_overlap_samples = int(conf.spec.spec_overlap * data.grid.samplerate)

    # make a list to store the bboxes in for each chunk
    bbox_dfs = []

    # get the frequency limits of the spectrogram
    flims = (
        np.min(data.track.freqs) - conf.spec.freq_pad,
        np.max(data.track.freqs) + conf.spec.freq_pad,
    )

    # iterate over the chunks
    overwritten = False
    msg = f"Detecting chirps in {data.path.name}..."
    prog.console.log(msg)
    logger.info(msg)
    for chunk_no in range(nchunks):
        # get start and stop indices for the current chunk
        # including some overlap to compensate for edge effects
        # this diffrers for the first and last chunk
        idx1, idx2 = make_chunk_indices(
            n_chunks=nchunks,
            current_chunk=chunk_no,
            chunksize=chunksize,
            window_overlap_samples=window_overlap_samples,
            max_end=data.grid.rec.shape[0],
        )

        # make a subset of for the current chunk
        chunk = subset(data, idx1, idx2, mode="index")

        # skip if there is no wavetracker tracking data in the current chunk
        if len(chunk.track.indices) == 0:
            continue

        # compute the spectrogram for each electrode of the current chunk
        startt = time.time()
        spec = compute_sum_spectrogam(chunk, nfft, hop_len)
        endt = time.time()
        msg = f"Computing spectrogram took {endt - startt:.2f} seconds."
        prog.console.log(msg)
        logger.debug(msg)

        # compute the time and frequency axes of the spectrogam
        spec_times, spec_freqs = make_spectrogram_axes(
            start=idx1,
            stop=idx2,
            nfft=nfft,
            hop_length=hop_len,
            samplerate=data.grid.samplerate,
        )

        # cut off everything outside the frequency limits
        spec, spec_freqs = cut_spec_to_frequency_limits(
            spec=spec,
            spec_freqs=spec_freqs,
            flims=flims,
        )

        # add the 3 channels, normalize to 0-1, etc
        img = spec_to_image(spec)

        # perform the detection
        startt = time.time()
        with torch.inference_mode():
            outputs = model([img])
        endt = time.time()
        msg = f"Detection took {endt - startt:.2f} seconds."
        prog.console.log(msg)
        logger.debug(msg)

        # make a path to save the spectrogram
        path = data.path / "chirpdetections"
        if path.exists() and overwritten is False:
            shutil.rmtree(path)
            overwritten = True
        path.mkdir(exist_ok=True)
        path /= f"cpd_detected_{chunk_no:05d}.png"

        # put the boxes, scores and labels into the dataset
        bbox_df, scores = convert_model_output_to_df(
            outputs, conf.det.threshold
        )
        if np.any(scores > conf.det.threshold):
            plot_detections(img, outputs[0], conf.det.threshold, path, conf)

        # convert the pixel coordinates to time and frequency
        bbox_df = pixel_bbox_to_time_frequency(bbox_df, spec_times, spec_freqs)

        # save df to list of dfs
        bbox_dfs.append(bbox_df)

    # concatenate all dataframes
    bbox_df = pd.concat(bbox_dfs)
    bbox_reset = bbox_df.reset_index(drop=True)

    # sort the dataframe by t1
    bbox_sorted = bbox_reset.sort_values(by="t1")

    # sort the columns
    bbox_sorted = bbox_sorted[
        ["label", "score", "x1", "y1", "x2", "y2", "t1", "f1", "t2", "f2"]
    ]

    # save the dataframe
    bbox_sorted.to_csv(data.path / "chirpdetector_bboxes.csv", index=False)


def detect_cli(input_path: pathlib.Path) -> None:
    """Terminal interface for the detection function.

    Parameters
    ----------
    - `path` : `str`

    Returns
    -------
    - `None`
    """
    logger = make_logger(__name__, input_path / "chirpdetector.log")
    datasets = [folder for folder in input_path.iterdir() if folder.is_dir()]
    confpath = input_path / "chirpdetector.toml"

    # load the config file and print a warning if it does not exist
    if confpath.exists():
        config = load_config(str(confpath))
    else:
        msg = (
            "The configuration file could not be found in the specified path."
            "Please run `chirpdetector copyconfig` and change the "
            "configuration file to your needs."
        )
        raise FileNotFoundError(msg)

    # detect chirps in all datasets in the specified path
    # and show a progress bar
    prog.console.rule("Starting detection")
    logger.info("Starting detection -----------------------------------------")
    with prog:
        task = prog.add_task("Detecting chirps...", total=len(datasets))
        for dataset in datasets:
            data = load(dataset)
            detect_chirps(config, data)
            prog.update(task, advance=1)
        prog.update(task, completed=len(datasets))
