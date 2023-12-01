"""Detect chirps on a spectrogram."""

import logging
import pathlib
import shutil
import time
from typing import Self, Tuple

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


class Timer:
    """A simple timer class."""

    def __enter__(self: Self) -> Self:
        """Start the timer."""
        self.start_time = time.time()
        return self

    def __exit__(
        self: Self, exc_type: str, exc_value: str, traceback: str
    ) -> None:
        """Stop the timer."""
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time


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
    outputs: torch.Tensor,
    threshold: float,
    spec_times: list,
    spec_freqs: list,
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

    dfs = []
    scores_out = []
    for i in range(len(outputs)):
        times = spec_times[i]
        freqs = spec_freqs[i]
        bboxes = outputs[i]["boxes"].detach().cpu().numpy()
        scores = outputs[i]["scores"].detach().cpu().numpy()
        labels = outputs[i]["labels"].detach().cpu().numpy()

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

        # convert the pixel coordinates to time and frequency
        bbox_df = pixel_bbox_to_time_frequency(bbox_df, times, freqs)
        scores_out.append(scores)
        dfs.append(bbox_df)

    dfs = pd.concat(dfs)
    bbox_df = dfs.reset_index(drop=True)
    scores = np.concatenate(scores_out)
    return bbox_df, scores


def collect_specs(
    conf: Config, data: Dataset, t1: float
) -> Tuple[list, list, list]:
    """Collect the spectrograms of a dataset.

    Collec a batch of  sum spectrograms of a certain length (e.g. 15 seconds)
    for a dataset subset (e.g. of 90 seconds) depending on the power of the
    GPU.

    Parameters
    ----------
    - `conf` : `Config`
        The configuration object.
    - `data` : `Dataset`
        The gridtools dataset to detect chirps on.
    - `t1` : `float`
        The start time of the dataset.

    Returns
    -------
    - `list`
        The spectrograms.
    - `list`
        The time axes of the spectrograms.
    - `list`
        The frequency axes of the spectrograms.
    """
    logger = logging.getLogger(__name__)

    # make spec config
    nfft = freqres_to_nfft(conf.spec.freq_res, data.grid.samplerate)  # samples
    hop_len = overlap_to_hoplen(conf.spec.overlap_frac, nfft)  # samples
    chunksize = int(conf.spec.time_window * data.grid.samplerate)  # samples
    window_overlap_samples = int(conf.spec.spec_overlap * data.grid.samplerate)

    # get the frequency limits of the spectrogram
    flims = (
        float(np.min(data.track.freqs) - conf.spec.freq_pad),
        float(np.max(data.track.freqs) + conf.spec.freq_pad),
    )

    # make the start and stop indices for all chunks
    # including some overlap to compensate for edge effects
    idx1 = np.arange(
        0,
        data.grid.rec.shape[0] - chunksize,
        chunksize - window_overlap_samples,
    )
    idx2 = idx1 + chunksize

    # save data here
    specs, times, freqs = [], [], []

    # iterate over the chunks
    for chunk_no, (start, stop) in enumerate(zip(idx1, idx2)):
        # make a subset of for the current chunk
        chunk = subset(data, start, stop, mode="index")

        # skip if there is no wavetracker tracking data in the current chunk
        if len(chunk.track.indices) == 0:
            continue

        # compute the spectrogram for each electrode of the current chunk
        with Timer() as t:
            spec = compute_sum_spectrogam(chunk, nfft, hop_len)
        msg = (
            f"Computing the sum spectrogram of chunk {chunk_no} took "
            f"{t.execution_time:.2f} seconds."
        )
        prog.console.log(msg)
        logger.debug(msg)

        # compute the time and frequency axes of the spectrogam
        spec_times, spec_freqs = make_spectrogram_axes(
            start=start,
            stop=stop,
            nfft=nfft,
            hop_length=hop_len,
            samplerate=data.grid.samplerate,
        )
        spec_times += t1

        # cut off everything outside the frequency limits
        spec, spec_freqs = cut_spec_to_frequency_limits(
            spec=spec,
            spec_freqs=spec_freqs,
            flims=flims,
        )

        # add the 3 channels, normalize to 0-1, etc
        img = spec_to_image(spec)

        # save the spectrogram
        specs.append(img)
        times.append(spec_times)
        freqs.append(spec_freqs)

    return specs, times, freqs


def handle_dataframes(
    bbox_dfs: list[pd.DataFrame], output_path: pathlib.Path
) -> None:
    """Handle concatenation and saving of dataframes.

    Parameters
    ----------
    - `bbox_dfs` : `list[pandas.DataFrame]`
        The list of dataframes to concatenate.
    - `output_path` : `pathlib.Path`
        The path to save the dataframe to.

    Returns
    -------
    - `None`
    """
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
    bbox_sorted.to_csv(output_path / "chirpdetector_bboxes.csv", index=False)


def detect_chirps(conf: Config, data: Dataset) -> None:
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

    window_duration = (
        conf.spec.time_window * conf.spec.batch_size
    ) - conf.spec.spec_overlap * (conf.spec.batch_size - 1)
    window_duration_samples = int(window_duration * data.grid.samplerate)

    msg = (
        f"Window duration: {window_duration:.2f} seconds.\n"
        f"Window duration in samples: {window_duration_samples} samples."
    )
    prog.console.log(msg)

    # make start and stop indices for all chunks with overlap
    window_overlap_samples = int(conf.spec.spec_overlap * data.grid.samplerate)
    idx1 = np.arange(
        0,
        int(data.grid.rec.shape[0] - window_duration_samples),
        int(window_duration_samples - window_overlap_samples),
    )
    idx2 = idx1 + window_duration_samples

    # make a list to store the bboxes in for each chunk
    bbox_dfs = []

    msg = f"Detecting chirps in {data.path.name}..."
    prog.console.log(msg)
    logger.info(msg)

    # iterate over the chunks
    overwritten = False
    for start, stop in zip(idx1, idx2):
        total_batch_startt = time.time()

        # make a subset of for the current chunk
        with Timer() as t:
            chunk = subset(data, start, stop, mode="index")
        msg = f"Creating the batch subset took {t.execution_time:.2f} seconds."
        prog.console.log(msg)

        # skip if there is no wavetracker tracking data in the current chunk
        if len(chunk.track.indices) == 0:
            continue

        # collect the spectrograms of the current batch
        t1 = start / data.grid.samplerate
        specs, spec_times, spec_freqs = collect_specs(conf, chunk, t1)

        # perform the detection
        with Timer() as t, torch.inference_mode():
            outputs = model(specs)
        msg = f"Detection took {t.execution_time:.2f} seconds."
        prog.console.log(msg)
        logger.debug(msg)

        # make a path to save the spectrogram
        path = data.path / "chirpdetections"
        if path.exists() and overwritten is False:
            shutil.rmtree(path)
            overwritten = True
        path.mkdir(exist_ok=True)

        # put the boxes, scores and labels into the dataset
        bbox_df, scores = convert_model_output_to_df(
            outputs, conf.det.threshold, spec_times, spec_freqs
        )

        num_chirps = len(scores[scores > conf.det.threshold])
        msg = f"Number of chirps detected: {num_chirps}"
        prog.console.log(msg)

        # startt = time.time()
        # if np.any(scores > conf.det.threshold):
        #     for spec_no, (img, out) in enumerate(zip(specs,outputs)):
        #         img_no  = chunk_no * len(specs) + spec_no
        #         img_path = path / f"cpd_detected_{img_no:05d}.png"
        #         plot_detections(
        #             img, out, conf.det.threshold, img_path, conf
        #         )
        # endt = time.time()
        # msg = f"Plotting to disk took {endt - startt:.2f} seconds."
        # prog.console.log(msg)
        # logger.debug(msg)

        # save df to list of dfs
        bbox_dfs.append(bbox_df)
        total_batch_endt = time.time()
        msg = (
            f"Total batch processing time: "
            f"{total_batch_endt - total_batch_startt:.2f} seconds.\n"
        )
        prog.console.log(msg)
        prog.console.rule("Next batch")
        logger.debug(msg)

    handle_dataframes(bbox_dfs, data.path)


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
            startt = time.time()
            data = load(dataset)
            stopt = time.time()
            msg = f"Loading the dataset took {stopt - startt:.2f} seconds.\n"
            prog.console.log(msg)
            detect_chirps(config, data)
            prog.update(task, advance=1)
        prog.update(task, completed=len(datasets))
