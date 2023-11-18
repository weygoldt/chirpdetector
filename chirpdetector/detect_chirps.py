#! /usr/bin/env python3

"""
Detect chirps on a spectrogram.
"""

import pathlib
import shutil
from IPython import embed

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from gridtools.datasets import Dataset, load, subset
from gridtools.utils.spectrograms import (
    decibel,
    freqres_to_nfft,
    overlap_to_hoplen,
    spectrogram,
)
from matplotlib.patches import Rectangle
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)

from .models.utils import get_device, load_fasterrcnn
from .utils.configfiles import Config, copy_config, load_config
from .utils.logging import make_logger

matplotlib.use("Agg")

prog = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
)


def float_index_interpolation(
    values: np.ndarray, index_arr: np.ndarray, data_arr: np.ndarray
) -> float:
    """
    Interpolate a value in the data dimension that is not necessarily on the data array.

    Parameters
    ----------
    - `value` : `float`
        The value to be interpolated.
    - `index_arr` : `numpy.ndarray`
        The array of indices.
    - `data_arr` : `numpy.ndarray`
        The array of data.

    Returns
    -------
    `numpy.ndarray`
        The interpolated value.
    """
    newvalues = np.zeros_like(values)
    for i, value in enumerate(values):
        nextlower = np.floor(value).astype(int)
        rest = value - nextlower

        if nextlower > len(data_arr) - 2:
            newvalues[i] = data_arr[-1]
            continue

        nextlower_data = data_arr[index_arr == nextlower][0]
        nexthigher_data = data_arr[index_arr == nextlower + 1][0]

        newvalue = nextlower_data + rest * (nexthigher_data - nextlower_data)
        newvalues[i] = newvalue

    return newvalues


def flip_boxes(boxes, img_height):
    """
    Flip the boxes vertically.
    """

    # correct the y coordinates
    boxes[:, 1], boxes[:, 3] = (
        img_height - boxes[:, 3],
        img_height - boxes[:, 1],
    )

    return boxes


def corner_coords_to_center_coords(boxes):
    """
    Convert box defined by corner coordinates to box defined by lower left, width
    and height.
    """
    new_boxes = np.zeros_like(boxes)
    new_boxes[:, 0] = boxes[:, 0]
    new_boxes[:, 1] = boxes[:, 1]
    new_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    new_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

    return new_boxes


def plot_detections(img_tensor, output, threshold, save_path, conf):
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)[..., 0]
    boxes = output["boxes"].detach().cpu().numpy()

    boxes = corner_coords_to_center_coords(boxes)
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
                )
            )
            ax.text(
                box[0],
                box[1],
                f"{scores[i]:.2f}",
                color="black",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=1),
            )
    plt.axis("off")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def spec_to_image(spec):
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
    scaled_tensor = normalized_tensor.float()

    return scaled_tensor


def detect_chirps(conf: Config, data: Dataset):
    n_electrodes = data.grid.rec.shape[1]

    # TODO: fix this ugly workaround mainly because detections
    # in time will be wrong!
    data.track.times -= data.track.times[0]

    # load the model and the checkpoint, and set it to evaluation mode
    device = get_device()
    model = load_fasterrcnn(num_classes=len(conf.hyper.classes))
    checkpoint = torch.load(
        f"{conf.hyper.modelpath}/model.pt", map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # make spec config
    nfft = freqres_to_nfft(conf.spec.freq_res, data.grid.samplerate)  # samples
    hop_len = overlap_to_hoplen(conf.spec.overlap_frac, nfft)  # samples
    chunksize = conf.spec.time_window * data.grid.samplerate  # samples
    nchunks = np.ceil(data.grid.rec.shape[0] / chunksize).astype(int)
    window_overlap_samples = int(conf.spec.spec_overlap * data.grid.samplerate)

    bbox_dfs = []

    # iterate over the chunks
    overwritten = False
    for chunk_no in range(nchunks):
        # get start and stop indices for the current chunk
        # including some overlap to compensate for edge effects
        # this diffrers for the first and last chunk

        if chunk_no == 0:
            idx1 = int(chunk_no * chunksize)
            idx2 = int((chunk_no + 1) * chunksize + window_overlap_samples)
        elif chunk_no == nchunks - 1:
            idx1 = int(chunk_no * chunksize - window_overlap_samples)
            idx2 = int((chunk_no + 1) * chunksize)
        else:
            idx1 = int(chunk_no * chunksize - window_overlap_samples)
            idx2 = int((chunk_no + 1) * chunksize + window_overlap_samples)

        # idx1 and idx2 now determine the window I cut out of the raw signal
        # to compute the spectrogram of.

        # compute the time and frequency axes of the spectrogram now that we
        # include the start and stop indices of the current chunk and thus the
        # right start and stop time. The `spectrogram` function does not know
        # about this and would start every time axis at 0.
        spec_times = np.arange(idx1, idx2 + 1, hop_len) / data.grid.samplerate
        spec_freqs = np.arange(0, nfft / 2 + 1) * data.grid.samplerate / nfft

        # create a subset from the grid dataset
        if idx2 > data.grid.rec.shape[0]:
            idx2 = data.grid.rec.shape[0] - 1
        chunk = subset(data, idx1, idx2, mode="index")

        # compute the spectrogram for each electrode of the current chunk
        spec = torch.zeros((len(spec_freqs), len(spec_times)))
        for el in range(n_electrodes):
            # get the signal for the current electrode
            sig = chunk.grid.rec[:, el]

            # compute the spectrogram for the current electrode
            chunk_spec, _, _ = spectrogram(
                data=sig.copy(),
                samplingrate=data.grid.rec.samplerate,
                nfft=nfft,
                hop_length=hop_len,
            )

            # sum spectrogram over all electrodes
            # the spec is a tensor
            if el == 0:
                spec = chunk_spec
            else:
                spec += chunk_spec

        # normalize spectrogram by the number of electrodes
        # the spec is still a tensor
        spec /= n_electrodes

        # convert the spectrogram to dB
        # .. still a tensor
        spec = decibel(spec)

        # cut off everything outside the upper frequency limit
        # the spec is still a tensor
        # TODO: THIS IS SKETCHY AS HELL! As a result, only time and frequency
        # bounding boxes can be used later! The spectrogram limits change for every
        # window!
        flims = (
            np.min(chunk.track.freqs) - conf.spec.freq_pad,
            np.max(chunk.track.freqs) + conf.spec.freq_pad,
        )
        spec = spec[(spec_freqs >= flims[0]) & (spec_freqs <= flims[1]), :]
        spec_freqs = spec_freqs[
            (spec_freqs >= flims[0]) & (spec_freqs <= flims[1])
        ]

        # make a path to save the spectrogram
        path = data.path / "chirpdetections"
        if path.exists() and overwritten is False:
            shutil.rmtree(path)
            overwritten = True
        path.mkdir(exist_ok=True)
        path /= f"chunk{chunk_no:05d}.png"

        # add the 3 channels, normalize to 0-1, etc
        img = spec_to_image(spec)

        # perform the detection
        with torch.inference_mode():
            outputs = model([img])

        plot_detections(img, outputs[0], conf.det.threshold, path, conf)

        # put the boxes, scores and labels into the dataset
        bboxes = outputs[0]["boxes"].detach().cpu().numpy()
        scores = outputs[0]["scores"].detach().cpu().numpy()
        labels = outputs[0]["labels"].detach().cpu().numpy()

        # remove all boxes with a score below the threshold
        bboxes = bboxes[scores > conf.det.threshold]
        labels = labels[scores > conf.det.threshold]
        scores = scores[scores > conf.det.threshold]

        # from scipy.signal import find_peaks
        # for bbox in bboxes:
        #     print(bbox)
        #     matplotlib.use("TkAgg")
        #     # cut out spec region of bbox
        #     bbox_spec = spec[
        #         int(bbox[1]) : int(bbox[3]),
        #         int(bbox[0]) : int(bbox[2]),
        #     ]
        #     bbox_spec_times = spec_times[int(bbox[0]) : int(bbox[2])]
        #     bbox_spec_freqs = spec_freqs[int(bbox[1]) : int(bbox[3])]
        #
        #     avg_spec = np.mean(bbox_spec.cpu().numpy(), axis=1)
        #     peaks, _ = find_peaks(avg_spec, prominence=5)
        #
        #     _, ax = plt.subplots(1,2, sharey=True, figsize=(20, 10))
        #     ax[0].imshow(bbox_spec.detach().cpu().numpy(), cmap="magma", origin="lower", extent = [bbox_spec_times[0], bbox_spec_times[-1], bbox_spec_freqs[0], bbox_spec_freqs[-1]], aspect = 'auto')
        #     ax[1].plot(avg_spec, bbox_spec_freqs)
        #     ax[1].scatter(avg_spec[peaks], bbox_spec_freqs[peaks], color="red")
        #
        #     for fish_id in chunk.track.ids:
        #         track = chunk.track.freqs[chunk.track.idents == fish_id]
        #         times = chunk.track.times[chunk.track.indices[chunk.track.idents == fish_id]]
        #         track = track[(times >= bbox_spec_times[0]) & (times <= bbox_spec_times[-1])]
        #         times = times[(times >= bbox_spec_times[0]) & (times <= bbox_spec_times[-1])]
        #         ax[0].plot(
        #             times,
        #             track,
        #             color="black",
        #             linewidth=1,
        #         )
        #
        #     plt.show()
        # exit()
        #
        bbox_df = pd.DataFrame(
            data=bboxes,
            columns=["x1", "y1", "x2", "y2"],
        )
        bbox_df["score"] = scores
        bbox_df["label"] = labels

        # convert x values to time on spec_times
        spec_times_index = np.arange(0, len(spec_times))
        bbox_df["t1"] = float_index_interpolation(
            bbox_df["x1"].values, spec_times_index, spec_times
        )
        bbox_df["t2"] = float_index_interpolation(
            bbox_df["x2"].values, spec_times_index, spec_times
        )

        # convert y values to frequency on spec_freqs
        spec_freqs_index = np.arange(0, len(spec_freqs))
        bbox_df["f1"] = float_index_interpolation(
            bbox_df["y1"].values, spec_freqs_index, spec_freqs
        )
        bbox_df["f2"] = float_index_interpolation(
            bbox_df["y2"].values, spec_freqs_index, spec_freqs
        )

        # save df to list
        bbox_dfs.append(bbox_df)

    # concatenate all dataframes
    bbox_df = pd.concat(bbox_dfs)
    bbox_df.reset_index(inplace=True, drop=True)

    # sort the dataframe by t1
    bbox_df.sort_values(by="t1", inplace=True)

    # sort the columns
    bbox_df = bbox_df[
        ["label", "score", "x1", "y1", "x2", "y2", "t1", "f1", "t2", "f2"]
    ]

    # save the dataframe
    bbox_df.to_csv(data.path / "chirpdetector_bboxes.csv", index=False)


def detect_cli(path):
    global logger  # pylint: disable=global-statement
    path = pathlib.Path(path)
    logger = make_logger(__name__, path / "chirpdetector.log")
    datasets = [dir for dir in path.iterdir() if dir.is_dir()]
    confpath = path / "chirpdetector.toml"

    if confpath.exists():
        config = load_config(str(confpath))
    else:
        raise FileNotFoundError(
            "The configuration file could not be found in the specified path."
            "Please run `chirpdetector copyconfig` and change the "
            "configuration file to your needs."
        )

    with prog:
        task = prog.add_task("Detecting chirps...", total=len(datasets))
        for dataset in datasets:
            msg = f"Detecting chirps in {dataset.name}..."
            prog.console.log(msg)
            logger.info(msg)

            data = load(dataset, grid=True)
            detect_chirps(config, data)
            prog.update(task, advance=1)
        prog.update(task, completed=len(datasets))
