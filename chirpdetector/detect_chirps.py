#! /usr/bin/env python3

"""
Detect chirps on a spectrogram.
"""

import argparse
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from IPython import embed
from matplotlib.patches import Rectangle
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)

from gridtools.datasets import Dataset, load, subset
from gridtools.utils.spectrograms import (
    decibel,
    freqres_to_nfft,
    overlap_to_hoplen,
    spectrogram,
)

from .models.helpers import get_device, load_fasterrcnn
from .utils.configfiles import Config, copy_config, load_config

matplotlib.use("Agg")

prog = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
)


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

    # load the model and the checkpoint, and set it to evaluation mode
    device = get_device()
    model = load_fasterrcnn(num_classes=len(conf.hyper.classes))
    checkpoint = torch.load(
        f"{conf.hyper.modelpath}/model.pt", map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # make spec config
    nfft = freqres_to_nfft(conf.det.freq_res, data.grid.samplerate)  # samples
    hop_len = overlap_to_hoplen(conf.det.overlap_frac, nfft)  # samples
    chunksize = conf.det.time_window * data.grid.samplerate  # samples
    nchunks = np.ceil(data.grid.rec.shape[0] / chunksize).astype(int)
    window_overlap_samples = int(conf.det.spec_overlap * data.grid.samplerate)

    # iterate over the chunks
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
        spec = spec[
            (spec_freqs >= conf.det.freq_window[0])
            & (spec_freqs <= conf.det.freq_window[1]),
            :,
        ]
        spec_freqs = spec_freqs[
            (spec_freqs >= conf.det.freq_window[0])
            & (spec_freqs <= conf.det.freq_window[1])
        ]

        # normalize the spectrogram to be between 0 and 1
        path = data.path / "chirpdetections"
        path.mkdir(exist_ok=True)
        path /= f"chunk{chunk_no:05d}.png"

        # add the 3 channels, normalize to 0-1, etc
        img = spec_to_image(spec)

        # perform the detection
        with torch.inference_mode():
            outputs = model([img])

        plot_detections(img, outputs[0], conf.det.threshold, path, conf)


def chirpdetector_cli():
    parser = argparse.ArgumentParser(
        description="Detect chirps on a spectrogram."
    )
    parser.add_argument(
        "--path",
        "-p",
        type=pathlib.Path,
        help="Path to the datasets.",
        required=True,
    )
    return parser.parse_args()


def detect(args):
    datasets = [dir for dir in args.path.iterdir() if dir.is_dir()]
    confpath = args.path / "chirpdetector.toml"

    if confpath.exists():
        config = load_config(str(confpath))
    else:
        raise FileNotFoundError(
            "The configuration file could not be found in the specified path."
            "Please run `chirpdetector copyconfig` and change the "
            "configuration file to your needs."
        )

    with prog:
        task = prog.add_task("[green]Detecting chirps...", total=len(datasets))
        for dataset in datasets:
            prog.console.log(f"Processing {dataset.name}...")
            data = load(dataset, grid=True)
            detect_chirps(config, data)
            prog.advance(task, 1)
        prog.update(task, completed=len(datasets))


def main():
    args = chirpdetector_cli()
    detect(args)
