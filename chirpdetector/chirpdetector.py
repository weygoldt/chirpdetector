#! /usr/bin/env python3

"""
Detect chirps on a spectrogram.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from matplotlib.patches import Rectangle
from PIL import Image

from gridtools.datasets import Dataset, load, subset
from gridtools.utils.spectrograms import (
    decibel,
    freqres_to_nfft,
    overlap_to_hoplen,
    spectrogram,
)

from .models.helpers import get_device, load_fasterrcnn
from .utils.configfiles import Config, load_config


def plot_detections(img_tensor, output, threshold, save_path):
    fig, ax = plt.subplots(constrained_layout=True)
    ax.imshow(
        img_tensor.cpu().squeeze().permute(1, 2, 0),
    )
    for (x0, y0, x1, y1), l, score in zip(
        output["boxes"].cpu(), output["labels"].cpu(), output["scores"].cpu()
    ):
        ax.text(
            x0,
            y0,
            f"{score:.2f}",
            ha="left",
            va="bottom",
            fontsize=10,
            color="white",
        )
        ax.add_patch(
            Rectangle(
                (x0, y0),
                (x1 - x0),
                (y1 - y0),
                fill=False,
                color="tab:red",
                linewidth=1,
                zorder=10,
            )
        )

    ax.set_axis_off()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
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

    # make sure image is float32
    scaled_tensor = reshaped_tensor.float()

    return scaled_tensor


def chirpdetector(conf: Config, data: Dataset):
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
        # spec = (spec - spec.min()) / (spec.max() - spec.min())
        path = data.path / "chirpdetections"
        path.mkdir(exist_ok=True)
        path /= f"chunk{chunk_no:05d}.png"

        # convert the spec tensor to the same format as a PIL image would be
        # put to the CPU and convert to numpy array
        spec = spec.detach().cpu().numpy()

        # flip the spectrogram upside down
        img = np.flipud(spec)

        # scale the spectrogram to be between 0 and 255
        img = np.uint8((img - img.min()) / (img.max() - img.min()) * 255)

        # convert to PIL image
        img = Image.fromarray(img)

        # convert to RGB
        img = img.convert("RGB")
        img.save(path)

        # make tensor
        img = F.to_tensor(img)

        # scale between 0 and 1
        img = img / 255

        # add batch dimension
        img = [img.to(device)]

        # perform the detection
        with torch.no_grad():
            outputs = model(img)

        print(outputs)
        # plot_detections(img, outputs[0], conf.det.threshold, path)


def chirpdetector_cli():
    parser = argparse.ArgumentParser(
        description="Detect chirps on a spectrogram."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=pathlib.Path,
        help="Path to the configuration file.",
        required=True,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=pathlib.Path,
        help="Path to the datasets.",
        required=True,
    )
    args = parser.parse_args()

    datasets = [dir for dir in args.input.iterdir() if dir.is_dir()]
    config = load_config(str(args.config))
    for dataset in datasets:
        data = load(dataset, grid=True)
        chirpdetector(config, data)
