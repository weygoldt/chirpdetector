#!/usr/bin/env python3

"""
This module contains functions and classes for converting data from one format
to another.
"""

import argparse
import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from rich.console import Console
from rich.progress import track

from gridtools.datasets import Dataset, load, subset
from gridtools.utils.spectrograms import (
    decibel,
    freqres_to_nfft,
    overlap_to_hoplen,
    sint,
    spectrogram,
)

con = Console()

freq_resolution = 6
overlap_fraction = 0.9
spectrogram_freq_limits = (100, 2200)


def make_file_tree(path: pathlib.Path) -> None:
    """
    Builds a file tree for the training dataset.

    Parameters
    ----------
    path : pathlib.Path
        The root directory of the dataset.
    """

    if path.exists():
        shutil.rmtree(path)

    path.mkdir(exist_ok=True, parents=True)

    train_imgs = path / "train" / "images"
    train_labels = path / "train" / "labels"
    train_imgs.mkdir(exist_ok=True, parents=True)
    train_labels.mkdir(exist_ok=True, parents=True)

    val_imgs = path / "val" / "images"
    val_labels = path / "val" / "labels"
    val_imgs.mkdir(exist_ok=True, parents=True)
    val_labels.mkdir(exist_ok=True, parents=True)


def numpy_to_pil(img: np.ndarray) -> Image:
    """Convert a numpy array to a PIL image.

    Parameters
    ----------
    img : np.ndarray
        The input image.

    Returns
    -------
    PIL.Image
        The converted image.
    """
    img = np.flipud(img)
    img = np.uint8((img - img.min()) / (img.max() - img.min()) * 255)
    img = Image.fromarray(img)
    return img


def chirp_bounding_boxes(data: Dataset, nfft: int) -> pd.DataFrame:
    assert hasattr(
        data.com.chirp, "params"
    ), "Dataset must have a chirp attribute with a params attribute"

    # Time padding is one NFFT window
    pad_time = nfft / data.grid.samplerate

    # Freq padding is fixed by the frequency resolution
    freq_res = data.grid.samplerate / nfft
    pad_freq = freq_res * 50

    boxes = []
    ids = []
    for fish_id in data.track.ids:
        freqs = data.track.freqs[data.track.idents == fish_id]
        times = data.track.times[
            data.track.indices[data.track.idents == fish_id]
        ]
        chirps = data.com.chirp.times[data.com.chirp.idents == fish_id]
        params = data.com.chirp.params[data.com.chirp.idents == fish_id]

        for chirp, param in zip(chirps, params):
            # take the two closest frequency points
            f_closest = freqs[np.argsort(np.abs(times - chirp))[:2]]

            # take the two closest time points
            t_closest = times[np.argsort(np.abs(times - chirp))[:2]]

            # compute the weighted average of the two closest frequency points
            # using the dt between chirp time and sampled time as weights
            f_closest = np.average(f_closest, weights=np.abs(t_closest - chirp))

            # we now have baseline eodf and time point of the chirp. Now
            # we get some parameters from the params to build the bounding box
            # for the chirp
            height = param[1]
            width = param[2]

            # now define bounding box as center coordinates, width and height
            t_center = chirp
            f_center = f_closest + height / 2

            bbox_height = height + pad_freq
            bbox_width = width + pad_time

            boxes.append((t_center, f_center, bbox_width, bbox_height))
            ids.append(fish_id)

    df = pd.DataFrame(
        boxes, columns=["t_center", "f_center", "width", "height"]
    )
    df["fish_id"] = ids
    return df


def make_spectrograms(
    data: Dataset, dataroot: pathlib.Path, plot_boxes: bool = False
) -> None:
    """
    Notes
    -----
    This function iterates through a raw recording in chunks and computes the
    sum spectrogram of each chunk. The chunk size needs to be chosen such that
    the images can be nicely fed to a detector. The function also computes
    the bounding boxes of chirps in that chunk and saves them to a dataframe
    and a txt file into a labels directory.
    """
    assert hasattr(data, "grid"), "Dataset must have a grid attribute"

    n_electrodes = data.grid.rec.shape[1]

    # How much time to put into each spectrogram
    time_window = 20  # seconds
    window_overlap = 1  # seconds
    window_overlap_samples = window_overlap * data.grid.samplerate  # samples

    # Spectrogram computation parameters
    nfft = freqres_to_nfft(freq_resolution, data.grid.samplerate)  # samples
    hop_len = overlap_to_hoplen(overlap_fraction, nfft)  # samples
    chunksize = time_window * data.grid.samplerate  # samples
    n_chunks = np.ceil(data.grid.rec.shape[0] / chunksize).astype(int)

    bbox_dfs = []

    for chunk_no in range(n_chunks):
        # get start and stop indices for the current chunk
        # including some overlap to compensate for edge effects
        # this diffrers for the first and last chunk

        if chunk_no == 0:
            idx1 = sint(chunk_no * chunksize)
            idx2 = sint((chunk_no + 1) * chunksize + window_overlap_samples)
        elif chunk_no == n_chunks - 1:
            idx1 = sint(chunk_no * chunksize - window_overlap_samples)
            idx2 = sint((chunk_no + 1) * chunksize)
        else:
            idx1 = sint(chunk_no * chunksize - window_overlap_samples)
            idx2 = sint((chunk_no + 1) * chunksize + window_overlap_samples)

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

        # skip if no chirps in this chunk
        if len(chunk.com.chirp.times) == 0:
            continue

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
            (spec_freqs >= spectrogram_freq_limits[0])
            & (spec_freqs <= spectrogram_freq_limits[1]),
            :,
        ]
        spec_freqs = spec_freqs[
            (spec_freqs >= spectrogram_freq_limits[0])
            & (spec_freqs <= spectrogram_freq_limits[1])
        ]

        # normalize the spectrogram to zero mean and unit variance
        # the spec is still a tensor
        spec = (spec - spec.mean()) / spec.std()

        # compute the bounding boxes for this chunk
        bboxes = chirp_bounding_boxes(chunk, nfft)

        # convert bounding box center coordinates to spectrogram coordinates
        # find the indices on the spec_times corresponding to the center times
        x = np.searchsorted(spec_times, bboxes.t_center)
        y = np.searchsorted(spec_freqs, bboxes.f_center)
        widths = np.searchsorted(spec_times - spec_times[0], bboxes.width)
        heights = np.searchsorted(spec_freqs, bboxes.height)

        # now we have center coordinates, widths and heights in indices. But PIL
        # expects coordinates in pixels in the format
        # (Upper left x coordinate, upper left y coordinate,
        # lower right x coordinate, lower right y coordinate)
        # In addiotion, an image starts in the top left corner so the bboxes
        # need to be mirrored horizontally.

        y = spec.shape[0] - y  # flip the y values to fit y=0 at the top
        lxs, lys = x - widths / 2, y - heights / 2
        rxs, rys = x + widths / 2, y + heights / 2

        # add them to the bboxes dataframe
        bboxes["upperleft_img_x"] = lxs
        bboxes["upperleft_img_y"] = lys
        bboxes["lowerright_img_x"] = rxs
        bboxes["lowerright_img_y"] = rys

        # most deep learning frameworks expect bounding box coordinates
        # as relative to the image size. So we normalize the coordinates
        # to the image size

        rel_lxs = lxs / spec.shape[1]
        rel_rxs = rxs / spec.shape[1]
        rel_lys = lys / spec.shape[0]
        rel_rys = rys / spec.shape[0]

        # add them to the bboxes dataframe
        bboxes["upperleft_img_x_norm"] = rel_lxs
        bboxes["upperleft_img_y_norm"] = rel_lys
        bboxes["lowerright_img_x_norm"] = rel_rxs
        bboxes["lowerright_img_y_norm"] = rel_rys

        # add chunk ID to the bboxes dataframe
        bboxes["chunk_id"] = chunk_no

        # put them into a dataframe to save for eahc spectrogram
        df = pd.DataFrame(
            {"lx": rel_lxs, "ly": rel_lys, "rx": rel_rxs, "ry": rel_rys}
        )

        # add as first colum instance id
        df.insert(0, "instance_id", np.ones_like(lxs, dtype=int))

        # convert the spectrogram to a PIL image
        spec = spec.detach().cpu().numpy()
        img = numpy_to_pil(spec)

        # draw the bounding boxes on the image
        if plot_boxes is True:
            for lx, ly, rx, ry in zip(lxs, lys, rxs, rys):
                img = img.convert("RGB")
                draw = ImageDraw.Draw(img)
                draw.rectangle((lx, ly, rx, ry), outline="red", width=1)

        # save image
        imgname = f"{data.path.name}_{chunk_no:06}.png"
        img.save(dataroot / "train" / "images" / f"{imgname}")

        # stash the bboxes dataframe for this chunk
        bboxes["image"] = imgname
        bbox_dfs.append(bboxes)

        # save dataframe for every spec without headers as txt
        df.to_csv(
            dataroot
            / "train"
            / "labels"
            / f"{data.path.name}_{chunk_no:06}.txt",
            header=False,
            index=False,
            sep=" ",
        )

    # concat all the bboxes dataframes
    bbox_df = pd.concat(bbox_dfs, ignore_index=True)
    # save the bboxes dataframe
    bbox_df.to_csv(
        dataroot / "train" / f"{data.path.name}_bboxes.csv", index=False
    )


def parse_datasets(
    input: pathlib.Path, output: pathlib.Path, make_boxes
) -> None:
    """
    Parse all datasets in a directory.

    Parameters
    ----------
    dataroot : pathlib.Path
        The root directory of the datasets.

    Returns
    -------
    None
    """

    make_file_tree(output)
    for path in track(list(input.iterdir()), description="Building datasets"):
        data = load(path, grid=True)

        if make_boxes is True:
            make_spectrograms(data, output, plot_boxes=False)
        else:
            print("Not implemented yet.")
