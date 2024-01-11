"""Functions and classes for converting data."""

import pathlib
import shutil
from typing import Union

import numpy as np
import pandas as pd
from gridtools.datasets import load, subset
from gridtools.datasets.models import Dataset
from gridtools.utils.spectrograms import (
    freqres_to_nfft,
    overlap_to_hoplen,
)
from PIL import Image
from rich.console import Console
from rich.progress import track

from chirpdetector.config import Config, load_config
from chirpdetector.datahandling.signal_processing import (
    compute_sum_spectrogam,
    make_chunk_indices,
    make_spectrogram_axes,
    zscore_standardize,
)

con = Console()


def make_file_tree(path: pathlib.Path, wipe: bool = True) -> None:
    """Build a file tree for the training dataset.

    Parameters
    ----------
    path : pathlib.Path
        The root directory of the dataset.
    """
    if not isinstance(path, pathlib.Path):
        msg = f"Path must be a pathlib.Path, not {type(path)}"
        raise TypeError(msg)

    if path.parent.exists() and path.parent.is_file():
        msg = (
            f"Parent directory of {path} is a file. "
            "Please specify a directory."
        )
        raise ValueError(msg)

    if path.exists() and wipe:
        shutil.rmtree(path)

    path.mkdir(exist_ok=True, parents=True)

    train_imgs = path / "images"
    train_labels = path / "labels"
    train_imgs.mkdir(exist_ok=True, parents=True)
    train_labels.mkdir(exist_ok=True, parents=True)


def numpy_to_pil(img: np.ndarray) -> Image.Image:
    """Convert a 2D numpy array to a PIL image.

    Parameters
    ----------
    img : np.ndarray
        The input image.

    Returns
    -------
    PIL.Image
        The converted image.
    """
    img_dimens = 2
    if len(img.shape) != img_dimens:
        msg = f"Image must be {img_dimens}D"
        raise ValueError(msg)

    if img.max() == img.min():
        msg = "Image must have more than one value"
        raise ValueError(msg)

    img = np.flipud(img)
    intimg = np.uint8((img - img.min()) / (img.max() - img.min()) * 255)
    return Image.fromarray(intimg)


def bboxes_from_simulated_chirps(data: Dataset, nfft: int) -> pd.DataFrame:
    """Make bounding boxes of simulated chirps using the chirp parameters.

    Parameters
    ----------
    - `data` : `Dataset`
        The dataset to make bounding boxes for.
    - `nfft` : int
        The number of samples in the FFT.

    Returns
    -------
    `pandas.DataFrame`
        A dataframe with the bounding boxes.
    """
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
            f_closest = np.average(
                f_closest,
                weights=np.abs(t_closest - chirp),
            )

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

    dataframe = pd.DataFrame(
        boxes,
        columns=["t_center", "f_center", "width", "height"],
    )
    dataframe["fish_id"] = ids
    return dataframe


def convert_bboxes_from_simulated_chirps(
    imgpath: pathlib.Path,
    spectrogram_times: np.ndarray,
    spectrogram_frequencies: np.ndarray,
    current_chunk: int,
    bboxes: pd.DataFrame,
) -> Union[pd.DataFrame, None]:
    """Generate labels of a simulated dataset.

    Parameters
    ----------
    - `imgpath` : `pathlib.Path`
        The path to the image.
    - `spectrogram_times` : `np.ndarray`
        The time axis of the spectrogram.
    - `spectrogram_frequencies` : `np.ndarray`
        The frequency axis of the spectrogram.
    - `current_chunk` : `int`
        The chunk number.
    - `bboxes` : `pd.DataFrame`
        The bounding boxes.

    Returns
    -------
    - `pandas.DataFrame`
        A dataframe with the bounding boxes.
    """
    # compute the bounding boxes for this chunk

    if len(bboxes) == 0:
        return None

    # convert bounding box center coordinates to spectrogram coordinates
    # find the indices on the spectrogram_times corresponding to the center
    # times
    x = np.searchsorted(spectrogram_times, bboxes.t_center)
    y = np.searchsorted(spectrogram_frequencies, bboxes.f_center)
    widths = np.searchsorted(
        spectrogram_times - spectrogram_times[0], bboxes.width
    )
    heights = np.searchsorted(
        spectrogram_frequencies - spectrogram_frequencies[0], bboxes.height
    )

    # now we have center coordinates, widths and heights in indices. But PIL
    # expects coordinates in pixels in the format
    # (Upper left x coordinate, upper left y coordinate,
    # lower right x coordinate, lower right y coordinate)
    # In addiotion, an image starts in the top left corner so the bboxes
    # need to be mirrored horizontally.

    y = len(spectrogram_frequencies) - y  # flip y values to fit y=0 at top
    lxs, lys = x - widths / 2, y - heights / 2
    rxs, rys = x + widths / 2, y + heights / 2

    # add them to the bboxes dataframe
    bboxes["upperleft_img_x"] = lxs
    bboxes["upperleft_img_y"] = lys
    bboxes["lowerright_img_x"] = rxs
    bboxes["lowerright_img_y"] = rys

    # yolo format is centerx, centery, width, height normalized to image size
    # convert xmin, ymin, xmax, ymax to centerx, centery, width, height
    centerx = (lxs + rxs) / 2
    centery = (lys + rys) / 2
    width = rxs - lxs
    height = rys - lys

    # most deep learning frameworks expect bounding box coordinates
    # as relative to the image size. So we normalize the coordinates
    # to the image size
    centerx_norm = centerx / len(spectrogram_times)
    centery_norm = centery / len(spectrogram_frequencies)
    width_norm = width / len(spectrogram_times)
    height_norm = height / len(spectrogram_frequencies)

    # add them to the bboxes dataframe
    # theses are the ones that will later make the label files
    bboxes["centerx_norm"] = centerx_norm
    bboxes["centery_norm"] = centery_norm
    bboxes["width_norm"] = width_norm
    bboxes["height_norm"] = height_norm

    # add chunk ID to the bboxes dataframe
    bboxes["chunk_id"] = current_chunk

    # add the image name to the bboxes dataframe
    bboxes["image"] = imgpath.name


def save_labels_for_simulated_chirps(
    bbox_df: pd.DataFrame, dataset_root: pathlib.Path
) -> None:
    """Save the labels for a simulated dataset.

    Parameters
    ----------
    - `bbox_df` : `pd.DataFrame`
        The bounding boxes.
    - `dataset_root` : `pathlib.Path`
        The root directory of the dataset.

    Returns
    -------
    - `None`
    """
    for img in bbox_df["image"].unique():
        x = bbox_df["centerx_norm"].loc[bbox_df["image"] == img]
        y = bbox_df["centery_norm"].loc[bbox_df["image"] == img]
        w = bbox_df["width_norm"].loc[bbox_df["image"] == img]
        h = bbox_df["height_norm"].loc[bbox_df["image"] == img]

        # make a dataframe with the labels
        label_df = pd.DataFrame({"cx": x, "cy": y, "w": w, "h": h})
        label_df.insert(0, "instance_id", np.ones_like(x, dtype=int))

        # save dataframe for every spec without headers as txt
        label_df.to_csv(
            dataset_root / "labels" / f"{img.stem}.txt",
            header=False,
            index=False,
            sep=" ",
        )


def detected_labels(
    output: pathlib.Path,
    chunk: Dataset,
    imgname: str,
    spec: np.ndarray,
    spectrogram_times: np.ndarray,
) -> None:
    """Use the detect_chirps to make a YOLO dataset.

    Parameters
    ----------
    - `output` : `pathlib.Path`
        The output directory.
    - `chunk` : `Dataset`
        The dataset to make bounding boxes for.
    - `imgname` : `str`
        The name of the image.
    - `spec` : `np.ndarray`
        The spectrogram.
    - `spectrogram_times` : `np.ndarray`
        The time axis of the spectrogram.

    Returns
    -------
    - `None`
    """
    # load the detected bboxes csv
    # TODO: This is a workaround. Instead improve the subset naming convention
    # in gridtools
    source_dataset = chunk.path.name.split("_")[1:-4]
    source_dataset = "_".join(source_dataset)
    source_dataset = chunk.path.parent / source_dataset

    dataframe = pd.read_csv(source_dataset / "chirpdetector_bboxes.csv")

    # get chunk start and stop time
    start, stop = spectrogram_times[0], spectrogram_times[-1]

    # get the bboxes for this chunk
    bboxes = dataframe[(dataframe.t1 >= start) & (dataframe.t2 <= stop)]

    # get the x and y coordinates of the bboxes in pixels as dataframe
    bboxes_xy = bboxes[["x1", "y1", "x2", "y2"]]

    # convert from x1, y1, x2, y2 to centerx, centery, width, height
    centerx = np.array((bboxes_xy["x1"] + bboxes_xy["x2"]) / 2)
    centery = np.array((bboxes_xy["y1"] + bboxes_xy["y2"]) / 2)
    width = np.array(bboxes_xy["x2"] - bboxes_xy["x1"])
    height = np.array(bboxes_xy["y2"] - bboxes_xy["y1"])

    # flip centery because origin is top left
    centery = spec.shape[0] - centery

    # make relative to image size
    centerx = centerx / spec.shape[1]
    centery = centery / spec.shape[0]
    width = width / spec.shape[1]
    height = height / spec.shape[0]
    labels = np.ones_like(centerx, dtype=int)

    # make a new dataframe with the relative coordinates
    new_bboxes = pd.DataFrame(
        {"l": labels, "x": centerx, "y": centery, "w": width, "h": height},
    )

    # save dataframe for every spec without headers as txt
    new_bboxes.to_csv(
        output / "labels" / f"{imgname[:-4]}.txt",
        header=False,
        index=False,
        sep=" ",
    )


def convert(
    data: Dataset,
    conf: Config,
    output: pathlib.Path,
    label_mode: str,
) -> None:
    """Convert a gridtools dataset to a YOLO dataset.

    Parameters
    ----------
    - `data` : `Dataset`
        The dataset to convert.
    - `conf` : `Config`
        The configuration.
    - `output` : `pathlib.Path`
        The output directory.
    - `label_mode` : `str`
        The label mode. Can be one of 'none', 'synthetic' or 'detected'.

    Returns
    -------
    - `None`

    Notes
    -----
    This function iterates through a raw recording in chunks and computes the
    sum spectrogram of each chunk. The chunk size needs to be chosen such that
    the images can be nicely fed to a detector. The function also computes
    the bounding boxes of chirps in that chunk and saves them to a dataframe
    and a txt file into a labels directory.
    """
    assert hasattr(data, "grid"), "Dataset must have a grid attribute"
    assert label_mode in [
        "none",
        "synthetic",
        "detected",
    ], "label_mode must be one of 'none', 'synthetic' or 'detected'"

    dataroot = output

    # How much time to put into each spectrogram
    time_window = conf.spec.time_window  # seconds
    window_overlap = conf.spec.spec_overlap  # seconds
    freq_pad = conf.spec.freq_pad  # Hz
    window_overlap_samples = int(window_overlap * data.grid.samplerate)
    spectrogram_freq_limits = (
        np.min(data.track.freqs) - freq_pad,
        np.max(data.track.freqs) + freq_pad,
    )

    # Spectrogram computation parameters
    nfft = freqres_to_nfft(conf.spec.freq_res, data.grid.samplerate)  # samples
    hop_len = overlap_to_hoplen(conf.spec.overlap_frac, nfft)  # samples
    chunksize = int(time_window * data.grid.samplerate)  # samples
    n_chunks = np.ceil(data.grid.rec.shape[0] / chunksize).astype(int)
    msg = (
        "Dividing recording of duration"
        f"{data.grid.rec.shape[0] / data.grid.samplerate} into {n_chunks}"
        f"chunks of {time_window} seconds each.",
    )
    con.log(msg)

    # stash here the dataframes with the bounding boxes
    bbox_dfs = []

    # shift the time of the tracks to start at 0
    # because a subset starts at the orignal time
    # TODO: Remove this when gridtools is fixed
    data.track.times -= data.track.times[0]

    for current_chunk in range(n_chunks):
        # get start and stop indices for the current chunk
        # including some overlap to compensate for edge effects
        # this diffrers for the first and last chunk

        idx1, idx2 = make_chunk_indices(
            n_chunks,
            current_chunk,
            chunksize,
            window_overlap_samples,
            data.grid.rec.shape[0],
        )

        # idx1 and idx2 now determine the window I cut out of the raw signal
        # to compute the spectrogram of.

        # compute the time and frequency axes of the spectrogram now that we
        # include the start and stop indices of the current chunk and thus the
        # right start and stop time. The `spectrogram` function does not know
        # about this and would start every time axis at 0.
        spectrogram_times, spectrogram_frequencies = make_spectrogram_axes(
            idx1,
            idx2,
            nfft,
            hop_len,
            data.grid.samplerate,
        )

        # If we reach the end of the recording, we need to cut off the last
        # chunk at the end of the recording.

        # make a subset of the current chunk
        chunk = subset(data, idx1, idx2, mode="index")

        # compute the spectrogram for each electrode of the current chunk
        spectrogram = compute_sum_spectrogam(chunk, nfft, hop_len)

        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.use("TkAgg")
        plt.imshow(spectrogram.cpu().numpy(), aspect="auto", origin="lower")
        plt.show()
        continue

        # cut off everything outside the upper frequency limit
        # the spec is still a tensor

        spectrogram = spectrogram[
            (spectrogram_frequencies >= spectrogram_freq_limits[0])
            & (spectrogram_frequencies <= spectrogram_freq_limits[1]),
            :,
        ]
        spectrogram_frequencies = spectrogram_frequencies[
            (spectrogram_frequencies >= spectrogram_freq_limits[0])
            & (spectrogram_frequencies <= spectrogram_freq_limits[1])
        ]

        # normalize the spectrogram to zero mean and unit variance
        # the spec is still a tensor
        spectrogram = zscore_standardize(spectrogram)

        # convert the spectrogram to a PIL image and save
        spectrogram = spectrogram.detach().cpu().numpy()

        img = numpy_to_pil(spectrogram)
        imgpath = dataroot / "images" / f"{chunk.path.name}.png"
        img.save(imgpath)

        if label_mode == "synthetic":
            bboxes = bboxes_from_simulated_chirps(chunk, nfft)
            bbox_df = convert_bboxes_from_simulated_chirps(
                imgpath,
                spectrogram_times,
                spectrogram_frequencies,
                current_chunk,
                bboxes,
            )
            if bbox_df is None:
                continue
            bbox_dfs.append(bbox_df)
            save_labels_for_simulated_chirps(bbox_df, dataroot)

        elif label_mode == "detected":
            detected_labels(
                dataroot, chunk, imgpath.name, spectrogram, spectrogram_times
            )

    if label_mode == "synthetic":
        bbox_df = pd.concat(bbox_dfs, ignore_index=True)
        bbox_df.to_csv(dataroot / f"{data.path.name}_bboxes.csv", index=False)

    # save the classes.txt file
    classes = ["__background__", "chirp"]
    with pathlib.Path.open(dataroot / "classes.txt", "w") as f:
        f.write("\n".join(classes))


def convert_cli(
    path: pathlib.Path,
    output: pathlib.Path,
    label_mode: str,
) -> None:
    """Parse all datasets in a directory and convert them to a YOLO dataset.

    Parameters
    ----------
    - `path` : `pathlib.Path`
        The root directory of the datasets.

    Returns
    -------
    - `None`
    """
    make_file_tree(output)
    config = load_config(str(path / "chirpdetector.toml"))

    for p in track(list(path.iterdir()), description="Building datasets"):
        if p.is_file():
            continue
        data = load(p)
        convert(data, config, output, label_mode)
