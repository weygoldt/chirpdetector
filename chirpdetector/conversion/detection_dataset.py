"""Detect chirps on a spectrogram."""

import gc
import logging
import pathlib
from typing import List, Self
from gridtools.utils.spectrograms import freqres_to_nfft
import shutil
import matplotlib as mpl
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch
from gridtools.datasets import load
from gridtools.datasets.models import Dataset
from gridtools.preprocessing.preprocessing import interpolate_tracks
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)
from PIL import Image
import pandas as pd

from chirpdetector.config import Config, load_config
from chirpdetector.datahandling.bbox_tools import (
    pixel_box_to_timefreq,
)
from chirpdetector.datahandling.dataset_parsing import (
    ArrayParser,
    make_batch_specs,
)
from chirpdetector.logging.logging import Timer, make_logger

# Use non-gui backend for matplotlib to avoid memory leaks
mpl.use("Agg")

# initialize the progress bar
prog = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
)


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


def chirp_height_width_to_bbox(
    data: Dataset,
    nfft: int,
) -> pd.DataFrame:

    pad_time = nfft / data.grid.samplerate * 0.8
    freq_res = data.grid.samplerate / nfft
    pad_freq = freq_res * 50
    pad_freq = freq_res * 40
    pad_freq = freq_res * 10

    boxes = []
    ids = []

    print(data.com.chirp.have_params)

    for fish_id in data.track.ids:

        freqs = data.track.freqs[data.track.idents == fish_id]
        times = data.track.times[data.track.indices[data.track.idents == fish_id]]
        chirps = data.com.chirp.times[data.com.chirp.idents == fish_id]
        print(np.shape(data.com.chirp.params))
        chirp_params = data.com.chirp.params[data.com.chirp.idents == fish_id]

        for chirp, param in zip(chirps, chirp_params):

            f_closest = freqs[np.argsort(np.abs(times - chirp))[:2]]
            t_closest = times[np.argsort(np.abs(times - chirp))[:2]]

            f_closest = np.average(
                f_closest, weights=np.abs(t_closest - chirp)
            )
            t_closest = np.average(
                t_closest, weights=np.abs(t_closest - chirp)
            )

            height = param[1]
            width = param[0]

            t_center = t_closest
            f_center = f_closest + height / 2

            bbox_height = height
            bbox_width = width + pad_time

            boxes.append([
                t_center - bbox_width / 2,
                (f_center - bbox_height / 2) - pad_freq,
                t_center + bbox_width / 2,
                (f_center + bbox_height / 2) + pad_freq / 2, # just a bit higher
            ])
            ids.append(fish_id)

    df = pd.DataFrame( # noqa
        boxes,
        columns=["t1", "f1", "t2", "f2"]
    )
    df["fish_id"] = ids
    return df


def convert_detections(
    detections: List,
    bbox_ids: List,
    metadata: List,
    times: List,
    freqs: List,
    cfg: Config,
) -> pd.DataFrame:
    """Convert the detected bboxes to a pandas DataFrame including metadata.

    Parameters
    ----------
    detections : List
        The detections for each spectrogram in the batch.
    metadata : List
        The metadata for each spectrogram in the batch.
    times : List
        The time axis for each spectrogram in the batch.
    freqs : List
        The frequency axis for each spectrogram in the batch.
    cfg : Config
        The configuration file.

    Returns
    -------
    out_df : pd.DataFrame
        The converted detections.
    """
    dataframes = []
    for i in range(len(detections)):

        # get the boxes and scores for the current spectrogram
        boxes = detections[i]["boxes"] # bbox coordinates in pixels
        scores = detections[i]["scores"] # confidence scores
        idents = bbox_ids[i] # unique ids for each bbox
        batch_spec_index = np.ones(len(boxes)) * i

        # discard boxes with low confidence
        boxes = boxes[scores >= cfg.det.threshold]
        idents = idents[scores >= cfg.det.threshold]
        batch_spec_index = batch_spec_index[scores >= cfg.det.threshold]
        scores = scores[scores >= cfg.det.threshold]

        # convert the boxes to time and frequency
        boxes_timefreq = pixel_box_to_timefreq(
            boxes=boxes,
            time=times[i],
            freq=freqs[i]
        )

        # put it all into a large dataframe
        dataframe = pd.DataFrame({
            "recording": [metadata[i]["recording"] for _ in range(len(boxes))],
            "batch": [metadata[i]["batch"] for _ in range(len(boxes))],
            "window": [metadata[i]["window"] for _ in range(len(boxes))],
            "spec": batch_spec_index,
            "box_ident": idents,
            "raw_indices": [metadata[i]["indices"] for _ in range(len(boxes))],
            "freq_range": [metadata[i]["frange"] for _ in range(len(boxes))],
            "x1": boxes[:, 0],
            "y1": boxes[:, 1],
            "x2": boxes[:, 2],
            "y2": boxes[:, 3],
            "t1": boxes_timefreq[:, 0],
            "f1": boxes_timefreq[:, 1],
            "t2": boxes_timefreq[:, 2],
            "f2": boxes_timefreq[:, 3],
            "score": scores
        })
        dataframes.append(dataframe)
    out_df = pd.concat(dataframes)
    return out_df.reset_index(drop=True)


def convert_cli(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    """Terminal interface for the detection function.

    Parameters
    ----------
    - `input_path` : `pathlib.Path`
        The path to the directory containing the raw dataset.
    - `output_path` : `pathlib.Path`
        The path to the directory to save the converted dataset to.

    Returns
    -------
    - `None`
    """
    logger = make_logger(__name__, input_path / "chirpdetector.log")
    datasets = [folder for folder in input_path.iterdir() if folder.is_dir()]
    confpath = input_path / "chirpdetector.toml"

    make_file_tree(output_path)

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

    with prog:
        task = prog.add_task("Detecting chirps...", total=len(datasets))
        for dataset in datasets:
            prog.console.log(f"Detecting chirps in {dataset.name}")
            data = load(dataset)
            data = interpolate_tracks(data, samplerate=120)
            cpd = Wavetracker2YOLOConverter(
                cfg=config,
                data=data,
                output_path=output_path,
                logger=logger,
            )
            cpd.convert()

            del data
            del cpd
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            prog.advance(task, 1)
        prog.update(task, completed=len(datasets))


class Wavetracker2YOLOConverter:
    """Parse a grid dataset into batches."""

    def __init__(
        self: Self,
        cfg: Config,
        data: Dataset,
        output_path: pathlib.Path,
        logger: logging.Logger
    ) -> None:
        """Initialize the converter.

        Parameters
        ----------
        cfg : Config
            Configuration file.
        data : Dataset
            Dataset to detect chirps on.
        logger: logging.Logger
            The logger to log to a logfile.
        """
        # Basic setup
        self.cfg = cfg
        self.data = data
        self.logger = logger
        self.bbox_df = chirp_height_width_to_bbox(
            data=data,
            nfft=freqres_to_nfft(cfg.spec.freq_res, data.grid.samplerate)
        )
        self.img_path = output_path / "images"
        self.label_path = output_path / "labels"

        # Batch and windowing setup
        self.parser = ArrayParser(
            length=data.grid.shape[0],
            samplingrate=data.grid.samplerate,
            batchsize=cfg.spec.batch_size,
            windowsize=cfg.spec.time_window,
            overlap=cfg.spec.spec_overlap,
            console=prog.console
        )

        msg = "Intialized Converter."
        self.logger.info(msg)
        prog.console.log(msg)

    def convert(self: Self) -> None:
        """Convert wavetracker data to spectrogram snippets."""
        prog.console.rule("[bold green]Starting parser")
        dataframes = []
        bbox_counter = 0
        for i, batch_indices in enumerate(self.parser.batches):

            prog.console.rule(
                f"[bold green]Batch {i} of {len(self.parser.batches)}"
            )

            # STEP 0: Create metadata for each batch
            batch_metadata = [{
                "recording": self.data.path.name,
                "batch": i,
                "window": j,
                "indices": indices,
                "frange": [],
            } for j, indices in enumerate(batch_indices)]

            # STEP 1: Load the raw data as a batch
            batch_raw = [
                np.array(self.data.grid.rec[idxs[0] : idxs[1], :])
                for idxs in batch_indices
            ]

            # STEP 2: Compute the spectrograms for each raw data snippet
            with Timer(prog.console, "Compute spectrograms"):
                batch_metadata, specs, times, freqs = make_batch_specs(
                    batch_indices,
                    batch_metadata,
                    batch_raw,
                    self.data.grid.samplerate,
                    self.cfg
                )

            # STEP 3: Use the chirp_params to estimate the bounding boxes
            # import matplotlib as mpl
            # import matplotlib.pyplot as plt
            # mpl.use("TkAgg")
            for spec, time, freq in zip(specs, times, freqs):

                # fig, ax = plt.subplots()
                # ax.pcolormesh(time, freq, spec.cpu().numpy()[1])
                # get the boxes for the current time and freq range
                boxes = self.bbox_df[
                    (self.bbox_df["t1"] >= time[0]) &
                    (self.bbox_df["t2"] <= time[-1])
                ].to_numpy()

                maxf = np.max(freq)
                minf = np.min(freq)

                # correct overshooting freq bounds of bboxes
                boxes[:, 1] = np.clip(boxes[:, 1], minf, maxf)
                boxes[:, 3] = np.clip(boxes[:, 3], minf, maxf)

                # plot the boxes
                # for box in boxes:
                #     ax.add_patch(Rectangle(
                #         (box[0], box[1]),
                #         box[2] - box[0],
                #         box[3] - box[1],
                #         linewidth=1,
                #         edgecolor="r",
                #         facecolor="none"
                #     ))
                # plt.show()

                # convert boxes to [label, x_center, y_center, width, height]
                # format in relative coordinates
                label = 1

                print(spec.shape)
                labels = np.ones(len(boxes), dtype=int) * label
                x = (boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2) / spec.shape[2]
                y = (boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2) / spec.shape[1]
                w = (boxes[:, 2] - boxes[:, 0]) / spec.shape[2]
                h = (boxes[:, 3] - boxes[:, 1]) / spec.shape[1]

                txt = np.stack([labels, x, y, w, h], axis=1)

                # save the boxes to the label directory
                # filename is batch metadata
                filename = (
                    f"{batch_metadata[0]['recording']}_"
                    f"{batch_metadata[0]['batch']}_"
                    f"{batch_metadata[0]['window']}.txt"
                )
                np.savetxt(
                    self.label_path / filename,
                    txt,
                    fmt=["%d", "%f", "%f", "%f", "%f"]
                )

                # convert the spec to PIL image
                img = numpy_to_pil(spec.cpu().numpy()[1])

                # save the image to the image directory
                filename = (
                    f"{batch_metadata[0]['recording']}_"
                    f"{batch_metadata[0]['batch']}_"
                    f"{batch_metadata[0]['window']}.png"
                )
                img.save(self.img_path / filename)

            del specs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

