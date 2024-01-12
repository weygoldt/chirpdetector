"""Detect chirps on a spectrogram."""

import gc
import logging
import pathlib
from typing import List, Self

import matplotlib as mpl
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


def convert_cli(input_path: pathlib.Path, make_training_data: bool) -> None:
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

    with prog:
        task = prog.add_task("Detecting chirps...", total=len(datasets))
        for dataset in datasets:
            prog.console.log(f"Detecting chirps in {dataset.name}")
            data = load(dataset)
            data = interpolate_tracks(data, samplerate=120)
            cpd = Wavetracker2YOLOConverter(
                cfg=config,
                data=data,
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
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            mpl.use("TkAgg")
            fig, ax = plt.subplots()
            for spec, time, freq in zip(specs, times, freqs):
                ax.pcolormesh(time, freq, spec.cpu().numpy()[1])
            plt.show()

            del specs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

