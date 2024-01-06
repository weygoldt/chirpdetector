"""Detect chirps on a spectrogram."""

import logging
import pathlib
import shutil
import time
import uuid
from typing import Self, Tuple, List, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gridtools.datasets import load, subset
from gridtools.datasets.models import Dataset
from gridtools.utils.spectrograms import (
    freqres_to_nfft,
    overlap_to_hoplen,
    compute_spectrogram,
    to_decibel
)
from matplotlib.patches import Rectangle
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)

from .convert_data import make_file_tree, numpy_to_pil
from .detect_chirps import spec_to_image
from .models.utils import get_device, load_fasterrcnn
from .utils.configfiles import Config, load_config
from .utils.logging import make_logger, Timer
from .utils.signal_processing import (
    compute_sum_spectrogam,
    make_spectrogram_axes,
)
from .dataset_parser import ArrayParser

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

class BatchBoxPredictor:
    """Predict boxes for a batch of spectrograms."""

    def __init__(
        self: Self,
        cfg: Config,
        model: torch.nn.Module,
        logger: logging.Logger,
    ) -> None:
        """Initialize the BatchBoxPredictor."""
        self.cfg = cfg
        self.model = model
        self.logger = logger
        self.device = get_device()
        self.model.to(self.device).eval()

    def predict(self: Self, batch: List) -> List:
        """Predict boxes for a batch of spectrograms."""
        with Timer(prog.console, "Predict boxes"), torch.no_grad():
            detections = self.model(batch)
        # TODO: Continue here so that output is always the same
        # no matter the used model


class BatchBoxAssigner:
    """Assign boxes on a batch of spectrograms to a wavetracker track ID."""

    def __init__(
        self: Self,
        cfg: Config,
        model: torch.nn.Module,
        logger: logging.Logger,
    ) -> None:
        """Initialize the BatchBoxAssigner."""
        pass
        # TODO: Implement assignment logic for a single batch here.

    def predict(self: Self, batch: List) -> List:
        """Predict boxes assignments for a batch of spectrograms."""
        pass


class DatasetParser:
    """Parse a grid dataset into batches."""

    def __init__(
        self: Self,
        cfg: Config,
        data: Dataset,
        model: torch.nn.Module,
        logger: logging.Logger
    ) -> None:
        """Initialize the ChirpDetector.

        Parameters
        ----------
        cfg : Config
            Configuration file.
        data : Dataset
            Dataset to detect chirps on.
        model : torch.nn.Module
            Model to use for detection.
        logger: logging.Logger
            The logger to log to a logfile.
        """
        # Basic setup
        self.cfg = cfg
        self.data = data
        self.device = get_device()
        self.logger = logger
        self.model = model

        # Model setup
        self.model.to(self.device).eval()

        # Batch and window setup
        self.parser = ArrayParser(
            length=data.grid.shape[0],
            samplingrate=data.grid.samplerate,
            batchsize=cfg.spec.batch_size,
            windowsize=cfg.spec.time_window,
            overlap=cfg.spec.spec_overlap,
        )

        msg = "Intialized ChirpDetector."
        self.logger.info(msg)
        prog.console.log(msg)

    def parse(self: Self) -> None:
        """Detect chirps on the dataset."""
        prog.console.rule("[bold green]Starting parser")

        for i, batch_indices in enumerate(self.parser.batches):

            # Generate a string handle for each spectrogram
            # e.g. for filenames in case of saving
            batch_handles = [
                f"{self.data.path.name}_batch-{i}_window-{j}"
                for j in range(len(batch_indices))
            ]

            # Get the raw data for the batch
            batch_raw = [
                np.array(self.data.grid.rec[idxs[0] : idxs[1], :])
                for idxs in batch_indices
            ]

            # Compute the spectrograms for the batch
            with Timer(prog.console, "Compute spectrograms"):
                handles, specs, times, freqs = make_batch_specs(
                    batch_indices,
                    batch_handles,
                    batch_raw,
                    self.data.grid.samplerate,
                    self.cfg
                )

            boxpredictor = BatchBoxPredictor()
            detections = boxpredictor.predict(specs)

            boxassigner = BatchBoxAssigner()
            assignments = boxassigner.predict(detections)

            # TODO: Save the output from above to a file

            # import matplotlib as mpl
            # mpl.use("TkAgg")
            # fig, ax = plt.subplots()
            # for spec, time, freq in zip(specs, times, freqs):
            #     spec = spec.cpu().numpy()
            #     ax.pcolormesh(time, freq, spec[0])
            #     ax.axvline(time[0], color="white", lw=1, ls="--")
            #     ax.axvline(time[-1], color="white", lw=1, ls="--")
            #     ax.axhline(freq[0], color="white", lw=1, ls="--")
            #     ax.axhline(freq[-1], color="white", lw=1, ls="--")
            # plt.show()



def convert_detections(
    detections, names, times, freqs, cfg: Config
    ):
    """Convert the detections to a pandas DataFrame."""
    for i in range(len(detections)):
        t = times[i]
        f = freqs[i]
        n = names[i]
        boxes = detections[i]["boxes"].cpu().numpy()
        scores = detections[i]["scores"].cpu().numpy()
        labels = detections[i]["labels"].cpu().numpy()

        boxes = boxes[(scores > cfg.det.threshold) & (labels == 1)]
        scores = scores[(scores > cfg.det.threshold) & (labels == 1)]
        labels = labels[(scores > cfg.det.threshold) & (labels == 1)]




def make_batch_specs(
    indices: List, handles: List, batch: List, samplerate: float, cfg: Config
    ) -> List:
    """Compute the spectrograms for a batch of windows."""
    batch = np.swapaxes(batch, 1, 2)
    nfft = freqres_to_nfft(
        freq_res=cfg.spec.freq_res,
        samplingrate=samplerate
    )
    hop_length = overlap_to_hoplen(
        nfft=nfft,
        overlap=cfg.spec.overlap_frac
    )
    batch_specs = [
        compute_spectrogram(
            data=signal,
            samplingrate=samplerate,
            nfft=nfft,
            hop_length=hop_length
        )[0] for signal in batch
    ]

    batch_specs_decibel = [
        to_decibel(spec) for spec in batch_specs
    ]
    # batch_specs_decible_cpu = [spec for spec in batch_specs_decibel]
    batch_sum_specs = [
        torch.sum(spec, dim=0) for spec in batch_specs_decibel
    ]
    axes = [
        make_spectrogram_axes(
            start=idxs[0],
            stop=idxs[1],
            nfft=nfft,
            hop_length=hop_length,
            samplerate=samplerate
        ) for idxs in indices
    ]
    batch_specs = [
        (spec, *ax) for spec, ax in zip(batch_sum_specs, axes)
    ]
    # Add the name to each spec tuple
    batch_specs = [
        (name, *spec) for name, spec in zip(
            handles, batch_specs
        )
    ]

    # Tile the spectrograms y-axis
    sliced_specs = tile_batch_specs(batch_specs, cfg)

    # Split the list into specs and axes
    handles, specs, times, freqs = zip(*sliced_specs)

    # Convert the spec tensors to mimic PIL images
    images = [spec_to_image(spec) for spec in specs]

    return handles, images, times, freqs


def tile_batch_specs(batch_specs: List, cfg: Config) -> List:
    """Tile the spectrograms of a batch."""
    freq_ranges = [(0, 1000), (500, 1500), (1000, 2000)]
    sliced_specs = []
    for start, end in freq_ranges:
        start_idx = np.argmax(batch_specs[0][3] >= start)
        end_idx = np.argmax(batch_specs[0][3] >= end)
        suffix = f"_frange-{start}-{end}"
        sliced_specs.extend([
            (name + suffix, spec[start_idx:end_idx, :], time, freq[start_idx:end_idx])
            for name, spec, time, freq in batch_specs
        ])

    return sliced_specs

def get_faster_rcnn(cfg: Config) -> torch.nn.Module:
    """Load the trained faster R-CNN model."""
    model = load_fasterrcnn(num_classes=len(cfg.hyper.classes))
    device = get_device()
    checkpoint = torch.load(
        f"{cfg.hyper.modelpath}/model.pt",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def detect_cli(input_path: pathlib.Path, make_training_data: bool) -> None:
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
            cpd = ChirpDetector(
                cfg=config,
                data=data,
                model=get_faster_rcnn(config),
                logger=logger,
            )
            cpd.detect_chirps()
            exit()

        prog.update(task, completed=len(datasets))





