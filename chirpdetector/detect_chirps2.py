"""Detect chirps on a spectrogram."""

import logging
import pathlib
import shutil
import time
import uuid
from typing import Self, Tuple, List

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

class ChirpDetector:
    """Detect chirps on a spectrogram."""

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
        self.model = model
        self.device = get_device()
        self.logger = logger

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

    def detect_chirps(self: Self) -> None:
        """Detect chirps on the dataset."""
        prog.console.rule("[bold green]Starting detection")

        for batch_indices in self.parser.batches:

            batch_raw = [
                np.array(self.data.grid.rec[idxs[0] : idxs[1], :])
                for idxs in batch_indices
            ]

            batch_specs = make_batch_specs(
                batch_indices, batch_raw, self.data.grid.samplerate, self.cfg
            )


def make_batch_specs(
        indices: List, batch: List, samplerate: float, cfg: Config
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
    batch_specs_decible_cpu = [spec.cpu().numpy() for spec in batch_specs_decibel]
    batch_sum_specs = [
        np.sum(spec, axis=0) for spec in batch_specs_decible_cpu
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
    return batch_specs


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





