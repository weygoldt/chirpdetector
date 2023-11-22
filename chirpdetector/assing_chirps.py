#!/usr/bin/env python3

"""Assign chirps detected on a spectrogram to wavetracker tracks."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from gridtools.datasets import Dataset, load
import pathlib
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)

from .utils.configfiles import Config, load_config
from .utils.logging import make_logger

# initialize the progress bar
prog = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
)


def clean_bboxes(chirp_df: pd.DataFrame) -> pd.DataFrame:
    # do some logic to remove duplicates
    # research NON-MAXIMUM SUPPRESSION

    for bbox in chirp_df:
        pass
        # remove bboxes that are too short
        # remove bboxes that are too long

    # remove bboxes that are too small
    # remove bboxes that are too large
    # remove bboxes that are too close together
    # remove bboxes that are too far apart
    return chirp_df


def bbox_to_chirptimes(chirp_df: pd.DataFrame) -> pd.DataFrame:
    chirp_df["chirp_times"] = np.mean(chirp_df[["t1", "t2"]], axis=1)
    return chirp_df


def assign_chirps(data: Dataset, chirp_df: pd.DataFrame) -> None:
    chirp_df = clean_bboxes(chirp_df)
    chirp_df = bbox_to_chirptimes(chirp_df)


def assing_chirps_cli(path: pathlib.Path):
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory")

    if not (path / "chirpdetector.toml").is_file():
        raise ValueError(
            f"{path} does not contain a chirpdetector.toml file"
            "Make sure you are in the correct directory"
        )

    logger = make_logger(__name__, path / "chirpdetector.log")
    config = load_config(path / "chirpdetector.toml")
    recs = list(path.iterdir())
    recs = [r for r in recs if r.is_dir()]

    msg = f"Found {len(recs)} recordings in {path}, starting assignment"
    logger.info(msg)

    for rec in recs:
        logger.info(f"Assigning chirps in {rec}")
        data = load(rec)
        chirp_df = pd.read_csv(rec / "chirpdetector_bboxes.csv")
        assign_chirps(data, chirp_df)
