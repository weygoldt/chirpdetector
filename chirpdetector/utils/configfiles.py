#! /usr/bin/env python3

"""
Read, write and handle files, such as configuration files, model files, etc.
"""


import pathlib
import shutil
from typing import List, Union

import toml
from pydantic import BaseModel, ConfigDict

from .logging import make_logger


class Hyperparams(BaseModel):
    classes: List
    num_epochs: int
    batch_size: int
    kfolds: int
    learning_rate: float
    momentum: float
    weight_decay: float
    num_workers: int
    modelpath: str


class Training(BaseModel):
    datapath: str


class Finetune(BaseModel):
    datapath: str


class Detection(BaseModel):
    threshold: float


class Spectrogram(BaseModel):
    time_window: float
    # freq_window: List
    freq_res: float
    freq_pad: float
    overlap_frac: float
    spec_overlap: float


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    path: str
    hyper: Hyperparams
    train: Training
    finetune: Finetune
    det: Detection
    spec: Spectrogram


def copy_config(path: str) -> None:
    """
    Copy the default config file from the package root directory into a
    specified path.
    """
    origin = pathlib.Path(__file__).parent.parent / "config.toml"
    if not origin.exists():
        raise FileNotFoundError(
            "Could not find the default config file. "
            "Please make sure that the file 'config.toml' exists in "
            "the package root directory."
        )

    destination = pathlib.Path(path)

    if destination.is_dir():
        shutil.copy(origin, destination / "chirpdetector.toml")

    elif destination.is_file():
        raise FileExistsError(
            "The specified path already exists and is a file. "
            "Please specify a directory or a non-existing path."
        )

    elif not destination.exists():
        raise FileNotFoundError("Please specify an existing directory.")


def load_config(path: Union[str, pathlib.Path]) -> Config:
    global logger
    logger = make_logger(
        __name__, pathlib.Path(path).parent / "chirpdetector.log"
    )

    logger.info(
        f"Moving default configuration file to {pathlib.Path(path).parent}"
    )

    file = toml.load(path)
    hy = Hyperparams(**file["hyperparameters"])
    tr = Training(**file["training"])
    fi = Finetune(**file["finetuning"])
    det = Detection(**file["detection"])
    spec = Spectrogram(**file["spectrogram"])
    config = Config(
        path=str(path),
        hyper=hy,
        train=tr,
        finetune=fi,
        det=det,
        spec=spec,
    )
    return config
