#! /usr/bin/env python3

"""
Read, write and handle files, such as configuration files, model files, etc.
"""


from typing import List

import toml
from pydantic import BaseModel, ConfigDict


def load_config(path: str) -> ConfigDict:
    file = toml.load(path)
    hy = Hyperparams(**file["hyper"])
    tr = Training(**file["train"])
    te = Testing(**file["test"])
    config = Config(
        hyper=hy,
        train=tr,
        test=te,
    )
    return config


class Hyperparams(BaseModel):
    classes: List(str)
    num_epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    model_path: str


class Training(BaseModel):
    datapath: str


class Testing(BaseModel):
    datapath: str


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    hyper: Hyperparams
    train: Training
    test: Testing
