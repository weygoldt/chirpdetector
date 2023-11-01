#! /usr/bin/env python3

"""
Read, write and handle files, such as configuration files, model files, etc.
"""


from typing import List

import toml
from pydantic import BaseModel, ConfigDict


def load_config(path: str) -> ConfigDict:
    path = path
    print(path)
    file = toml.load(path)
    hy = Hyperparams(**file["hyperparameters"])
    tr = Training(**file["training"])
    te = Testing(**file["evaluation"])
    config = Config(
        path=path,
        hyper=hy,
        train=tr,
        test=te,
    )
    return config


class Hyperparams(BaseModel):
    width: int
    height: int
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


class Testing(BaseModel):
    datapath: str


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    path: str
    hyper: Hyperparams
    train: Training
    test: Testing
