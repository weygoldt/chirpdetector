#! /usr/bin/env python3

"""
Train and test the neural network specified in the config file.
"""

import torch
from rich.console import Console
from rich.progress import track
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold

from .models.model import get_device, load_fasterrcnn
from .utils.configfiles import load_config

config = load_config("config.toml")
model = load_fasterrcnn(len(config.hyper.classes))
con = Console()


def train_model() -> None:
    pass
