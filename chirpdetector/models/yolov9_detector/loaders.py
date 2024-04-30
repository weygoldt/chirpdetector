"""Loaders for the YOLOv9 model."""

import pathlib

import torch
from ultralytics import YOLO

from chirpdetector.config import Config


def load_finetuned_yolov8(cfg: Config) -> torch.nn.Module:
    """Load the trained YOLOv9 model.

    Parameters
    ----------
    cfg : Config
        The configuration file.

    Returns
    -------
    model : torch.nn.Module
        The trained model.
    """
    path = pathlib.Path(cfg.hyper.modelpath) / "yolov9e.pt"
    return YOLO(str(path))
