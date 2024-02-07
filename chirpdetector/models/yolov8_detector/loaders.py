"""Loaders for the YOLOV8 model."""

import pathlib
from ultralytics import YOLO
import torch
from chirpdetector.config import Config

def load_finetuned_yolov8(cfg: Config) -> torch.nn.Module:
    """Load the trained YOLOV8 model.

    Parameters
    ----------
    cfg : Config
        The configuration file.

    Returns
    -------
    model : torch.nn.Module
        The trained model.
    """
    path = pathlib.Path(cfg.hyper.modelpath) / "yolov8n.pt"
    return YOLO(str(path))
