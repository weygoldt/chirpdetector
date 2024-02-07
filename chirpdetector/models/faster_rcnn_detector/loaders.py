"""Loaders for the Faster R-CNN model."""

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from chirpdetector.config import Config
from chirpdetector.models.utils import get_device


def load_pretrained_faster_rcnn(num_classes: int) -> torch.nn.Module:
    """Create a pretrained Faster RCNN Model and replaces the final predictor.

    Replace the final predictor of the Faster RCNN Model to fit to a specific
    detection task.

    Parameters
    ----------
    - `num_classes`: `int`
        Number of classes (+1) that shall be detected with the model.
        One more class is required because of background.

    Returns
    -------
    - `model`: `torch.nn.Module`
        Adapted FasterRCNN Model
    """
    if not isinstance(num_classes, int):
        msg = "num_classes must be an integer"
        raise TypeError(msg)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def load_finetuned_faster_rcnn(cfg: Config) -> torch.nn.Module:
    """Load the trained faster R-CNN model.

    Parameters
    ----------
    cfg : Config
        The configuration file.

    Returns
    -------
    model : torch.nn.Module
        The trained model.
    """
    model = load_pretrained_faster_rcnn(num_classes=len(cfg.hyper.classes))
    device = get_device()
    checkpoint = torch.load(
        f"{cfg.hyper.modelpath}/faster-rcnn.pt",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
