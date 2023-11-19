#!/usr/bin/env python3

"""
Load, save and handle models.
"""

import albumentations as A
import torch
import torch.nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_device():
    """
    Check if a CUDA-enabled GPU is available, and return the appropriate device.

    Returns
    -------
    - `device`: `torch.device`
        The device to use for PyTorch computations. If a CUDA-enabled GPU is available, returns a device object
        representing that GPU. If an Apple M1 GPU is available, returns a device object representing that GPU.
        Otherwise, returns a device object representing the CPU.
    """
    if torch.cuda.is_available() is True:
        device = torch.device("cuda")  # nvidia / amd gpu
    elif torch.backends.mps.is_available() is True:
        device = torch.device("mps")  # apple m1 gpu
    else:
        device = torch.device("cpu")  # no gpu
    return device


def get_transforms(width, height, train: bool):
    """
    Define the transformations that should be applied to the images.
    """
    if train:
        return A.Compose(
            [
                A.PixelDropout(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.Resize(width, height),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["labels"]
            ),
        )
    return A.Compose(
        [
            A.Resize(width, height),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate
    function (to be passed to the DataLoader).

    Parameters
    ----------
    - `batch`: `list`
        A list of the data loaded from the dataset.
    """
    return tuple(zip(*batch))


def load_fasterrcnn(num_classes: int) -> torch.nn.Module:
    """
    Create a pretrained Faster RCNN Model and replaces the final predictor in order to fit
    to a specific detection task.

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
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
