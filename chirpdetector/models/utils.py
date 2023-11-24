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
    """Check if a CUDA-enabled GPU is available, and return the correct device.

    Returns
    -------
    - `device`: `torch.device`
        The device to use for PyTorch computations. If a CUDA-enabled GPU is
        available, returns a device object
        representing that GPU. If an Apple M1 GPU is available, returns a
        device object representing that GPU.
        Otherwise, returns a device object representing the CPU.
    """
    if torch.cuda.is_available() is True:
        device = torch.device("cuda")  # nvidia / amd gpu
    elif torch.backends.mps.is_available() is True:
        device = torch.device("mps")  # apple m1 gpu
    else:
        device = torch.device("cpu")  # no gpu
    return device


def get_transforms(width: int, height: int, train: bool) -> A.Compose:
    """Define the transformations that should be applied to the images.

    Transforms are used to augment the training data and to standardize the
    validation data. Augmentations help to increase the robustness of the model
    and to prevent overfitting.

    Parameters
    ----------
    - `width`: `int`
        The width of the images.
    - `height`: `int`
        The height of the images.
    - `train`: `bool`
        Whether the images are used for training or validation.

    Returns
    -------
    - `transforms`: `A.Compose`
        The transformations that should be applied to the images.
    """
    assert isinstance(width, int), "width must be an integer"
    assert isinstance(height, int), "height must be an integer"
    assert isinstance(train, bool), "train must be a boolean"

    if train:
        return A.Compose(
            [
                A.PixelDropout(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.Resize(width, height),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
            ),
        )
    return A.Compose(
        [
            A.Resize(width, height),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


def collate_fn(batch: list) -> tuple:
    """
    Since each image may have a different number of objects, we need a collate
    function (to be passed to the DataLoader).

    Parameters
    ----------
    - `batch`: `list`
        A list of the data loaded from the dataset.

    Returns
    -------
    - `tuple`
        A tuple containing the images and the targets.
    """
    return tuple(zip(*batch))


def load_fasterrcnn(num_classes: int) -> torch.nn.Module:
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
        raise TypeError("num_classes must be an integer")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
