"""Evaluate detection model"""

import gc
import logging
import pathlib
from typing import List, Self, Tuple

import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)

from chirpdetector.config import Config, load_config
from chirpdetector.datahandling.bbox_tools import (
    dataframe_nms,
    pixel_box_to_timefreq,
)
from chirpdetector.detection.assignment_models import (
    AbstractBoxAssigner,
    SpectrogramPowerTroughBoxAssignerMLP,
)
from chirpdetector.detection.detection_models import (
    AbstractDetectionModel,
    YOLOv8,
    FasterRCNN,
)
from chirpdetector.logging.logging import Timer, make_logger
from chirpdetector.models.mlp_assigner import load_trained_mlp
from chirpdetector.models.utils import get_device
from chirpdetector.models.yolov8_detector import load_finetuned_yolov8
from chirpdetector.models.faster_rcnn_detector import (
    load_finetuned_faster_rcnn,
)

# initialize the progress bar
prog = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
)


class DetectionEvaluator:
    """Evaluate detection model."""

    def __init__(
        self: Self,
        detectors: List[AbstractDetectionModel],
        detection_threshold: float,
        iou_threshold: float,
    ) -> None:
        self.detectors = detectors
        self.detection_threshold = detection_threshold
        self.iou_threshold = iou_threshold

    def evaluate(self: Self, dataset: pathlib.Path) -> pd.DataFrame:
        """Evaluate the detection model on the given dataset."""

        img_paths = list(dataset.glob("images/*.png"))
        labels_paths = list(dataset.glob("labels/*.txt"))
        img_paths.sort()
        labels_paths.sort()

        for detector in self.detectors:
            for img_path, label_path in zip(img_paths, labels_paths):
                img = Image.open(img_path)
                img = torchvision.transforms.ToTensor()(img)
                # add 3 channels
                img = img.repeat(3, 1, 1)
                img_pred = [img]

                labels = np.loadtxt(label_path)
                labels = labels[:, 1:]
                labels[:, 0] = labels[:, 0] * img.shape[2]
                labels[:, 1] = labels[:, 1] * img.shape[1]
                labels[:, 2] = labels[:, 2] * img.shape[2]
                labels[:, 3] = labels[:, 3] * img.shape[1]

                result = detector.predict(img_pred)
                pred_boxes = result[0]["boxes"]
                pred_scores = result[0]["scores"]

                plt.imshow(img.permute(1, 2, 0))

                for box, score in zip(pred_boxes, pred_scores):
                    x1, y1, x2, y2 = box
                    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "r-")
                    plt.text(x1, y1, f"{score:.2f}", color="red")

                for label in labels:
                    x, y, w, h = label
                    # convert to x1, y1, x2, y2
                    x1 = x - w / 2
                    y1 = y - h / 2
                    x2 = x + w / 2
                    y2 = y + h / 2
                    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "g-")

                plt.show()


def eval_detection_cli():
    path = "data/interrim/yolofull"
    print(f"Loading dataset from {path}")

    yolov8 = load_finetuned_yolov8(
        load_config("data/raw/local/competition_subsets/chirpdetector.toml")
    )
    yolov8 = YOLOv8(yolov8)
    evaler = DetectionEvaluator([yolov8], 0.5, 0.5)
    evaler.evaluate(pathlib.Path(path) / "val")
