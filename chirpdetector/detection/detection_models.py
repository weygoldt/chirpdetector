"""Collection of algorithms to predict bounding boxes around chirps."""

import gc
from abc import ABC, abstractmethod
from typing import List, Self

import torch


class AbstractDetectionModel(ABC):
    """Abstract base class for model wrappers.

    Standard output format:
    [
        {
            "boxes": torch.Tensor,
            "scores": torch.Tensor,
        },
        ...
    ]
    One dict for each spectrogram in the batch.
    Boxes follow the format [x1, y1, x2, y2] in pixels.
    We dont need labels as we only detect one class.
    """

    def __init__(self: Self, model: torch.nn.Module) -> None:
        """Initialize the model wrapper."""
        self.model = model

    def predict(self: Self, batch: List) -> List:
        """Predict boxes for a batch of spectrograms."""
        output = self.predictor(batch)
        result = self.convert_to_standard_format(output)
        del output
        gc.collect()
        return result

    @abstractmethod
    def predictor(self: Self, batch: List) -> List:
        """Predict boxes for a batch of spectrograms."""
        pass

    @abstractmethod
    def convert_to_standard_format(self: Self, model_output: List) -> List:
        """Convert the model output to a standardized format."""
        pass


class FasterRCNN(AbstractDetectionModel):
    """Wrapper for the Faster R-CNN model."""

    def predictor(self: Self, batch: List) -> List:
        """Predict boxes for a batch of spectrograms."""
        with torch.no_grad():
            return self.model(batch)

    def convert_to_standard_format(self: Self, model_output: List) -> List:
        """Convert the model output to a standardized format."""
        output = []
        for i in range(len(model_output)):
            boxes = model_output[i]["boxes"].detach().cpu().numpy()
            scores = model_output[i]["scores"].detach().cpu().numpy()
            labels = model_output[i]["labels"].detach().cpu().numpy()

            boxes = boxes[labels == 1]
            scores = scores[labels == 1]
            output.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                }
            )
        return output

