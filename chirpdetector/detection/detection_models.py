"""Collection of algorithms to predict bounding boxes around chirps."""

import gc
from abc import ABC, abstractmethod
from typing import List, Self

import torch
from torchvision.transforms import Resize

# TODO: Refactor the ABC to always get an input transform and an output transform
# method and pass it the raw output from spectrogram computations.


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


class YOLOv8(AbstractDetectionModel):
    """Wrapper for the YOLOv model from ultralytics."""

    def predictor(self: Self, batch: List) -> List:
        """Predict boxes for a batch of spectrograms."""
        orig_batch = batch.copy()
        oldx = batch[0].shape[2]
        oldy = batch[0].shape[1]

        # Resize the spectrograms so that the longest side is 640 pixels
        newx = 800
        newy = int(batch[0].shape[1] * newx // batch[0].shape[2])

        # vertically mirror the spectrograms
        for i in range(len(batch)):
            batch[i] = torch.flip(batch[i], [1])

        # print("ORIGINAL")
        # for i in range(len(batch)):
        #     print(batch[i].shape)

        # print(newx, newy)

        # print("RESIZED")
        resize = Resize((newy, newx))
        for i in range(len(batch)):
            batch[i] = resize(batch[i])
            # print(batch[i].shape)

        # Add padding to y axis so that the spectrograms are newx x newx
        # print("PADDED")
        for i in range(len(batch)):
            batch[i] = torch.nn.functional.pad(
                batch[i],
                # pad=(0, 0, newx - newy, 0),
                pad=(0, 0, 0, newx - newy),
                mode="constant",
                value=0.5,
            )
            # print(batch[i].shape)

        # convert list of tensors to tensor
        batch = torch.stack(batch)

        model_output = self.model.predict(batch, save=False)

        transformed_model_output = []
        for i in range(len(model_output)):
            boxes = model_output[i].boxes.xyxy.detach().cpu().numpy()
            labels = model_output[i].boxes.cls.detach().cpu().numpy()
            scores = model_output[i].boxes.conf.detach().cpu().numpy()

            # print("BOXES")
            # print(boxes)
            # print("LABELS")
            # print(labels)
            # print("SCORES")
            # print(scores)

            # flip the boxes back so that y(0,0) is the bottom left corner
            boxes[:, 1] = newy - boxes[:, 1]
            boxes[:, 3] = newy - boxes[:, 3]
            boxes[:, [1, 3]] = boxes[:, [3, 1]]

            # scale the boxes back to the original size
            boxes[:, 0] = boxes[:, 0] * oldx / newx
            boxes[:, 2] = boxes[:, 2] * oldx / newx
            boxes[:, 1] = boxes[:, 1] * oldy / newy
            boxes[:, 3] = boxes[:, 3] * oldy / newy

            # truncate the boxes to the original size
            boxes[:, 0] = boxes[:, 0].clip(0, oldx)
            boxes[:, 2] = boxes[:, 2].clip(0, oldx)
            boxes[:, 1] = boxes[:, 1].clip(0, oldy)
            boxes[:, 3] = boxes[:, 3].clip(0, oldy)

            # print("FLIPPED BOXES")
            # print(boxes)

            # plot the boxes
            # mpl.use("TkAgg")
            # fig, ax = plt.subplots()
            # ax.imshow(orig_batch[i].detach().cpu().numpy().squeeze()[0, :, :], cmap="magma")
            # for j in range(len(boxes)):
            #     box = boxes[j]
            #     rect = mpl.patches.Rectangle(
            #         (box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor="r", facecolor="none"
            #     )
            #     ax.add_patch(rect)
            # plt.show()
            # exit()

            output = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
            transformed_model_output.append(output)
        return transformed_model_output

    def convert_to_standard_format(self: Self, model_output: List) -> List:
        """Convert the model output to a standardized format."""
        output = []
        for i in range(len(model_output)):
            boxes = model_output[i]["boxes"]
            scores = model_output[i]["scores"]
            labels = model_output[i]["labels"]

            boxes = boxes[labels == 0]
            scores = scores[labels == 0]
            output.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                }
            )
        return output
