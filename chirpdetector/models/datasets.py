#! /usr/bin/env python3

"""
Dataset classes to train and test the model.
"""

import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision
import torchvision.transforms.functional as F
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader, Dataset

from ..utils.configfiles import load_config
from .helpers import collate_fn


class CustomDataset(set):
    """
    Custom dataset class for the training and testing data.
    """

    def __init__(
        self,
        path: str,
        classes: List[str],
        width: int = None,
        height: int = None,
        transforms: torchvision.transforms.Compose = None,
    ):
        self.transforms = transforms
        self.path = pathlib.Path(path)
        self.image_dir = pathlib.Path(path) / "images"
        self.label_dir = pathlib.Path(path) / "labels"
        self.height = height
        self.width = width
        self.classes = classes

        # get all the image paths in sorted order
        self.image_paths = list(self.image_dir.glob("*.png"))
        self.images = sorted([str(x.name) for x in self.image_paths])

    def __getitem__(self, idx):
        # get the path of the image and load it
        image_name = self.images[idx]
        image_path = self.image_dir / image_name
        img = PIL.Image.open(image_path)
        img = F.to_tensor(img.convert("RGB"))
        img_width = img.shape[2]
        img_height = img.shape[1]

        # get the corresponding label of the image
        label_path = self.label_dir / (image_name[:-4] + ".txt")
        annotations = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)
        boxes = []
        labels = []

        for annot in annotations:
            labels.append(int(annot[0]))

            # resize the normed bounding boxes according to the image width
            # and height
            x1 = annot[1] * img_width
            y1 = annot[2] * img_height
            x2 = annot[3] * img_width
            y2 = annot[4] * img_height
            boxes.append([x1, y1, x2, y2])

        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([idx])

        # apply the image transforms
        if self.transforms:
            sample = self.transforms(
                image=img, bboxes=target["boxes"], labels=labels
            )
            img = sample["image"]
            target["boxes"] = torch.Tensor(sample["bboxes"])

        return img, target

    def __len__(self):
        return len(self.images)


def main():
    """
    Instantiate the CustomDataset class and visualize the images and bounding
    boxes.
    """

    config = load_config(
        "/home/weygoldt/Projects/mscthesis/src/chirpdetector/chirpdetector/config.toml"
    )
    dataset = CustomDataset(
        config.train.datapath,
        config.hyper.width,
        config.hyper.height,
        config.hyper.classes,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.hyper.batch_size,
        shuffle=True,
        num_workers=config.hyper.num_workers,
        collate_fn=collate_fn,
    )

    print(f"Number of training images: {len(dataset)}")
    for samples, targets in loader:
        for s, t in zip(samples, targets):
            _, ax = plt.subplots()
            ax.imshow(s.permute(1, 2, 0), aspect="auto")
            for (x0, y0, x1, y1), l in zip(t["boxes"], t["labels"]):
                print(x0, y0, x1, y1, l)
                ax.add_patch(
                    Rectangle(
                        (x0, y0),
                        (x1 - x0),
                        (y1 - y0),
                        fill=False,
                        color="white",
                        linewidth=2,
                        zorder=10,
                    )
                )
            plt.show()


if __name__ == "__main__":
    main()
