"""Train and test the neural network."""

import json
import pathlib
from collections import Counter
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import BaseModel
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from .models.datasets import CustomDataset
from .models.utils import collate_fn, get_device, load_fasterrcnn
from .utils.configfiles import Config, load_config
from .utils.logging import make_logger

# Use non-gui backend to prevent memory leaks
mpl.use("Agg")

# Initialize the logger and progress bar
con = Console()
progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    console=con,
)


class PerformanceMetrics(BaseModel):
    """Performance metrics for object detection models.

    Parameters
    ----------
    - `classes`: `List[int]`
        The classes.
    - `precision`: `List[List[float]]`
        The precision per class per target (bbox).
    - `recall`: `List[List[float]]`
        The recall per class per target (bbox).
    - `f1`: `List[List[float]]`
        The f1 score per class per target (bbox).
    - `scores`: `List[List[float]]`
        The scores per class per target (bbox).
    - `average_precision`: `List[float]`
        The average precision per class.
    - `mean_avg_prec`: `float`
        The mean average precision.
    """

    classes: List[int]  # list of classes, here only one class
    precision: List[List[float]]  # precision per class per target (bbox)
    recall: List[List[float]]  # recall per class per target (bbox)
    f1: List[List[float]]  # f1 score per class per target (bbox)
    scores: List[List[float]]  # scores per class per target (bbox)

    average_precision: List[float]  # average precision per class
    mean_avg_prec: float  # mean average precision
    checkpoint: bool = False  # whether this is a checkpoint


class FoldMetrics(BaseModel):
    """Metrics for each fold.

    Parameters
    ----------
    - `n_epochs`: `int`
        How many epochs were trained.
    - `iou_thresholds`: `List[float]`
        Which IoU thresholds were used to compute the models performance
        metrics.
    - `avg_train_loss`: `List[float]`
        The average training loss for each epoch.
    - `avg_val_loss`: `List[float]`
        The average validation loss for each epoch.
    - `metrics`: `List[List[PerformanceMetrics]]`
        The performance metrics for each epoch for each IoU threshold.
    """

    n_epochs: int  # how many epochs were trained
    iou_thresholds: List[float]  # which IoU thresholds were used
    avg_train_loss: List[float]  # average training loss per epoch
    avg_val_loss: List[float]  # average validation loss per epoch
    metrics: List[
        List[PerformanceMetrics]
    ]  # metrics (eg AP) per epoch per IoU


def collapse_all_dims(arr: np.ndarray) -> np.ndarray:
    """Collapse all dimensions of an array.

    Parameters
    ----------
    - `np.ndarray`: `np.ndarray`
        The array to collapse.

    Returns
    -------
    - `np.ndarray`
        The collapsed array.
    """
    while len(np.shape(arr)) > 1:
        arr = np.squeeze(arr)

    return arr


def intersection_over_union(
    boxes_preds: torch.Tensor,
    boxes_labels: torch.Tensor,
    box_format: str = "corners",
) -> torch.Tensor:
    """Calculate intersection over union.

    Adapted from:
    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master
    /ML/Pytorch/object_detection/metrics/iou.py

    Parameters
    ----------
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns
    -------
        tensor: Intersection over union for all examples
    """
    if box_format not in ["midpoint", "corners"]:
        msg = (
            f"Unknown box format {box_format}. Must be one of: "
            "'midpoint', 'corners'."
        )
        raise ValueError(msg)

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    else:
        msg = "Provided box format is correct but failed to compute boxes."
        raise ValueError(msg)

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection
    # to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def mean_average_precision(  # noqa
    pred_boxes: list,
    true_boxes: list,
    iou_threshold: float = 0.5,
    box_format: str = "corners",
    num_classes: int = 1,
) -> PerformanceMetrics:
    """Calculate mean average precision and metrics used in it.

    Adapted from:
    https://github.com/aladdinpersson/Machine-Learning-Collection
    /blob/master/ML/Pytorch/object_detection/metrics/mean_avg_precision.py

    Parameters
    ----------
    - `pred_boxes` : `list`
        list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
    - `true_boxes` : `list`
        Similar as pred_boxes except all the correct ones. Score is set to 1.
    - `iou_threshold` : `float`
        IOU threshold where predicted bboxes is correct. See
        intersection_over_union function.
    - `box_format` : `str`
        "midpoint" or "corners" used to specify bboxes
        Midpoint is YOLO format: [x, y, width, height] and corners is
        e.g. COCO format: [x1, y1, x2, y2]. This model outputs
        "corners" format.
    - `num_classes` : `int`
        number of classes

    Returns
    -------
    - `PerformanceMetrics`
        The performance metrics.
    """
    # list storing all AP for respective classes
    all_scores = []
    all_precisions = []
    all_recalls = []
    all_f1 = []
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    # __background__ is a reserved class so there is no class 0
    # in the model output
    classes = np.arange(num_classes) + 1
    classes = classes.tolist()

    # This function is created for multiclass but in this
    # case we have only one class
    # So this is not actually the mean average precision
    # but just the average precision
    for c in classes:
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)

        # Make zeros for every detection and then later set
        # it to 1 if it is a true positive or false positive
        TP = torch.zeros((len(detections)))  # noqa
        FP = torch.zeros((len(detections)))  # noqa

        # Collect the number of total true bboxes for this class
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        # Otherwise we now loob over each prediction
        # and get the corresponding ground truth
        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            best_iou = 0
            best_gt_idx = None

            # Now get the ground truth bbox that has the highest
            # iou with this detection bbox
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            # Once we're done looking for the best match,
            # we check if the IoU is greater than the threshold
            # If it is then it's a true positive, else a false positive
            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1

                # if the bounding box has already been detected,
                # it is a false positive so a duplicate
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        # Compute how many true positives and false positives we have
        TP_cumsum = torch.cumsum(TP, dim=0)  # noqa
        FP_cumsum = torch.cumsum(FP, dim=0)  # noqa

        # Compute corresponding detection scores
        scores = torch.tensor([detection[2] for detection in detections])
        scores = torch.cat((torch.tensor([1]), scores))
        all_scores.append(scores.tolist())

        # Compute precision and recall
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        all_recalls.append(recalls.tolist())
        all_precisions.append(precisions.tolist())

        # We need to 1 to the precision so that numerical integration
        # is correct.
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # compute the f1 scores
        f1 = 2 * (precisions * recalls) / (precisions + recalls + epsilon)
        all_f1.append(f1.tolist())

        # Integral of prec-rec curve: torch.trapz for numerical integration
        # (AP = area under prec-rec curve)
        avg_prec = torch.trapz(precisions, recalls)
        average_precisions.append(avg_prec.item())

    # mean average precision is the mean of all the average precisions
    mean_avg_prec = sum(average_precisions) / len(average_precisions)
    mean_avg_prec = float(mean_avg_prec)

    # instantiate the performance metrics
    return PerformanceMetrics(
        classes=classes,
        precision=all_precisions,
        recall=all_recalls,
        f1=all_f1,
        scores=all_scores,
        average_precision=average_precisions,
        mean_avg_prec=mean_avg_prec,
    )


def save_model(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: pathlib.Path,
) -> None:
    """Save the model state dict.

    Parameters
    ----------
    - `epoch`: `int`
        The current epoch.
    - `model`: `torch.nn.Module`
        The model to save.
    - `optimizer`: `torch.optim.Optimizer`
        The optimizer to save.
    - `path`: `pathlib.Path`
        The path to save the model to.

    Returns
    -------
    - `None`
    """
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path / "model.pt",
    )


def plot_epochs(
    epoch_train_loss: list,
    epoch_val_loss: list,
    epoch_avg_train_loss: list,
    epoch_avg_val_loss: list,
    path: pathlib.Path,
) -> None:
    """Plot the loss for each epoch.

    Parameters
    ----------
    - `epoch_train_loss`: `list`
        The training loss for each epoch.
    - `epoch_val_loss`: `list`
        The validation loss for each epoch.
    - `epoch_avg_train_loss`: `list`
        The average training loss for each epoch.
    - `epoch_avg_val_loss`: `list`
        The average validation loss for each epoch.
    - `path`: `pathlib.Path`
        The path to save the plot to.

    Returns
    -------
    - `None`
    """
    _, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    x_train = np.arange(len(epoch_train_loss[0])) + 1
    x_val = np.arange(len(epoch_val_loss[0])) + len(epoch_train_loss[0]) + 1

    for train_loss, val_loss in zip(epoch_train_loss, epoch_val_loss):
        ax[0].plot(x_train, train_loss, c="tab:blue", label="_")
        ax[0].plot(x_val, val_loss, c="tab:orange", label="_")
        x_train = np.arange(len(epoch_train_loss[0])) + x_val[-1]
        x_val = np.arange(len(epoch_val_loss[0])) + x_train[-1]

    x_avg = np.arange(len(epoch_avg_train_loss)) + 1
    ax[1].plot(
        x_avg,
        epoch_avg_train_loss,
        label="Training Loss",
        c="tab:blue",
    )
    ax[1].plot(
        x_avg,
        epoch_avg_val_loss,
        label="Validation Loss",
        c="tab:orange",
    )

    ax[0].set_ylabel("Loss")
    ax[0].set_xlabel("Batch")
    ax[0].set_ylim(bottom=0)
    ax[0].set_title("Loss per batch")

    ax[1].set_ylabel("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()
    ax[1].set_ylim(bottom=0)
    ax[1].set_title("Avg loss per epoch")

    plt.savefig(path)
    plt.close()


def plot_folds(
    fold_avg_train_loss: list,
    fold_avg_val_loss: list,
    path: pathlib.Path,
) -> None:
    """Plot the loss for each fold.

    Parameters
    ----------
    - `fold_avg_train_loss`: `list`
        The average training loss for each fold.
    - `fold_avg_val_loss`: `list`
        The average validation loss for each fold.
    - `path`: `pathlib.Path`
        The path to save the plot to.

    Returns
    -------
    - `None`
    """
    _, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    for train_loss, val_loss in zip(fold_avg_train_loss, fold_avg_val_loss):
        x = np.arange(len(train_loss)) + 1
        ax.plot(x, train_loss, c="tab:blue", alpha=0.3, label="_")
        ax.plot(x, val_loss, c="tab:orange", alpha=0.3, label="_")

    avg_train = np.mean(fold_avg_train_loss, axis=0)
    avg_val = np.mean(fold_avg_val_loss, axis=0)
    x = np.arange(len(avg_train)) + 1
    ax.plot(
        x,
        avg_train,
        label="Training Loss",
        c="tab:blue",
    )
    ax.plot(
        x,
        avg_val,
        label="Validation Loss",
        c="tab:orange",
    )

    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.savefig(path)
    plt.close()


def train_epoch(
    dataloader: DataLoader,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> List:
    """Train the model for one epoch.

    Parameters
    ----------
    - `dataloader`: `DataLoader`
        The dataloader for the training data.
    - `device`: `torch.device`
        The device to train on.
    - `model`: `torch.nn.Module`
        The model to train.
    - `optimizer`: `torch.optim.Optimizer`
        The optimizer to use.

    Returns
    -------
    - `train_loss`: `List`
        The training loss for each batch.
    """
    train_loss = []

    for samples, targets in dataloader:
        images = [sample.to(device) for sample in samples]
        bboxes = [
            {k: v.to(device) for k, v in t.items() if k != "image_name"}
            for t in targets
        ]

        model.train()
        loss_dict = model(images, bboxes)
        losses = sum(loss for loss in loss_dict.values())
        train_loss.append(losses.item())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    return train_loss


def bbox_dict_to_list(
    predicted_bboxes: List[dict],
    groundtruth_bboxes: List[dict],
) -> Tuple[List[List], List[List]]:
    """Convert the predicted and groundtruth bboxes to lists.

    Format will be as follows:
    [img_idx, label, score, x1, y1, x2, y2]

    For ground truth the score will always be 1.

    Parameters
    ----------
    - `predicted_bboxes`: `List[dict]`
        The predicted bboxes.
    - `groundtruth_bboxes`: `List[dict]`
        The groundtruth bboxes.

    Returns
    -------
    - `pred_boxes`: `List`
        The predicted bboxes.
    - `true_boxes`: `List`
        The groundtruth bboxes.
    """
    pred_boxes = []
    true_boxes = []
    for img_idx in range(len(predicted_bboxes)):
        for bbox_idx in range(len(predicted_bboxes[img_idx]["boxes"])):
            # gather data of predictions
            pred_img = img_idx
            pred_label = (
                predicted_bboxes[img_idx]["labels"][bbox_idx].cpu().numpy()
            )
            pred_score = (
                predicted_bboxes[img_idx]["scores"][bbox_idx].cpu().numpy()
            )
            pred_bbox = (
                predicted_bboxes[img_idx]["boxes"][bbox_idx].cpu().numpy()
            )

            # collapse all remaining dimensions
            pred_label = collapse_all_dims(pred_label).tolist()
            pred_score = collapse_all_dims(pred_score).tolist()
            pred_bbox = collapse_all_dims(pred_bbox).tolist()

            pred_total_bbox = [pred_img, pred_label, pred_score]
            pred_total_bbox.extend(pred_bbox)
            pred_boxes.append(pred_total_bbox)

        for bbox_idx in range(len(groundtruth_bboxes[img_idx]["boxes"])):
            # gather data of groundtruth
            true_img = img_idx
            true_label = (
                groundtruth_bboxes[img_idx]["labels"][bbox_idx].cpu().numpy()
            )
            true_score = 1
            true_bbox = (
                groundtruth_bboxes[img_idx]["boxes"][bbox_idx].cpu().numpy()
            )

            # collapse all remaining dimensions
            true_label = collapse_all_dims(true_label).tolist()
            true_bbox = collapse_all_dims(true_bbox).tolist()

            true_total_bbox = [true_img, true_label, true_score]
            true_total_bbox.extend(true_bbox)
            true_boxes.append(true_total_bbox)

    return pred_boxes, true_boxes


def val_epoch(
    dataloader: DataLoader,
    device: torch.device,
    model: torch.nn.Module,
) -> Tuple[List, List, List]:
    """Validate the model for one epoch.

    Parameters
    ----------
    - `dataloader`: `DataLoader`
        The dataloader for the validation data.
    - `device`: `torch.device`
        The device to train on.
    - `model`: `torch.nn.Module`
        The model to train.

    Returns
    -------
    - `loss_dict`: `dict`
        The loss dictionary.
    """
    val_loss = []
    groundtruth_bboxes = []
    predicted_bboxes = []
    for samples, targets in dataloader:
        images = [sample.to(device) for sample in samples]
        bboxes = [
            {k: v.to(device) for k, v in t.items() if k != "image_name"}
            for t in targets
        ]

        # get losses for each image
        with torch.inference_mode():
            # to get the loss_dict
            model.train()
            loss_dict = model(images, bboxes)

            # to get the predicted boxes
            model.eval()
            predicted_bboxes.extend(model(images))

        losses = sum(loss for loss in loss_dict.values())
        groundtruth_bboxes.extend(bboxes)
        val_loss.append(losses.item())

    pred_boxes, true_boxes = bbox_dict_to_list(
        predicted_bboxes, groundtruth_bboxes
    )

    return val_loss, pred_boxes, true_boxes


def train(config: Config, mode: str = "pretrain") -> None:  # noqa
    """Train the model.

    Parameters
    ----------
    - `config`: `Config`
        The config file.
    - `mode`: `str`
        The mode to train in. Either `pretrain` or `finetune`.

    Returns
    -------
    - `None`
    """
    # Load a pretrained model from pytorch if in pretrain mode,
    # otherwise open an already trained model from the
    # model state dict.
    if mode == "pretrain":
        assert config.train.datapath is not None
        datapath = config.train.datapath
    elif mode == "finetune":
        assert config.finetune.datapath is not None
        datapath = config.finetune.datapath
    else:
        msg = f"Unknown mode {mode}. Must be one of: 'pretrain', 'finetune'."
        raise ValueError(msg)

    # Check if the path to the data actually exists
    if not pathlib.Path(datapath).exists():
        msg = f"Path {datapath} does not exist."
        raise FileNotFoundError(msg)

    # Initialize the logger and progress bar, make the logger global
    logger = make_logger(
        __name__,
        pathlib.Path(config.path).parent / "chirpdetector.log",
    )

    # Get the device (e.g. GPU or CPU)
    device = get_device()

    # Print information about starting training
    progress.console.rule("Starting training")
    msg = (
        f"Device: {device}, Config: {config.path},"
        f" Mode: {mode}, Data: {datapath}"
    )
    progress.console.log(msg)
    logger.info(msg)

    # initialize the dataset
    data = CustomDataset(
        path=datapath,
        classes=config.hyper.classes,
    )

    # initialize the k-fold cross-validation
    splits = KFold(n_splits=config.hyper.kfolds, shuffle=True, random_state=42)

    # initialize the IoU threshold sweep for the validation
    iou_thresholds = np.round(np.arange(0.5, 1.0, 0.05), 2)

    # initialize the best validation loss to a large number
    best_val_loss = float("inf")

    # iterate over the folds for k-fold cross-validation
    with progress:
        # save loss across all epochs and folds
        fold_train_loss = []
        fold_val_loss = []
        fold_avg_train_loss = []
        fold_avg_val_loss = []

        # Add kfolds progress bar that runs alongside the epochs progress bar
        task_folds = progress.add_task(
            f"[blue]{config.hyper.kfolds}-Fold Crossvalidation",
            total=config.hyper.kfolds,
        )

        # iterate over the folds
        for fold, (train_idx, val_idx) in enumerate(
            splits.split(np.arange(len(data))),
        ):
            # initialize the model and optimizer
            model = load_fasterrcnn(num_classes=len(config.hyper.classes)).to(
                device,
            )

            # If the mode is finetune, load the model state dict from
            # previous training
            if mode == "finetune":
                modelpath = pathlib.Path(config.hyper.modelpath) / "model.pt"
                checkpoint = torch.load(modelpath, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])

            # Initialize stochastic gradient descent optimizer
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(
                params,
                lr=config.hyper.learning_rate,
                momentum=config.hyper.momentum,
                weight_decay=config.hyper.weight_decay,
            )

            # make train and validation dataloaders for the current fold
            train_data = torch.utils.data.Subset(data, train_idx)
            val_data = torch.utils.data.Subset(data, val_idx)

            # this is for training
            train_loader = DataLoader(
                train_data,
                batch_size=config.hyper.batch_size,
                shuffle=True,
                num_workers=config.hyper.num_workers,
                collate_fn=collate_fn,
            )

            # this is only for validation
            val_loader = DataLoader(
                val_data,
                batch_size=config.hyper.batch_size,
                shuffle=True,
                num_workers=config.hyper.num_workers,
                collate_fn=collate_fn,
            )

            # save loss across all epochs
            epoch_avg_train_loss = []
            epoch_avg_val_loss = []
            epoch_train_loss = []
            epoch_val_loss = []
            epoch_metrics = []

            # train the model for the specified number of epochs
            task_epochs = progress.add_task(
                f"{config.hyper.num_epochs} Epochs for fold k={fold + 1}",
                total=config.hyper.num_epochs,
            )

            # iterate across n epochs
            for epoch in range(config.hyper.num_epochs):
                # print information about the current epoch
                msg = (
                    f"Training epoch {epoch + 1} of {config.hyper.num_epochs} "
                    f"for fold {fold + 1} of {config.hyper.kfolds}"
                )
                progress.console.log(msg)
                logger.info(msg)

                # train the epoch
                train_loss = train_epoch(
                    dataloader=train_loader,
                    device=device,
                    model=model,
                    optimizer=optimizer,
                )

                # validate the epoch
                val_loss, predicted_bboxes, true_bboxes = val_epoch(
                    dataloader=val_loader,
                    device=device,
                    model=model,
                )

                # Compute model performance metrics
                # for this epoch
                metric_sweep = []
                for iou_threshold in iou_thresholds:
                    metrics = mean_average_precision(
                        pred_boxes=predicted_bboxes,
                        true_boxes=true_bboxes,
                        iou_threshold=iou_threshold,
                        box_format="corners",
                        num_classes=1,
                    )
                    metric_sweep.append(metrics)

                # save losses for this epoch
                epoch_train_loss.append(train_loss)
                epoch_val_loss.append(val_loss)

                # save the average loss for this epoch
                epoch_avg_train_loss.append(np.median(train_loss))
                epoch_avg_val_loss.append(np.median(val_loss))

                # save the model if it is the best so far
                if np.mean(val_loss) < best_val_loss:
                    best_val_loss = sum(val_loss) / len(val_loss)

                    # get the mean average precision
                    ap = np.mean(
                        [metric.mean_avg_prec for metric in metric_sweep]
                    )

                    # write checkpoint to the metrics
                    for metric in metric_sweep:
                        metric.checkpoint = True

                    msg = (
                        f"New best validation loss: {best_val_loss:.4f}, \n"
                        f"Current AP@[.50:.05:.95] {ap:.4f}, \n"
                        "saving model..."
                    )
                    progress.console.log(msg)
                    logger.info(msg)

                    modelpath = pathlib.Path(config.hyper.modelpath)
                    save_model(
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        path=modelpath,
                    )

                # save the metrics for this epoch
                epoch_metrics.append(metric_sweep)

                # plot the losses for this epoch
                plot_epochs(
                    epoch_train_loss=epoch_train_loss,
                    epoch_val_loss=epoch_val_loss,
                    epoch_avg_train_loss=epoch_avg_train_loss,
                    epoch_avg_val_loss=epoch_avg_val_loss,
                    path=pathlib.Path(config.hyper.modelpath)
                    / f"fold{fold + 1}.png",
                )

                # update the progress bar for the epochs
                progress.update(task_epochs, advance=1)

            # update the progress bar for the epochs and hide it if done
            progress.update(task_epochs, visible=False)

            # save the losses for this fold
            fold_train_loss.append(epoch_train_loss)
            fold_val_loss.append(epoch_val_loss)
            fold_avg_train_loss.append(epoch_avg_train_loss)
            fold_avg_val_loss.append(epoch_avg_val_loss)

            plot_folds(
                fold_avg_train_loss=fold_avg_train_loss,
                fold_avg_val_loss=fold_avg_val_loss,
                path=pathlib.Path(config.hyper.modelpath) / "losses.png",
            )

            # save the losses for this fold
            fold_metrics = FoldMetrics(
                n_epochs=config.hyper.num_epochs,
                iou_thresholds=iou_thresholds.tolist(),
                avg_train_loss=epoch_avg_train_loss,
                avg_val_loss=epoch_avg_val_loss,
                metrics=epoch_metrics,
            )

            filename = f"fold_{fold + 1}.json"
            filepath = pathlib.Path(config.hyper.modelpath) / filename
            with filepath.open("w") as outfile:
                json.dump(fold_metrics.dict(), outfile)

            # update the progress bar for the folds
            progress.update(task_folds, advance=1)

        # update the progress bar for the folds and hide it if done
        progress.update(task_folds, visible=False)

        # print information about the training
        msg = (
            "Average validation loss of last epoch across folds: "
            f"{np.mean(fold_val_loss):.4f}"
        )
        progress.console.log(msg)
        logger.info(msg)
        progress.console.rule("[bold blue]Finished training")


def train_cli(config_path: pathlib.Path, mode: str) -> None:
    """Train the model from the command line.

    Parameters
    ----------
    - `config_path`: `pathlib.Path`
        The path to the config file.
    - `mode`: `str`
        The mode to train in. Either `pretrain` or `finetune`.

    Returns
    -------
    - `None`
    """
    config = load_config(config_path)
    train(config, mode=mode)
