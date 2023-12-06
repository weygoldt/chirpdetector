"""Evaluate the metadata saved during training."""

import json
import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel

from .train_model import FoldMetrics


class AvgEpochMetrics(BaseModel):
    """Average metrics for each epoch across all folds."""

    precision: List[float]
    recall: List[float]
    f1: List[float]
    score: List[float]
    avg_precision: List[float]


def load_metrics(path: pathlib.Path) -> List[FoldMetrics]:
    """Load the metrics from a JSON file."""
    files = list(path.glob("*.json"))
    metrics = []
    for file in files:
        with file.open("r") as f:
            data = json.load(f)
        metrics.append(FoldMetrics(**data))
    return metrics


def average_epoch_metrics(metrics: List[FoldMetrics]) -> None:  # noqa
    """Average the metrics over all epochs."""
    # epoch_avg_precisions = []

    epoch_med_precisions = []
    epoch_q1_precisions = []
    epoch_q3_precisions = []
    epoch_med_recalls = []
    epoch_q1_recalls = []
    epoch_q3_recalls = []
    # epoch_med_f1s = []
    # epoch_q1_f1s = []
    # epoch_q3_f1s = []
    # epoch_med_scores = []
    # epoch_q1_scores = []
    # epoch_q3_scores = []
    epoch_mean_avg_precisions = []
    epoch_std_avg_precisions = []
    all_iou_epoch_mean_avg_precisions = []

    # get max number of bboxes per epoch for resampling
    metamax = 0
    iou_thresh = 0

    for metric in metrics:
        for epoch in metric.metrics:
            # first IOU threshold because they are all the same length
            # and first array in scores because we only have one class
            metamax = max(metamax, len(epoch[0].scores[0]))

    # first iterate over the epochs
    # we ignore the IoU threshold for now and just use the first one (0.5)
    for epoch in range(metrics[iou_thresh].n_epochs):
        fold_epoch_precisions = []
        fold_epoch_recalls = []
        fold_epoch_f1s = []
        fold_epoch_scores = []
        fold_epoch_avg_precisions = []
        all_iou_fold_epoch_mean_avg_precisions = []

        # now iterate over the folds
        for metric in metrics:
            # get the metrics of the current epoch
            metric_epoch = metric.metrics[epoch]

            # get the AP at all IOU thresholds
            fold_mean_ap = []
            for iou in range(len(metric_epoch)):
                fold_mean_ap.append(metric_epoch[iou].mean_avg_prec)
            fold_mean_ap = np.array(fold_mean_ap)
            all_iou_fold_epoch_mean_avg_precisions.append(fold_mean_ap)

            # just get the first IOU threshold for now which is 0.5
            metric_epoch = metric_epoch[iou_thresh]

            # get the metrics of the current epoch
            fold_epoch_precisions.append(metric_epoch.precision[0])
            fold_epoch_recalls.append(metric_epoch.recall[0])
            fold_epoch_f1s.append(metric_epoch.f1[0])
            fold_epoch_scores.append(metric_epoch.scores[0])
            fold_epoch_avg_precisions.append(metric_epoch.mean_avg_prec)

        # average the metrics over the folds
        # this requires resampling the metrics to the same length
        # since the number of bboxes varies for each training sample

        # resample the metrics
        num_samples = 1000
        fold_epoch_precisions = [
            np.interp(
                np.linspace(0, metamax, num_samples), np.arange(len(x)), x
            )
            for x in fold_epoch_precisions
        ]
        fold_epoch_recalls = [
            np.interp(
                np.linspace(0, metamax, num_samples), np.arange(len(x)), x
            )
            for x in fold_epoch_recalls
        ]
        fold_epoch_f1s = [
            np.interp(
                np.linspace(0, metamax, num_samples), np.arange(len(x)), x
            )
            for x in fold_epoch_f1s
        ]
        fold_epoch_scores = [
            np.interp(
                np.linspace(0, metamax, num_samples), np.arange(len(x)), x
            )
            for x in fold_epoch_scores
        ]

        # average the metrics
        fold_epoch_precisions = np.array(fold_epoch_precisions)
        fold_epoch_recalls = np.array(fold_epoch_recalls)
        fold_epoch_f1s = np.array(fold_epoch_f1s)
        fold_epoch_scores = np.array(fold_epoch_scores)

        epoch_med_precisions.append(np.median(fold_epoch_precisions, axis=0))
        epoch_q1_precisions.append(
            np.quantile(fold_epoch_precisions, 0.25, axis=0)
        )
        epoch_q3_precisions.append(
            np.quantile(fold_epoch_precisions, 0.75, axis=0)
        )

        epoch_med_recalls.append(np.median(fold_epoch_recalls, axis=0))
        epoch_q1_recalls.append(np.quantile(fold_epoch_recalls, 0.25, axis=0))
        epoch_q3_recalls.append(np.quantile(fold_epoch_recalls, 0.75, axis=0))

        epoch_mean_avg_precisions.append(np.median(fold_epoch_avg_precisions))
        epoch_std_avg_precisions.append(np.std(fold_epoch_avg_precisions))

        all_iou_epoch_mean_avg_precisions.append(
            np.mean(all_iou_fold_epoch_mean_avg_precisions)
        )

    import matplotlib as mpl

    mpl.use("TkAgg")
    for i in range(len(epoch_med_precisions)):
        # fill the area under the PR curve
        plt.fill_between(
            epoch_med_recalls[i], epoch_med_precisions[i], alpha=1, zorder=-i
        )
        # plt.fill_between(
        #     epoch_med_recalls[i],
        #     epoch_q1_precisions[i],
        #     epoch_q3_precisions[i],
        #     alpha=0.2,
        # )
        print(
            f"Epoch {i} AP@.5: {epoch_mean_avg_precisions[i]} ",
            f"+- {epoch_std_avg_precisions[i]}",
        )
        print(
            f"Epoch {i} AP@.50:.05:.95: ",
            f"{all_iou_epoch_mean_avg_precisions[i]}",
        )

    plt.legend()
    plt.show()


def eval_detection_cli(path: pathlib.Path) -> None:
    """Command line interface for evaluating detection."""
    metrics = load_metrics(path)
    average_epoch_metrics(metrics)
