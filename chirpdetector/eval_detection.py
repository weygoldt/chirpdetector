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


def average_epoch_metrics(metrics: List[FoldMetrics]) -> None:
    """Average the metrics over all epochs."""
    epoch_avg_precisions = []

    # first iterate over the epochs
    for epoch in range(metrics[0].n_epochs):
        fold_epoch_precisions = []
        fold_epoch_recalls = []
        fold_epoch_f1s = []
        fold_epoch_scores = []
        fold_epoch_avg_precisions = []

        # now iterate over the folds
        for metric in metrics:
            # get the metrics of the current epoch
            metric_epoch = metric.metrics[epoch]

            # just get the first IOU threshold for now which is 0.5
            metric_epoch = metric_epoch[0]

            # get the metrics of the current epoch
            fold_epoch_precisions.append(metric_epoch.precision[0])
            fold_epoch_recalls.append(metric_epoch.recall[0])
            fold_epoch_f1s.append(metric_epoch.f1[0])
            fold_epoch_scores.append(metric_epoch.scores[0])
            fold_epoch_avg_precisions.append(metric_epoch.mean_avg_prec)

        # average the metrics over the folds
        # this requires resampling the metrics to the same length
        # since the number of bboxes varies for each training sample
        epoch_avg_precisions.append(fold_epoch_avg_precisions)

    x = np.arange(len(epoch_avg_precisions))
    mean = np.mean(epoch_avg_precisions, axis=1)
    std = np.std(epoch_avg_precisions, axis=1)

    avg_val_losess = [metric.avg_val_loss for metric in metrics]
    avg_train_losess = [metric.avg_train_loss for metric in metrics]

    import matplotlib as mpl

    mpl.use("TkAgg")

    plt.plot(x, mean)
    for i in range(len(epoch_avg_precisions[0])):
        plt.plot(x, np.array(epoch_avg_precisions)[:, i], alpha=0.2, color="k")
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.show()

    for val, train in zip(avg_val_losess, avg_train_losess):
        plt.plot(val, alpha=0.2, color="blue")
        plt.plot(train, alpha=0.2, color="orange")
    plt.show()


def eval_detection_cli(path: pathlib.Path) -> None:
    """Command line interface for evaluating detection."""
    metrics = load_metrics(path)
    average_epoch_metrics(metrics)
