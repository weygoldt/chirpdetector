#! /usr/bin/env python3

"""
Train and test the neural network specified in the config file.
"""
import argparse
import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import embed
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, track
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from .models.datasets import CustomDataset
from .models.helpers import (
    collate_fn,
    get_device,
    get_transforms,
    load_fasterrcnn,
)
from .utils.configfiles import Config, load_config

progress = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
)


def save_model(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
) -> None:
    """
    Save the model.
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


def plot_losses(
    path: str,
    filename: str,
    train_loss: np.ndarray,
    train_spread: np.ndarray,
    val_loss: np.ndarray,
    val_spread: np.ndarray,
) -> None:
    """
    Plot the training and validation losses.
    """
    _, ax = plt.subplots()
    x = np.arange(len(train_loss))

    ax.fill_between(
        x,
        train_spread[0],
        train_spread[1],
        alpha=0.1,
    )
    ax.plot(x, train_loss, label="Training Loss")
    ax.fill_between(
        x,
        val_spread[0],
        val_spread[1],
        alpha=0.1,
    )
    ax.plot(val_loss, label="Validation Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_ylim(bottom=0)

    savepath = pathlib.Path(path) / filename
    plt.savefig(savepath)


def plot_roc(
    path: str,
    filename: str,
    output: List,
    labels: List,
) -> None:
    """
    Plot the ROC curve.
    """
    fpr, tpr, _ = roc_curve(output, labels)
    _, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", lw=0.5, color="k")
    ax.plot(fpr, tpr)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")

    savepath = pathlib.Path(path) / filename
    plt.savefig(savepath)


def train_epoch(
    dataloader: DataLoader,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> List:
    """
    Train the model for one epoch.
    """
    train_loss = []

    with progress:
        task = progress.add_task("Training...   ", total=len(dataloader))
        for samples, targets in dataloader:
            images = list(sample.to(device) for sample in samples)
            targets = [
                {k: v.to(device) for k, v in t.items() if k != "image_name"}
                for t in targets
            ]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            train_loss.append(losses.item())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            progress.update(task, advance=1)

    return train_loss


def val_epoch(
    dataloader: DataLoader,
    device: torch.device,
    model: torch.nn.Module,
) -> List:
    """
    Validate the model for one epoch.
    """
    val_loss = []

    with progress:
        task = progress.add_task("Validating... ", total=len(dataloader))

        for samples, targets in dataloader:
            images = list(sample.to(device) for sample in samples)
            targets = [
                {k: v.to(device) for k, v in t.items() if k != "image_name"}
                for t in targets
            ]

            with torch.inference_mode():
                loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            val_loss.append(losses.item())
            progress.update(task, advance=1)

    return loss_dict, val_loss


def train(config: Config) -> None:
    """
    Train the model.
    """
    device = get_device()
    progress.console.log("[bold green]Start training...[/bold green]")
    progress.console.log(f"Using device: {device}")
    progress.console.log(f"Using config file: {config.path}")

    data = CustomDataset(
        path=config.train.datapath,
        classes=config.hyper.classes,
        width=config.hyper.width,
        height=config.hyper.height,
    )

    # initialize the model and optimizer and kfolds
    model = load_fasterrcnn(len(config.hyper.classes)).to(device)
    splits = KFold(n_splits=config.hyper.kfolds, shuffle=True, random_state=42)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=config.hyper.learning_rate, momentum=0.9, weight_decay=0.0005
    )

    # initialize the best validation loss to a large number
    best_val_loss = float("inf")

    # iterate over the folds for k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(
        splits.split(np.arange(len(data)))
    ):
        progress.console.log(f"Fold: {fold + 1} of {config.hyper.kfolds}")
        train_data = torch.utils.data.Subset(data, train_idx)
        val_data = torch.utils.data.Subset(data, val_idx)

        train_loader = DataLoader(
            train_data,
            batch_size=config.hyper.batch_size,
            shuffle=True,
            num_workers=config.hyper.num_workers,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=config.hyper.batch_size,
            shuffle=True,
            num_workers=config.hyper.num_workers,
            collate_fn=collate_fn,
        )

        epoch_train_loss = []
        epoch_train_loss_spread = []
        epoch_val_loss = []
        epoch_val_loss_spread = []

        # train the model for the specified number of epochs
        for epoch in range(config.hyper.num_epochs):
            progress.console.log(
                f"Epoch: {epoch + 1} of {config.hyper.num_epochs}"
            )
            train_loss = train_epoch(
                dataloader=train_loader,
                device=device,
                model=model,
                optimizer=optimizer,
            )
            _, val_loss = val_epoch(
                dataloader=val_loader,
                device=device,
                model=model,
            )

            epoch_train_loss.append(np.median(train_loss))
            epoch_train_loss_spread.append(np.percentile(train_loss, [25, 75]))
            epoch_val_loss.append(np.median(val_loss))
            epoch_val_loss_spread.append(np.percentile(val_loss, [25, 75]))

            print(np.shape(np.array(epoch_train_loss_spread)))
            print(epoch)
            print(np.mean(val_loss))

            # save the model if it is the best so far
            if (epoch > 1) and (np.mean(val_loss) < best_val_loss):
                best_val_loss = sum(val_loss) / len(val_loss)

                progress.console.log(
                    f"New best validation loss: {best_val_loss:.4f}, saving model..."
                )

                save_model(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    path=config.hyper.modelpath,
                )

                plot_losses(
                    path=config.hyper.modelpath,
                    filename="best_model_losses.png",
                    train_loss=np.array(epoch_train_loss),
                    train_spread=np.array(epoch_train_loss_spread),
                    val_loss=np.array(epoch_val_loss),
                    val_spread=np.array(epoch_val_loss_spread),
                )

        epoch_train_loss = np.array(epoch_train_loss)
        epoch_train_loss_spread = np.array(epoch_train_loss_spread)
        epoch_val_loss = np.array(epoch_val_loss)
        epoch_val_loss_spread = np.array(epoch_val_loss_spread)

        plot_losses(
            path=config.hyper.modelpath,
            filename=f"fold_{fold + 1}_losses.png",
            train_loss=epoch_train_loss,
            train_spread=epoch_train_loss_spread,
            val_loss=epoch_val_loss,
            val_spread=epoch_val_loss_spread,
        )


def train_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to config file.",
    )
    config = load_config(parser.parse_args().config)
    train(config)
