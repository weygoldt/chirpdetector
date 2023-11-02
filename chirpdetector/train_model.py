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
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)
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

prog = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    transient=True,
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
    # train_spread: np.ndarray,
    val_loss: np.ndarray,
    # val_spread: np.ndarray,
) -> None:
    """
    Plot the training and validation losses.
    """
    _, ax = plt.subplots()
    x = np.arange(len(train_loss))

    ax.plot(x, train_loss, label="Training Loss")
    ax.plot(val_loss, label="Validation Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_ylim(bottom=0)

    savepath = pathlib.Path(path) / filename
    plt.savefig(savepath)
    plt.close()


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

    return loss_dict, val_loss


def train(config: Config) -> None:
    """
    Train the model.
    """
    device = get_device()
    prog.console.log(f"Using device: [bold blue]{device}[reset]")
    prog.console.log(f"Using config file: [bold blue]{config.path}[reset]")
    prog.console.log(
        "[bold magenta] ------------------ Starting Training ------------------ [reset]"
    )
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
    fold_val_loss = []

    # iterate over the folds for k-fold cross-validation
    with prog:
        task_folds = prog.add_task("[blue]Crossval", total=config.hyper.kfolds)
        for fold, (train_idx, val_idx) in enumerate(
            splits.split(np.arange(len(data)))
        ):
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

            all_train_loss = []
            all_val_loss = []

            # train the model for the specified number of epochs
            task_epochs = prog.add_task(
                f"Epochs k={fold + 1}", total=config.hyper.num_epochs
            )
            for epoch in range(config.hyper.num_epochs):
                prog.console.log(
                    f"Training epoch {epoch + 1} of {config.hyper.num_epochs} for fold {fold + 1} of {config.hyper.kfolds}"
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

                all_train_loss.append(train_loss)
                all_val_loss.append(val_loss)

                epoch_train_loss.append(np.median(train_loss))
                epoch_train_loss_spread.append(
                    np.percentile(train_loss, [25, 75])
                )
                epoch_val_loss.append(np.median(val_loss))
                epoch_val_loss_spread.append(np.percentile(val_loss, [25, 75]))

                # save the model if it is the best so far
                if np.mean(val_loss) < best_val_loss:
                    best_val_loss = sum(val_loss) / len(val_loss)

                    prog.console.log(
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
                        filename="best_model_losses_epoch.png",
                        train_loss=np.array(epoch_train_loss),
                        val_loss=np.array(epoch_val_loss),
                    )

                    plot_losses(
                        path=config.hyper.modelpath,
                        filename="best_model_losses_all.png",
                        train_loss=np.array(all_train_loss).reshape((-1)),
                        val_loss=np.array(all_val_loss).reshape((-1)),
                    )

                prog.update(task_epochs, advance=1)

            fold_val_loss.append(np.mean(val_loss))

            epoch_train_loss = np.array(epoch_train_loss)
            epoch_train_loss_spread = np.array(epoch_train_loss_spread)
            epoch_val_loss = np.array(epoch_val_loss)
            epoch_val_loss_spread = np.array(epoch_val_loss_spread)

            plot_losses(
                path=config.hyper.modelpath,
                filename=f"fold_{fold + 1}_losses.png",
                train_loss=epoch_train_loss,
                val_loss=epoch_val_loss,
            )
            prog.update(task_folds, advance=1)

        prog.console.log(
            "[bold magenta] ------------------ Training Complete ------------------ [reset]"
        )
        prog.console.log(
            f"Average validation loss of last epoch: {np.mean(fold_val_loss)}"
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
