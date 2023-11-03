#! /usr/bin/env python3

"""
Train and test the neural network specified in the config file.
"""
import argparse
import pathlib
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import embed
from rich.progress import (
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

matplotlib.use("Agg")

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


def plot_avg_losses(
    path: str,
    train_loss: np.ndarray,
    val_loss: np.ndarray,
    checkpoint: bool = False,
) -> None:
    """
    Plot the training and validation losses.
    """
    _, ax = plt.subplots()
    x = np.arange(len(train_loss))

    ax.plot(x, train_loss, label="Training Loss")
    ax.plot(val_loss, label="Validation Loss")
    if checkpoint:
        ax.axvline(
            x[-1], color="gray", linestyle="--", lw=1, label="Checkpoint"
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_ylim(bottom=0)

    savepath = pathlib.Path(path) / "avg_losses.png"
    plt.savefig(savepath)
    plt.close()


def plot_all_losses(
    path: str,
    train_loss: np.ndarray,
    val_loss: np.ndarray,
) -> None:
    """
    Plot the training and validation losses.
    """
    _, ax = plt.subplots(figsixe=(10, 20))

    x_train = np.arange(len(train_loss[0]))
    x_val = np.arange(len(val_loss[0])) + len(train_loss[0])
    for train_epoch_loss, val_epoch_loss in zip(train_loss, val_loss):
        ax.plot(x_train, train_epoch_loss, label="Training Loss", c="tab:blue")
        ax.plot(x_val, val_epoch_loss, label="Validation Loss", c="tab:orange")
        x_train += len(train_loss[0])
        x_val += len(val_loss[0])

    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_ylim(bottom=0)

    savepath = pathlib.Path(path) / "all_losses.png"
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
    model = load_fasterrcnn(len(config.hyper.classes)).to(device).train()
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
        # save loss across all epochs and folds
        all_train_loss = []
        all_val_loss = []
        epoch_train_loss = []
        epoch_val_loss = []

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
                epoch_val_loss.append(np.median(val_loss))

                plot_all_losses(
                    path=config.hyper.modelpath,
                    train_loss=np.array(all_train_loss),
                    val_loss=np.array(all_val_loss),
                )

                # save the model if it is the best so far
                checkpoint = False
                if np.mean(val_loss) < best_val_loss:
                    checkpoint = True
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

                plot_avg_losses(
                    path=config.hyper.modelpath,
                    train_loss=np.array(epoch_train_loss),
                    val_loss=np.array(epoch_val_loss),
                    checkpoint=True,
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


def verify_detections(
    image: torch.Tensor, target: dict, output: dict, path: str
) -> None:
    """
    Verify the detections.
    """

    _, ax = plt.subplots()
    ax.imshow(image.cpu().numpy().transpose((1, 2, 0)))

    for box in target["boxes"]:
        box = box.cpu().numpy()
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                fill=False,
                edgecolor="red",
                linewidth=1,
            )
        )

    for box in output["boxes"]:
        box = box.cpu().numpy()
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                fill=False,
                edgecolor="blue",
                linewidth=1,
            )
        )
        ax.text(
            box[0],
            box[1],
            f"{output['scores'][0].cpu().numpy():.2f}",
            color="blue",
        )

    plt.savefig(path)
    plt.close()


def inference_on_trainingdata(conf: Config) -> None:
    """
    Perform inference on the training data.
    """
    device = get_device()
    prog.console.log(f"Using device: [bold blue]{device}[reset]")
    prog.console.log(f"Using config file: [bold blue]{conf.path}[reset]")
    prog.console.log(
        "[bold magenta] ------------------ Starting Inference ------------------ [reset]"
    )
    data = CustomDataset(
        path=conf.train.datapath,
        classes=conf.hyper.classes,
        width=conf.hyper.width,
        height=conf.hyper.height,
    )

    modelpath = pathlib.Path(conf.hyper.modelpath) / "model.pt"

    model = load_fasterrcnn(len(conf.hyper.classes)).to(device)
    model.load_state_dict(torch.load(modelpath)["model_state_dict"])
    model.eval()

    # with prog:
    # task = prog.add_task("Inference", total=len(data))
    for idx in range(len(data)):
        image, target = data[idx]
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        with torch.inference_mode():
            output = model([image])

        path = pathlib.Path(conf.hyper.modelpath) / "verify"
        path.mkdir(parents=True, exist_ok=True)
        path = path / f"{idx}.png"
        verify_detections(
            image=image,
            target=target,
            output=output[0],
            path=path,
        )


def inference_cli():
    parser = argparse.ArgumentParser(
        description="Perform inference on the training data."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to config file.",
    )
    config = load_config(parser.parse_args().config)
    inference_on_trainingdata(config)
