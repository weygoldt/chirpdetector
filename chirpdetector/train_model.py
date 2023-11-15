#! /usr/bin/env python3

"""
Train and test the neural network specified in the config file.
"""
import pathlib
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
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

matplotlib.use("Agg")

con = Console()
progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    console=con,
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


def plot_epochs(
    epoch_train_loss: np.ndarray,
    epoch_val_loss: np.ndarray,
    epoch_avg_train_loss: np.ndarray,
    epoch_avg_val_loss: np.ndarray,
    path: str,
) -> None:
    _, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    x_train = np.arange(len(epoch_train_loss[0])) + 1
    x_val = np.arange(len(epoch_val_loss[0])) + len(epoch_train_loss[0]) + 1

    for train_loss, val_loss in zip(epoch_train_loss, epoch_val_loss):
        ax[0].plot(x_train, train_loss, c="tab:blue", label="_")
        ax[0].plot(x_val, val_loss, c="tab:orange", label="_")
        x_train = np.arange(len(epoch_train_loss[0])) + x_val[-1]
        x_val = np.arange(len(epoch_val_loss[0])) + x_train[-1]

    x_avg = np.arange(len(epoch_avg_train_loss)) + 1
    ax[1].plot(x_avg, epoch_avg_train_loss, label="Training Loss", c="tab:blue")
    ax[1].plot(
        x_avg, epoch_avg_val_loss, label="Validation Loss", c="tab:orange"
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
    fold_avg_train_loss: np.ndarray,
    fold_avg_val_loss: np.ndarray,
    path: str,
) -> None:
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


def train(config: Config, mode: str = "pretrain") -> None:
    """
    Train the model.
    """

    assert mode in ["pretrain", "finetune"]
    if mode == "pretrain":
        assert config.train.datapath is not None
        datapath = config.train.datapath
    elif mode == "finetune":
        assert config.hyper.modelpath is not None
        datapath = config.hyper.modelpath

    if not pathlib.Path(datapath).exists():
        raise FileNotFoundError(f"Path {datapath} does not exist.")

    global logger
    logger = make_logger(
        __name__, pathlib.Path(config.path).parent / "chirpdetector.log"
    )
    device = get_device()

    progress.console.rule("Starting training")
    msg = f"Device: {device}, Config: {config.path}, Mode: {mode}, Data: {datapath}"
    progress.console.log(msg)
    logger.info(msg)

    data = CustomDataset(
        path=datapath,
        classes=config.hyper.classes,
        # width=config.hyper.width,
        # height=config.hyper.height,
    )

    # initialize the k-fold cross-validation
    splits = KFold(n_splits=config.hyper.kfolds, shuffle=True, random_state=42)

    # initialize the best validation loss to a large number
    best_val_loss = float("inf")

    # iterate over the folds for k-fold cross-validation
    with progress:
        # save loss across all epochs and folds
        fold_train_loss = []
        fold_val_loss = []
        fold_avg_train_loss = []
        fold_avg_val_loss = []

        task_folds = progress.add_task(
            f"[blue]{config.hyper.kfolds}-Fold Crossvalidation",
            total=config.hyper.kfolds,
        )
        for fold, (train_idx, val_idx) in enumerate(
            splits.split(np.arange(len(data)))
        ):
            # initialize the model and optimizer
            model = load_fasterrcnn(num_classes=len(config.hyper.classes)).to(
                device
            )

            if mode == "finetune":
                modelpath = pathlib.Path(config.hyper.modelpath) / "model.pt"
                checkpoint = torch.load(modelpath, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])

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

            epoch_avg_train_loss = []
            epoch_avg_val_loss = []
            epoch_train_loss = []
            epoch_val_loss = []

            # train the model for the specified number of epochs
            task_epochs = progress.add_task(
                f"{config.hyper.num_epochs} Epochs for fold k={fold + 1}",
                total=config.hyper.num_epochs,
            )
            for epoch in range(config.hyper.num_epochs):
                msg = (
                    f"Training epoch {epoch + 1} of {config.hyper.num_epochs} "
                    f"for fold {fold + 1} of {config.hyper.kfolds}"
                )
                progress.console.log(msg)
                logger.info(msg)

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

                epoch_train_loss.append(train_loss)
                epoch_val_loss.append(val_loss)

                epoch_avg_train_loss.append(np.median(train_loss))
                epoch_avg_val_loss.append(np.median(val_loss))

                # save the model if it is the best so far
                if np.mean(val_loss) < best_val_loss:
                    best_val_loss = sum(val_loss) / len(val_loss)

                    msg = (
                        f"New best validation loss: {best_val_loss:.4f}, "
                        "saving model..."
                    )
                    progress.console.log(msg)
                    logger.info(msg)

                    save_model(
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        path=config.hyper.modelpath,
                    )

                plot_epochs(
                    epoch_train_loss=epoch_train_loss,
                    epoch_val_loss=epoch_val_loss,
                    epoch_avg_train_loss=epoch_avg_train_loss,
                    epoch_avg_val_loss=epoch_avg_val_loss,
                    path=pathlib.Path(config.hyper.modelpath)
                    / f"fold{fold + 1}.png",
                )

                progress.update(task_epochs, advance=1)

            progress.update(task_epochs, visible=False)

            fold_train_loss.append(epoch_train_loss)
            fold_val_loss.append(epoch_val_loss)
            fold_avg_train_loss.append(epoch_avg_train_loss)
            fold_avg_val_loss.append(epoch_avg_val_loss)

            plot_folds(
                fold_avg_train_loss=fold_avg_train_loss,
                fold_avg_val_loss=fold_avg_val_loss,
                path=pathlib.Path(config.hyper.modelpath) / "losses.png",
            )

            progress.update(task_folds, advance=1)

        progress.update(task_folds, visible=False)

        msg = f"Average validation loss of last epoch across folds: {np.mean(fold_val_loss):.4f}"
        progress.console.log(msg)
        logger.info(msg)
        progress.console.rule("[bold blue]Finished training")


def train_cli(config_path: pathlib.Path, mode: str) -> None:
    """
    Train the model.
    """
    config = load_config(config_path)
    train(config, mode=mode)


def verify_detections(
    image: torch.Tensor, target: dict, output: dict, path: str
) -> None:
    """
    Verify the detections by plotting them on the image.
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


def inference_on_trainingdata(conf: Config, path) -> None:
    """
    Perform inference on the training data.
    """
    device = get_device()
    con.log(f"Using device: [bold blue]{device}[reset]")
    con.log(f"Using config file: [bold blue]{conf.path}[reset]")
    con.log(
        "[bold magenta] ------------------ Starting Inference ------------------ [reset]"
    )
    data = CustomDataset(
        path=path,
        classes=conf.hyper.classes,
        # width=conf.hyper.width,
        # height=conf.hyper.height,
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
