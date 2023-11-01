#! /usr/bin/env python3

"""
Train and test the neural network specified in the config file.
"""

import argparse
from typing import List

import torch
from rich.console import Console
from rich.progress import Progress, track
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from .models.datasets import CustomDataset
from .models.helpers import collate_fn, get_device, load_fasterrcnn
from .utils.configfiles import Config, load_config

con = Console()


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

    with Progress() as progbar:
        task = progbar.add_task("Training...", total=len(dataloader))

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

            progbar.console.print(f"Loss: {losses.item()}", end="\r")
            progbar.update(task, advance=1)

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

    with Progress() as progbar:
        task = progbar.add_task("Validating...", total=len(dataloader))

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

            progbar.console.print(f"Loss: {losses.item()}", end="\r")
            progbar.update(task, advance=1)

    return val_loss


def train(config: Config) -> None:
    """
    Train the model.
    """
    device = get_device()
    con.print("[bold green]Start training...[/bold green]")
    con.print(f"Using device: {device}")
    con.print(f"Using config file: {config.path}")

    train_dataset = CustomDataset(
        path=config.train.datapath,
        classes=config.hyper.classes,
        width=config.hyper.width,
        height=config.hyper.height,
        transforms=get_transform(train=True),
    )

    val_dataset = CustomDataset(
        path=config.test.datapath,
        classes=config.hyper.classes,
        width=config.hyper.width,
        height=config.hyper.height,
        transforms=get_transform(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.hyper.batch_size,
        shuffle=True,
        num_workers=config.hyper.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.hyper.batch_size,
        shuffle=False,
        num_workers=config.hyper.num_workers,
        collate_fn=collate_fn,
    )

    model = load_fasterrcnn(len(config.hyper.classes)).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=config.hyper.learning_rate, momentum=0.9, weight_decay=0.0005
    )

    for epoch in range(config.hyper.num_epochs):
        con.print(f"Epoch: {epoch + 1} of {config.hyper.num_epochs}")
        train_loss = train_epoch(
            dataloader=train_loader,
            device=device,
            model=model,
            optimizer=optimizer,
        )
        val_loss = val_epoch(
            dataloader=val_loader,
            device=device,
            model=model,
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
