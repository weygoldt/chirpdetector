#!/usr/bin/env python

"""
Chirpdetector - Detect chirps of fish on a spectrogram.
This is the main entry point of the chirpdetector command line tool.
"""

import pathlib

import rich_click as click
import toml

from .convert_data import parse_datasets
from .dataset_utils import (
    clean_yolo_dataset,
    merge_yolo_datasets,
    plot_yolo_dataset,
    subset_yolo_dataset,
)
from .detect_chirps import detect_cli
from .plot_dataset import plot_yolo_dataset_cli
from .train_model import train_cli
from .utils.configfiles import copy_config

click.rich_click.USE_RICH_MARKUP = True
click.rich_click.USE_MARKDOWN = True

pyproject = toml.load(pathlib.Path(__file__).parent.parent / "pyproject.toml")
__version__ = pyproject["tool"]["poetry"]["version"]


def add_version(f):
    """
    Add the version of the chirpdetector to the help heading.
    """
    doc = f.__doc__
    f.__doc__ = (
        "Welcome to Chirpdetector Version: " + __version__ + "\n\n" + doc
    )
    return f


@click.group()
@click.version_option(
    __version__, "-V", "--version", message="Chirpdetector, version %(version)s"
)
@add_version
def cli():
    """Detect chirps of fish on a spectrogram.

    The chirpdetector command line tool is a collection of commands that
    make it easier to detect chirps of wave-type weakly electric
    fish on a spectrogram. It provides a set of managing functions e.g.
    to copy the default config file to your dataset as well as a suite to
    train, test and use a faster-R-CNN to detect chirps.

    The usual workflow is:
    1. `copyconfig` to copy the configfile to your dataset.
    2. `convert` to convert your dataset to training data.
    3. label your data, e.g. in label-studio.
    4. `train` to train the detector.
    5. `detect` to detect chirps on your dataset.

    Repeat this cycle from (2) to (5) until you are satisfied with the
    detection performance.

    For more information including a tutorial, see the documentation at
    https://weygoldt.com/chirpdetector

    Happy chirp detecting :fish::zap:
    """
    pass


@cli.command()
@click.argument("mode", type=click.Choice(["train", "detected"]))
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True),
    required=True,
    help="Path to the dataset.",
)
@click.option(
    "--n_images",
    "-n",
    type=int,
    required=True,
    help="Number of images to show.",
)
def show(mode, path, n_images):
    """
    Visualize chirps on spectrograms for the training dataset
    or detected chirps on wavetracker datasets.
    """
    if mode == "train":
        plot_yolo_dataset_cli(path, n_images)


@cli.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    help="Path to the dataset.",
    required=True,
)
def copyconfig(path):
    """Copy the default config file to your dataset."""
    copy_config(path)


@cli.command()
@click.option(
    "--input_path",
    "-i",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    required=True,
    help="Path to the input dataset.",
)
@click.option(
    "--output_path",
    "-o",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    required=True,
    help="Path to the output dataset.",
)
@click.option(
    "--labels",
    "-l",
    type=click.Choice(["none", "synthetic", "detected"]),
    required=True,
    help="Whether labels are not there yet (none), simulated (synthetic) or inferred by the detector (detected).",
)
def convert(input_path, output_path, labels):
    """Convert a wavetracker dataset to labeled or unlabeled spectrogram images to train the model."""
    parse_datasets(input_path, output_path, labels)


@cli.command()
@click.option(
    "--config_path",
    "-c",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    required=True,
    help="Path to the configuration file.",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["pretrain", "finetune"]),
    required=True,
    help="""Whether to train the model with synthetic data or to finetune a 
        model with real data.""",
)
def train(config_path, mode):
    """Train the model."""
    train_cli(config_path, mode)


@cli.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    required=True,
    help="Path to the dataset.",
)
def detect(path):
    """Detect chirps on a spectrogram."""
    detect_cli(path)


@cli.group()
def yoloutils():
    """Utilities to manage YOLO-style training datasets."""
    pass


@yoloutils.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    required=True,
    help="Path to the dataset.",
)
@click.option(
    "--img_ext",
    "-e",
    type=str,
    required=True,
    help="The image extension, e.g. .png or .jpg",
)
def clean(path, img_ext):
    """Remove all images where the label file is empty."""
    clean_yolo_dataset(path, img_ext)


@yoloutils.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    required=True,
    help="Path to the dataset.",
)
@click.option(
    "--img_ext",
    "-e",
    type=str,
    required=True,
    help="The image extension, e.g. .png or .jpg",
)
@click.option(
    "--n",
    "-n",
    type=int,
    required=True,
    help="The size of the subset",
)
def subset(path, img_ext, n):
    """Create a subset of a dataset. Useful for manually labeling a small subset."""
    subset_yolo_dataset(path, img_ext, n)


@yoloutils.command()
@click.option(
    "--dataset1",
    "-d1",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    required=True,
    help="Path to the first dataset.",
)
@click.option(
    "--dataset2",
    "-d2",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    required=True,
    help="Path to the second dataset.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    required=True,
    help="Path to the output dataset.",
)
def merge(dataset1, dataset2, output):
    """Merge two datasets."""
    merge_yolo_datasets(dataset1, dataset2, output)


if __name__ == "__main__":
    cli()
