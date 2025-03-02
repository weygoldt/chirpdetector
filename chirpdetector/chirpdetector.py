"""Chirpdetector - Detect chirps of fish on a spectrogram.

This is the main entry point of the chirpdetector command line tool.
"""

import pathlib
from typing import Callable

import rich_click as click
import toml

from chirpdetector.config import copy_config

# from chirpdetector.conversion.convert_data import convert_cli
from chirpdetector.datahandling.convert_to_training_data import convert_cli
from chirpdetector.datahandling.yolo_dataset_utils import (
    clean_yolo_dataset,
    merge_yolo_datasets,
    plot_yolo_dataset,
    subset_yolo_dataset,
)
from chirpdetector.detection.detect_chirps import detect_cli
from chirpdetector.models.faster_rcnn_detector.train import train_cli
from chirpdetector.visualization.plot_detections import (
    clean_all_plots_cli,
    clean_plots_cli,
    plot_all_detections_cli,
    plot_detections_cli,
)

click.rich_click.USE_MARKDOWN = True
# click.rich_click.SHOW_ARGUMENTS = True
# click.rich_click.COMMAND_GROUPS = {
#     "main": [
#         {
#             "name": "Commands for detecting chirps",
#             "commands": ["copyconfig", "train", "detect", "assign"],
#         },
#         {
#             "name": "Commands for managing datasets",
#             "commands": ["convert", "datautils"],
#         },
#         {
#             "name": "Commands for visualizations",
#             "commands": ["show", "plot"],
#         },
#     ]
# }

pyproject = toml.load(pathlib.Path(__file__).parent.parent / "pyproject.toml")
__version__ = pyproject["tool"]["poetry"]["version"]


def add_version(f: Callable) -> Callable:
    """Add the version of the chirpdetector to the help heading."""
    doc = f.__doc__
    f.__doc__ = (
        "**Welcome to Chirpdetector Version: "
        + __version__
        + " :fish::zap:**\n\n"
        + doc
    )
    return f


@click.group()
@click.version_option(
    __version__,
    "-V",
    "--version",
    prog_name="Chirpdetector",
    message="Chirpdetector, version %(version)s",
)
@add_version
def chirpdetector() -> None:
    """The chirpdetector command line tool is a collection of commands that
    make it easier to detect chirps of wave-type weakly electric
    fish on a spectrogram. The usual workflow is:

    1. `copyconfig` to copy the configfile to your dataset.
    2. `convert` to convert your dataset to training data.
    3. label your data, e.g. in label-studio.
    4. `train` to train the detector.
    5. `detect` to detect chirps on your dataset.
    6. `assign` to assign detections to the tracks of individual fish.

    Repeat this cycle from (2) to (5) until you are satisfied with the
    detection performance.

    For more information including a tutorial, see the documentation at
    *https://weygoldt.com/chirpdetector*
    """  # noqa


@chirpdetector.command()
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
    help="Path to the dataset.",
    required=True,
)
def copyconfig(input_path: pathlib.Path) -> None:
    """Copy the default config file to your dataset."""
    copy_config(input_path)


@chirpdetector.command()
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
    help="""
    Whether labels are not there yet (none), simulated
    (synthetic) or inferred by the detector (detected).
    """,
)
def convert(
    input_path: pathlib.Path, output_path: pathlib.Path, labels: str
) -> None:
    """Convert a wavetracker dataset to YOLO.

    Convert wavetracker dataset to labeled or unlabeled
    spectrogram images to train the model.
    """
    convert_cli(input_path, output_path, labels)


@chirpdetector.command()
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
def train(config_path: pathlib.Path, mode: str) -> None:
    """Train the model."""
    train_cli(config_path, mode)


@chirpdetector.command()
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
    "--make_training_data",
    "-t",
    is_flag=True,
    default=False,
    help="Whether to make training data. Not recommended for large datasets.",
)
def detect(path: pathlib.Path, make_training_data: bool) -> None:
    """Detect chirps on a spectrogram."""
    # detect_cli(path, make_training_data)
    detect_cli(path)


#
# @chirpdetector.command()
# @click.option(
#     "--path",
#     "-p",
#     type=click.Path(
#         exists=True,
#         file_okay=False,
#         dir_okay=True,
#         resolve_path=True,
#         path_type=pathlib.Path,
#     ),
#     required=True,
#     help="Path to the dataset.",
# )
# def assign(path: pathlib.Path) -> None:
#     """Detect chirps on a spectrogram."""
#     assign_cli(path)


# @chirpdetector.command()
# @click.option(
#     "--path",
#     "-p",
#     type=click.Path(
#         exists=True,
#         file_okay=False,
#         dir_okay=True,
#         resolve_path=True,
#         path_type=pathlib.Path,
#     ),
#     required=True,
#     help="Path to the dataset.",
# )
# def evaltrain(path: pathlib.Path) -> None:
#     """Detect chirps on a spectrogram."""
#     eval_detection_cli(path)


@chirpdetector.command()
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
    "--all",
    "-a",
    is_flag=True,
    default=False,
    help="Whether to iterate over multiple datasets.",
)
@click.option(
    "--clean",
    "-c",
    is_flag=True,
    default=False,
    help="Just delete plots in the current dataset.",
)
def plot(path: pathlib.Path, all: bool, clean: bool) -> None:  # noqa
    """Plot detected chirps on a spectrogram.

    You can supply a path to a **single** recording and plot all chirp
    detections for it or delete all plots if you supply the `--clean` option.

    Alternatively, you can supply a path to a **folder** containing multiple
    recordings and plot all chirp detections for all recordings if you supply
    the `--all` option. You can also delete all plots for all recordings if
    you supply the `--all` and `--clean` options.
    """
    if clean:
        if all:
            clean_all_plots_cli(path)
        else:
            clean_plots_cli(path)
    elif all:
        plot_all_detections_cli(path)
    else:
        plot_detections_cli(path)


@chirpdetector.group()
def datautils() -> None:
    """Utilities to manage YOLO-style training datasets."""
    pass


@datautils.command()
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
def clean(path: pathlib.Path, img_ext: str) -> None:
    """Remove all images where the label file is empty."""
    clean_yolo_dataset(path, img_ext)


@datautils.command()
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
def subset(path: pathlib.Path, img_ext: str, n: int) -> None:
    """Create a subset of a dataset.

    Useful for manually labeling a small subset.
    """
    subset_yolo_dataset(path, img_ext, n)


@datautils.command()
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
def merge(
    dataset1: pathlib.Path, dataset2: pathlib.Path, output: pathlib.Path
) -> None:
    """Merge two datasets."""
    merge_yolo_datasets(dataset1, dataset2, output)


@datautils.command()
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
    required=False,
    help="Path to the dataset.",
)
@click.option(
    "--number",
    "-n",
    type=int,
    required=False,
    default=5,
    help="Number of images to show.",
)
def show(path: pathlib.Path, number: int) -> None:
    """Show the dataset."""
    plot_yolo_dataset(path, number)


if __name__ == "__main__":
    chirpdetector()
