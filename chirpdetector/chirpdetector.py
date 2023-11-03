import argparse
import pathlib

from rich.console import Console
from rich.markdown import Markdown
from rich_argparse import RichHelpFormatter

from .detect_chirps import detect
from .train_model import train
from .utils.configfiles import copy_config, load_config

description = """
Detect chirps on a spectrogram.
the chirpdetector package provides a command line interface (CLI) to train a
computer vision model to detect chirps on a spectrogram.

A valid dataset needs to be tracked using the `wavetracker` package to 
correctly assign chirps after detection.

To get started, run `chirpdetector copyconfig` to copy the default config file
into the root directory of your dataset. Then, change the configuration file
to provide paths to training data and to directories where the model weights
and loss plots are saved. 

Run `chirpdetector train` to train the model. If you want to use a pretrained
model, you can download one from the GitHub repository and specify the path
to the model weights in the configuration file.

Finally, run `chirpdetector detect` to detect chirps on a spectrogram. 
This will save the detected chirps as spectrogram images in a subdirectory of 
each recording. Additionally, the bounding boxes of the detected chirps are
saved as a CSV file alongside the recording.

Have fun!
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=Markdown(description),
        formatter_class=RichHelpFormatter,
    )

    # Copy the default config file to a specified directory.
    subparser = parser.add_subparsers(dest="command")
    copyconfig = subparser.add_parser(
        "copyconfig",
        help="Copy the default config file to a specified directory.",
        formatter_class=parser.formatter_class,
    )
    copyconfig.add_argument(
        "--path",
        "-p",
        type=pathlib.Path,
        help="The destination directory.",
        required=True,
    )

    # Train the model.
    train = subparser.add_parser(
        "train",
        help="Train the model.",
        formatter_class=parser.formatter_class,
    )
    train.add_argument(
        "--config",
        "-c",
        type=pathlib.Path,
        help="Path to the configuration file.",
        required=True,
    )

    # Detect chirps on a spectrogram.
    detect = subparser.add_parser(
        "detect",
        help="Detect chirps on a spectrogram.",
        formatter_class=parser.formatter_class,
    )
    detect.add_argument(
        "--path",
        "-p",
        type=pathlib.Path,
        help="Path to the dataset.",
        required=True,
    )

    return parser.parse_args()


def chirpdetector_cli():
    args = parse_args()

    if args.command == "copyconfig":
        copy_config(str(args.path))

    elif args.command == "train":
        conf = load_config(str(args.config))
        train(conf)

    elif args.command == "detect":
        detect(args)

    else:
        raise ValueError("Unknown command. See --help for more information.")
