import argparse
import pathlib

from rich.console import Console
from rich.markdown import Markdown
from rich_argparse import RichHelpFormatter

from .convert_data import parse_datasets
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
into the root directory of the dataset on which you want to detect chirps. Then, 
change the configuration file to provide paths to training data and to 
directories where the model weights and loss plots are saved. 

If you want to create a training dataset from a wavetracker dataset, run
`chirpdetector convert`. This will create a new dataset with spectrogram images
into a specified directory. If the dataset is synthetic, you can also infer
the bounding boxes from the chirp parameters. Otherwise, you need to label the
bounding boxes manually. I recommend using the `label-studio` package for this.

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
    train.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["pretrain", "finetune"],
        required=True,
        help="""Whether to train the model with synthetic data or to finetune a 
            model with real data.""",
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

    convert = subparser.add_parser(
        "convert",
        help="Convert a wavetracker dataset to labeled or unlabeled spectrogram images to train the model.",
        formatter_class=parser.formatter_class,
    )
    convert.add_argument(
        "--input",
        "-i",
        type=pathlib.Path,
        help="Path to the input dataset.",
        required=True,
    )
    convert.add_argument(
        "--output",
        "-o",
        type=pathlib.Path,
        help="Path to the output dataset.",
        required=True,
    )
    convert.add_argument(
        "--labels",
        "-l",
        type=str,
        choices=["none", "synthetic", "detected"],
        help="""Whether to, and how to, add labes to the dataset,
            none: Don't make labels, just spectrograms,
            synthetic: Make labels from the chirp parameters used to generate 
            the synthetic data, detected: Make labels from the detected chirps""",
        required=True,
    )

    return parser.parse_args()


def chirpdetector_cli():
    args = parse_args()

    if args.command == "copyconfig":
        copy_config(str(args.path))

    elif args.command == "train":
        conf = load_config(str(args.config))
        mode = args.mode
        train(conf, mode)

    elif args.command == "detect":
        detect(args)

    elif args.command == "convert":
        parse_datasets(args.input, args.output, args.labels)

    else:
        raise ValueError("Unknown command. See --help for more information.")
