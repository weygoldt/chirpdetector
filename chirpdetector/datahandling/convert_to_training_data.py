"""
Convert a dataset with or without already detected chirps to training data.

... meaning: .png images of spectrograms. If chirp ids are known, it also
genereates the assignment training dataset for the multilayer perceptron.
"""

import gc
import logging
import pathlib
import shutil
from typing import List, Self

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gridtools.datasets import load
from gridtools.datasets.models import Dataset
from gridtools.preprocessing.preprocessing import interpolate_tracks
from gridtools.utils.spectrograms import freqres_to_nfft
from PIL import Image
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)
from scipy.signal import find_peaks

from chirpdetector.config import Config, load_config
from chirpdetector.datahandling.bbox_tools import (
    reverse_float_index_interpolation,
)
from chirpdetector.datahandling.dataset_parsing import (
    ArrayParser,
    make_batch_specs,
)
from chirpdetector.logging.logging import Timer, make_logger

# Use non-gui backend for matplotlib to avoid memory leaks
mpl.use("Agg")

# initialize the progress bar
prog = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
)

try:
    base = "/home/weygoldt/Projects/mscthesis/src/base.mplstyle"
    darkbg = "/home/weygoldt/Projects/mscthesis/src/light_background.mplstyle"
    plt.style.use([base, darkbg])
except FileNotFoundError:
    msg = "Could not find the coustom matplotlib style, freestyling."
    prog.console.log(msg)
    pass


def numpy_to_pil(img: np.ndarray) -> Image.Image:
    """Convert a 2D numpy array to a PIL image.

    Parameters
    ----------
    img : np.ndarray
        The input image.

    Returns
    -------
    PIL.Image
        The converted image.
    """
    img_dimens = 2
    if len(img.shape) != img_dimens:
        msg = f"Image must be {img_dimens}D"
        raise ValueError(msg)

    if img.max() == img.min():
        msg = "Image must have more than one value"
        raise ValueError(msg)

    img = np.flipud(img)
    intimg = np.uint8((img - img.min()) / (img.max() - img.min()) * 255)
    return Image.fromarray(intimg)


def make_file_tree(path: pathlib.Path, wipe: bool = True) -> None:
    """Build a file tree for the training dataset.

    Parameters
    ----------
    path : pathlib.Path
        The root directory of the dataset.
    """
    if not isinstance(path, pathlib.Path):
        msg = f"Path must be a pathlib.Path, not {type(path)}"
        raise TypeError(msg)

    if path.parent.exists() and path.parent.is_file():
        msg = (
            f"Parent directory of {path} is a file. "
            "Please specify a directory."
        )
        raise ValueError(msg)

    if path.exists() and wipe:
        shutil.rmtree(path)

    path.mkdir(exist_ok=True, parents=True)

    train_imgs = path / "images"
    train_labels = path / "labels"
    assign_path = path / "assignment"
    train_imgs.mkdir(exist_ok=True, parents=True)
    train_labels.mkdir(exist_ok=True, parents=True)
    assign_path.mkdir(exist_ok=True, parents=True)


def chirp_height_width_to_bbox(
    data: Dataset,
    nfft: int,
) -> pd.DataFrame:
    pad_time = nfft / data.grid.samplerate * 0.8
    freq_res = data.grid.samplerate / nfft
    pad_freq = freq_res * 50
    pad_freq = freq_res * 40
    pad_freq = freq_res * 10

    boxes = []
    ids = []

    for fish_id in data.track.ids:
        freqs = data.track.freqs[data.track.idents == fish_id]
        times = data.track.times[
            data.track.indices[data.track.idents == fish_id]
        ]
        chirps = data.com.chirp.times[data.com.chirp.idents == fish_id]
        chirp_params = data.com.chirp.params[data.com.chirp.idents == fish_id]

        for chirp, param in zip(chirps, chirp_params):
            f_closest = freqs[np.argsort(np.abs(times - chirp))[:2]]
            t_closest = times[np.argsort(np.abs(times - chirp))[:2]]

            f_closest = np.average(
                f_closest, weights=np.abs(t_closest - chirp)
            )
            t_closest = np.average(
                t_closest, weights=np.abs(t_closest - chirp)
            )

            height = param[1]
            trough = param[2]
            width = param[0]

            t_center = t_closest
            f_center = f_closest + height / 2

            bbox_height = height
            bbox_width = width + pad_time

            boxes.append(
                [
                    t_center - bbox_width / 2,
                    (f_center - bbox_height / 2) + trough - pad_freq / 2,
                    t_center + bbox_width / 2,
                    (f_center + bbox_height / 2)
                    + pad_freq / 2,  # just a bit higher
                ]
            )
            ids.append(fish_id)

    df = pd.DataFrame(  # noqa
        boxes, columns=["t1", "f1", "t2", "f2"]
    )
    df["fish_id"] = ids
    return df


def extract_assignment_training_data(
    data: Dataset,
    boxes: np.ndarray,
    spec: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
) -> List[np.ndarray]:
    # iterate through the boxes
    # for each box, get the baseline EODf of the fish from the track
    # then get the index of the closest y value to the baseline EODf
    # that one is labeled 1, the rest 0
    # then get the spectrogram snippet
    y = []
    x = []
    for i, box in enumerate(boxes):
        # get the fish id
        fish_id = box[-1]

        # get the baseline EODf
        track = data.track.freqs[data.track.idents == fish_id]
        track_times = data.track.times[
            data.track.indices[data.track.idents == fish_id]
        ]

        # get the track values in the box
        track = track[(track_times >= box[0]) & (track_times <= box[2])]
        track_times = track_times[
            (track_times >= box[0]) & (track_times <= box[2])
        ]

        # baseline is median in the box
        baseline = np.median(track)

        # get the spectrogram window in box
        spec_window = spec[
            :,
            (times >= box[0]) & (times <= box[2]),
        ]
        spec_window = spec_window[
            (freqs >= box[1]) & (freqs <= (box[3] - (box[3] - box[1]) / 2)), :
        ]
        window_freqs = freqs[
            (freqs >= box[1]) & (freqs <= (box[3] - (box[3] - box[1]) / 2))
        ]
        window_times = times[(times >= box[0]) & (times <= box[2])]

        # interpolate the spec window and axes to 100x100
        res = 100
        spec_window = np.array(Image.fromarray(spec_window).resize((res, res)))

        window_freqs = np.linspace(window_freqs[0], window_freqs[-1], res)
        window_times = np.linspace(window_times[0], window_times[-1], res)

        # get the baseline EODfs in the window by finding peaks on the
        # power spectrum
        target_prom = (np.max(spec_window) - np.min(spec_window)) * 0.01
        # print(f"Spec window shape: {spec_window.shape}")
        power = np.mean(spec_window, axis=1)
        peaks = find_peaks(power, prominence=target_prom)[0]

        # peak start and stop are either where the sign of the diff switches
        # or where the window is 10 Hz wide
        window_radius = 5

        # get the start and end of the peak
        starts = []
        ends = []
        for idx in peaks:
            # descend the peak until the sign of the diff changes
            # or the window is 10 Hz wide
            # this is the start of the peak
            start = idx
            while True:
                if start == 0:
                    break
                if window_freqs[idx] - window_freqs[start] > window_radius:
                    break
                if np.sign(power[idx] - power[start]) == -1:
                    break
                start -= 1

            # descend the other side of the peak until the sign of the diff
            # changes or the window is 10 Hz wide
            # this is the end of the peak
            end = idx
            while True:
                if end > len(power) - 1:
                    break
                if window_freqs[end] - window_freqs[idx] > window_radius:
                    break
                if np.sign(power[idx] - power[end]) == -1:
                    break
                end += 1
            starts.append(start)
            ends.append(end)

        closest = np.argmin(np.abs(window_freqs[peaks] - baseline))

        # compute average power in the window determined by peak start and end
        powers = []
        for s, e in zip(starts, ends):
            mu = np.mean(spec_window[s:e, :], axis=0)
            # print(f"Freq power mean range start: {s}, end: {e}")
            # print(f"Shape: {mu.shape}")
            powers.append(mu)
        powers = np.array(powers)

        # # plot
        # if len(np.shape(powers)) == 1:
        # import matplotlib as mpl
        # import matplotlib.pyplot as plt
        # mpl.use("TkAgg")
        #
        # cm = 1/2.54  # centimeters in inches
        # size = (16*cm, 8*cm)
        # fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=size, constrained_layout=True)
        #
        # # get 3 cool colors from magma palette
        # # c1, c2, c3 = plt.cm.magma(np.linspace(0.5, 1, 3))
        # c2 = "tab:orange"
        # c1 = "tab:blue"
        #
        # # ax1.pcolormesh(window_times, window_freqs, spec_window, alpha=0.8)
        # ax1.imshow(
        #     spec_window,
        #     aspect="auto",
        #     origin="lower",
        #     extent=[window_times[0], window_times[-1], window_freqs[0], window_freqs[-1]],
        #     cmap="viridis",
        # )
        # ax1.plot(track_times, track, color=c2, lw=2)
        #
        # ax2.plot(power, window_freqs, color="black")
        # ax2.plot(power[peaks], window_freqs[peaks], ".", color=c1)
        #
        # for s, e in zip(starts, ends):
        #     ax2.fill_betweenx(
        #         window_freqs[s:e],
        #         power[s:e],
        #         color=c1,
        #         alpha=0.5,
        #     )
        #
        # for ix, p in enumerate(powers):
        #     if ix == closest:
        #         ax3.plot(window_times, p, color=c2, lw=2, zorder=1000)
        #     else:
        #         ax3.plot(window_times, p, color="grey", lw=1)
        #
        # ax1.set_ylim([window_freqs[0], window_freqs[-1]])
        # ax1.set_xlim([window_times[0], window_times[-1]])
        # ax2.set_ylim([window_freqs[0], window_freqs[-1]])
        # ax2.set_xlim([0, 1])
        # ax3.set_ylim([0, np.max(powers)])
        # ax3.set_xlim([window_times[0], window_times[-1]])
        #
        # ax1.set_xlabel("Time [s]")
        # ax1.set_ylabel("Frequency [Hz]")
        # ax2.set_xlabel("Power [a.u.]")
        # ax3.set_xlabel("Time [s]")
        # ax3.set_ylabel("Power [a.u.]")
        #
        # # remove ax2 y labels and axes and y spine
        # ax2.spines["left"].set_visible(False)
        # ax2.set_yticklabels([])
        # ax2.set_yticks([])
        # ax2.set_ylabel("")
        #
        # # make margins nicer
        # plt.savefig(f"plots/{uuid4()}.svg")
        # plt.show()
        #
        # prepare data and labels for export
        labels = np.zeros(len(powers))
        labels[closest] = 1

        # append to x and y
        x.extend(powers)
        y.extend(labels)

    x = np.array(x)
    y = np.array(y)

    return x, y


def convert_cli(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    """Terminal interface for the detection function.

    Parameters
    ----------
    - `input_path` : `pathlib.Path`
        The path to the directory containing the raw dataset.
    - `output_path` : `pathlib.Path`
        The path to the directory to save the converted dataset to.

    Returns
    -------
    - `None`
    """
    logger = make_logger(__name__, input_path / "chirpdetector.log")
    datasets = [folder for folder in input_path.iterdir() if folder.is_dir()]
    confpath = input_path / "chirpdetector.toml"

    make_file_tree(output_path)

    # load the config file and print a warning if it does not exist
    if confpath.exists():
        config = load_config(str(confpath))
    else:
        msg = (
            "The configuration file could not be found in the specified path."
            "Please run `chirpdetector copyconfig` and change the "
            "configuration file to your needs."
        )
        raise FileNotFoundError(msg)

    with prog:
        task = prog.add_task(
            "Converting to training data ...", total=len(datasets)
        )
        for dataset in datasets:
            prog.console.log(f"Detecting chirps in {dataset.name}")
            data = load(dataset)
            data = interpolate_tracks(data, samplerate=120)
            cpd = Wavetracker2YOLOConverter(
                cfg=config,
                data=data,
                output_path=output_path,
                logger=logger,
            )
            cpd.convert()

            del data
            del cpd
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            prog.advance(task, 1)
        prog.update(task, completed=len(datasets))


class Wavetracker2YOLOConverter:
    """Parse a grid dataset into batches."""

    def __init__(
        self: Self,
        cfg: Config,
        data: Dataset,
        output_path: pathlib.Path,
        logger: logging.Logger,
    ) -> None:
        """Initialize the converter.

        Parameters
        ----------
        cfg : Config
            Configuration file.
        data : Dataset
            Dataset to detect chirps on.
        logger: logging.Logger
            The logger to log to a logfile.
        """
        # Basic setup
        self.cfg = cfg
        self.data = data
        self.logger = logger
        self.bbox_df = chirp_height_width_to_bbox(
            data=data,
            nfft=freqres_to_nfft(cfg.spec.freq_res, data.grid.samplerate),
        )
        self.img_path = output_path / "images"
        self.label_path = output_path / "labels"
        self.assignment_path = output_path / "assignment"

        # Batch and windowing setup
        self.parser = ArrayParser(
            length=data.grid.shape[0],
            samplingrate=data.grid.samplerate,
            batchsize=cfg.spec.batch_size,
            windowsize=cfg.spec.time_window,
            overlap=cfg.spec.spec_overlap,
            console=prog.console,
        )

        msg = "Intialized Converter."
        self.logger.info(msg)
        prog.console.log(msg)

    def convert(self: Self) -> None:
        """Convert wavetracker data to spectrogram snippets."""
        prog.console.rule("[bold green]Starting parser")
        dataframes = []
        assingment_x = []
        assignment_y = []
        bbox_counter = 0
        for i, batch_indices in enumerate(self.parser.batches):
            prog.console.rule(
                f"[bold green]Batch {i} of {len(self.parser.batches)}"
            )

            # STEP 0: Create metadata for each batch
            batch_metadata = [
                {
                    "recording": self.data.path.name,
                    "batch": i,
                    "window": j,
                    "indices": indices,
                    "frange": [],
                }
                for j, indices in enumerate(batch_indices)
            ]

            # STEP 1: Load the raw data as a batch
            batch_raw = [
                np.array(self.data.grid.rec[idxs[0] : idxs[1], :])
                for idxs in batch_indices
            ]

            # STEP 2: Compute the spectrograms for each raw data snippet
            with Timer(prog.console, "Compute spectrograms"):
                batch_metadata, specs, times, freqs = make_batch_specs(
                    batch_indices,
                    batch_metadata,
                    batch_raw,
                    self.data.grid.samplerate,
                    self.cfg,
                )

            # STEP 3: Use the chirp_params to estimate the bounding boxes
            # import matplotlib as mpl
            # import matplotlib.pyplot as plt
            # mpl.use("TkAgg")
            for i, (meta, spec, time, freq) in enumerate(
                zip(batch_metadata, specs, times, freqs)
            ):
                # fig, ax = plt.subplots()
                # ax.pcolormesh(time, freq, spec.cpu().numpy()[1])
                # get the boxes for the current time and freq range
                boxes = self.bbox_df[
                    (self.bbox_df["t1"] >= time[0])
                    & (self.bbox_df["t2"] <= time[-1])
                ].to_numpy()

                if len(boxes) == 0:
                    continue

                # extract the assignment training data
                x, y = extract_assignment_training_data(
                    data=self.data,
                    boxes=boxes,
                    spec=spec.cpu().numpy()[1],
                    times=time,
                    freqs=freq,
                )
                assingment_x.append(x)
                assignment_y.append(y)

                # correct overshooting freq bounds of bboxes
                maxf = np.max(freq)
                minf = np.min(freq)
                boxes[:, 1] = np.clip(boxes[:, 1], minf, maxf)
                boxes[:, 3] = np.clip(boxes[:, 3], minf, maxf)

                # get box fish ids
                idents = boxes[:, -1]

                # get box coordinates
                t1 = boxes[:, 0]
                f1 = boxes[:, 1]
                t2 = boxes[:, 2]
                f2 = boxes[:, 3]

                # import matplotlib as mpl
                # import matplotlib.pyplot as plt
                # from matplotlib.patches import Rectangle
                # mpl.use("TkAgg")
                # fig, ax = plt.subplots()
                # ax.pcolormesh(time, freq, spec.cpu().numpy()[1])
                # for x1, y1, x2, y2 in zip(t1, f1, t2, f2):
                #     ax.add_patch(
                #         Rectangle(
                #             (x1, y1),
                #             x2 - x1,
                #             y2 - y1,
                #             linewidth=1,
                #             edgecolor="w",
                #             facecolor="none",
                #         )
                #     )
                # plt.show()

                # convert to indices on the spectrogram
                x1 = reverse_float_index_interpolation(
                    boxes[:, 0], time, np.arange(len(time))
                )
                y1 = reverse_float_index_interpolation(
                    boxes[:, 1], freq, np.arange(len(freq))
                )
                x2 = reverse_float_index_interpolation(
                    boxes[:, 2], time, np.arange(len(time))
                )
                y2 = reverse_float_index_interpolation(
                    boxes[:, 3], freq, np.arange(len(freq))
                )

                # make relative to the spectrogram
                x1 /= spec.shape[-1]
                y1 /= spec.shape[-2]
                x2 /= spec.shape[-1]
                y2 /= spec.shape[-2]

                # convert to x, y, w, h
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1

                # flip y axis as y=0 is at the top
                y = 1 - y

                # make labels
                labels = np.ones(len(boxes), dtype=int)

                # make label txt
                txt = np.stack([labels, x, y, w, h], axis=1)

                # save the boxes to the label directory
                # filename is batch metadata
                filename = (
                    f"{batch_metadata[0]['recording']}_"
                    f"{batch_metadata[0]['batch']}_"
                    f"{batch_metadata[0]['window']}_"
                    f"{i}.txt"
                )
                np.savetxt(
                    self.label_path / filename,
                    txt,
                    fmt=["%d", "%f", "%f", "%f", "%f"],
                )

                # convert the spec to PIL image
                img = numpy_to_pil(spec.cpu().numpy()[1])

                # save the image to the image directory
                filename = (
                    f"{batch_metadata[0]['recording']}_"
                    f"{batch_metadata[0]['batch']}_"
                    f"{batch_metadata[0]['window']}_"
                    f"{i}.png"
                )
                img.save(self.img_path / filename)

                dataframe = pd.DataFrame(
                    {
                        "recording": [
                            meta["recording"] for _ in range(len(boxes))
                        ],
                        "batch": [meta["batch"] for _ in range(len(boxes))],
                        "window": [meta["window"] for _ in range(len(boxes))],
                        "spec": i,
                        "box_ident": idents,
                        "raw_indices": [
                            meta["indices"] for _ in range(len(boxes))
                        ],
                        "freq_range": [
                            meta["frange"] for _ in range(len(boxes))
                        ],
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "t1": t1,
                        "f1": f1,
                        "t2": t2,
                        "f2": f2,
                        "score": np.ones(len(x1), dtype=int),
                    }
                )
                dataframes.append(dataframe)

            del specs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # for x in assingment_x:
        #     print(f"Assignment x shape: {x.shape}")

        assignment_x = np.concatenate(assingment_x)
        assignment_y = np.concatenate(assignment_y)

        assignment_x_name = self.data.path.name + "_data.npy"
        assignment_y_name = self.data.path.name + "_labels.npy"

        np.save(self.assignment_path / assignment_x_name, assignment_x)
        np.save(self.assignment_path / assignment_y_name, assignment_y)

        dataframes = pd.concat(dataframes)
        dataframes = dataframes.reset_index(drop=True)
        dataframes.to_csv(self.data.path / "labels.csv", index=False)
