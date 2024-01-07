"""Detect chirps on a spectrogram."""

import logging
import pathlib
import shutil
import time
import uuid
from typing import Self, Tuple, List, Any
from abc import ABC, abstractmethod

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gridtools.datasets import load, subset
from gridtools.datasets.models import Dataset
from gridtools.utils.spectrograms import (
    freqres_to_nfft,
    overlap_to_hoplen,
    compute_spectrogram,
    to_decibel
)
from matplotlib.patches import Rectangle
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)

from torchvision.ops import nms
from gridtools.preprocessing.preprocessing import interpolate_tracks

from .convert_data import make_file_tree, numpy_to_pil
from .detect_chirps import spec_to_image, float_index_interpolation
from .models.utils import get_device, load_fasterrcnn
from .utils.configfiles import Config, load_config
from .utils.logging import make_logger, Timer
from .utils.signal_processing import (
    compute_sum_spectrogam,
    make_spectrogram_axes,
)
from .dataset_parser import ArrayParser

# Use non-gui backend for matplotlib to
# avoid memory leaks
mpl.use("Agg")

# initialize the progress bar
prog = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
)


def get_faster_rcnn(cfg: Config) -> torch.nn.Module:
    """Load the trained faster R-CNN model."""
    model = load_fasterrcnn(num_classes=len(cfg.hyper.classes))
    device = get_device()
    checkpoint = torch.load(
        f"{cfg.hyper.modelpath}/model.pt",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def make_batch_specs(
    indices: List, handles: List, batch: List, samplerate: float, cfg: Config
    ) -> List:
    """Compute the spectrograms for a batch of windows."""
    batch = np.swapaxes(batch, 1, 2)
    nfft = freqres_to_nfft(
        freq_res=cfg.spec.freq_res,
        samplingrate=samplerate
    )
    hop_length = overlap_to_hoplen(
        nfft=nfft,
        overlap=cfg.spec.overlap_frac
    )
    batch_specs = [
        compute_spectrogram(
            data=signal,
            samplingrate=samplerate,
            nfft=nfft,
            hop_length=hop_length
        )[0] for signal in batch
    ]

    batch_specs_decibel = [
        to_decibel(spec) for spec in batch_specs
    ]
    # batch_specs_decible_cpu = [spec for spec in batch_specs_decibel]
    batch_sum_specs = [
        torch.sum(spec, dim=0) for spec in batch_specs_decibel
    ]
    axes = [
        make_spectrogram_axes(
            start=idxs[0],
            stop=idxs[1],
            nfft=nfft,
            hop_length=hop_length,
            samplerate=samplerate
        ) for idxs in indices
    ]
    batch_specs = [
        (spec, *ax) for spec, ax in zip(batch_sum_specs, axes)
    ]
    # Add the name to each spec tuple
    batch_specs = [
        (name, *spec) for name, spec in zip(
            handles, batch_specs
        )
    ]

    # Tile the spectrograms y-axis
    sliced_specs = tile_batch_specs(batch_specs, cfg)

    # Split the list into specs and axes
    handles, specs, times, freqs = zip(*sliced_specs)

    # Convert the spec tensors to mimic PIL images
    images = [spec_to_image(spec) for spec in specs]

    return handles, images, times, freqs


def tile_batch_specs(batch_specs: List, cfg: Config) -> List:
    """Tile the spectrograms of a batch."""
    freq_ranges = [(0, 1000), (500, 1500), (1000, 2000)]
    sliced_specs = []
    for start, end in freq_ranges:
        start_idx = np.argmax(batch_specs[0][3] >= start)
        end_idx = np.argmax(batch_specs[0][3] >= end)
        suffix = f"_frange-{start}-{end}"
        sliced_specs.extend([
            (name + suffix, spec[start_idx:end_idx, :], time, freq[start_idx:end_idx])
            for name, spec, time, freq in batch_specs
        ])

    return sliced_specs


def pixel_to_timefreq(
        boxes: np.ndarray, time: np.ndarray, freq: np.ndarray
    ) -> np.ndarray:
    """Convert the pixel coordinates of a box to time and frequency."""
    freq_indices = np.arange(len(freq))
    time_indices = np.arange(len(time))

    # convert the pixel coordinates to time and frequency
    t1 = float_index_interpolation(boxes[:, 0], time_indices, time)
    f1 = float_index_interpolation(boxes[:, 1], freq_indices, freq)
    t2 = float_index_interpolation(boxes[:, 2], time_indices, time)
    f2 = float_index_interpolation(boxes[:, 3], freq_indices, freq)

    # turn into same shape as input boxes
    t1 = np.expand_dims(t1, axis=1)
    f1 = np.expand_dims(f1, axis=1)
    t2 = np.expand_dims(t2, axis=1)
    f2 = np.expand_dims(f2, axis=1)

    return np.concatenate([t1, f1, t2, f2], axis=1)


def convert_detections(
    detections: List,
    names: List,
    times: List,
    freqs: List,
    cfg: Config,
) -> pd.DataFrame():
    """Convert the detections to a pandas DataFrame."""
    dataframes = []
    for i in range(len(detections)):

        # get the boxes and scores for the current spectrogram
        boxes = detections[i]["boxes"] # bbox coordinates in pixels
        scores = detections[i]["scores"] # confidence scores
        batch_spec_index = np.ones(len(boxes)) * i

        # discard boxes with low confidence
        boxes = boxes[scores >= cfg.det.threshold]
        batch_spec_index = batch_spec_index[scores >= cfg.det.threshold]
        scores = scores[scores >= cfg.det.threshold]

        # add the name to each box
        name = [names[i]] * len(boxes)

        # convert the boxes to time and frequency
        boxes_timefreq = pixel_to_timefreq(
            boxes=boxes,
            time=times[i],
            freq=freqs[i]
        )

        # put it all into a large dataframe
        dataframe = pd.DataFrame({
            "name": name,
            "batch_spec_index": batch_spec_index,
            "x1": boxes[:, 0],
            "y1": boxes[:, 1],
            "x2": boxes[:, 2],
            "y2": boxes[:, 3],
            "t1": boxes_timefreq[:, 0],
            "f1": boxes_timefreq[:, 1],
            "t2": boxes_timefreq[:, 2],
            "f2": boxes_timefreq[:, 3],
            "score": scores
        })
        dataframes.append(dataframe)
    out_df = pd.concat(dataframes)
    return out_df.reset_index(drop=True)


def non_max_suppression(
    chirp_df: pd.DataFrame,
    overlapthresh: float,
) -> List:
    """Non maximum suppression with the torchvision nms implementation."""
    # convert the dataframe to a list of boxes
    boxes = chirp_df[["t1", "f1", "t2", "f2"]].to_numpy()

    # convert the boxes to the format expected by torchvision
    boxes = torch.tensor(boxes, dtype=torch.float32).to(get_device())

    # convert the scores to the format expected by torchvision
    scores = torch.tensor(
        chirp_df["score"].to_numpy(), dtype=torch.float32
    ).to(get_device())

    # perform non-maximum suppression
    indices = nms(boxes, scores, overlapthresh)

    # retrieve the indices from the gpu if necessary
    if indices.is_cuda:
        indices = indices.cpu()

    return indices.tolist()


def detect_cli(input_path: pathlib.Path, make_training_data: bool) -> None:
    """Terminal interface for the detection function.

    Parameters
    ----------
    - `path` : `str`

    Returns
    -------
    - `None`
    """
    logger = make_logger(__name__, input_path / "chirpdetector.log")
    datasets = [folder for folder in input_path.iterdir() if folder.is_dir()]
    confpath = input_path / "chirpdetector.toml"

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

    # detect chirps in all datasets in the specified path
    # and show a progress bar
    model = get_faster_rcnn(config)
    model.to(get_device()).eval()
    predictor = FasterRCNN(
        model=model
    )
    with prog:
        task = prog.add_task("Detecting chirps...", total=len(datasets))
        for dataset in datasets:
            data = load(dataset)
            data = interpolate_tracks(data)
            cpd = ChirpDetector(
                cfg=config,
                data=data,
                model=predictor,
                logger=logger,
            )
            cpd.detect()
            exit()

        prog.update(task, completed=len(datasets))


class AbstractDetectionModel(ABC):
    """Abstract base class for model wrappers.

    Standard output format:
    [
        {
            "boxes": torch.Tensor,
            "scores": torch.Tensor,
        },
        ...
    ]
    One dict for each spectrogram in the batch.
    Boxes follow the format [x1, y1, x2, y2] in pixels.
    """

    def __init__(self: Self, model: torch.nn.Module) -> None:
        """Initialize the model wrapper."""
        self.model = model

    def predict(self: Self, batch: List) -> List:
        """Predict boxes for a batch of spectrograms."""
        output = self.predictor(batch)
        return self.convert_to_standard_format(output)

    @abstractmethod
    def predictor(self: Self, batch: List) -> List:
        """Predict boxes for a batch of spectrograms."""
        pass

    @abstractmethod
    def convert_to_standard_format(self: Self, model_output: List) -> List:
        """Convert the model output to a standardized format."""
        pass


class FasterRCNN(AbstractDetectionModel):
    """Wrapper for the Faster R-CNN model."""

    def predictor(self: Self, batch: List) -> List:
        """Predict boxes for a batch of spectrograms."""
        with torch.no_grad():
            return self.model(batch)

    def convert_to_standard_format(self: Self, model_output: List) -> List:
        """Convert the model output to a standardized format."""
        output = []
        for i in range(len(model_output)):
            boxes = model_output[i]["boxes"].detach().cpu().numpy()
            scores = model_output[i]["scores"].detach().cpu().numpy()
            output.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                }
            )
        return output


class YOLONAS(AbstractDetectionModel):
    """Wrapper for the deci-ai YOLO-NAS model."""

    def convert_to_standard_format(self: Self, model_output: List) -> List:
        """Convert the model output to a standardized format."""
        pass


class AbstractBoxAssigner(ABC):
    """Wrapper around different box assignment methods."""

    def __init__(
        self: Self,
        batch_indices: List,
        batch_specs: List,
        batch_times: List,
        batch_freqs: List,
        batch_detections: pd.DataFrame,
        data: Dataset,
        cfg: Config,
    ) -> None:
        """Initialize the BoxAssigner."""
        self.batch_indices = batch_indices
        self.batch_specs = batch_specs
        self.batch_times = batch_times
        self.batch_freqs = batch_freqs
        self.batch_detections = batch_detections
        self.data = data
        self.cfg = cfg

    @abstractmethod
    def assign(self: Self) -> pd.DataFrame:
        """Assign boxes to tracks."""
        pass


class TroughBoxAssigner(AbstractBoxAssigner):
    """Assign boxes to tracks based on power troughs."""

    def assign(self: Self) -> pd.DataFrame:
        """Assign boxes to tracks by troughts in power.

        Assignment by checking which of the tracks has a trough in spectrogram
        power in the spectrogram bbox.
        """
        subdata = self.data

        # retrieve frequency and time for each fish id
        tracks = [
            subdata.track.freqs[subdata.track.idents == ident]
            for ident in np.unique(subdata.track.ids)
        ]
        times = [
            subdata.track.times[subdata.track.indices[subdata.track.idents == ident]]
            for ident in np.unique(subdata.track.ids)
        ]
        ids = np.unique(subdata.track.ids)

        import matplotlib as mpl
        mpl.use("TkAgg")
        # plt.plot(subdata.track.times)
        for t, f in zip(times, tracks):
            plt.plot(t, f)
        plt.show()

        for i, (spec, time, freq) in enumerate(zip(
            self.batch_specs, self.batch_times, self.batch_freqs
        )):

            spec = spec.cpu().numpy()
            spec_boxes = self.batch_detections[
                self.batch_detections["batch_spec_index"] == i
            ][["t1", "f1", "t2", "f2"]].to_numpy()
            plt.pcolormesh(time, freq, spec[0])

            for box in spec_boxes:
                x1, y1, x2, y2 = box
                plt.plot([x1, x2], [y1, y2], color="red", lw=1, ls="--")

            for track_id, t, f in zip(ids, times, tracks):
                continue
            plt.show()



class ChirpDetector:
    """Parse a grid dataset into batches."""

    def __init__(
        self: Self,
        cfg: Config,
        data: Dataset,
        model: AbstractDetectionModel,
        logger: logging.Logger
    ) -> None:
        """Initialize the ChirpDetector.

        Parameters
        ----------
        cfg : Config
            Configuration file.
        data : Dataset
            Dataset to detect chirps on.
        model : torch.nn.Module
            Model to use for detection.
        logger: logging.Logger
            The logger to log to a logfile.
        """
        # Basic setup
        self.cfg = cfg
        self.data = data
        self.logger = logger
        self.model = model

        # Batch and windowing setup
        self.parser = ArrayParser(
            length=data.grid.shape[0],
            samplingrate=data.grid.samplerate,
            batchsize=cfg.spec.batch_size,
            windowsize=cfg.spec.time_window,
            overlap=cfg.spec.spec_overlap,
            console=prog.console
        )

        msg = "Intialized ChirpDetector."
        self.logger.info(msg)
        prog.console.log(msg)

    def detect(self: Self) -> None:
        """Detect chirps on the dataset."""
        prog.console.rule("[bold green]Starting parser")

        prog.console.log("Interpolating wavetracker tracks")

        for i, batch_indices in enumerate(self.parser.batches):

            # Generate a string handle for each spectrogram
            # e.g. for filenames in case of saving
            batch_handles = [
                f"{self.data.path.name}_batch-{i}_window-{j}"
                for j in range(len(batch_indices))
            ]

            # STEP 1: Load the raw data as a batch
            batch_raw = [
                np.array(self.data.grid.rec[idxs[0] : idxs[1], :])
                for idxs in batch_indices
            ]

            # STEP 2: Compute the spectrograms for each raw data snippet
            with Timer(prog.console, "Compute spectrograms"):
                handles, specs, times, freqs = make_batch_specs(
                    batch_indices,
                    batch_handles,
                    batch_raw,
                    self.data.grid.samplerate,
                    self.cfg
                )

            # STEP 3: Predict boxes for each spectrogram
            with Timer(prog.console, "Detect chirps"):
                predictions = self.model.predict(specs)

            # STEP 4: Convert pixel values to time and frequency
            # and save everything in a dataframe
            with Timer(prog.console, "Convert detections"):
                batch_df = convert_detections(
                    predictions,
                    handles,
                    times,
                    freqs,
                    self.cfg
                )

            # STEP 5: Remove overlapping boxes by non-maximum suppression
            with Timer(prog.console, "Non-maximum suppression"):
                good_box_indices = non_max_suppression(
                    batch_df,
                    overlapthresh=0.5,
                )
                nms_batch_df = batch_df.iloc[good_box_indices]

            # STEP 6: Assign boxes to wavetracker tracks
            # TODO: Implement this
            assigner = TroughBoxAssigner(
                batch_indices=batch_indices,
                batch_specs=specs,
                batch_times=times,
                batch_freqs=freqs,
                batch_detections=nms_batch_df,
                data=self.data,
                cfg=self.cfg,
            )
            assigner.assign()

            import matplotlib as mpl
            mpl.use("TkAgg")
            fig, ax = plt.subplots()
            for spec, time, freq in zip(specs, times, freqs):
                spec = spec.cpu().numpy()
                ax.pcolormesh(time, freq, spec[0])
                ax.axvline(time[0], color="white", lw=1, ls="--")
                ax.axvline(time[-1], color="white", lw=1, ls="--")
                ax.axhline(freq[0], color="white", lw=1, ls="--")
                ax.axhline(freq[-1], color="white", lw=1, ls="--")

            for i in range(len(batch_df)):
                x1 = batch_df["t1"].iloc[i]
                y1 = batch_df["f1"].iloc[i]
                x2 = batch_df["t2"].iloc[i]
                y2 = batch_df["f2"].iloc[i]
                score = batch_df["score"].iloc[i]
                ax.add_patch(
                    Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        color="red",
                        lw=1,
                    )
                )
                ax.text(
                    x1,
                    y1,
                    f"{score:.2f}",
                    color="red",
                    fontsize=8,
                    ha="left",
                    va="bottom",
                )

            for i in range(len(nms_batch_df)):
                x1 = nms_batch_df["t1"].iloc[i]
                y1 = nms_batch_df["f1"].iloc[i]
                x2 = nms_batch_df["t2"].iloc[i]
                y2 = nms_batch_df["f2"].iloc[i]
                score = nms_batch_df["score"].iloc[i]
                ax.add_patch(
                    Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        color="green",
                        lw=1,
                    )
                )
                ax.text(
                    x1,
                    y1,
                    f"{score:.2f}",
                    color="green",
                    fontsize=8,
                    ha="left",
                    va="bottom",
                )

            plt.show()

            # TODO: Save the output from above to a file







