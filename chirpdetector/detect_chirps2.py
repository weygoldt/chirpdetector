"""Detect chirps on a spectrogram."""

import logging
import pathlib
from abc import ABC, abstractmethod
from typing import List, Self, Tuple
import gc

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gridtools.datasets import load
from gridtools.datasets.models import Dataset
from gridtools.preprocessing.preprocessing import interpolate_tracks
from gridtools.utils.spectrograms import (
    compute_spectrogram,
    freqres_to_nfft,
    overlap_to_hoplen,
    to_decibel,
)
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)
from torchvision.ops import nms
from scipy.signal import find_peaks

from .dataset_parser import ArrayParser
from .detect_chirps import float_index_interpolation, spec_to_image
from .models.utils import get_device, load_fasterrcnn
from .utils.configfiles import Config, load_config
from .utils.logging import Timer, make_logger
from .utils.signal_processing import (
    make_spectrogram_axes,
)

# Use non-gui backend for matplotlib to avoid memory leaks
mpl.use("Agg")

# initialize the progress bar
prog = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
)


def get_trained_faster_rcnn(cfg: Config) -> torch.nn.Module:
    """Load the trained faster R-CNN model.

    Parameters
    ----------
    cfg : Config
        The configuration file.

    Returns
    -------
    model : torch.nn.Module
        The trained model.
    """
    model = load_fasterrcnn(num_classes=len(cfg.hyper.classes))
    device = get_device()
    checkpoint = torch.load(
        f"{cfg.hyper.modelpath}/model.pt",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def make_batch_specs(
    indices: List,
    metadata: List,
    batch: np.ndarray,
    samplerate: float,
    cfg: Config
) -> Tuple[List, List, List, List]:
    """Compute the spectrograms for a batch of windows.

    Gets the snippets of raw data for one batch and computes the sum
    spectrogram for each snippet. The sum spectrogram is then converted
    to decibel and the spectrograms are tiled along the frequency axis
    and converted into 0-255 uint8 images.

    Parameters
    ----------
    indices : List
        The indices of the raw data snippets in the original recording.
    metadata : List
        The metadata for each snippet.
    batch : np.ndarray
        The raw data snippets.
    samplerate : float
        The sampling rate of the raw data.
    cfg : Config
        The configuration file.

    Returns
    -------
    metadata : List
        The metadata for each snippet.
    images : List
        The spectrograms as images.
    times : List
        The time axis for each spectrogram.
    freqs : List
        The frequency axis for each spectrogram.
    """
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
    # Add the metadata to each spec tuple
    batch_specs = [
        (meta, *spec) for meta, spec in zip(
            metadata, batch_specs
        )
    ]

    # Tile the spectrograms y-axis
    sliced_specs = tile_batch_specs(batch_specs, cfg)

    # Split the list into specs and axes
    metadata, specs, times, freqs = zip(*sliced_specs)

    # Convert the spec tensors to mimic PIL images
    images = [spec_to_image(spec) for spec in specs]

    return metadata, images, times, freqs


def tile_batch_specs(batch_specs: List, cfg: Config) -> List:
    """Tile the spectrograms of a batch.

    Parameters
    ----------
    batch_specs : List
        The spectrograms of a batch.
    cfg : Config
        The configuration file.

    Returns
    -------
    sliced_specs : List
        The tiled spectrograms.
    """
    freq_ranges = [(0, 1000), (500, 1500), (1000, 2000)]
    sliced_specs = []
    for i, (start, end) in enumerate(freq_ranges):
        start_idx = np.argmax(batch_specs[0][3] >= start)
        end_idx = np.argmax(batch_specs[0][3] >= end)
        for meta, spec, time, freq in batch_specs:
            newmeta = meta.copy()
            newmeta["frange"] = (start, end)
            sliced_specs.append(
                (
                    newmeta,
                    spec[start_idx:end_idx, :],
                    time,
                    freq[start_idx:end_idx]
                )
            )
    return sliced_specs


def pixel_box_to_timefreq(
        boxes: np.ndarray, time: np.ndarray, freq: np.ndarray
    ) -> np.ndarray:
    """Convert the pixel coordinates of a box to time and frequency.

    Parameters
    ----------
    boxes : np.ndarray
        The boxes to convert.
    time : np.ndarray
        The time axis of the spectrogram.
    freq : np.ndarray
        The frequency axis of the spectrogram.

    Returns
    -------
    boxes_timefreq : np.ndarray
        The converted boxes.
    """
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
    bbox_ids: List,
    metadata: List,
    times: List,
    freqs: List,
    cfg: Config,
) -> pd.DataFrame:
    """Convert the detected bboxes to a pandas DataFrame including metadata.

    Parameters
    ----------
    detections : List
        The detections for each spectrogram in the batch.
    metadata : List
        The metadata for each spectrogram in the batch.
    times : List
        The time axis for each spectrogram in the batch.
    freqs : List
        The frequency axis for each spectrogram in the batch.
    cfg : Config
        The configuration file.

    Returns
    -------
    out_df : pd.DataFrame
        The converted detections.
    """
    dataframes = []
    for i in range(len(detections)):

        # get the boxes and scores for the current spectrogram
        boxes = detections[i]["boxes"] # bbox coordinates in pixels
        scores = detections[i]["scores"] # confidence scores
        idents = bbox_ids[i] # unique ids for each bbox
        batch_spec_index = np.ones(len(boxes)) * i

        # discard boxes with low confidence
        boxes = boxes[scores >= cfg.det.threshold]
        idents = idents[scores >= cfg.det.threshold]
        batch_spec_index = batch_spec_index[scores >= cfg.det.threshold]
        scores = scores[scores >= cfg.det.threshold]

        # convert the boxes to time and frequency
        boxes_timefreq = pixel_box_to_timefreq(
            boxes=boxes,
            time=times[i],
            freq=freqs[i]
        )

        # put it all into a large dataframe
        dataframe = pd.DataFrame({
            "recording": [metadata[i]["recording"] for _ in range(len(boxes))],
            "batch": [metadata[i]["batch"] for _ in range(len(boxes))],
            "window": [metadata[i]["window"] for _ in range(len(boxes))],
            "spec": batch_spec_index,
            "box_ident": idents,
            "raw_indices": [metadata[i]["indices"] for _ in range(len(boxes))],
            "freq_range": [metadata[i]["frange"] for _ in range(len(boxes))],
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


def dataframe_nms(
    chirp_df: pd.DataFrame,
    overlapthresh: float,
) -> List:
    """Non maximum suppression with the torchvision nms implementation.

    ...but with a pandas dataframe as input.

    Parameters
    ----------
    chirp_df : pd.DataFrame
        The dataframe with the detections.
    overlapthresh : float
        The overlap threshold for non-maximum suppression.

    Returns
    -------
    indices : List
        The indices of the boxes to keep after non-maximum suppression.
    """
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

    # Get the box predictor
    model = get_trained_faster_rcnn(config)
    model.to(get_device()).eval()
    predictor = FasterRCNN(
        model=model
    )

    # get the box assigner
    assigner = TroughBoxAssigner(config)

    with prog:
        task = prog.add_task("Detecting chirps...", total=len(datasets))
        for dataset in datasets:
            prog.console.log(f"Detecting chirps in {dataset.name}")
            data = load(dataset)
            data = interpolate_tracks(data, samplerate=120)
            cpd = ChirpDetector(
                cfg=config,
                data=data,
                detector=predictor,
                assigner=assigner,
                logger=logger,
            )
            cpd.detect()

            del data
            del cpd
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            prog.advance(task, 1)
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
    We dont need labels as we only detect one class.
    """

    def __init__(self: Self, model: torch.nn.Module) -> None:
        """Initialize the model wrapper."""
        self.model = model

    def predict(self: Self, batch: List) -> List:
        """Predict boxes for a batch of spectrograms."""
        output = self.predictor(batch)
        result = self.convert_to_standard_format(output)
        del output
        gc.collect()
        return result

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
            labels = model_output[i]["labels"].detach().cpu().numpy()

            boxes = boxes[labels == 1]
            scores = scores[labels == 1]
            output.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                }
            )
        return output


class AbstractBoxAssigner(ABC):
    """Default wrapper around different box assignment methods."""

    def __init__( # noqa
        self: Self,
        cfg: Config,
    ) -> None:
        """Initialize the BoxAssigner."""
        self.cfg = cfg

    @abstractmethod
    def assign( # noqa
        self: Self,
        batch_specs: List,
        batch_times: List,
        batch_freqs: List,
        batch_detections: pd.DataFrame,
        data: Dataset,
    ) -> pd.DataFrame:
        """Assign boxes to tracks."""
        pass


class TroughBoxAssigner(AbstractBoxAssigner):
    """Assign boxes to tracks based on power troughs.

    The idea is to assign boxes to tracks by checking which of the tracks
    has a trough in spectrogram power in the spectrogram bbox.

    This is done by combining the information included the chirp detection
    spectrogram, which has a fine temporal resolution, and the approximation
    of the fishes fundamental frequencies, which have fine frequency
    resolution.

    To do this, I take the chirp detection spectrogram and extract the powers
    that lie below a frequency track of a fish for each bounding box. During
    chirps, there usually is a trough in power. If a fish did not chirp but the
    chirp of another fish crosses its frequency band, there should be a peak
    in power as the signal of the chirper and the signal of the non-chirper
    add up. In short: Chirping fish have a trough in power, non-chirping fish
    have a peak in power (in the ideal world).

    This method uses this notion and just assigns chirps by peak detection.
    """

    def assign( #noqa
        self: Self,
        batch_specs: List,
        batch_times: List,
        batch_freqs: List,
        batch_detections: pd.DataFrame,
        data: Dataset,
    ) -> pd.DataFrame:
        """Assign boxes to tracks by troughts in power.

        Assignment by checking which of the tracks has a trough in spectrogram
        power in the spectrogram bbox.
        """
        padding = 0.05 # seconds before and after bbox bounds to ad

        # retrieve frequency and time for each fish id
        track_ids = np.unique(data.track.ids)
        track_freqs = [
            data.track.freqs[data.track.idents == ident]
            for ident in track_ids
        ]
        track_times = [
            data.track.times[data.track.indices[
            data.track.idents == ident
        ]]
            for ident in track_ids
        ]

        # Plan: First go though each box and check if there is a track in them.
        # If there is, compute the power in the box for each track.
        # If, not skip the box.
        # Then, check if there is a trough in power for each track.
        # If there is, assign the box to the track.
        # If not, skip the box.

        assigned_ids = []

        for i in range(len(batch_detections)):

            # get the current box
            box = batch_detections.iloc[i]

            # get the time and frequency indices for the box
            t1 = box["t1"]
            f1 = box["f1"]
            t2 = box["t2"]
            f2 = box["f2"]
            spec_idx = box["spec"].astype(int)

            # get the power in the box for each track
            box_powers = []
            box_power_times = []
            box_power_ids = []
            # import matplotlib as mpl
            # mpl.use("TkAgg")
            # fig, (ax1, ax2) = plt.subplots(2,1)
            # colors = ["red", "green", "blue", "yellow", "orange", "purple"]
            # plot the box and the track
            # ax1.pcolormesh(
            #     batch_times[spec_idx],
            #     batch_freqs[spec_idx],
            #     batch_specs[spec_idx].cpu().numpy()[0]
            # )
            # ax1.add_patch(
            #     Rectangle(
            #         (t1, f1),
            #         t2 - t1,
            #         f2 - f1,
            #         fill=False,
            #         color="white",
            #         lw=1,
            #     )
            # )
            for j, (track_id, track_freq, track_time) in enumerate(zip(
                track_ids, track_freqs, track_times
            )):
                # get the time indices for the track
                # as the dataset is interpolated, time and freq indices
                # the same
                track_t1_idx = np.argmin(np.abs(track_time - (t1 - padding)))
                track_t2_idx = np.argmin(np.abs(track_time - (t2 + padding)))

                # get the track snippet in the current bbox
                track_freq_snippet = track_freq[track_t1_idx:track_t2_idx]
                track_time_snippet = track_time[track_t1_idx:track_t2_idx]

                # Check if the frequency values of the snippet are
                # inside the bbox
                if (np.min(track_freq_snippet) > f2) or \
                    (np.max(track_freq_snippet) < f1):
                    # the track does not lie in the box
                    continue

                # Now get the power on spec underneath the track
                # and plot it
                spec_powers = batch_specs[spec_idx].cpu().numpy()[0]
                spec_times = batch_times[spec_idx]
                spec_freqs = batch_freqs[spec_idx]

                spec_t1_idx = np.argmin(np.abs(spec_times - (t1 - padding)))
                spec_t2_idx = np.argmin(np.abs(spec_times - (t2 + padding)))

                spec_powers = spec_powers[:, spec_t1_idx:spec_t2_idx]
                spec_times = spec_times[spec_t1_idx:spec_t2_idx]

                spec_f_indices = [
                    np.argmin(np.abs(spec_freqs - freq))
                    for freq in track_freq_snippet
                ]

                spec_powers = [
                    spec_powers[f_idx, t_idx] for f_idx, t_idx in zip(
                        spec_f_indices, range(len(spec_times))
                    )
                ]

                # store the powers
                box_powers.append(spec_powers)
                box_power_times.append(spec_times)
                box_power_ids.append(track_id)
                # ax1.plot(track_time_snippet, track_freq_snippet, color=colors[j])

            # shift the track power baseline to same level
            starts = [power[0] for power in box_powers]
            box_powers = [
                power - start for power, start in zip(box_powers, starts)
            ]

            # detect peaks in the power
            ids = []
            costs = []
            for j, (power, time, track_id) in enumerate(zip(
                box_powers, box_power_times, box_power_ids
            )):
                peaks, props = find_peaks(-power, prominence=0)
                proms = props["prominences"]
                if len(proms) == 0:
                    # no peaks found
                    continue

                # takes the highest peak
                peak = peaks[np.argmax(proms)]
                prom = proms[np.argmax(proms)]

                # ax2.plot(power, label=track_id, color=colors[j])
                # ax2.plot(peak, power[peak], "o", color=colors[j])

                # Compute peak distance to box center
                box_center = (t1 + t2) / 2
                peak_dist = np.abs(box_center - time[peak])

                # cost is high when peak prominence is low and peak is far away
                # from box center
                cost = (1 / prom) * peak_dist

                # plot
                # ax2.text(
                #     peak,
                #     power[peak],
                #     f"{cost:.2f}",
                #     fontsize=8,
                #     ha="left",
                #     va="bottom",
                # )
                ids.append(track_id)
                costs.append(cost)

            # assign the box to the track with the lowest cost
            if len(costs) != 0:
                best_id = ids[np.argmin(costs)]
                assigned_ids.append(best_id)
                # print(best_id)
                # print(colors[np.argmin(costs)])
            else:
                best_id = np.nan
                assigned_ids.append(best_id)
            # plt.close()

            # print("done")
            # print(len(assigned_ids))
            # print(len(batch_detections))

        batch_detections.loc[:, "track_id"] = assigned_ids

        # drop all boxes that were not assigned
        batch_detections = batch_detections.dropna()

        return batch_detections


class ChirpDetector:
    """Parse a grid dataset into batches."""

    def __init__(
        self: Self,
        cfg: Config,
        data: Dataset,
        detector: AbstractDetectionModel,
        assigner: AbstractBoxAssigner,
        logger: logging.Logger
    ) -> None:
        """Initialize the ChirpDetector.

        Parameters
        ----------
        cfg : Config
            Configuration file.
        data : Dataset
            Dataset to detect chirps on.
        detector: AbstractDetectionModel
            Model to use for detection.
        assigner: AbstractBoxAssigner
            Assigns bboxes to frequency tracks.
        logger: logging.Logger
            The logger to log to a logfile.
        """
        # Basic setup
        self.cfg = cfg
        self.data = data
        self.logger = logger
        self.detector = detector
        self.assigner = assigner

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
        dataframes = []
        bbox_counter = 0
        for i, batch_indices in enumerate(self.parser.batches):

            batch_metadata = [{
                "recording": self.data.path.name,
                "batch": i,
                "window": j,
                "indices": indices,
                "frange": [],
            } for j, indices in enumerate(batch_indices)]

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
                    self.cfg
                )

            # STEP 3: Predict boxes for each spectrogram
            with Timer(prog.console, "Detect chirps"):
                predictions = self.detector.predict(specs)

            # STEP 4: Convert pixel values to time and frequency
            # and save everything in a dataframe
            with Timer(prog.console, "Convert detections"):
                # give every bbox a unique identifier
                # first get total number of new bboxes
                n_bboxes = 0
                for prediction in predictions:
                    n_bboxes += len(prediction["boxes"])
                # then create the ids
                bbox_ids = np.arange(bbox_counter, bbox_counter + n_bboxes)
                bbox_counter += n_bboxes
                # now split the bbox ids in sublists with same length as
                # detections for each spec
                split_ids = []
                for j, prediction in enumerate(predictions):
                    sub_ids = bbox_ids[:len(prediction["boxes"])]
                    bbox_ids = bbox_ids[len(prediction["boxes"]):]
                    split_ids.append(sub_ids)
                batch_df = convert_detections(
                    predictions,
                    split_ids,
                    batch_metadata,
                    times,
                    freqs,
                    self.cfg
                )

            # STEP 5: Remove overlapping boxes by non-maximum suppression
            with Timer(prog.console, "Non-maximum suppression"):
                good_box_indices = dataframe_nms(
                    batch_df,
                    overlapthresh=0.5,
                )
                nms_batch_df = batch_df.iloc[good_box_indices]

            # STEP 6: Assign boxes to wavetracker tracks
            # TODO: Implement this
            with Timer(prog.console, "Assign boxes to wavetracker tracks"):
                assigned_batch_df = self.assigner.assign(
                    batch_specs=specs,
                    batch_times=times,
                    batch_freqs=freqs,
                    batch_detections=nms_batch_df,
                    data=self.data,
                )

            dataframes.append(assigned_batch_df)
            del specs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Save the dataframe
        dataframes = pd.concat(dataframes)
        dataframes = dataframes.reset_index(drop=True)
        # sort by t1
        dataframes = dataframes.sort_values(by=["t1"])
        savepath = self.data.path / "chirpdetector_bboxes.csv"
        dataframes.to_csv(savepath, index=False)

            # import matplotlib as mpl
            # mpl.use("TkAgg")
            # fig, ax = plt.subplots()
            # for spec, time, freq in zip(specs, times, freqs):
            #     spec = spec.cpu().numpy()
            #     ax.pcolormesh(time, freq, spec[0])
            #     ax.axvline(time[0], color="white", lw=1, ls="--", alpha=0.2)
            #     ax.axvline(time[-1], color="white", lw=1, ls="--", alpha=0.2)
            #     ax.axhline(freq[0], color="white", lw=1, ls="--", alpha=0.2)
            #     ax.axhline(freq[-1], color="white", lw=1, ls="--", alpha=0.2)
            #
            # for i in range(len(batch_df)):
            #     x1 = batch_df["t1"].iloc[i]
            #     y1 = batch_df["f1"].iloc[i]
            #     x2 = batch_df["t2"].iloc[i]
            #     y2 = batch_df["f2"].iloc[i]
            #     score = batch_df["score"].iloc[i]
            #     ax.add_patch(
            #         Rectangle(
            #             (x1, y1),
            #             x2 - x1,
            #             y2 - y1,
            #             fill=False,
            #             color="white",
            #             lw=1,
            #             alpha=0.2
            #         )
            #     )
            #     ax.text(
            #         x1,
            #         y1,
            #         f"{score:.2f}",
            #         color="white",
            #         fontsize=8,
            #         ha="left",
            #         va="bottom",
            #         alpha=0.2,
            #     )
            #
            # for i in range(len(nms_batch_df)):
            #     x1 = nms_batch_df["t1"].iloc[i]
            #     y1 = nms_batch_df["f1"].iloc[i]
            #     x2 = nms_batch_df["t2"].iloc[i]
            #     y2 = nms_batch_df["f2"].iloc[i]
            #     score = nms_batch_df["score"].iloc[i]
            #     ax.add_patch(
            #         Rectangle(
            #             (x1, y1),
            #             x2 - x1,
            #             y2 - y1,
            #             fill=False,
            #             color="grey",
            #             lw=1,
            #             alpha=0.5,
            #         )
            #     )
            #     ax.text(
            #         x1,
            #         y1,
            #         f"{score:.2f}",
            #         color="grey",
            #         fontsize=8,
            #         ha="left",
            #         va="bottom",
            #         alpha=0.5,
            #     )
            #
            # colors = ["#1f77b4", "#ff7f0e"]
            # for j, track_id in enumerate(self.data.track.ids):
            #     track_freqs = self.data.track.freqs[self.data.track.idents == track_id]
            #     track_time = self.data.track.times[self.data.track.indices[self.data.track.idents == track_id]]
            #     ax.plot(track_time, track_freqs, color=colors[j], lw=1.5)
            #
            # patches = []
            # for j in range(len(assigned_batch_df)):
            #     t1 = assigned_batch_df["t1"].iloc[j]
            #     f1 = assigned_batch_df["f1"].iloc[j]
            #     t2 = assigned_batch_df["t2"].iloc[j]
            #     f2 = assigned_batch_df["f2"].iloc[j]
            #     assigned_id = assigned_batch_df["track_id"].iloc[j]
            #
            #     if assigned_id not in self.data.track.ids:
            #         continue
            #
            #     # print(assigned_id)
            #     # print(self.data.track.ids)
            #
            #     color = np.array(colors)[self.data.track.ids == assigned_id][0]
            #     # print(color)
            #
            #     patches.append(Rectangle(
            #             (t1, f1),
            #             t2 - t1,
            #             f2 - f1,
            #             fill=False,
            #             color=color,
            #             lw=1.5,
            #             alpha=1,
            #     ))
            #
            # ax.add_collection(PatchCollection(patches, match_original=True))
            # ax.set_xlim(np.min(times), np.max(times))
            # plt.show()
            # plt.close()
            #
            # # TODO: Save the output from above to a file







