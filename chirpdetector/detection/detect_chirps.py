"""Detect chirps on a spectrogram."""

import gc
import logging
import pathlib
from typing import List, Self

import numpy as np
import pandas as pd
import torch
from gridtools.datasets import load
from gridtools.datasets.models import Dataset
from gridtools.preprocessing.preprocessing import interpolate_tracks
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)

from chirpdetector.config import Config, load_config
from chirpdetector.datahandling.bbox_tools import (
    dataframe_nms,
    pixel_box_to_timefreq,
)
from chirpdetector.datahandling.dataset_parsing import (
    ArrayParser,
    make_batch_specs,
)
from chirpdetector.detection.assignment_models import (
    AbstractBoxAssigner,
    SpectrogramPowerTroughBoxAssignerMLP,
)
from chirpdetector.detection.detection_models import (
    AbstractDetectionModel,
    YOLOv8,
)
from chirpdetector.detection.visualization_functions import (
    plot_batch_detections,
    plot_raw_batch,
    plot_spec_tiling,
)
from chirpdetector.logging.logging import Timer, make_logger
from chirpdetector.models.mlp_assigner import load_trained_mlp
from chirpdetector.models.utils import get_device
from chirpdetector.models.yolov8_detector import load_finetuned_yolov8

# initialize the progress bar
prog = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
)


def assign_ffreqs_to_tracks(
    bbox_df: pd.DataFrame, data: Dataset
) -> pd.DataFrame:
    """Assign a bounding box to a track."""
    dist_threshold = 20  # Hz
    assigned_tracks = []
    for i in range(len(bbox_df)):
        chirptime = (bbox_df["t1"].iloc[i] + bbox_df["t2"].iloc[i]) / 2
        emitter_ffreq = bbox_df["emitter_eodf"].iloc[i]

        candidate_tracks = []
        candidate_freqs = []
        candidate_times = []
        for track_id in data.track.ids[~np.isnan(data.track.ids)]:
            time = data.track.times[
                data.track.indices[data.track.idents == track_id]
            ]
            freq = data.track.freqs[data.track.idents == track_id]
            closest_time = time[np.argmin(np.abs(time - chirptime))]
            closest_freq = freq[np.argmin(np.abs(time - chirptime))]
            candidate_tracks.append(track_id)
            candidate_freqs.append(closest_freq)
            candidate_times.append(closest_time)

        # find the closest track
        distances = np.abs(np.array(candidate_freqs) - emitter_ffreq)
        closest_track = candidate_tracks[np.argmin(distances)]

        if np.min(distances) > dist_threshold:
            assigned_tracks.append(np.nan)
        else:
            assigned_tracks.append(closest_track)
    assigned_tracks = np.array(assigned_tracks)
    bbox_df["track_id"] = assigned_tracks
    return bbox_df


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
        boxes = detections[i]["boxes"]  # bbox coordinates in pixels
        scores = detections[i]["scores"]  # confidence scores
        idents = bbox_ids[i]  # unique ids for each bbox
        batch_spec_index = np.ones(len(boxes)) * i

        # discard boxes with low confidence
        boxes = boxes[scores >= cfg.det.threshold]
        idents = idents[scores >= cfg.det.threshold]
        batch_spec_index = batch_spec_index[scores >= cfg.det.threshold]
        scores = scores[scores >= cfg.det.threshold]

        # convert the boxes to time and frequency
        boxes_timefreq = pixel_box_to_timefreq(
            boxes=boxes, time=times[i], freq=freqs[i]
        )

        # put it all into a large dataframe
        dataframe = pd.DataFrame(
            {
                "recording": [
                    metadata[i]["recording"] for _ in range(len(boxes))
                ],
                "batch": [metadata[i]["batch"] for _ in range(len(boxes))],
                "window": [metadata[i]["window"] for _ in range(len(boxes))],
                "spec": batch_spec_index,
                "box_ident": idents,
                "raw_indices": [
                    metadata[i]["indices"] for _ in range(len(boxes))
                ],
                "freq_range": [
                    metadata[i]["frange"] for _ in range(len(boxes))
                ],
                "x1": boxes[:, 0],
                "y1": boxes[:, 1],
                "x2": boxes[:, 2],
                "y2": boxes[:, 3],
                "t1": boxes_timefreq[:, 0],
                "f1": boxes_timefreq[:, 1],
                "t2": boxes_timefreq[:, 2],
                "f2": boxes_timefreq[:, 3],
                "score": scores,
            }
        )
        dataframes.append(dataframe)
    out_df = pd.concat(dataframes)
    return out_df.reset_index(drop=True)


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
    # det_model = load_finetuned_faster_rcnn(config)
    # det_model.to(get_device()).eval()
    # predictor = FasterRCNN(model=det_model)
    model = load_finetuned_yolov8(config)
    # exit()
    # model.to(get_device()).eval()
    predictor = YOLOv8(model=model)

    # get the box assigner
    ass_model = load_trained_mlp(config)
    ass_model.to(get_device()).eval()
    assigner = SpectrogramPowerTroughBoxAssignerMLP(ass_model)

    with prog:
        task = prog.add_task("Detecting chirps...", total=len(datasets))
        # for dataset in datasets[10:]:
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


class ChirpDetector:
    """Parse a grid dataset into batches."""

    def __init__(  # noqa
        self: Self,
        cfg: Config,
        data: Dataset,
        detector: AbstractDetectionModel,
        assigner: AbstractBoxAssigner,
        logger: logging.Logger,
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
            console=prog.console,
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

            if i == 1:
                plot_raw_batch(self.data, batch_indices, batch_raw)

            # STEP 2: Compute the spectrograms for each raw data snippet
            with Timer(prog.console, "Compute spectrograms"):
                batch_metadata, specs, times, freqs = make_batch_specs(
                    batch_indices,
                    batch_metadata,
                    batch_raw,
                    self.data.grid.samplerate,
                    self.cfg,
                )

            if i == 1:
                plot_spec_tiling(specs, times, freqs)

            # STEP 3: Predict boxes for each spectrogram
            with Timer(prog.console, "Detect chirps"):
                predictions = self.detector.predict(specs.copy())

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
                for prediction in predictions:
                    sub_ids = bbox_ids[: len(prediction["boxes"])]
                    bbox_ids = bbox_ids[len(prediction["boxes"]) :]
                    split_ids.append(sub_ids)

                batch_df = convert_detections(
                    predictions,
                    split_ids,
                    batch_metadata,
                    times,
                    freqs,
                    self.cfg,
                )

            # STEP 5: Remove overlapping boxes by non-maximum suppression
            with Timer(prog.console, "Non-maximum suppression"):
                good_box_indices = dataframe_nms(
                    batch_df,
                    overlapthresh=0.2,
                )
                nms_batch_df = batch_df.iloc[good_box_indices]

            # STEP 6: Predict the fundamental frequency of the emitter
            # for each box
            with Timer(prog.console, "Assign boxes to wavetracker tracks"):
                assigned_batch_df = self.assigner.assign(
                    batch_specs=specs,
                    batch_times=times,
                    batch_freqs=freqs,
                    batch_detections=nms_batch_df,
                    data=self.data,
                )
                if len(assigned_batch_df) == 0:
                    prog.console.log(f"No detections in batch {i}.")
                    continue

            # STEP 7: Associate the fundamental frequency of the emitter
            # to the closest wavetracker track
            with Timer(prog.console, "Associate emitter frequency to tracks"):
                assigned_batch_df = assign_ffreqs_to_tracks(
                    assigned_batch_df, self.data
                )

            # STEP 8: plot the detections
            plot_batch_detections(
                specs,
                times,
                freqs,
                batch_df,
                nms_batch_df,
                assigned_batch_df,
                self.data,
                i,
                ylims="full",
                interpolate=False,
            )

            dataframes.append(assigned_batch_df)
            del specs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Save the dataframe
        dataframes = pd.concat(dataframes)
        dataframes = dataframes.reset_index(drop=True)
        dataframes = dataframes.sort_values(by=["t1"])
        savepath = self.data.path / "chirpdetector_bboxes.csv"
        dataframes.to_csv(savepath, index=False)

        # save chirp times and identities as numpy files
        chirp_times = dataframes["t1"] + (
            (dataframes["t2"] - dataframes["t1"]) / 2
        )
        chirp_times = chirp_times.to_numpy()
        chirp_ids = dataframes["track_id"].to_numpy()
        # print(np.shape(chirp_times))
        # print(np.shape(chirp_ids))

        # drop unassigned
        chirp_times = chirp_times[~np.isnan(chirp_ids)]
        chirp_ids = chirp_ids[~np.isnan(chirp_ids)]

        # save the arrays
        np.save(self.data.path / "chirp_times_rcnn.npy", chirp_times)
        np.save(self.data.path / "chirp_ids_rcnn.npy", chirp_ids)
