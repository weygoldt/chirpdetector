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


def extract_spec_snippets(specs, times, freqs, batch_df):
    """Extract the spectrogram snippet for each bbox.

    This is useful to save the spectrogram snippet of each
    detected chirp to disk, e.g., for clustering purposes.
    The saving step has not been implemented yet.
    """
    spec_snippets = []
    time_snippets = []
    freq_snippets = []
    for i in range(len(batch_df)):
        # get bbox
        t1 = batch_df["t1"].iloc[i]
        t2 = batch_df["t2"].iloc[i]
        f1 = batch_df["f1"].iloc[i]
        f2 = batch_df["f2"].iloc[i]
        batch_spec_idx = batch_df["spec"].iloc[i]
        batch_spec_idx = int(batch_spec_idx)

        # get the spec
        spec = specs[batch_spec_idx][0, :, :]
        time = times[batch_spec_idx]
        freq = freqs[batch_spec_idx]

        # get snippet indices
        t1_idx = np.argmin(np.abs(time - t1))
        t2_idx = np.argmin(np.abs(time - t2))
        f1_idx = np.argmin(np.abs(freq - f1))
        f2_idx = np.argmin(np.abs(freq - f2))

        # extract snippet
        spec_snippet = spec[f1_idx:f2_idx, t1_idx:t2_idx]
        time_snippet = time[t1_idx:t2_idx]
        freq_snippet = freq[f1_idx:f2_idx]

        spec_snippets.append(spec_snippet)
        time_snippets.append(time_snippet)
        freq_snippets.append(freq_snippet)

    return spec_snippets, time_snippets, freq_snippets


def resize_spec_snippets(spec_snippets, time_snippets, freq_snippets, size):
    """Resize the spectrogram snippets to the same size."""
    resized_specs = []
    resized_times = []
    resized_freqs = []
    orig_sizes = []
    for i in range(len(spec_snippets)):
        spec = spec_snippets[i]
        time = time_snippets[i]
        freq = freq_snippets[i]
        orig_sizes.append(spec.shape)

        # resize the spec
        spec = spec.resize((size, size))
        time = np.linspace(time[0], time[-1], size)
        freq = np.linspace(freq[0], freq[-1], size)

        resized_specs.append(spec)
        resized_times.append(time)
        resized_freqs.append(freq)

    return resized_specs, resized_times, resized_freqs


def assign_ffreqs_to_tracks(
    bbox_df: pd.DataFrame, data: Dataset
) -> pd.DataFrame:
    """Assign a bounding box to a wavetracker track."""
    dist_threshold = 20  # Hz
    assigned_tracks = []

    times = []
    freqs = []
    ids = []
    for track_id in data.track.ids[~np.isnan(data.track.ids)]:
        time = data.track.times[
            data.track.indices[data.track.idents == track_id]
        ]
        freq = data.track.freqs[data.track.idents == track_id]
        times.append(time)
        freqs.append(freq)
        ids.append(track_id)

    for i in range(len(bbox_df)):
        chirptime = (bbox_df["t1"].iloc[i] + bbox_df["t2"].iloc[i]) / 2
        emitter_ffreq = bbox_df["emitter_eodf"].iloc[i]

        closest_times = [np.argmin(abs(t - chirptime)) for t in times]
        print("closest times")
        print(closest_times)
        distances = []
        for i, f in enumerate(freqs):
            track_f = f[closest_times[i]]
            dist = np.abs(track_f - emitter_ffreq)
            distances.append(dist)

        emitter_id = ids[np.argmin(np.abs(distances))]

        print(f"Emitter id: {emitter_id}")
        print(f"All ids: {ids}")
        print(f"Corresponding distances: {distances}")

        if np.min(distances) > dist_threshold:
            assigned_tracks.append(np.nan)
        else:
            assigned_tracks.append(emitter_id)
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

    # Get the detection predictor
    model = load_finetuned_yolov8(config)
    predictor = YOLOv8(model=model)

    # get the emitter EODf predictor
    ass_model = load_trained_mlp(config)
    ass_model.to(get_device()).eval()
    assigner = SpectrogramPowerTroughBoxAssignerMLP(ass_model)

    good_datasets = []
    for dataset in datasets:
        # check if .raw or .wav file is in dir
        checktypes = ["raw", "wav"]
        filenames = [str(file.name) for file in dataset.iterdir()]

        foundfile = False
        for filetype in checktypes:
            if any(filetype in filename for filename in filenames):
                foundfile = True

        if not foundfile:
            continue

        # if pathlib.Path(dataset / "chirpdetector_bboxes.csv").exists():
        #     print("chirpdetector_bboxes.csv exists, skipping")
        #     continue

        good_datasets.append(dataset)

    print(
        f"Out of {len(datasets)} a total of {len(good_datasets)} still need detecting"
    )
    with prog:
        task = prog.add_task("Detecting chirps...", total=len(good_datasets))
        for dataset in good_datasets:
            prog.console.log(f"Detecting chirps in {dataset.name}")
            data = load(dataset)
            data = interpolate_tracks(data, samplerate=120)

            # TODO: This is a mess,standardize this
            chirpfiles = [
                "chirps.h5",
                "chirp_times_rcnn.npy",
                "chirp_ids_rcnn.npy",
                "chirpdetector_bboxes.csv",
                "plots/",
                "chirpdetector/",
            ]

            # remove old files
            for file in chirpfiles:
                file = dataset / file
                if file.exists():
                    file.unlink()

            # Initialize the chirp detector
            cpd = ChirpDetector(
                cfg=config,
                data=data,
                detector=predictor,
                assigner=assigner,
                logger=logger,
            )
            # Detect chirps
            cpd.detect()

            # Clean up mo
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

            # STEP 2: Compute the spectrograms for each raw data snippet
            with Timer(prog.console, "Compute spectrograms"):
                batch_metadata, specs, times, freqs = make_batch_specs(
                    batch_indices,
                    batch_metadata,
                    batch_raw,
                    self.data.grid.samplerate,
                    self.cfg,
                )

            # STEP 3: Predict boxes for each spectrogram
            with Timer(prog.console, "Detect chirps"):
                predictions = self.detector.predict(specs.copy())

            # STEP 4: Convert pixel values to time and frequency
            # and save everything in a dataframe
            with Timer(prog.console, "Convert detections"):
                # give every bbox a unique integer identifier
                # first get total number of new bboxes
                n_bboxes = 0
                # TODO: This does not require a for-loop
                for prediction in predictions:
                    n_bboxes += len(prediction["boxes"])

                # then create the ids
                bbox_ids = np.arange(bbox_counter, bbox_counter + n_bboxes)
                bbox_counter += n_bboxes

                # now split the bbox ids in sublists with same length as
                # detections for each spec
                split_ids = []
                # TODO: This can probably be done more elegantly
                for prediction in predictions:
                    sub_ids = bbox_ids[: len(prediction["boxes"])]
                    bbox_ids = bbox_ids[len(prediction["boxes"]) :]
                    split_ids.append(sub_ids)

                # Convert detections from pixel to time and frequency
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
                    overlapthresh=0.2,  # TODO: Move into config
                )
                nms_batch_df = batch_df.iloc[good_box_indices]

            # STEP 6: Predict the fundamental frequency of the emitter
            # for each box
            with Timer(prog.console, "Predicting emitter EODfs"):
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
            # TODO: This should be all done at once after the full file is processed

            # to the closest wavetracker track
            # with Timer(prog.console, "Associate emitter frequency to tracks"):
            #     assigned_batch_df = assign_ffreqs_to_tracks(
            #         assigned_batch_df, self.data
            #     )

            # spec_snippets, time_snippets, freq_snippets = extract_spec_snippets(
            #     specs, times, freqs, assigned_batch_df
            # )

            # TODO: Move shape to config

            # spec_snippets, time_snippets, freq_snippets, orig_shapes = resize_spec_snippets(
            #     spec_snippets, time_snippets, freq_snippets, 256
            # )

            # STEP 8: plot the detections
            with Timer(prog.console, "Saving plot for current batch"):
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
                    save_data=False,
                )

            # TODO: Add function here that extracts chirp spec snippets, makes chirp dataset
            # and saves it to disk with h5py

            dataframes.append(assigned_batch_df)
            del specs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Save the dataframe
        if len(dataframes) > 0:
            dataframes = pd.concat(dataframes)
            dataframes = dataframes.reset_index(drop=True)
            dataframes = dataframes.sort_values(by=["t1"])

            # Associate emitter eodfs with wavetracker track
            with Timer(prog.console, "Associate emitter frequency to tracks"):
                dataframes = assign_ffreqs_to_tracks(dataframes, self.data)

            savepath = self.data.path / "chirpdetector_bboxes.csv"
            dataframes.to_csv(savepath, index=False)

            # save chirp times and identities as numpy files
            chirp_times = dataframes["t1"] + (
                (dataframes["t2"] - dataframes["t1"]) / 2
            )
            chirp_times = chirp_times.to_numpy()
            chirp_ids = dataframes["track_id"].to_numpy()

            # drop unassigned
            chirp_times = chirp_times[~np.isnan(chirp_ids)]
            chirp_ids = chirp_ids[~np.isnan(chirp_ids)]

            # save the arrays
            np.save(self.data.path / "chirp_times_rcnn.npy", chirp_times)
            np.save(self.data.path / "chirp_ids_rcnn.npy", chirp_ids)
        else:
            savepath = self.data.path / "chirpdetector_bboxes.csv"
            empty_df = pd.DataFrame()
            empty_df.to_csv(savepath, index=False)
            np.save(self.data.path / "chirp_times_rcnn.npy", np.array([]))
            np.save(self.data.path / "chirp_ids_rcnn.npy", np.array([]))
