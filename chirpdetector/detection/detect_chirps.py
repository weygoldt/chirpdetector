"""Detect chirps on a spectrogram."""

import gc
import logging
import pathlib
from typing import List, Self

import matplotlib as mpl
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

from chirpdetector.datahandling.dataset_parsing import ArrayParser
from chirpdetector.models.utils import get_device
from chirpdetector.models.faster_rcnn_detector import (
    load_finetuned_faster_rcnn,
)
from chirpdetector.config import Config, load_config
from chirpdetector.logging.logging import Timer, make_logger
from chirpdetector.datahandling.bbox_tools import (
    dataframe_nms,
    pixel_box_to_timefreq,
)
from chirpdetector.datahandling.dataset_parsing import make_batch_specs
from chirpdetector.detection.detection_models import (
    AbstractDetectionModel,
    FasterRCNN,
)
from chirpdetector.detection.assignment_models import (
    AbstractBoxAssigner,
    SpectrogramPowerTroughBoxAssigner,
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
    model = load_finetuned_faster_rcnn(config)
    model.to(get_device()).eval()
    predictor = FasterRCNN(
        model=model
    )

    # get the box assigner
    assigner = SpectrogramPowerTroughBoxAssigner(config)

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

            prog.console.rule(
                f"[bold green]Batch {i} of {len(self.parser.batches)}"
            )

            # STEP 0: Create metadata for each batch
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
                for prediction in predictions:
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







