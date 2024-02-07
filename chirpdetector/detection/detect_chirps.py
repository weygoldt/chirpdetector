"""Detect chirps on a spectrogram."""

import gc
import logging
import pathlib
from typing import List, Self

import matplotlib as mpl
import matplotlib.pyplot as plt
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
    SpectrogramPowerTroughBoxAssigner,
    SpectrogramPowerTroughBoxAssignerMLP,
)
from chirpdetector.detection.detection_models import (
    AbstractDetectionModel,
    FasterRCNN,
)
from chirpdetector.detection.visualization_functions import (
    plot_raw_batch,
    plot_spec_tiling,
)
from chirpdetector.logging.logging import Timer, make_logger
from chirpdetector.models.faster_rcnn_detector import (
    load_finetuned_faster_rcnn,
)
from chirpdetector.models.utils import get_device

# Use non-gui backend for matplotlib to avoid memory leaks
mpl.use("TkAgg")

try:
    basestyle = "/home/weygoldt/Projects/mscthesis/src/base.mplstyle"
    background = "/home/weygoldt/Projects/mscthesis/src/light_background.mplstyle"
    plt.style.use([basestyle, background])
except FileNotFoundError:
    pass

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
    # model = load_finetuned_yolov8(config)
    # predictor = YOLOV8(model=model)

    # get the box assigner
    assigner = SpectrogramPowerTroughBoxAssignerMLP(config)

    with prog:
        task = prog.add_task("Detecting chirps...", total=len(datasets))
        # for dataset in datasets[10:]:
        for dataset in datasets[10:]:
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

            if i == 1:
                plot_raw_batch(self.data, batch_indices, batch_raw)

            # STEP 2: Compute the spectrograms for each raw data snippet
            with Timer(prog.console, "Compute spectrograms"):
                batch_metadata, specs, times, freqs = make_batch_specs(
                    batch_indices,
                    batch_metadata,
                    batch_raw,
                    self.data.grid.samplerate,
                    self.cfg
                )

            if i == 1:
                plot_spec_tiling(specs, times, freqs)

            # STEP 3: Predict boxes for each spectrogram
            with Timer(prog.console, "Detect chirps"):
                predictions = self.detector.predict(specs)

            # if i == 1:
            #     fig, ax = plt.subplots()
            #     from matplotlib.patches import Rectangle
            #     ax.pcolormesh(specs[4][0].cpu().numpy())
            #     for j in range(len(predictions[4]["boxes"])):
            #         box = predictions[4]["boxes"][j]
            #         score = predictions[4]["scores"][j]
            #         ax.add_patch(
            #             Rectangle(
            #                 (box[0], box[1]),
            #                 box[2] - box[0],
            #                 box[3] - box[1],
            #                 fill=False,
            #                 color="grey",
            #                 lw=1,
            #                 alpha=1
            #             )
            #         )
            #         ax.text(
            #             box[0],
            #             box[1],
            #             f"{score:.2f}",
            #             color="grey",
            #             fontsize=8,
            #             ha="left",
            #             va="bottom",
            #             alpha=1,
            #         )
            #     ax.axis("off")
            #     plt.show()
            #     exit()
            #

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

        #     dataframes.append(assigned_batch_df)
        #     del specs
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()
        #     gc.collect()
        #
        # # Save the dataframe
        # dataframes = pd.concat(dataframes)
        # dataframes = dataframes.reset_index(drop=True)
        # dataframes = dataframes.sort_values(by=["t1"])
        # savepath = self.data.path / "chirpdetector_bboxes.csv"
        # dataframes.to_csv(savepath, index=False)

            import matplotlib as mpl
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Rectangle
            mpl.use("TkAgg")

            for speci in range(4):
                cm = 1/2.54
                fig, ax = plt.subplots(figsize=(16*cm, 9*cm), constrained_layout=True)
                iter = 0
                for spec, time, freq in zip(specs, times, freqs):
                    spec = spec.cpu().numpy()
                    ax.pcolormesh(time, freq, spec[0, :, :])
                    # ax.imshow(spec[0], aspect="auto", origin="lower",
                    #           extent=[time[0], time[-1], freq[0], freq[-1]],
                    #           interpolation="gaussian",)
                    iter += 1

                # # before nms
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
                #             lw=2,
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

                # after nms
                # for i in range(len(nms_batch_df)):
                #     x1 = nms_batch_df["t1"].iloc[i]
                #     y1 = nms_batch_df["f1"].iloc[i]
                #     x2 = nms_batch_df["t2"].iloc[i]
                #     y2 = nms_batch_df["f2"].iloc[i]
                #     score = nms_batch_df["score"].iloc[i]
                #     ax.add_patch(
                #         Rectangle(
                #             (x1-0.1, y1),
                #             (x2 - x1) + 0.2,
                #             y2 - y1,
                #             fill=False,
                #             color="white",
                #             lw=1.5,
                #             alpha=1,
                #             zorder=1000
                #         )
                #     )
                #     ax.text(
                #         x1,
                #         y2,
                #         f"{score:.2f}",
                #         color="black",
                #         fontsize=10,
                #         ha="left",
                #         va="bottom",
                #         alpha=1,
                #         backgroundcolor="white"
                #     )

                colors = ["#1f77b4", "#ff7f0e"]
                colors = ["tab:red", "tab:orange"]
                for j, track_id in enumerate(self.data.track.ids):
                    track_freqs = self.data.track.freqs[self.data.track.idents == track_id]
                    track_time = self.data.track.times[self.data.track.indices[self.data.track.idents == track_id]]
                    ax.plot(track_time, track_freqs, color=colors[j], lw=2, label=f"Fish {j+1}")

                patches = []
                for j in range(len(assigned_batch_df)):
                    t1 = assigned_batch_df["t1"].iloc[j]
                    f1 = assigned_batch_df["f1"].iloc[j]
                    t2 = assigned_batch_df["t2"].iloc[j]
                    f2 = assigned_batch_df["f2"].iloc[j]
                    score = assigned_batch_df["score"].iloc[j]

                    # assigned_id = assigned_batch_df["track_id"].iloc[j]

                    assigned_eodf = assigned_batch_df["emitter_eodf"].iloc[j]
                    print(f"Assigned EODF: {assigned_eodf}")
                    print(f"Corresponding time: {t2-t1}")

                    ax.scatter(t1 + (t2-t1)/2, assigned_eodf, color="black", s=10, zorder=1000)

                    # if assigned_id not in self.data.track.ids:
                        # continue

                    ax.text(
                        t1,
                        f2,
                        f"score{score:.2f} id2",
                        color="black",
                        fontsize=10,
                        ha="left",
                        va="bottom",
                        alpha=1,
                        backgroundcolor="white"
                    )

                    # print(assigned_id)
                    # print(self.data.track.ids)

                    # color = np.array(colors)[self.data.track.ids == assigned_id][0]
                    # print(color)
                    color = "white"

                    patches.append(Rectangle(
                            (t1 - 0.1, f1),
                            t2 - t1 + 0.2,
                            f2 - f1,
                            fill=False,
                            color=color,
                            lw=1.5,
                            alpha=1,
                    ))

                ax.add_collection(PatchCollection(patches, match_original=True))

                # go through in 15 second intervals
                ax.set_ylim(np.min(self.data.track.freqs) - 100, np.max(self.data.track.freqs) + 300)
                ax.set_xlim(times[0][0] + speci * 15, times[0][0] + (speci + 1) * 15)
                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Frequency [Hz]")
                ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, ncol=2)
                from uuid import uuid4
                # plt.savefig(f"plots/{self.data.path.name}_{speci}_{uuid4()}.svg", dpi=300)
                plt.show()

            plt.close()
            #
            # # TODO: Save the output from above to a file







