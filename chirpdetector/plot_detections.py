"""Functions to visualize detections on images."""

import pathlib

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch
from gridtools.datasets import Dataset, load, subset
from gridtools.utils.spectrograms import (
    decibel,
    freqres_to_nfft,
    overlap_to_hoplen,
    spectrogram,
)

from .utils.configfiles import Config, load_config

from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, BarColumn, TimeElapsedColumn
from rich.console import Console

console = Console()
prog = Progress(
    SpinnerColumn(),
    "[progress.description]{task.description}",
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    TimeElapsedColumn(),
    MofNCompleteColumn(),
    console=console,
)





def plot_detections(
    data: Dataset,
    chirp_df: pd.DataFrame,
    conf: Config,
) -> None:
    """Plot detections on spectrograms.

    Parameters
    ----------
    data : Dataset
        The dataset.
    chirp_df : pd.DataFrame
        The dataframe containing the chirp detections.
    conf : Config
        The config file.
    """
    time_window = 15
    n_electrodes = data.grid.rec.shape[1]

    nfft = freqres_to_nfft(conf.spec.freq_res, data.grid.samplerate)  # samples
    hop_len = overlap_to_hoplen(conf.spec.overlap_frac, nfft)  # samples
    chunksize = time_window * data.grid.samplerate  # samples
    nchunks = np.ceil(data.grid.rec.shape[0] / chunksize).astype(int)
    window_overlap_samples = int(conf.spec.spec_overlap * data.grid.samplerate)

    for chunk_no in range(nchunks):
        # get start and stop indices for the current chunk
        # including some overlap to compensate for edge effects
        # this diffrers for the first and last chunk

        if chunk_no == 0:
            idx1 = int(chunk_no * chunksize)
            idx2 = int((chunk_no + 1) * chunksize + window_overlap_samples)
        elif chunk_no == nchunks - 1:
            idx1 = int(chunk_no * chunksize - window_overlap_samples)
            idx2 = int((chunk_no + 1) * chunksize)
        else:
            idx1 = int(chunk_no * chunksize - window_overlap_samples)
            idx2 = int((chunk_no + 1) * chunksize + window_overlap_samples)

        # idx1 and idx2 now determine the window I cut out of the raw signal
        # to compute the spectrogram of.

        # compute the time and frequency axes of the spectrogram now that we
        # include the start and stop indices of the current chunk and thus the
        # right start and stop time. The `spectrogram` function does not know
        # about this and would start every time axis at 0.
        spec_times = np.arange(idx1, idx2 + 1, hop_len) / data.grid.samplerate
        spec_freqs = np.arange(0, nfft / 2 + 1) * data.grid.samplerate / nfft

        # create a subset from the grid dataset
        if idx2 > data.grid.rec.shape[0]:
            idx2 = data.grid.rec.shape[0] - 1
        chunk = subset(data, idx1, idx2, mode="index")

        if len(chunk.com.chirp.times) == 0:
            continue

        # compute the spectrogram for each electrode of the current chunk
        spec = torch.zeros((len(spec_freqs), len(spec_times)))
        for el in range(n_electrodes):
            # get the signal for the current electrode
            sig = chunk.grid.rec[:, el]

            # compute the spectrogram for the current electrode
            chunk_spec, _, _ = spectrogram(
                data=sig.copy(),
                samplingrate=data.grid.samplerate,
                nfft=nfft,
                hop_length=hop_len,
            )

            # sum spectrogram over all electrodes
            if el == 0:
                spec = chunk_spec
            else:
                spec += chunk_spec

        # normalize spectrogram by the number of electrodes
        spec /= n_electrodes

        # convert the spectrogram to dB
        spec = decibel(spec)
        spec = spec.detach().cpu().numpy()

        # Set y limits
        flims = (
            np.min(data.track.freqs) - 200,
            np.max(data.track.freqs) + 700,
        )
        spec = spec[(spec_freqs >= flims[0]) & (spec_freqs <= flims[1]), :]
        spec_freqs = spec_freqs[
            (spec_freqs >= flims[0]) & (spec_freqs <= flims[1])
        ]

        # Extract the bounding boxes for the current chunk
        chunk_t1 = idx1 / data.grid.samplerate
        chunk_t2 = idx2 / data.grid.samplerate
        chunk_df = chirp_df[
            (chirp_df["t1"] >= chunk_t1) & (chirp_df["t2"] <= chunk_t2)
        ]

        # get t1, t2, f1, f2 from chunk_df
        bboxes = chunk_df[["score", "t1", "f1", "t2", "f2"]].to_numpy()

        # get chirp times and chirp ids
        chirp_times = chunk_df["envelope_trough_time"]
        chirp_ids = chunk_df["assigned_track"]

        mpl.use("TkAgg")

        _, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

        # plot bounding boxes
        ax.imshow(
            spec,
            aspect="auto",
            origin="lower",
            interpolation="gaussian",
            extent=[
                spec_times[0],
                spec_times[-1],
                spec_freqs[0],
                spec_freqs[-1],
            ],
            cmap="magma",
            vmin=-80,
            vmax=-45,
        )
        idx = 0
        for bbox in bboxes:
            ax.add_patch(
                Rectangle(
                    (bbox[1], bbox[2]),
                    bbox[3] - bbox[1],
                    bbox[4] - bbox[2],
                    fill=False,
                    color="gray",
                    linewidth=1,
                    label="faster-R-CNN predictions",
                ),
            )
            ax.text(
                bbox[1],
                bbox[4] + 15,
                f"{bbox[0]:.2f}",
                color="gray",
                fontsize=10,
                verticalalignment="bottom",
                horizontalalignment="left",
                rotation=90,
            )

        # plot chirp times and frequency traces
        for track_id in np.unique(data.track.idents):
            ctimes = chirp_times[chirp_ids == track_id]

            freqs = data.track.freqs[data.track.idents == track_id]
            times = data.track.times[
                data.track.indices[data.track.idents == track_id]
            ]
            freqs = freqs[
                (times >= spec_times[0] - 10) & (times <= spec_times[-1] + 10)
            ]
            times = times[
                (times >= spec_times[0] - 10) & (times <= spec_times[-1] + 10)
            ]

            # get freqs where times are closest to ctimes
            cfreqs = np.zeros_like(ctimes)
            for i, ctime in enumerate(ctimes):
                indx = np.argmin(np.abs(times - ctime))
                cfreqs[i] = freqs[indx]

            ax.plot(
                times,
                freqs,
                lw=2,
                color="black",
                label="Frequency traces",
            )

            ax.scatter(
                ctimes,
                cfreqs,
                marker="o",
                lw=1,
                facecolor="white",
                edgecolor="black",
                s=25,
                zorder=10,
                label="Chirp assignments",
            )

        ax.set_ylim(flims[0] + 5, flims[1] - 5)
        ax.set_xlim([spec_times[0], spec_times[-1]])
        ax.set_xlabel("Time [s]", fontsize=12)
        ax.set_ylabel("Frequency [Hz]", fontsize=12)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(
            by_label.values(),
            by_label.keys(),
            bbox_to_anchor=(0.5, 1.02),
            loc="lower center",
            mode="None",
            borderaxespad=0,
            ncol=3,
            fancybox=False,
            framealpha=0,
        )

        savepath = data.path / "chirpdetections"
        savepath.mkdir(exist_ok=True)
        plt.savefig(
            savepath / f"cpd_{chunk_no}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()
        plt.clf()
        plt.cla()
        plt.close("all")

def clean_plots_cli(path: pathlib.Path) -> None:
    """Remove all plots from the chirpdetections folder.

    Parameters
    ----------
    path : pathlib.Path
        Path to the config file.
    """
    savepath = path / "chirpdetections"
    for f in savepath.iterdir():
        f.unlink()

def plot_detections_cli(path: pathlib.Path) -> None:
    """Plot detections on images.

    Parameters
    ----------
    path : pathlib.Path
        Path to the config file.
    """
    conf = load_config(path.parent / "chirpdetector.toml")
    data = load(path)
    chirp_df = pd.read_csv(path / "chirpdetector_bboxes.csv")
    plot_detections(data, chirp_df, conf)

def plot_all_detections_cli(path: pathlib.Path) -> None:
    """Plot detections on images.

    Parameters
    ----------
    path : pathlib.Path
        Path to the config file.
    """
    conf = load_config(path / "chirpdetector.toml")

    dirs = [dataset for dataset in path.iterdir() if dataset.is_dir()]
    with prog:
        task = prog.add_task("Plotting detections...", total=len(dirs))
        for dataset in dirs:
            prog.console.log(f"Plotting detections for {dataset.name}")
            data = load(dataset)
            chirp_df = pd.read_csv(dataset / "chirpdetector_bboxes.csv")
            plot_detections(data, chirp_df, conf)
            prog.advance(task)

def clean_all_plots_cli(path: pathlib.Path) -> None:
    """Remove all plots from the chirpdetections folder.

    Parameters
    ----------
    path : pathlib.Path
        Path to the config file.
    """
    dirs = [dataset for dataset in path.iterdir() if dataset.is_dir()]
    with prog:
        task = prog.add_task("Cleaning plots...", total=len(dirs))
        for dataset in dirs:
            prog.console.log(f"Cleaning plots for {dataset.name}")
            clean_plots_cli(dataset)
            prog.advance(task)
