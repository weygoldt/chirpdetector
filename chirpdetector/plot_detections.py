"""Functions to visualize detections on images."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gridtools.datasets import Dataset, load, subset
from gridtools.utils.spectrograms import (
    freqres_to_nfft,
    overlap_to_hoplen,
)
from matplotlib.patches import Rectangle
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)

from .utils.configfiles import Config, load_config
from .utils.signal_processing import (
    compute_sum_spectrogam,
    make_chunk_indices,
    make_spectrogram_axes,
)

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


class Spectrogram(BaseModel):
    """Class to store spectrogram parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    powers: np.ndarray
    freqs: np.ndarray
    times: np.ndarray


def plot_detections(
    data: Dataset,
    spec: Spectrogram,
    bboxes: np.ndarray,
    save_path: pathlib.Path,
    file_index: str,
) -> None:
    """Plot a spectrogram with tracks, bounding boxes and chirp times.

    Parameters
    ----------
    - `data` : `Dataset`
        The dataset from gridtools.datasets.
    - `spec` : `Spectrogram`
        The spectrogram.
    - `bboxes` : `np.ndarray`
        The bounding boxes, as returned by `chirpdetector`.
    - `file_index` : `str`
        The file index, used for saving the plot.

    Returns
    -------
    - `None`
    """
    _, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    # plot the spectrogram
    extent = (
        spec.times[0],
        spec.times[-1],
        spec.freqs[0],
        spec.freqs[-1],
    )
    ax.imshow(
        spec.powers,
        aspect="auto",
        origin="lower",
        interpolation="gaussian",
        extent=extent,
        cmap="magma",
    )

    # plot the bounding boxes
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
        chirptimes = data.com.chirp.times[data.com.chirp.idents == track_id]
        freqs = data.track.freqs[data.track.idents == track_id]
        times = data.track.times[
            data.track.indices[data.track.idents == track_id]
        ]
        freqs = freqs[(times >= extent[0] - 10) & (times <= extent[1] + 10)]
        times = times[(times >= extent[0] - 10) & (times <= extent[1] + 10)]

        # get freqs where times are closest to ctimes
        chirp_eodfs = np.zeros_like(chirptimes)
        for i, ctime in enumerate(chirptimes):
            try:
                indx = np.argmin(np.abs(times - ctime))
                chirp_eodfs[i] = freqs[indx]
            except ValueError:
                msg = (
                    "Failed to find track time closest to chirp time "
                    f"in chunk {file_index}, check the plots."
                )
                prog.console.log(msg)

        if len(times) != 0:
            ax.plot(
                times,
                freqs,
                lw=2,
                color="black",
                label="Frequency traces",
            )
        ax.scatter(
            chirptimes,
            chirp_eodfs,
            marker="o",
            lw=1,
            facecolor="white",
            edgecolor="black",
            s=25,
            zorder=10,
            label="Chirp assignments",
        )

    ax.set_ylim(spec.freqs[0] + 5, spec.freqs[-1] - 5)
    ax.set_xlim([spec.times[0], spec.times[-1]])

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

    save_path = (
        save_path / "chirpdetections" / f"cpd_assigned_{file_index}.png"
    )
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
    plt.clf()
    plt.cla()
    plt.close("all")


def plot_all_detections(
    data: Dataset,
    chirp_df: pd.DataFrame,
    conf: Config,
) -> None:
    """Plot all chirp detections of a full recording on spectrograms.

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
    nfft = freqres_to_nfft(conf.spec.freq_res, data.grid.samplerate)  # samples
    hop_len = overlap_to_hoplen(conf.spec.overlap_frac, nfft)  # samples
    chunksize = int(time_window * data.grid.samplerate)  # samples
    nchunks = np.ceil(data.grid.rec.shape[0] / chunksize).astype(int)
    window_overlap_samples = int(conf.spec.spec_overlap * data.grid.samplerate)

    # Set y limits for the spectrogram
    flims = (
        np.min(data.track.freqs) - 200,
        np.max(data.track.freqs) + 700,
    )

    for chunk_no in range(nchunks):
        # get start and stop indices for the current chunk
        # including some overlap to compensate for edge effects
        # this diffrers for the first and last chunk
        idx1, idx2 = make_chunk_indices(
            n_chunks=nchunks,
            current_chunk=chunk_no,
            chunksize=chunksize,
            window_overlap_samples=window_overlap_samples,
            max_end=data.grid.rec.shape[0],
        )

        # subset the data to the current chunk
        chunk = subset(data, idx1, idx2, mode="index")
        chunk.track.times += idx1 / data.grid.samplerate

        # dont plot chunks without chirps
        if len(chunk.com.chirp.times) == 0:
            continue

        if len(chunk.track.indices) == 0:
            continue

        # compute the time and frequency axes of the spectrogram
        spec_times, spec_freqs = make_spectrogram_axes(
            start=idx1,
            stop=idx2,
            nfft=nfft,
            hop_length=hop_len,
            samplerate=data.grid.samplerate,
        )
        # compute the spectrogram for each electrode of the current chunk
        spec = compute_sum_spectrogam(
            data=chunk,
            nfft=nfft,
            hop_len=hop_len,
        )

        # detach from GPU and convert to numpy
        spec = spec.detach().cpu().numpy()
        spec = spec[(spec_freqs >= flims[0]) & (spec_freqs <= flims[1]), :]
        spec_freqs = spec_freqs[
            (spec_freqs >= flims[0]) & (spec_freqs <= flims[1])
        ]
        spectrogram = Spectrogram(
            powers=spec,
            freqs=spec_freqs,
            times=spec_times,
        )

        # Extract the bounding boxes for the current chunk
        chunk_t1 = idx1 / data.grid.samplerate
        chunk_t2 = idx2 / data.grid.samplerate
        chunk_df = chirp_df[
            (chirp_df["t1"] >= chunk_t1) & (chirp_df["t2"] <= chunk_t2)
        ]

        # get t1, t2, f1, f2 from chunk_df
        bboxes = chunk_df[["score", "t1", "f1", "t2", "f2"]].to_numpy()

        plot_detections(
            data=data,
            spec=spectrogram,
            bboxes=bboxes,
            save_path=data.path,
            file_index=f"{chunk_no:05d}",
        )


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
    plot_all_detections(data, chirp_df, conf)


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
            plot_all_detections(data, chirp_df, conf)
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
