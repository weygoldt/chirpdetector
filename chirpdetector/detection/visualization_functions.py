"""Visualization functions for the data and the model."""

import pathlib
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gridtools.datasets.models import Dataset
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from PIL import Image
from rich.console import Console
import seaborn as sns

console = Console()
backend = "Agg"
mpl.use(backend)


def plot_batch_detections(
    specs: List,
    times: List,
    freqs: List,
    batch_df: pd.DataFrame,
    nms_batch_df: pd.DataFrame,
    assigned_batch_df: pd.DataFrame,
    data: Dataset,
    batch_no: int,
    ylims: str = "full",
    interpolate: bool = False,
    save_data: bool = False,
) -> None:
    """Plot the detections for each batch."""
    assert ylims in [
        "full",
        "fit",
    ], "Invalid ylims argument. Must be 'full' or 'fit'."

    cm = 1 / 2.54
    lenmult = 1
    width = (np.max(times) - np.min(times)) * lenmult
    figsize = (width * cm, 30 * cm)
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    vmin = np.min([np.min(spec.cpu().numpy()) for spec in specs])
    vmax = np.max([np.max(spec.cpu().numpy()) for spec in specs])
    for i in range(len(specs)):
        spec = specs[i].cpu().numpy()

        # interpolate as it looks better
        if interpolate:
            factor = 5
            x = np.linspace(times[i][0], times[i][-1], len(times[i]) * factor)
            y = np.linspace(freqs[i][0], freqs[i][-1], len(freqs[i]) * factor)
            img = Image.fromarray(spec[0, :, :])
            img = img.resize((len(times[i]) * factor, len(freqs[i]) * factor))
            spec = np.array(img)
            ax.pcolormesh(
                x,
                y,
                spec,
                cmap="magma",
                rasterized=True,
                vmin=vmin,
                vmax=vmax,
            )
            # print(f"Interpolated dims: {spec.shape}")
        else:
            ax.pcolormesh(
                times[i],
                freqs[i],
                spec[0, :, :],
                cmap="magma",
                rasterized=True,
                vmin=vmin,
                vmax=vmax,
            )

    patches = []
    # bboxes before nms
    for j in range(len(batch_df)):
        t1 = batch_df["t1"].iloc[j]
        f1 = batch_df["f1"].iloc[j]
        t2 = batch_df["t2"].iloc[j]
        f2 = batch_df["f2"].iloc[j]
        score = batch_df["score"].iloc[j]
        patches.append(
            plt.Rectangle(
                (t1, f1),
                t2 - t1,
                f2 - f1,
                fill=False,
                color="lightgrey",
                lw=0.5,
                alpha=0.2,
            )
        )
        ax.text(
            t1,
            f2,
            f"{score:.2f}",
            color="lightgrey",
            fontsize=8,
            ha="left",
            va="bottom",
            alpha=0.2,
            rotation=45,
        )

    # bboxes after nms
    for j in range(len(nms_batch_df)):
        t1 = nms_batch_df["t1"].iloc[j]
        f1 = nms_batch_df["f1"].iloc[j]
        t2 = nms_batch_df["t2"].iloc[j]
        f2 = nms_batch_df["f2"].iloc[j]
        score = nms_batch_df["score"].iloc[j]
        patches.append(
            plt.Rectangle(
                (t1, f1),
                t2 - t1,
                f2 - f1,
                fill=False,
                color="white",
                lw=0.5,
                alpha=1,
            )
        )
        ax.text(
            t1,
            f2,
            f"{score:.2f}",
            color="white",
            fontsize=8,
            ha="left",
            va="bottom",
            alpha=1,
            rotation=45,
        )

    # get bboxes after assignment
    for j in range(len(assigned_batch_df)):
        t1 = assigned_batch_df["t1"].iloc[j]
        f1 = assigned_batch_df["f1"].iloc[j]
        t2 = assigned_batch_df["t2"].iloc[j]
        f2 = assigned_batch_df["f2"].iloc[j]
        score = assigned_batch_df["score"].iloc[j]
        predicted_eodf = assigned_batch_df["emitter_eodf"].iloc[j]

        color = "white"

        patches.append(
            plt.Rectangle(
                (t1, f1),
                t2 - t1,
                f2 - f1,
                fill=False,
                color=color,
                lw=0.5,
                alpha=1,
            )
        )
        ax.text(
            t1,
            f2,
            f"{score:.2f}",
            color=color,
            fontsize=8,
            ha="left",
            va="bottom",
            alpha=1,
            rotation=45,
        )

        ax.scatter(
            t1 + (t2 - t1) / 2,
            predicted_eodf,
            color=color,
            s=10,
            zorder=1000,
            edgecolor="black",
        )

    ax.add_collection(PatchCollection(patches, match_original=True))

    ax.set_xlim(np.min(times), np.max(times))
    ax.set_xlabel("Time [s]", fontsize=16)
    ax.set_ylabel("Frequency [Hz]", fontsize=16)

    savepath = pathlib.Path(f"{data.path}/chirpdetection")
    savepath.mkdir(exist_ok=True, parents=True)

    if ylims == "full":
        filename = savepath / f"batch_{batch_no}_full.png"
        ax.set_ylim(np.min(freqs), np.max(freqs))
        plt.savefig(filename, dpi=300)
    if ylims == "fit":
        filename = savepath / f"batch_{batch_no}_fit.png"
        # ax.set_ylim(min_freq - 200, max_freq + 500)
        plt.savefig(filename, dpi=300)

    if save_data:
        datapath = pathlib.Path(f"{data.path}/chirpdetection/{filename.stem}")
        datapath.mkdir(exist_ok=True, parents=True)

        # save all input data
        spec = {
            "specs": specs,
            "times": times,
            "freqs": freqs,
        }
        np.save(datapath / "specs.npy", spec)

        # save all detections
        batch_df.to_csv(datapath / "batch_df.csv", index=False)
        nms_batch_df.to_csv(datapath / "nms_batch_df.csv", index=False)
        assigned_batch_df.to_csv(
            datapath / "assigned_batch_df.csv", index=False
        )

        raw_dataset_path = data.path
        # save path as txt
        with open(datapath / "raw_dataset_path.txt", "w") as f:
            f.write(str(raw_dataset_path))
            f.close()

    plt.close()


def plot_raw_batch(data, batch_indices, batch_raw):
    _, ax = plt.subplots()
    start = batch_indices[0][0] - 10 * data.grid.samplerate
    end = batch_indices[-1][-1] + 10 * data.grid.samplerate
    times = np.arange(start, end) / data.grid.samplerate

    plot_margin = (
        np.max(data.grid.rec[start:end, :])
        - np.min(data.grid.rec[start:end, :]) * 0.2
    )

    for el in range(data.grid.shape[1])[:4]:
        ax.plot(
            times,
            data.grid.rec[start:end, el] + el * plot_margin,
            color="grey",
        )

    for batch_idxs, batch in zip(batch_indices, batch_raw):
        x = np.arange(batch_idxs[0], batch_idxs[1]) / data.grid.samplerate
        ax.axvline(x[0], color="white", lw=0.5, alpha=1, zorder=1000)
        ax.axvline(x[-1], color="white", lw=0.5, alpha=1, zorder=1000)
        for el in range(batch.shape[1])[:4]:
            ax.plot(
                x, batch[:, el] + el * plot_margin, alpha=0.8, color="black"
            )

    ax.axvline(
        batch_indices[0][0] / data.grid.samplerate,
        color="white",
        lw=2,
        alpha=1,
        zorder=1001,
    )
    ax.axvline(
        batch_indices[-1][-1] / data.grid.samplerate,
        color="white",
        lw=2,
        alpha=1,
        zorder=1001,
    )

    ax.axis("off")
    plt.savefig("/home/weygoldt/Projects/mscthesis/plots/batch.svg")
    plt.close("all")


def plot_spec_tiling(specs, times, freqs):
    _, ax = plt.subplots()
    for j, (spec, time, freq) in enumerate(zip(specs, times, freqs)):
        spec = spec.cpu().numpy()
        ax.pcolormesh(time, freq, spec[0], rasterized=True)
        if j in [0, 3, 5, 7, 10, 11]:
            ax.add_patch(
                Rectangle(
                    (time[0], freq[0]),
                    time[-1] - time[0],
                    freq[-1] - freq[0],
                    fill=True,
                    color="white",
                    lw=1,
                    alpha=0.3,
                    zorder=999,
                )
            )
    ax.set_xlim(np.min(times), np.max(times))
    ax.set_ylim(np.min(freqs), np.max(freqs))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    plt.savefig("/home/weygoldt/Projects/mscthesis/plots/spec_tiling.svg")
    plt.close("all")
