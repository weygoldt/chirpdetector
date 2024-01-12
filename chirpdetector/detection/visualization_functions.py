"""Visualization functions for the data and the model."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def plot_raw_batch(data, batch_indices, batch_raw):
    mpl.use("TkAgg")
    _, ax = plt.subplots()
    start = batch_indices[0][0] - 10 * data.grid.samplerate
    end = batch_indices[-1][-1] + 10 * data.grid.samplerate
    times = np.arange(start, end) / data.grid.samplerate

    plot_margin = np.max(data.grid.rec[start:end, :]) - np.min(data.grid.rec[start:end, :]) * 0.2

    for el in range(data.grid.shape[1])[:4]:
        ax.plot(times, data.grid.rec[start:end, el] + el * plot_margin, color="grey")

    for batch_idxs, batch in zip(batch_indices, batch_raw):
        x = np.arange(batch_idxs[0], batch_idxs[1]) / data.grid.samplerate
        ax.axvline(x[0], color="white", lw=.5, alpha=1, zorder=1000)
        ax.axvline(x[-1], color="white", lw=.5, alpha=1, zorder=1000)
        for el in range(batch.shape[1])[:4]:
            ax.plot(x, batch[:, el] + el * plot_margin, alpha=0.8, color="black")

    ax.axvline(batch_indices[0][0] / data.grid.samplerate, color="white", lw=2, alpha=1, zorder=1001)
    ax.axvline(batch_indices[-1][-1] / data.grid.samplerate, color="white", lw=2, alpha=1, zorder=1001)

    ax.axis("off")
    plt.savefig("/home/weygoldt/Projects/mscthesis/plots/batch.svg")
    plt.close("all")


def plot_spec_tiling(specs, times, freqs):
    _, ax = plt.subplots()
    for j, (spec, time, freq) in enumerate(zip(specs, times, freqs)):
        spec = spec.cpu().numpy()
        ax.pcolormesh(time, freq, spec[0])
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
                    zorder=999
                )
            )
    ax.set_xlim(np.min(times), np.max(times))
    ax.set_ylim(np.min(freqs), np.max(freqs))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    plt.savefig("/home/weygoldt/Projects/mscthesis/plots/spec_tiling.png")
    plt.close("all")
