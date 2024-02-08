"""Tools to parse large datasets in batches."""

from typing import List, Self, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from gridtools.utils.spectrograms import (
    compute_spectrogram,
    freqres_to_nfft,
    overlap_to_hoplen,
    to_decibel,
)
from rich.console import Console

from chirpdetector.config import Config
from chirpdetector.datahandling.signal_processing import (
    make_spectrogram_axes,
    spec_to_image,
)


def tile_batch_specs(batch_specs: List, cfg: Config) -> List:
    """Tile the spectrograms of a batch in the frequency dimension.

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
    freq_ranges = cfg.spec.freq_ranges
    # group ranges into tuples of 2 ints
    freq_ranges = [
        (freq_ranges[i], freq_ranges[i + 1])
        for i in range(0, len(freq_ranges), 2)
    ]
    sliced_specs = []
    for start, end in freq_ranges:
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
                    freq[start_idx:end_idx],
                )
            )
    return sliced_specs


def make_batch_specs(
    indices: List,
    metadata: List,
    batch: np.ndarray,
    samplerate: float,
    cfg: Config,
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
    nfft = freqres_to_nfft(freq_res=cfg.spec.freq_res, samplingrate=samplerate)
    hop_length = overlap_to_hoplen(nfft=nfft, overlap=cfg.spec.overlap_frac)
    batch_specs = [
        compute_spectrogram(
            data=signal,
            samplingrate=samplerate,
            nfft=nfft,
            # nfft=4096,
            hop_length=hop_length,
        )[0]
        for signal in batch
    ]

    batch_specs_decibel = [to_decibel(spec) for spec in batch_specs]
    # batch_specs_decible_cpu = [spec for spec in batch_specs_decibel]
    batch_sum_specs = [torch.sum(spec, dim=0) for spec in batch_specs_decibel]
    axes = [
        make_spectrogram_axes(
            start=idxs[0],
            stop=idxs[1],
            nfft=nfft,
            hop_length=hop_length,
            samplerate=samplerate,
        )
        for idxs in indices
    ]
    batch_specs = [(spec, *ax) for spec, ax in zip(batch_sum_specs, axes)]
    # Add the metadata to each spec tuple
    batch_specs = [(meta, *spec) for meta, spec in zip(metadata, batch_specs)]

    # Tile the spectrograms y-axis
    sliced_specs = tile_batch_specs(batch_specs, cfg)

    # Split the list into specs and axes
    metadata, specs, times, freqs = zip(*sliced_specs)

    # Convert the spec tensors to mimic PIL images
    images = [spec_to_image(spec) for spec in specs]

    return metadata, images, times, freqs


class ArrayParser:
    """Generate indices to iterate large arrays in batches."""

    def __init__(  # noqa
        self: Self,
        length: int,
        samplingrate: float,
        batchsize: int,
        windowsize: float,
        overlap: float,
        console: Console,
    ) -> None:
        """Initialize DatasetParser.

        Parameters
        ----------
        length : int
            Length of the dataset in samples.
        samplingrate : float
            Sampling rate of the dataset in Hz.
        batchsize : int
            Number of windows in each batch.
        windowsize : float
            Size of each window in seconds.
        overlap : float
            Overlap between windows in seconds.
        """
        self.batchsize = batchsize
        self.samplingrate = samplingrate
        self.overlap_seconds = overlap
        self.windowsize_seconds = windowsize
        self.datasetsize_samples = length
        self.console = console
        self.set_indices()
        self.set_batches()

    def set_indices(self: Self) -> None:
        """Set indices for batches and windows."""
        self.overlap_samples = self.overlap_seconds * self.samplingrate
        if self.overlap_samples % 1 != 0:
            self.overlap_samples = int(np.round(self.overlap_samples))
            self.overlap_seconds = self.overlap_samples / self.samplingrate
            msg = (
                "⚠️ Overlap was not an integer number of samples. "
                f"Rounded to {self.overlap_seconds} seconds."
            )
            self.console.log(msg)
        self.windowsize_samples = self.windowsize_seconds * self.samplingrate
        if self.windowsize_samples % 1 != 0:
            self.windowsize_samples = int(np.round(self.windowsize_samples))
            self.windowsize_seconds = (
                self.windowsize_samples / self.samplingrate
            )
            msg = (
                "⚠️ Window size was not an integer number of samples. "
                f"Rounded to {self.windowsize_seconds} seconds."
            )
            self.console.log(msg)
        self.batchsize_samples = self.windowsize_samples * self.batchsize
        self.nbatches = int(
            np.ceil(self.datasetsize_samples / self.batchsize_samples)
        )

        # check if last batch has at least one window
        if self.nbatches > 1:
            last_batch_start = self.batchsize_samples * (self.nbatches - 1)
            last_batch_end = self.datasetsize_samples
            last_batch_size = last_batch_end - last_batch_start
            if last_batch_size < (
                self.windowsize_samples + self.overlap_samples
            ):
                self.nbatches -= 1
                msg = (
                    "⚠️ Last batch has less than one window. "
                    "Dropping last batch. This means that the last "
                    f"{last_batch_size / self.samplingrate} seconds of "
                    "the dataset will be lost."
                )
                self.console.log(msg)

        msg = (
            "✅ Initialized DatasetParser with "
            f"{self.nbatches} batches of {self.batchsize} seconds. "
            f"Subbatches with {self.windowsize_seconds} seconds windows and "
            f"{self.overlap_seconds} seconds overlap."
        )
        self.console.log(msg)

    def set_batches(self: Self) -> None:
        """Compute the indices for each batch and window."""
        indices = np.arange(0, self.nbatches)

        starts = indices * self.batchsize_samples
        ends = starts + self.batchsize_samples

        window_start_indices = [
            np.arange(start, end, self.windowsize_samples)
            for start, end in zip(starts, ends)
        ]

        window_end_indices = [
            (item + self.windowsize_samples + self.overlap_samples)
            for item in window_start_indices
        ]

        # check if indices are out of range
        window_end_indices = [
            ends[starts < self.datasetsize_samples]
            for ends, starts in zip(window_end_indices, window_start_indices)
        ]
        window_start_indices = [
            starts[starts < self.datasetsize_samples]
            for starts in window_start_indices
        ]
        window_start_indices = [
            starts[ends <= self.datasetsize_samples]
            for starts, ends in zip(window_start_indices, window_end_indices)
        ]
        window_end_indices = [
            ends[ends <= self.datasetsize_samples]
            for ends in window_end_indices
        ]

        self.batches = [
            list(zip(start, end))
            for start, end in zip(window_start_indices, window_end_indices)
        ]

        remaining = self.datasetsize_samples - self.batches[-1][-1][1]
        if remaining != 0:
            msg = (
                "⚠️ Last window does not end at the end of the dataset. "
                f"Dropping last window. This means that the last "
                f"{remaining / self.samplingrate} seconds of "
                "the dataset will be lost."
            )
            self.console.log(msg)

    def viz(self: Self) -> None:
        """Visualize the batches."""
        _, ax = plt.subplots()
        for i in range(self.nbatches):
            batchstart = self.batches[i][0][0]
            batchend = self.batches[i][-1][1]
            ax.axvline(batchstart, color="black", lw=1)
            ax.axvline(batchend, color="black", lw=1)
            for start, end in self.batches[i]:
                ax.axvspan(start, end, alpha=0.5, color="red", label="Windows")
        ax.axvline(0, color="blue", lw=1, label="Origin", zorder=100)
        ax.axvline(self.datasetsize_samples, color="blue", lw=1, zorder=100)
        ax.set_xlabel("Index")
        ax.set_ylabel("Batch")
        ax.set_title("Batch visualization")
        plt.show()

    def __getitem__(self: Self, index: int) -> list:
        """Get a batch of windows."""
        return self.batches[index]

    def __len__(self: Self) -> int:
        """Return the number of batches."""
        return self.nbatches

    def __repr__(self: Self) -> str:
        """Return a representation of the DatasetParser."""
        return f"DatasetParser with {self.nbatches} batches."

    def __str__(self: Self) -> str:
        """Return a string representation of the DatasetParser."""
        return (
            f"DatasetParser with {self.nbatches} batches and "
            f"{self.batchsize} windows each."
        )


def arrayparser_demo() -> None:
    """Run a test."""
    samplingrate = 20000
    length = 100023423
    batchsize = 5
    windowsize = 15
    overlap = 1
    dsp = ArrayParser(
        length, samplingrate, batchsize, windowsize, overlap, Console()
    )
    print(f"Has {dsp.nbatches} batches.")
    dsp.viz()


def main() -> None:
    """Run the demos."""
    arrayparser_demo()


if __name__ == "__main__":
    main()
