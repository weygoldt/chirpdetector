"""Parse a dataset in batches safely."""

from typing import Self

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console


class ArrayParser:
    """Generate indices to iterate large arrays in batches."""

    def __init__( # noqa
        self: Self,
        length: int,
        samplingrate: float,
        batchsize: int,
        windowsize: float,
        overlap: float,
        console: Console
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
        self.nbatches = int(np.ceil(
            self.datasetsize_samples / self.batchsize_samples)
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

        window_start_indices = [np.arange(
            start, end, self.windowsize_samples
        ) for start, end in zip(starts, ends)]

        window_end_indices = [
            (
                item + self.windowsize_samples + self.overlap_samples
            ) for item in window_start_indices
        ]

        # check if indices are out of range
        window_end_indices = [
            ends[starts < self.datasetsize_samples] for ends, starts in zip(
                window_end_indices, window_start_indices
            )
        ]
        window_start_indices = [
            starts[starts < self.datasetsize_samples] for starts in \
                window_start_indices
        ]
        window_start_indices = [
            starts[ends <= self.datasetsize_samples] for starts, ends in zip(
                window_start_indices, window_end_indices
            )
        ]
        window_end_indices = [
            ends[ends <= self.datasetsize_samples] for ends in \
            window_end_indices
        ]

        self.batches = [
            list(zip(start, end)) for start, end in zip(
                window_start_indices, window_end_indices
            )
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


def main() -> None:
    """Run a test."""
    samplingrate = 20000
    length = 100023423
    batchsize = 5
    windowsize = 15
    overlap = 1
    dsp = DatasetParser(length, samplingrate, batchsize, windowsize, overlap)
    print(f"Has {dsp.nbatches} batches.")
    dsp.viz()

if __name__ == "__main__":
    main()

