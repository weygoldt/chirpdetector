"""
Specifies the output data model for the application.

This module contains the class definitions for the output data model
as well as functions and classes to conveniently serialize and
deserialize the output data model to and from HDF5 files.
"""

import pathlib
from typing import Self, Union
import h5py
import numpy as np
from numpy import typing as npt
import pandas as pd
from pydantic import BaseModel
from dataclasses import dataclass

class ChirpDataset:
    """
    A class to hold the extracted chirps of a single dataset.

    This object holds the spectrogram of each chirp, the bounding boxes of
    the chirps, the assigned emitter for each chirp and much more.
    """

    def __init__(  # noqa
        self: Self,
        recording: str,
        detector: str,
        are_detected: bool,
        info: str,
        spectrogram_batch: npt.ArrayLike,
        waveform_indices: npt.ArrayLike,
        spectrogram_batch_index: npt.ArrayLike,
        spectrogram_window: npt.ArrayLike,
        spectrogram_frequency_range: npt.ArrayLike,
        bbox_id: npt.ArrayLike,
        bbox_xyxy: npt.ArrayLike,
        bbox_ftft: npt.ArrayLike,
        bbox_confidence: npt.ArrayLike,
        bbox_spec_powers: npt.ArrayLike,
        bbox_spec_times: npt.ArrayLike,
        bbox_spec_freqs: npt.ArrayLike,
        bbox_spec_orig_shape: npt.ArrayLike,
        assigned_emitter_eodf: npt.ArrayLike,
        assigned_emitter_id: npt.ArrayLike,
    ) -> None:
        """
        Initialize a ChirpDataset object.

        This object holds the extracted chirps of a single dataset, i.e. a
        single recording. The object contains the spectrogram of each chirp,
        the bounding boxes of the chirps, and the assigned emitter for each
        chirp.

        The data arrays can either be numpy arrays or h5py datasets. The
        h5py datasets are used to load the data from disk only when needed.
        This is useful when the data is too large to fit into memory.

        Parameters
        ----------
        - `recording` : `str`
            The name of the corresponding recording, i.e. name of the directory
            containing the raw data files.
        - `detector` : `str`
            The name of the detector that was run to generate the data. Could
            be e.g. YOLOv8 or Faster-RCNN.
        - `are_detected` : `bool`
            False if the detector was run yet, True if the detector was run and
            chirps were detected. Why is this here? Well, if we load the
            ChirpDataset from many subdatasets in a loop and some have no
            ChirpDataset, yet, we still return a ChirpDataset object instead of
            None. If you process many datasets in a loop, you can check this
            flag to see if the detector was run yet.
        - `info` : `str`
            Additional information about the dataset.
        - `spectrogram_batch` : `npt.ArrayLike`
            The batch number that the spectrogram belongs to.
        - `waveform_indices` : `npt.ArrayLike`
            The start and stop indices of the raw recording from which the
            batch was extracted.
        - `spectrogram_batch_index` : `npt.ArrayLike`
            The index of the spectrogram within the batch.
        - `spectrogram_window` : `npt.ArrayLike`
            The window of the spectrogram, i.e. the the index to the frequency
            range of the spectrogram. The next attribute
            `spectrogram_frequency_range` might be more informative.
        - `spectrogram_frequency_range` : `npt.ArrayLike`
            The frequency range of the spectrogram.
        - `bbox_id` : `npt.ArrayLike`
            The id of the bounding box.
        - `bbox_xyxy` : `npt.ArrayLike`
            The bounding box in xyxy format as pixel coordinates.
        - `bbox_ftft` : `npt.ArrayLike`
            The bounding box in ftft format in time and frequency in seconds
            and Hz.
        - `bbox_confidence` : `npt.ArrayLike`
            The confidence of the bounding box as given by the object detector.
        - `bbox_spec_powers` : `npt.ArrayLike`
            The powers of the spectrogram within the bounding box interpolated
            to be a square matrix.
        - `bbox_spec_times` : `npt.ArrayLike`
            The times of the spectrogram within the bounding box.
        - `bbox_spec_freqs` : `npt.ArrayLike`
            The frequencies of the spectrogram within the bounding box.
        - `bbox_spec_orig_shape` : `npt.ArrayLike`
            The original shape of the spectrogram within the bounding box.
        - `assigned_emitter_eodf` : `npt.ArrayLike`
            The eodf of the assigned emitter.
        - `assigned_emitter_id` : `npt.ArrayLike`
            The wavetracker ID of the assigned emitter.
        """
        self.recording = recording
        self.detector = detector
        self.are_detected = are_detected
        self.info = info

        self.spectrogram_batch = spectrogram_batch
        self.waveform_indices = waveform_indices
        self.spectrogram_batch_index = spectrogram_batch_index
        self.spectrogram_window = spectrogram_window
        self.spectrogram_frequency_range = spectrogram_frequency_range
        self.bbox_id = bbox_id
        self.bbox_xyxy = bbox_xyxy
        self.bbox_ftft = bbox_ftft
        self.bbox_confidence = bbox_confidence
        self.bbox_spec_powers = bbox_spec_powers
        self.bbox_spec_times = bbox_spec_times
        self.bbox_spec_freqs = bbox_spec_freqs
        self.bbox_spec_orig_shape = bbox_spec_orig_shape
        self.assigned_emitter_eodf = assigned_emitter_eodf
        self.assigned_emitter_id = assigned_emitter_id

    def __repr__(self: Self) -> str:
        """Return a string representation of the object."""
        return f"ChirpDataset({self.recording})"

    def __str__(self: Self) -> str:
        """Return a string representation of the object."""
        return f"ChirpDataset({self.recording})"

    def _check_health(self: Self) -> None:
        """Check the health of the object."""
        #TODO: Implement this method
        pass


class ChirpDatasetSaver:
    """Methods to handle saving of the ChirpDataset object to disk."""

    def __init__(
        self: Self, chirp_dataset: ChirpDataset, path: pathlib.Path
    ) -> None:
        """Initialize the ChirpDatasetSaver object."""
        self.chirp_dataset = chirp_dataset
        self.path = path

    def save(self: Self) -> None:
        """Save the ChirpDataset object to disk."""
        # Check if the path exists
        if not self.path.exists():
            msg = f"The path {self.path} does not exist."
            raise FileNotFoundError(msg)

        # Check if a chirps.h5 file exists
        chirps_h5 = self.path / "chirps.h5"
        if chirps_h5.exists():
            self._append_h5()
        else:
            self._init_h5()

    def _init_h5(self: Self) -> None:
        """Initialize the HDF5 file."""
        with h5py.File(self.path / "chirps.h5", "w") as f:
            # save the metadata
            f.create_dataset("recording", data=self.chirp_dataset.recording)
            f.create_dataset("detector", data=self.chirp_dataset.detector)
            f.create_dataset(
                "are_detected",
                data=self.chirp_dataset.are_detected,
                dtype="b1",
            )
            f.create_dataset("info", data=self.chirp_dataset.info)

            # save the data
            f.create_dataset(
                "spectrogram_batch",
                data=self.chirp_dataset.spectrogram_batch,
                dtype=np.int8,
                chunks=True,
                maxshape=(None,),
            )
            f.create_dataset(
                "waveform_indices",
                data=self.chirp_dataset.waveform_indices,
                dtype=np.int8,
                chunks=True,
                maxshape=(None, 2),
            )
            f.create_dataset(
                "spectrogram_batch_index",
                data=self.chirp_dataset.spectrogram_batch_index,
                dtype=np.int8,
                chunks=True,
                maxshape=(None,),
            )
            f.create_dataset(
                "spectrogram_window",
                data=self.chirp_dataset.spectrogram_window,
                dtype=np.int8,
                chunks=True,
                maxshape=(None,),
            )
            f.create_dataset(
                "spectrogram_frequency_range",
                data=self.chirp_dataset.spectrogram_frequency_range,
                dtype=np.float32,
                chunks=True,
                maxshape=(None, 2),
            )
            f.create_dataset(
                "bbox_id",
                data=self.chirp_dataset.bbox_id,
                dtype=np.int8,
                chunks=True,
                maxshape=(None,),
            )
            f.create_dataset(
                "bbox_xyxy",
                data=self.chirp_dataset.bbox_xyxy,
                dtype=np.float32,
                chunks=True,
                maxshape=(None, 4),
            )
            f.create_dataset(
                "bbox_ftft",
                data=self.chirp_dataset.bbox_ftft,
                dtype=np.float32,
                chunks=True,
                maxshape=(None, 4),
            )
            f.create_dataset(
                "bbox_confidence",
                data=self.chirp_dataset.bbox_confidence,
                dtype=np.float32,
                chunks=True,
                maxshape=(None,),
            )
            f.create_dataset(
                "bbox_spec_powers",
                data=self.chirp_dataset.bbox_spec_powers,
                dtype=np.float32,
                chunks=True,
                maxshape=(None, 512, 512),
            )
            f.create_dataset(
                "bbox_spec_times",
                data=self.chirp_dataset.bbox_spec_times,
                dtype=np.float32,
                chunks=True,
                maxshape=(None, 512),
            )
            f.create_dataset(
                "bbox_spec_freqs",
                data=self.chirp_dataset.bbox_spec_freqs,
                dtype=np.float32,
                chunks=True,
                maxshape=(None, 512),
            )
            f.create_dataset(
                "bbox_spec_orig_shape",
                data=self.chirp_dataset.bbox_spec_orig_shape,
                dtype=np.int8,
                chunks=True,
                maxshape=(None, 2),
            )
            f.create_dataset(
                "assigned_emitter_eodf",
                data=self.chirp_dataset.assigned_emitter_eodf,
                dtype=np.float32,
                chunks=True,
                maxshape=(None,),
            )
            f.create_dataset(
                "assigned_emitter_id",
                data=self.chirp_dataset.assigned_emitter_id,
                dtype=np.int8,
                chunks=True,
                maxshape=(None,),
            )
            f.close()

    def _append_h5(self: Self) -> None:
        """Append the ChirpDataset object to an existing HDF5 file."""
        # open the HDF5 file
        with h5py.File(self.path / "chirps.h5", "a") as f:
            # append the data
            resize_append(
                f["spectrogram_batch"], self.chirp_dataset.spectrogram_batch
            )
            resize_append(
                f["waveform_indices"], self.chirp_dataset.waveform_indices
            )
            resize_append(
                f["spectrogram_batch_index"],
                self.chirp_dataset.spectrogram_batch_index,
            )
            resize_append(
                f["spectrogram_window"], self.chirp_dataset.spectrogram_window
            )
            resize_append(
                f["spectrogram_frequency_range"],
                self.chirp_dataset.spectrogram_frequency_range,
            )
            resize_append(f["bbox_id"], self.chirp_dataset.bbox_id)
            resize_append(f["bbox_xyxy"], self.chirp_dataset.bbox_xyxy)
            resize_append(f["bbox_ftft"], self.chirp_dataset.bbox_ftft)
            resize_append(
                f["bbox_confidence"], self.chirp_dataset.bbox_confidence
            )
            resize_append(
                f["bbox_spec_powers"], self.chirp_dataset.bbox_spec_powers
            )
            resize_append(
                f["bbox_spec_times"], self.chirp_dataset.bbox_spec_times
            )
            resize_append(
                f["bbox_spec_freqs"], self.chirp_dataset.bbox_spec_freqs
            )
            resize_append(
                f["bbox_spec_orig_shape"],
                self.chirp_dataset.bbox_spec_orig_shape,
            )
            resize_append(
                f["assigned_emitter_eodf"],
                self.chirp_dataset.assigned_emitter_eodf,
            )
            resize_append(
                f["assigned_emitter_id"],
                self.chirp_dataset.assigned_emitter_id,
            )
            f.close()


def resize_append(h5_dataset: h5py.Dataset, data: npt.NDArray) -> None:
    """Resize and append a numpy array to an h5py dataset."""
    if not isinstance(h5_dataset, h5py.Dataset):
        msg = "The first argument must be an h5py dataset."
        raise TypeError(msg)
    if not isinstance(data, np.ndarray):
        msg = "The second argument must be a numpy array."
        raise TypeError(msg)

    h5_dataset.resize((h5_dataset.shape[0] + data.shape[0]), axis=0)
    h5_dataset[-data.shape[0] :] = data
