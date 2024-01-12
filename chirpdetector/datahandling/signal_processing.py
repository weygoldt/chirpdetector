"""Utilities for signal processing."""

from typing import Tuple, TypeVar

import numpy as np
import torch
from gridtools.datasets.models import Dataset
from gridtools.utils.spectrograms import compute_spectrogram, sint, to_decibel
from scipy.signal import butter, sosfiltfilt

ArrayLike = TypeVar("ArrayLike", np.ndarray, torch.Tensor)


def zscore_standardize(array: ArrayLike) -> ArrayLike:
    """Z-score standardize a matrix.

    Parameters
    ----------
    - `array` : `ArrayLike`
        The array to be standardized.

    Returns
    -------
    - `ArrayLike`
        The standardized array.
    """
    return (array - array.mean()) / array.std()


def bandpass_filter(
    signal: np.ndarray,
    samplerate: float,
    lowf: float,
    highf: float,
) -> np.ndarray:
    """Bandpass filter a signal.

    Parameters
    ----------
    signal : np.ndarray
        The data to be filtered
    rate : float
        The sampling rate
    lowf : float
        The low cutoff frequency
    highf : float
        The high cutoff frequency

    Returns
    -------
    np.ndarray
        The filtered data
    """
    sos = butter(2, (lowf, highf), "bandpass", fs=samplerate, output="sos")
    return sosfiltfilt(sos, signal)


def envelope(
    signal: np.ndarray,
    samplerate: float,
    cutoff_frequency: float,
) -> np.ndarray:
    """Calculate the envelope of a signal using a lowpass filter.

    Parameters
    ----------
    signal : np.ndarray
        The signal to calculate the envelope of
    samplingrate : float
        The sampling rate of the signal
    cutoff_frequency : float
        The cutoff frequency of the lowpass filter

    Returns
    -------
    np.ndarray
        The envelope of the signal
    """
    sos = butter(2, cutoff_frequency, "lowpass", fs=samplerate, output="sos")
    return np.sqrt(2) * sosfiltfilt(sos, np.abs(signal))


def make_chunk_indices(
    n_chunks: int,
    current_chunk: int,
    chunksize: int,
    window_overlap_samples: int,
    max_end: int,
) -> Tuple[int, int]:
    """Get start and stop indices for the current chunk.

    Parameters
    ----------
    - `n_chunks` : `int`
        The number of chunks.
    - `current_chunk` : `int`
        The current chunk number.
    - `chunksize` : `int`
        The chunk size in samples.
    - `window_overlap_samples` : `int`
        The window overlap in samples.
    - `max_end` : `int`
        The maximum end index, i.e. the length of the recording.

    Returns
    -------
    - `Tuple[int, int]`
        The start and stop indices.
    """
    if current_chunk == 0:
        start = sint(current_chunk * chunksize)
        stop = sint((current_chunk + 1) * chunksize + window_overlap_samples)
    elif current_chunk == n_chunks - 1:
        start = sint(current_chunk * chunksize - window_overlap_samples)
        stop = sint((current_chunk + 1) * chunksize)
    else:
        start = sint(current_chunk * chunksize - window_overlap_samples)
        stop = sint((current_chunk + 1) * chunksize + window_overlap_samples)

    if stop > max_end:
        stop = max_end

    return start, stop


def make_spectrogram_axes(
    start: int, stop: int, nfft: int, hop_length: int, samplerate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the time and frequency axes of a spectrogram.

    Parameters
    ----------
    - `start` : `int`
        The start index.
    - `stop` : `int`
        The stop index.
    - `nfft` : `int`
        The number of samples in the FFT.
    - `hop_length` : `int`
        The hop length in samples.
    - `samplerate` : `float`
        The sampling rate.

    Returns
    -------
    - `Tuple[np.ndarray, np.ndarray]`
        The time and frequency axes.
    """
    spectrogram_times = np.arange(start, stop + 1, hop_length) / samplerate
    spectrogram_freqs = np.arange(0, nfft / 2 + 1) * samplerate / nfft
    return spectrogram_times, spectrogram_freqs


def spec_to_image(spec: torch.Tensor) -> torch.Tensor:
    """Convert a spectrogram to an image.

    Add 3 color channels, normalize to 0-1, etc.

    Parameters
    ----------
    - `spec` : `torch.Tensor`

    Returns
    -------
    - `torch.Tensor`
    """
    # make sure the spectrogram is a tensor
    if not isinstance(spec, torch.Tensor):
        msg = (
            "The spectrogram must be a torch.Tensor.\n"
            f"Type of spectrogram: {type(spec)}"
        )
        raise TypeError(msg)

    # make sure the spectrogram is 2-dimensional
    spec_dims = 2
    if len(spec.size()) != spec_dims:
        msg = (
            "The spectrogram must be a 2-dimensional matrix.\n"
            f"Shape of spectrogram: {spec.size()}"
        )
        raise ValueError(msg)

    # Get the dimensions of the original matrix
    original_shape = spec.size()

    # Calculate the number of rows and columns in the matrix
    num_rows, num_cols = original_shape

    # duplicate the matrix 3 times
    spec = spec.repeat(3, 1, 1)

    # Reshape the matrix to the desired shape (3, num_rows, num_cols)
    desired_shape = (3, num_rows, num_cols)
    reshaped_tensor = spec.view(desired_shape)

    # normalize the spectrogram to be between 0 and 1
    normalized_tensor = (reshaped_tensor - reshaped_tensor.min()) / (
        reshaped_tensor.max() - reshaped_tensor.min()
    )

    # make sure image is float32
    return normalized_tensor.float()

# def compute_spectrogram_worker(
#     args: Tuple[int, np.ndarray, float, int, int]
# ) -> torch.Tensor:
#     """Compute the spectrogram for a single electrode.
#
#     Parameters
#     ----------
#     - `args` : `Tuple[int, np.ndarray, float, int, int]`
#         The arguments for the worker.
#
#     Returns
#     -------
#     - `torch.Tensor`
#         The spectrogram.
#     """
#     _, signal, sampling_rate, nfft, hop_len = args
#     electrode_spectrogram, _, _ = compute_spectrogram(
#         data=signal.copy(),
#         samplingrate=sampling_rate,
#         nfft=nfft,
#         hop_length=hop_len,
#     )
#     return electrode_spectrogram
#
#
# def compute_sum_spectrogram_parallel(
#     data: Dataset, nfft: int, hop_len: int
# ) -> torch.Tensor:
#     """Compute the sum spectrogram of a chunk.
#
#     Parameters
#     ----------
#     - `data` : `Dataset`
#         The dataset to make bounding boxes for.
#     - `nfft` : `int`
#         The number of samples in the FFT.
#     - `hop_len` : `int`
#
#     Returns
#     -------
#     - `torch.tensor`
#         The sum spectrogram.
#     """
#     with suppress(RuntimeError):
#         set_start_method("spawn")
#
#     n_electrodes = data.grid.rec.shape[1]
#
#     # Create arguments for parallel processing
#     args_list = [
#         (
#             electrode,
#             data.grid.rec[:, electrode],
#             data.grid.samplerate,
#             nfft,
#             hop_len,
#         )
#         for electrode in range(n_electrodes)
#     ]
#
#     # Use multiprocessing to parallelize spectrogram computation
#     with Pool() as pool:
#         electrode_spectrograms = pool.map(
#             compute_spectrogram_worker, args_list
#         )
#
#     # Sum spectrograms over all electrodes
#     spectrogram = torch.sum(torch.stack(electrode_spectrograms), dim=0)
#
#     # Normalize spectrogram by the number of electrodes
#     spectrogram /= n_electrodes
#
#     # Convert the spectrogram to dB
#     return to_decibel(spectrogram)
#
#
def compute_sum_spectrogam(
    data: Dataset, nfft: int, hop_len: int
) -> torch.Tensor:
    """Compute the sum spectrogram of a chunk.

    Parameters
    ----------
    - `chunk` : `Dataset`
        The dataset to make bounding boxes for.
    - `nfft` : `int`
        The number of samples in the FFT.
    - `hop_len` : `int`
        The hop length in samples.

    Returns
    -------
    - `torch.tensor`
        The sum spectrogram.
    """
    n_electrodes = data.grid.rec.shape[1]
    spectrogram = None
    for electrode in range(n_electrodes):
        # get the signal for the current electrode
        signal = data.grid.rec[:, electrode]

        # compute the spectrogram for the current electrode
        electrode_spectrogram, _, _ = compute_spectrogram(
            data=signal.copy(),
            samplingrate=data.grid.samplerate,
            nfft=nfft,
            hop_length=hop_len,
        )

        # sum spectrogram over all electrodes
        # the spec is a tensor
        if electrode == 0:
            spectrogram = electrode_spectrogram
        else:
            spectrogram += electrode_spectrogram

    if spectrogram is None:
        msg = "Failed to compute spectrogram."
        raise ValueError(msg)

    # normalize spectrogram by the number of electrodes
    # the spec is still a tensor
    spectrogram /= n_electrodes

    # convert the spectrogram to dB
    # .. still a tensor
    return to_decibel(spectrogram)
