"""Utilities for signal processing."""

import pathlib

import torch
import numpy as np

from scipy.signal import butter, sosfiltfilt


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
    filtered_signal = sosfiltfilt(sos, signal)

    return filtered_signal


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
    envelope = np.sqrt(2) * sosfiltfilt(sos, np.abs(signal))

    return envelope
