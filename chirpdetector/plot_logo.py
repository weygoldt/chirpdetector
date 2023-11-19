#!/usr/bin/env python

"""
Plot the logo of the chirpdetector package.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from thunderfish.fakefish import wavefish_eods, chirps
from gridtools.utils.spectrograms import (
    decibel,
    freqres_to_nfft,
    overlap_to_hoplen,
    sint,
    spectrogram,
)


def make_signal():
    """
    Make a signal with a chirp.

    Returns
    -------
    - `sig` : `np.ndarray`
        The signal.
    """

    f, a = chirps(
        eodf=500,
        samplerate=20000,
        duration=5,
        chirp_freq=0.2,
        chirp_size=150,
        chirp_width=0.03,
        chirp_kurtosis=2,
        chirp_contrast=0.8,
    )

    sig = wavefish_eods(
        fish="Alepto",
        frequency=f,
        samplerate=20000,
        duration=5,
        phase0=0,
        noise_std=0.01,
    )

    sig *= a
    return sig, f


def make_spectrogram(sig):
    """
    Make a spectrogram of the signal.

    Parameters
    ----------
    - `sig` : `np.ndarray`
        The signal.

    Returns
    -------
    - `spec` : `np.ndarray`
        The spectrogram.
    - `freq` : `np.ndarray`
        The frequencies.
    - `time` : `np.ndarray`
        The times.
    """

    nfft = freqres_to_nfft(6, 20000)
    hop_len = overlap_to_hoplen(0.9, 20000)

    spec, freq, time = spectrogram(
        data=sig,
        samplingrate=20000,
        nfft=nfft,
        hop_length=hop_len,
    )

    spec = decibel(spec)
    spec = spec.cpu().numpy()

    return spec, freq, time


def plot_logo(spec, freq, time, f):
    """
    Plot the logo.

    Parameters
    ----------
    - `spec` : `np.ndarray`
        The spectrogram.
    - `freq` : `np.ndarray`
        The frequencies.
    - `time` : `np.ndarray`
        The times.
    """

    ftime = np.linspace(0, 5, len(f))
    print(f)

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.imshow(
        spec,
        aspect="auto",
        cmap="magma",
        origin="lower",
        extent=[freq[0], freq[-1], time[0], time[-1]],
        interpolation="gaussian",
    )
    ax.plot(ftime, f, color="black", lw=1.5, zorder=1000)
    ax.set_xlim(1.5, 3.5)
    ax.set_ylim(350, 750)
    plt.savefig(
        "../assets/logo_raw.svg", dpi=300, bbox_inches="tight", transparent=True
    )


def main():
    """
    Main function.
    """

    sig, f = make_signal()
    spec, freq, time = make_spectrogram(sig)
    plot_logo(spec, freq, time, f)


if __name__ == "__main__":
    main()
