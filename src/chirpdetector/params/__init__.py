"""Exploratory: decode a chirp's physiological parameters from its spectrogram.

Parameters (definitions in `chirpdetector.params.truth`): height (peak
excursion above the emitter's baseline, Hz), width (FWHM and 10 % width of
the excursion, s) and undershoot (deepest drop below baseline, Hz).

Modules
-------
truth
    Parameter definitions on an exact frequency trace.
crops
    Inputs for a decoder: the baseline-rectified multi-resolution crop
    around a detection box, and the detector-view crop.
estimators
    Non-learned baselines (box arithmetic and ridge tracking).
model
    A compact CNN regressor with heteroscedastic Gaussian heads.
"""
