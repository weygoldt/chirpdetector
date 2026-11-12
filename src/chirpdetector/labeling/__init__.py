"""Hand-label chirp emitters on a grid of cutouts (``chirpdetector label-emitters``).

A matplotlib window shows a page of spectrogram cutouts, one v1 detection
each. Left click on the emitter's baseline = chirp from that fish, right
click = not a chirp, middle click = chirp with unclear emitter.

Modules
-------
queue
    Which detections to label, stratified by the frequency difference of
    the two closest live tracks; persisted so sessions resume exactly.
render
    Electrode-summed spectrogram cutouts.
store
    Append-only event log of clicks and the derived audian CSV files.
gui
    The matplotlib labeller.

Labelling is blind: the queue never stores v1's assigned emitter, and every
live track is marked the same way.
"""
