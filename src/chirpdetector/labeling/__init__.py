"""Hand-label chirps and their emitters (``chirpdetector label-emitters``).

One candidate box at a time, in a maximised matplotlib window or in a
browser (served locally, for an SSH port-forward): click on the baseline of
the emitting fish (the click snaps to the nearest live track) or press the
track's number; keys for "not a chirp", "emitter unclear", skip and undo.

Modules
-------
queue
    Which v1 detections to label, stratified by the frequency difference of
    the two closest live tracks; persisted so sessions resume exactly.
candidates
    The candidates file: any list of boxes to label (stable format).
render
    Electrode-summed spectrograms of a time-frequency window.
view
    What a single-candidate frontend shows (overview and baseline zoom,
    numbered tracks) and how a click maps to a frequency and a track.
session
    Queue position, actions, skip, undo and rendering cache shared by the
    frontends.
store
    Append-only event log and the derived audian CSV files.
single
    The matplotlib single-candidate frontend.
server
    The browser frontend (stdlib HTTP server and ``labeller.html``).
gui
    The former matplotlib grid of cutouts (``--view grid``).
summary
    Counts of labels by decision, source and ΔF bin.

Labelling is blind by default: v1 queues never store v1's assigned emitter,
every live track is marked the same way and numbered by frequency, and a
candidate's predicted emitter is shown only on request (recorded with each
label as ``prediction_shown``).
"""
