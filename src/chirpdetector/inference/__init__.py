"""Run a model bundle on recordings.

`predict_span` detects chirps in a time span of a recording, the way the
detector datasets were made: the bundle's preprocessing (click removal
where the grid allows it), its spectrogram front end, its tiling (tiles of
a fixed number of frames with a fixed overlap, each normalised on its own
frames), the model, and the exact pixel -> time/frequency map. Detections
of overlapping tiles are de-duplicated (`merge_tiles`).

Given a params bundle, `predict_span` also estimates every detection's
height, widths and undershoot (`chirpdetector.params`).
`detect_in_spectrogram` is the detection step on an already computed
spectrogram (it makes the decoder's training boxes).

Streaming inference over whole recordings (chunked, with global
de-duplication and the legacy output tables) builds on this later.
"""

from chirpdetector.inference.spans import (
    DUPLICATE_TOLERANCE_S,
    TiledSpan,
    detect_in_spectrogram,
    merge_tiles,
    predict_span,
    render_span,
    render_tiles,
    span_bounds,
    span_spectrogram,
    tile_starts,
)

__all__ = [
    "DUPLICATE_TOLERANCE_S",
    "TiledSpan",
    "detect_in_spectrogram",
    "merge_tiles",
    "predict_span",
    "render_span",
    "render_tiles",
    "span_bounds",
    "span_spectrogram",
    "tile_starts",
]
