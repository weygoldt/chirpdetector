# Post-processing

Two kinds of false positives survive the detector and the merge of
overlapping tiles: copies of chirps on a fish's second EOD harmonic, and
two boxes on one chirp. `chirpdetector.postprocess` flags both with a
reason; `predict_span` drops the flagged rows by default, and the pipeline
keeps them in its table with `kept == False` and the reason in
`pp_drop_reason`. Nothing is dropped silently.

```python
from chirpdetector.postprocess import PostprocessConfig

predict_span(..., postprocess=False)  # off
predict_span(..., keep_dropped=True)  # flag only, keep every row
predict_span(..., postprocess=PostprocessConfig(trackless="drop"))  # other settings
```

## Emitter frequency of a detection

Both rules need the frequency of a detection's emitter. The dual-head
detector predicts it (`emitter_hz`, *head* rows). For a box-only model
(YOLO26) the emitter's baseline lies a little above the box's lower edge,
because the box includes the power drop at the baseline and the analysis
window's smear: on the simulated test tiles the true baseline was 41 Hz
above `f_low_hz` (median; 0.5 % and 99.5 % quantiles 18 and 142 Hz), so
*edge* rows use windows relative to the lower edge instead of a single
frequency.

## Harmonic copies

A wave-type fish's EOD has harmonics. The second harmonic of a fish below
750 Hz lies inside the detector's band (500-1500 Hz), and every chirp of
that fish appears there too: at the same time, at twice the baseline, with
twice the excursion. Training treated these copies as ignore regions, so
the detector is free to box them, and the dual-head detector often does,
with high scores.

A detection is a harmonic copy if

1. twice the frequency of a track alive at its centre time lies in its
   harmonic interval, and
2. no track alive at that time lies in its fundamental interval, and it has
   no assigned emitter.

| interval | head rows (`emitter_hz` = e) | edge rows (`f_low_hz` = l) |
|---|---|---|
| harmonic | e ± 15 Hz | l + 15 to l + 170 Hz |
| fundamental | e ± 50 Hz | l - 20 to l + 170 Hz |

Rule 2 protects every detection whose emitter matches a live fundamental,
including the chirps of a fish that happens to sit at twice another fish's
frequency. The tolerances come from the simulated test tiles and the four
hand-labelled recordings; `chirpdetector.postprocess` documents the
measurements behind each one.

A track is alive across gaps of up to 10 s (`max_gap_s`, the pipeline's
liveness rule). wavetracker tracks have gaps; with a 1-s rule, 45 of the
dual-head's 335 true positives on the hand labels had no live track within
15 Hz of their `emitter_hz`, with 10 s all lay within 6.4 Hz of one.

### Re-attribution

A copy whose fundamental was not detected is the only trace of a real
chirp. With an emitter head, such a copy is kept and reported at the
fundamental instead of dropped: `emitter_hz`, `f_low_hz` and `f_high_hz`
are halved (the detected values move to `harmonic_emitter_hz`,
`harmonic_f_low_hz`, `harmonic_f_high_hz`), the emitter is the track whose
harmonic it is, and `reattributed == "harmonic"`. The decoder never saw
harmonic crops, so its values on these rows are NaN (`params_flag =
"harmonic"`).

A copy counts as detected at the fundamental if a kept detection overlaps
it in time (temporal IoU >= 0.3) and either spans the fish's frequency or
is assigned to it. Re-attribution is on by default for emitter-head models
and off for box-only ones (`reattribute_harmonics=None`; `True`/`False`
override): on the simulated test tiles at score 0.5, the dual-head detector
re-attributed 1 copy, which matched its missed chirp, while YOLO26s
re-attributed 12, all false positives. On the hand-labelled recordings and
about 12 h of dry runs nothing was re-attributed: a fish's fundamental is
much stronger than its second harmonic and is practically always detected.

Re-attribution depends on which fundamentals are detected, so, unlike the
other rules, it does not commute with the score threshold: it is applied to
the reported detections (`predict_span` at its threshold, the pipeline at
the operating threshold).

### Without tracks

Without tracks, rule 2 cannot be checked. The *pair rule* then marks a
detection that overlaps another, at least as confident detection in time
(temporal IoU >= 0.5) at half its emitter frequency (head rows: within
20 Hz; edge rows: with `f_low_hz + 41 Hz` as the emitter, within 40 Hz).
It cannot tell a copy from the chirp of a second fish at twice the
frequency (e.g. *Eigenmannia* at 300-600 Hz next to *A. leptorhynchus* at
600-1000 Hz), so by default it only flags (`harmonic_pair_of`);
`trackless="drop"` drops as well, `"off"` skips it.

## Duplicates

Two boxes on one chirp. Detections are visited by decreasing score; a
detection is a duplicate of an already kept one if the two overlap in
frequency and

- head rows (both): `emitter_hz` within 5 Hz (about the 90 % quantile of
  the emitter error), the same assigned emitter, and temporal IoU >= 0.7;
- otherwise: lower edges within 2 Hz, temporal IoU >= 0.5 and frequency IoU
  >= 0.5.

The higher-scoring one is kept. As in greedy non-maximum suppression, the
result at any threshold equals post-processing only the detections above
it. Two chirps of one fish in quick succession and simultaneous chirps of
two fish are not merged: "centre inside the other box" would have merged
7.8 % of the dual-head's pairs on two chirps of one fish, so the rules use
overlap thresholds that merged none of the measured pairs on different
chirps.

## Columns

| column | meaning |
|---|---|
| `drop_reason` | `""` (kept), `"harmonic"` or `"duplicate"` |
| `harmonic_of_id` | the track whose second harmonic the detection lies on |
| `harmonic_pair_of` | pair rule: row of the detection at half the frequency |
| `duplicate_of` | row of the kept twin |
| `reattributed` | `"harmonic"` for a re-attributed copy |
| `harmonic_emitter_hz`, `harmonic_f_low_hz`, `harmonic_f_high_hz` | the detected values of a re-attributed copy |

In the pipeline's table these are `pp_drop_reason`, `pp_harmonic_of_track`,
`pp_harmonic_pair_of`, `pp_duplicate_of` (as `chirp_id`s),
`pp_reattributed` and `pp_harmonic_*`.

## Effect

On the simulated test tiles (9,401 chirps) at score 0.5, post-processing
removed 111 of the dual-head detector's 336 false positives and one true
positive (precision 0.964 -> 0.976, recall 0.957 -> 0.957), and 72 of
YOLO26s's 173 (precision 0.981 -> 0.989, recall 0.952). On the hand-labelled
recordings no true positive was lost at any threshold; the copies there lie
above the labelled bands and did not count either way.
