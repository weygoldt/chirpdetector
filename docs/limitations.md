# Limitations

What the evaluation does and does not show, and where chirpdetector is
likely to fail.

## The evidence

- **Real detection is measured on few chirps.** 365 chirps in 6 labelled
  regions of 4 recordings (2019, 2020, 2022 tube competitions; *A.
  leptorhynchus*). 335 of them were labelled by hand before v2 existed; 30
  were found among the boxes the new detectors reported and the original
  labels missed, and confirmed by the author in `label-emitters` with the
  box shown but the models' scores and emitters hidden. Recall on the
  original 335 was 1.000 for both v2 detectors, so their real recall rests
  on these labels and the labeller's judgement. The confidence interval of
  the dual-head detector's recall at 0.5 is 0.976-0.998 (exact binomial).
- **The released bundle's attached metrics are on the 335 original labels**
  (precision 0.868, recall 1.000): 27 of its 51 "false positives" there
  were real chirps that the original labels missed.
- **Emitter assignment is measured on simulations only.** No real
  recording with known emitters exists (no single-fish or empty-tank
  recordings). The assignment accuracy (0.914, 0.875 below 10 Hz ΔF) and
  the assignment confidence are simulation numbers; simulated tracks are
  noise-free, so on real tracks the error is larger by the tracker's error
  and the confidence is an upper bound.
- **Simulations are the generator's own domain.** Test scenes use held-out
  background recordings and fish whose chirp templates were never trained
  on, but the chirp shapes, timing, movement, field and EOD waveforms come
  from the same simulator.
- **Chirp parameters are validated on simulations**, and on real data only
  against an independent trace estimate of 513 big, long chirps (mostly of
  one fish). Real small chirps have no independent reference.
- **The human benchmark** used one expert labeller on simulated items.

## Where it can fail

- **Recordings unlike the training data.** The training backgrounds come
  from the 2019, 2020 and 2022 tube competitions and from two field
  deployments; chirp templates from *A. leptorhynchus* (and a group at
  320-440 Hz that may be *Eigenmannia*). Other species, recording systems,
  sampling chains or very different noise are untested. The front end
  covers 500-1500 Hz: fish whose baseline lies outside it are not seen.
- **Close fish.** Assignment is hardest when the two closest fish are less
  than 10 Hz apart (0.875 correct in simulation, against 0.97-1.00 above
  10 Hz).
- **Tracks.** Assignment needs wavetracker tracks. A chirp of an untracked
  fish, or of a fish whose track is in a gap longer than 10 s, is assigned
  to another fish within 20 Hz or left unassigned; a tracker that follows
  chirp excursions or swaps identities propagates into the assignment.
- **Harmonics.** Second-harmonic copies are dropped using the tracks
  (the pair rule without tracks only flags them). A fish within 50 Hz of
  another fish's second harmonic protects that fish's copies, which are then
  kept. In the 2019 recordings the fish lie at 582-903 Hz, so this does not
  occur there.
- **Clicks with few electrodes.** Click removal needs at least 9
  electrodes; below that the detector has to reject clicks itself.
- **Recording edges.** Chirps within half a tile of a recording's start or
  end are seen with less context; the decoder flags crops that cross the
  edge (`params_flag = "edge"`).
- **Box-only models** (YOLO26) cannot assign emitters in the pipeline.
