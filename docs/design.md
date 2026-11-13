# Design notes

Why chirpdetector works the way it does. The decisions are recorded with
their evidence in the chirp competition project's plan (decisions 1-24) and
in the module docstrings.

## Linear frequency, not mel

Audio models usually work on log-mel spectrograms, which compress high
frequencies. For electric fish that throws away the information that
matters: several fish sit within a few hundred Hz of each other, often less
than 10 Hz apart, and telling which baseline lost power during a chirp, or
which fish's frequency a chirp starts from, needs fine and uniform frequency
resolution across the band. The front end is therefore a linear-frequency
power spectrogram of 500-1500 Hz with 4.88 Hz bins.

The bins come from the long analysis window (0.2048 s), which is what
separates close fish but smears a 10-30 ms chirp over 200 ms. Shorter
windows (0.0512 s and 0.0128 s) show the chirp's shape. Rather than trading
one for the other, the three windows are computed on one frequency grid
(shorter windows zero-padded to the long window's FFT length, all frames
centred on the same sample) and stacked as the image's three channels. The
detector sees the long window's frequency resolution and the short windows'
time resolution at once.

Electrodes are summed in linear power, never in dB (v1 summed dB values,
i.e. multiplied powers). Each tile is normalised to a fixed 60 dB range
that ends 10 dB above the median over frames of the strongest fish's peak:
the anchor is on the fish, which are in every recording, so a global gain
changes nothing and the noise floor, not the fish, is what clips.

## Physical units everywhere

Every spectrogram, click-removal, plotting and footprint parameter is in
seconds and Hz; the sampling rate comes from the recording. Recordings at
any rate get the same time and frequency grid (to within rounding), so a
bundle trained at 20 kHz applies at any rate, and a model never sees a
rescaled image: images are fed at their native size, padded to the model's
stride.

## Footprint boxes

What should a chirp's box be? v1 used the template's width and height plus
fixed padding. chirpdetector derives the box from the chirp's *footprint* in
the spectrogram the detector actually sees: the simulator re-renders the
emitter over a short span with and without that chirp, both are passed
through the same front end, and the box spans the cells where the two
differ by more than 30 dB below the difference's peak, connected to the
chirp's trajectory. The box thus includes the power drop at the emitter's
baseline during the chirp and the smear of the analysis window, at whatever
window the detector uses.

The same comparison against everything else in the recording (other fish,
noise, hum, clicks) gives each chirp an SNR. Chirps below the visibility
threshold become *ignore regions*, not negatives: they neither train the
detector to hallucinate nor punish it for finding a faint chirp. Second
harmonic footprints and chirps cut by a tile edge are ignore regions too.
Both detectors mask their classification loss there.

## The emitter head and assignment

v1 assigned emitters with a separate network on hand-made features that
differed between training and inference, never evaluated on real data. In
v2 the detector itself predicts, with every box, the baseline EODf of the
fish that emitted the chirp: one more small head on each decoder query of
RT-DETRv2, regressing the emitter's frequency relative to the predicted
box's lower edge. Assignment is then the nearest live track to that
frequency, a step with no learned parameters that works with any number of
fish and any electrode layout.

RT-DETR was chosen because set prediction is NMS-free: Hungarian matching
trains one query per chirp, so near-simultaneous chirps of different fish
are never merged by non-maximum suppression (a v1 failure), and each
query's emitter belongs to exactly one chirp. The loss (matching,
varifocal and box losses on every decoder layer and the denoising groups)
is reused from `transformers`; the emitter head adds one L1 term.

On the simulated test chirps this assignment was correct for 0.914 of the
detected chirps (0.875 when the two closest fish were less than 10 Hz
apart), against 0.648 for the nearest track to the box's lower edge plus
the median offset. YOLO26, also NMS-free, is kept as the box-only
alternative; its boxes are as good, but it cannot assign.

## Simulated training data on real backgrounds

There are a few hundred hand-labelled real chirps and no real data with
known emitters.
The detectors therefore train on simulations only
([gridtools](https://github.com/weygoldt/gridtools)): real chirp templates
(not parametric shapes, which are unrealistically smooth) on simulated fish
with drift, close frequencies and crossings, placed on real background
segments of grid recordings (real noise, hum, clicks and non-chirping real
fish). Background recordings and chirp templates (by individual fish) are
split between training, validation and test, so simulated test numbers
measure new fish on new recordings. Real data is the arbiter: the
hand-labelled recordings are never used for training or model selection.

## Calibration of the chirp parameters

A regressor trained on a population shrinks its estimates toward that
population's middle, and how much depends on the training mix. Regressing
the truth on the estimate afterwards (e.g. isotonic) would not remove this:
it gives another posterior mean, again tied to the training population.
The decoder's bundle instead stores the bias curve h(y) = median(estimate |
truth y) and reports h⁻¹(estimate), which is median-unbiased at every true
value, independent of how often each value occurs. Because how much an
estimate shrinks depends on how well the chirp is seen, there is one curve
per tercile of the decoder's own uncertainty. The raw estimates stay in the
output for population analyses ([Chirp parameters](decoder.md)).

The assignment confidence is calibrated the same way in spirit: an explicit
error model (Student-t, fitted on simulated true positives) and a prior over
live tracks, checked on held-out chirps (expected calibration error 0.013).

## Models ship as bundles

Weights travel with their front end, tiling, preprocessing, threshold,
training provenance and the sha256 of the weights; metrics can only be
attached to the weights that produced the evaluated predictions. v1's
reported assignment metrics came from a different model than the deployed
one; a bundle makes that impossible.
