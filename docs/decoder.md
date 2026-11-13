# Chirp parameters

`chirpdetector.params` is a second-stage decoder that estimates, for every
detected chirp,

- **height** (`height_hz`): the largest frequency excursion above the
  emitter's baseline, in Hz;
- **duration** as full width at half height (`fwhm_s`) and width at 10 % of
  the height (`width10_s`), in s;
- **undershoot** (`undershoot_hz`): the deepest drop below the baseline
  after the peak, in Hz;

each with a calibrated standard deviation. The pipeline runs it on every
detection; `predict_span(..., params=...)` and `chirpdetector showcase
--params` do too.

## Definitions

The ground truth the decoder learns is the simulator's noise-free
instantaneous frequency of the emitter, one value per EOD period
(`chirpdetector.params.truth`); nothing is read from a spectrogram.

- Cycle boundaries are the times where the emitter's unwrapped phase
  crosses 2πm (computed with only this chirp in place, so that a neighbour
  in a burst cannot lend its peak or undershoot).
- Period k has the frequency 1/T_k; its excursion is 1/T_k minus the
  period mean of the chirp-free baseline (with drift).
- Height is the largest excursion. FWHM and the 10 % width are the times
  between the crossings of 50 % and 10 % of the height that bracket the
  peak, interpolated between period midpoints. Undershoot is the deepest
  excursion below zero after the peak (0 if there is none).

On a linear frequency flank the interpolated crossings are exact, and a
plateau of H Hz lasting a whole period gives a height of H.

## Input

The decoder sees the detector's own spectrogram (all three analysis
windows) on a 3.2-s canvas centred on the box, normalised like a detector
tile, resampled to 96 x 96 over the box plus 15 % margins, plus the log box
duration and height. `chirpdetector.params.crops.decoder_inputs` is the one
function that makes this input for training, evaluation and inference. A
decoder is trained on the boxes of one detector and only valid with it: its
bundle records the detector's weight hash, and `predict_span` and the
pipeline refuse another detector.

## Calibration

A regressor shrinks toward the middle of its training population: small
chirps are overestimated, large ones underestimated, and by how much depends
on the training mix (here: an equalised mix with many big and long chirps;
real recordings are dominated by small chirps). The bundle therefore stores,
per parameter, the bias curve h(y) = median(estimate | true value y), fitted
on the validation scenes (bin medians smoothed by a local linear fit,
slopes at least 0.5, one curve per tercile of the decoder's own uncertainty,
because poorly seen chirps shrink more), and reports h⁻¹(estimate). This
estimate is median-unbiased at every true value, so its meaning does not
depend on the mix of chirps in a recording.

The price: a wider per-chirp error than the raw estimate (the inverse of a
curve with slope below 1 amplifies the noise), and a population
distribution that is the true one blurred by that error. The uncalibrated
estimates are therefore kept as well.

Standard deviations are propagated through h⁻¹ and rescaled so that ±1 sd
covers 68 % of the validation chirps. The 95 % interval is multiplicative:
`(value + offset) * exp(±k95 * sd / (value + offset)) - offset`, with
`offset` 1 Hz for the undershoot and 0 otherwise, and `k95` in the bundle
(`calibration.k95`). `value ± k95 * sd` approximates it only for small sd.

## Output columns

| column | meaning |
|---|---|
| `height_hz`, `fwhm_s`, `width10_s`, `undershoot_hz` | calibrated estimates |
| `height_sd_hz`, `fwhm_sd_s`, `width10_sd_s`, `undershoot_sd_hz` | calibrated standard deviations (first order) |
| `height_raw_hz`, `fwhm_raw_s`, `width10_raw_s`, `undershoot_raw_hz` | uncalibrated estimates (no sd) |
| `params_ok` | the box lies in the decoder's validated domain |
| `params_flag` | why not, comma-separated: `edge` (the crop crosses the recording's start or end), `band` (the box reaches the edge of the 500-1500 Hz band: the chirp may continue beyond it), `size` (box duration or height outside the range seen in training), `uncertain` (an sd above the 99.5th percentile of the validation chirps), `harmonic` (a re-attributed harmonic copy; values NaN) |

Which to use:

- per chirp (comparisons of single chirps, regressions on chirp size): the
  calibrated columns, with `params_ok`;
- distributions over many chirps: the raw ones are pulled toward the
  training mix, the calibrated ones are blurred by the amplified error, so
  compare both; the evaluation report measures each against the truth.

## Accuracy

On 7,678 simulated test chirps of fish whose templates the decoder never
saw (detected at score >= 0.5, on real background recordings), calibrated
estimates, median absolute error: height 9.54 Hz (6.7 %), FWHM 1.16 ms
(9.8 %), 10 % width 1.58 ms (7.0 %), undershoot 1.52 Hz (36.0 %). The
shrinkage slope (estimate on truth, logs) rose from 0.933 to 0.976 for the
height and from 0.970 to 1.038 for the FWHM. Reading the parameters off the
detection box instead gives a median height error of 24 Hz and a FWHM error
of 234 ms. 5.2 % of matched test boxes were flagged outside the validated
domain.

On 513 big, long chirps extracted from the competition recordings, the
decoder agreed with an independent per-period trace estimate to a median
of 17.7 Hz (height), 24.5 ms (FWHM) and 7.5 ms (10 % width). Small real
chirps were not checked against an independent reference
([Limitations](limitations.md)).

Training and evaluating a decoder: [Training the decoder](decoder-training.md).
