<p align="center">
  <img src="assets/logo.png" width="100" alt="chirpdetector logo">
</p>

# chirpdetector

Detect chirps of wave-type weakly electric fish in multi-electrode grid
recordings and assign each chirp to the fish that emitted it. Ships with the
tools to build training data from simulations, train and evaluate the models,
and label real data.

> **Status:** version 2 is being rebuilt on the `v2` branch. The code used for
> the chirp competition paper (2024) is available as tag
> `paper-chirp-competition-v1`.

## Installation

```sh
uv add "chirpdetector @ git+https://github.com/weygoldt/chirpdetector@v2"
```

## Labelling chirp emitters

A matplotlib window shows pages of spectrogram cutouts of chirpdetector v1
detections. Left click on the baseline the chirp rises from (chirp, emitter
= nearest live track), right click if it is not a chirp, middle click if the
emitter is unclear; space for the next page, `u` to undo, `h` for help.
Every click is saved at once; the same command resumes where you stopped.

```sh
uv sync --extra label
uv run chirpdetector label-emitters /mnt/data2/2019_tube_competition/2019-12-09-10_00 \
    --out /mnt/data1/chirp_v2/labels/emitters --per-bin 20 --seed 0 --labeler pw
```

## Evaluating detections

`chirpdetector evaluate` scores predictions against hand labels (audian files
with `labeled_region` spans), `label-emitters` output, or simulated scenes
(`chirps.csv`). The default matching rule is that of the v1 detector
benchmark: centre times within 100 ms and overlapping frequency ranges,
matched one-to-one (most pairs, then smallest total |Δt|); only predictions
inside a labelled region and its frequency band count.

```sh
uv run chirpdetector evaluate /mnt/data2/2019_tube_competition/2019-12-09-10_00 \
    --labels /mnt/data2/2019_tube_competition/2019-12-09-10_00/traces-grid1-editable-labels.csv \
    --out /mnt/data1/chirp_v2/eval/v1-2019-12-09 --plot
```

The output directory holds `metrics.json` (with the sha256 of every input)
and CSV tables: precision, recall and F1 per recording, dataset and pooled
for thresholds 0.25-0.95; exact binomial and region-bootstrap CIs; PR curves
and average precision; the outcome of every label and prediction; and
emitter accuracy, coverage, calibration and end-to-end metrics by ΔF. v1's
pre-NMS boxes need `--nms-iou 0.2`. From Python, use
`chirpdetector.evaluation.evaluate`.

## Detector datasets

A detector dataset is a directory with `dataset.json` (front end, tiling,
sources, group-wise splits, per-image geometry, content hashes),
`boxes.parquet` (chirp boxes and ignore regions in absolute time and
frequency, with the emitter's EOD frequency), `images/*.png` and YOLO
`labels/*.txt`. Builders write it with `chirpdetector.data.DatasetWriter`;
the contract is documented in `chirpdetector.data.contract`. Check a
dataset with

```sh
uv run chirpdetector dataset validate /mnt/data1/chirp_v2/datasets/sim_v1
```

## Training a detector

```sh
scripts/guarded --name train-r1 --mem 24G --vram 0.6 -- \
    uv run chirpdetector train detector train.toml \
        --dataset /mnt/data1/chirp_v2/datasets/sim_v1 --run-dir /mnt/data1/chirp_v2/runs/r1
```

The config (`chirpdetector.training.config`) selects the model (default
YOLO26s), optimisation, augmentation (only optional time reversal; no
vertical flips, colour jitter, rotation, scaling, translation or mosaic,
which break the time and frequency axes of a spectrogram) and grouped
k-fold cross-validation over the `train` groups; a final model is trained
on all `train` groups with the `val` groups for early stopping. `--fold k`
or `--fold final` trains a single fit. The run directory holds the
resolved config, git SHAs (with dirty flags) of chirpdetector and
gridtools, the `uv.lock` hash, the dataset hash, versions, hardware, the
epoch curves and one model bundle per fit.

**GPU memory.** The GPU is shared. Run training (and any GPU job) through
the workspace's `scripts/guarded`: it caps RAM in a systemd scope and sets
`CHIRP_MAX_VRAM_FRACTION`, which training and inference apply with
`torch.cuda.set_per_process_memory_fraction` before building a model (the
smaller of it and the config's `max_vram_fraction`). Batch sizes are fixed
in the config; AutoBatch is not used.

**Bundles.** `chirpdetector bundle inspect BUNDLE` verifies the weights'
sha256 and prints the model, front end, tiling, preprocessing, training
provenance and attached metrics. Metrics are attached with
`chirpdetector.evaluation.attach_to_bundle` and only for predictions made
by the bundle's own weights (`chirpdetector.prediction` records their
hash).

## Chirp parameters

`chirpdetector.params` is a second-stage decoder that estimates, for every
detected chirp, its **height** (largest frequency excursion above the
emitter's baseline, Hz), its **duration** as full width at half height
(`fwhm_s`) and width at 10 % of the height (`width10_s`), and its
**undershoot** (deepest drop below the baseline after the peak, Hz), each with
a calibrated standard deviation.

**Ground truth** (`chirpdetector.params.truth`) is the simulator's
noise-free instantaneous frequency of the emitter, one value per EOD period:
cycle boundaries are the 2π crossings of the emitter's phase
(`SceneModel.phase(..., chirps="all")`), and each period gets `1/T_k` minus
the period mean of the chirp-free baseline (with drift). Height is the
largest of these excursions; the widths are the times between the crossings
of 50 % and 10 % of the height that bracket the peak (interpolated between
period midpoints); undershoot is the deepest excursion below zero after the
peak. Nothing is read from a spectrogram.

**Input** (`chirpdetector.params.crops.decoder_inputs`): the detector's own
3-window spectrogram, normalised like a detector tile on a 3.2 s canvas
centred on the box, resampled to 96 x 96 over the box plus 15 % margins,
plus the log box duration and height. Training, evaluation and inference
make it with this one function.

**Training** (`chirpdetector train params`) runs the detector bundle over the
simulated scenes, matches its boxes one-to-one to the true chirps (centre
within 100 ms, overlapping frequency ranges) and trains on those *predicted*
boxes: a params bundle is only valid with the detector it was trained with,
whose weight hash it records (`predict_span` refuses another detector).
Splits follow the detector datasets (seeds 0-1599 train, 1600-1799 val,
1800-1999 test); by default only scenes on real background segments are used
(`scene_background = "hybrid"`).

```sh
scripts/guarded --name params --mem 12G --vram 0.1 -- \
    uv run chirpdetector train params params.toml \
        --scenes /mnt/data1/chirp_v2/scenes/train_v3 \
        --detector /mnt/data1/chirp_v2/runs/<detector run>/final/bundle \
        --run-dir /mnt/data1/chirp_v2/params/runs/p1
uv run chirpdetector evaluate params /mnt/data1/chirp_v2/params/runs/p1/bundle \
    --scenes /mnt/data1/chirp_v2/scenes/train_v3
```

**Calibration** (`chirpdetector.params.calibration`). A regressor shrinks
toward the middle of its training population (small chirps overestimated,
large ones underestimated), and how much depends on that population. The
bundle stores, per parameter, the monotone bias curve
`h(y) = median(estimate | true = y)` fitted on the validation scenes at the
detector's threshold, and reports `h⁻¹(estimate)`: median-unbiased at every
true value, so its meaning does not depend on the mix of chirps in a
recording. Standard deviations are propagated through `h⁻¹` and rescaled so
that ±1 sd covers 68 % of the validation chirps (`k95` sd cover 95 %).

**Inference.** `predict_span(..., params=BUNDLE)` (and `chirpdetector
showcase --params BUNDLE`) adds `height_hz, height_sd_hz, fwhm_s, fwhm_sd_s,
width10_s, width10_sd_s, undershoot_hz, undershoot_sd_hz`, plus `params_ok`
and `params_flag` (`edge`: the crop crosses the recording's edge; `band`: it
reaches past the front end's band; `size`: a box larger or smaller than any
in training; `uncertain`: a sd above the 99.5th percentile of the validation
chirps).

**Evaluation** (`chirpdetector evaluate params`, report in `BUNDLE/eval/`):
errors by family, SNR, ΔF, overlap and decile of the true value, before and
after calibration (shrinkage slopes); coverage of the sds; whether the
*distribution* of estimates over a population matches the true one
(KS/Wasserstein distances, quantiles, joint height x FWHM histograms), also
after re-weighting the test chirps to the natural template mix of the
library (training scenes over-sample big and long chirps); whether separated
chirp types stay separated and a continuum stays a continuum; and agreement
with the independently extracted frequency traces of 533 harvested real
chirps.

**Caveats.** Accuracy is measured on simulations (real chirp templates on
simulated and real background); on real recordings only the agreement on
big, long harvested chirps is checked, not small chirps. The parameters are
only valid for boxes of the detector recorded in the bundle.

## Development

chirpdetector depends on [gridtools](https://github.com/weygoldt/gridtools),
expected as a sibling checkout (`../gridtools`, branch `v2`).

```sh
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

## License

MIT
