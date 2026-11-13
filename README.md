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

`chirpdetector label-emitters` shows one candidate box at a time: an
overview of the whole chirp on top and, below it, a zoom on the baselines of
the fish near the box (204.8 ms window; a few Hz span many pixels). Every
live EOD frequency track is marked at both edges with its number (1, 2, ...
from the lowest baseline); tracks are bridged across wavetracker gaps of up
to `--track-gap-s` (10 s), and the gap is shown. Click on the baseline the chirp comes from: the click snaps to
the highlighted track (the nearest one within `--tolerance-hz`, at the time
you click); a click farther from every track is rejected, not guessed.

| input | label |
|---|---|
| left click on a baseline, or `1`-`9` | chirp from that track |
| `0` or middle click | chirp, emitter unclear |
| `x` or right click | not a chirp |
| `s` or space | skip (comes back after the open items) |
| `u`, backspace, ctrl+z | undo the last action |
| left / right arrow | previous / next candidate |
| `w` | overview window 204.8 ms / 51.2 ms (chirp shape); the zoom stays at 204.8 ms |
| `e` | show / hide the model's predicted emitter (hidden by default) |
| `f`, `h` | full screen, help |

Every action is saved at once (`<out>/events.jsonl`); running the same
command again resumes the same queue, in either frontend.

```sh
uv sync --extra label
# matplotlib window (maximised) on this machine
uv run chirpdetector label-emitters --candidates candidates.parquet \
    --out /mnt/data1/chirp_v2/labels/emitters --labeler pw
# the same in a browser, e.g. from home
uv run chirpdetector label-emitters --candidates candidates.parquet \
    --out /mnt/data1/chirp_v2/labels/emitters --labeler pw --serve --port 8765
#   on the laptop: ssh -N -L 8765:localhost:8765 USER@HOST, then open
#   http://localhost:8765/ (the server only listens on 127.0.0.1)
# counts by decision, source and ΔF bin
uv run chirpdetector label-summary /mnt/data1/chirp_v2/labels/emitters
```

Instead of `--candidates`, trial directories sample v1 detections
(`chirpdetector_bboxes.csv`) stratified by ΔF (`--per-bin`, `--seed`).
`--view grid` shows the former grid of small cutouts (matplotlib only).

### Candidates file

A CSV or Parquet table, one row per box. This format is stable; the
production pipeline writes its validation samples in it.

| column | | meaning |
|---|---|---|
| `recording` | required | recording directory (relative paths: relative to the file) |
| `t_start_s`, `t_end_s` | required | box in seconds from the recording start |
| `f_low_hz`, `f_high_hz` | required | box in Hz |
| `candidate_id` | optional | id of the box within its recording (no `#`); the same id in another file is the same item. Default: a hash of recording name and box |
| `score` | optional | detector score |
| `emitter_hz` | optional | the model's predicted emitter EODf |
| `track_id` | optional | the model's predicted emitter track |
| `source` | optional | free text: model, run, sample |
| `tracks` | optional | wavetracker directory, if not the recording directory |

Other columns are kept and stored with each label as `candidate_<name>`.
The queue keeps every row (shuffled with `--seed`; `--keep-order` keeps the
file order).

### Label files

`<out>/labels/<recording>.csv`, one row per labelled box and labeller,
readable by `chirpdetector evaluate --labels`: the audian columns with the
box, then `label` (`chirp`, `not_chirp`, `chirp_emitter_unclear`),
`emitter_id` (track id), `clicked_hz`, `clicked_t_s`, `labeler`,
`labeled_at`, `tool_version`, `queue_id`, `item_id`
(`<recording>#<candidate_id>`), `det_index`, `df_bin`, then

| column | meaning |
|---|---|
| `emitter_baseline_hz` | emitter track's frequency at the box start, from the track (not the click) |
| `click_offset_hz` | click minus the emitter track's frequency at the click time |
| `via` | `mouse` or `key` |
| `frontend` | `mpl`, `mpl-grid` or `browser` |
| `window_s` | spectrogram window shown |
| `prediction_shown` | whether the model's prediction was visible |
| `n_candidates`, `delta_f_hz` | live tracks at the box centre, ΔF of the closest two |
| `recording_dir`, `tracks_dir` | where recording and tracks were read |
| `candidate_id`, `candidate_score`, `candidate_emitter_hz`, `candidate_track_id`, `candidate_source`, `candidate_*` | the candidates-file columns |

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

**Dual-head model.** `[model] type = "dualhead"` (needs `uv sync --extra
dualhead`) trains an RT-DETRv2 detector (COCO weights, ResNet18-vd by
default) whose queries predict each chirp's box *and* its emitter's
baseline EOD frequency; `chirpdetector.evaluation.assign_by_frequency`
then picks the nearest live track. Predictions (`predict_images`,
`predict_span`) carry an `emitter_hz` column. See
`chirpdetector.models.dualhead` for the design and the `[model]` keys.
A dual-head fit that is stopped (`systemctl stop` on its guarded scope,
Ctrl-C) ships its best validated epoch as the bundle; `[model] resume =
"RUN/final/dualhead/state.pt"` continues it in a new run directory.

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
cycle boundaries are the 2π crossings of the emitter's phase with only that
chirp (its other chirps left out, so that a neighbour in a burst cannot lend
its peak or undershoot), and each period gets `1/T_k` minus
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
A scene's split is the `train`/`val`/`test` directory it lies in (as
`gridtools sim run` writes them, and as the detector datasets use them with
`--split-dir`); `split = "train=0:1600,..."` assigns by seed instead. Only
scenes on real background segments are used (`scene_background = "hybrid"`).

```sh
scripts/guarded --name params --mem 12G --vram 0.1 -- \
    uv run chirpdetector train params params.toml \
        --scenes /mnt/data1/chirp_v2/scenes/train_v4 \
        --detector /mnt/data1/chirp_v2/runs/<detector run>/final/bundle \
        --run-dir /mnt/data1/chirp_v2/params/runs/p1
uv run chirpdetector evaluate params /mnt/data1/chirp_v2/params/runs/p1/bundle \
    --scenes /mnt/data1/chirp_v2/scenes/train_v4
```

The config keys and their defaults are listed in `chirpdetector.params.training`
(every key is optional); the sample set (crops, truth, matched boxes) lands in
`RUN/samples` unless `--samples` points to one made with the same settings.

**Calibration** (`chirpdetector.params.calibration`). A regressor shrinks
toward the middle of its training population (small chirps overestimated,
large ones underestimated), and how much depends on that population. The
bundle stores, per parameter, the monotone bias curve
`h(y) = median(estimate | true = y)` fitted on the validation scenes at the
detector's threshold (bin medians smoothed by a local linear fit, slopes at
least 0.5; one curve per tercile of the decoder's own sigma, because poorly
seen chirps shrink more), and reports `h⁻¹(estimate)`: median-unbiased at
every true value, so its meaning does not depend on the mix of chirps in a
recording.
The price is a wider per-chirp error than the raw estimate (the inverse of a
curve with slope < 1 amplifies the noise) and a population distribution that
is the true one blurred by that error; the evaluation report shows both. Standard deviations are propagated through `h⁻¹` and rescaled so
that ±1 sd covers 68 % of the validation chirps (`k95` sd cover 95 %).

**Inference.** `predict_span(..., params=BUNDLE)` (and `chirpdetector
showcase --params BUNDLE`) adds `height_hz, height_sd_hz, fwhm_s, fwhm_sd_s,
width10_s, width10_sd_s, undershoot_hz, undershoot_sd_hz` (sds are first
order; the calibrated intervals are multiplicative, `(value + offset) *
exp(±k95 * sd / (value + offset)) - offset` with offset 1 Hz for undershoot
and 0 otherwise), plus `params_ok`
and `params_flag` (`edge`: the crop crosses the recording's edge; `band`: the
box reaches the edge of the front end's band; `size`: a box larger or smaller than any
in training; `uncertain`: a sd above the 99.5th percentile of the validation
chirps).

**Evaluation** (`chirpdetector evaluate params`, report in `BUNDLE/eval/`):
errors by family, SNR, ΔF, overlap, template reuse and decile of the true
value, before and after calibration (shrinkage slopes); whether the error is
coupled to the chirp's own parameters *within* SNR strata (quintiles of each
true parameter, rank correlations of every error with every true parameter);
coverage of the sds; whether the *distribution* of estimates over a
population matches the true one (KS/Wasserstein distances, quantiles, joint
height x FWHM histograms), also after re-weighting the test chirps to the
library's template mix (training scenes over-sample big and long chirps);
whether separated chirp types stay separated and a continuum stays a
continuum; and agreement with the independently extracted frequency traces of
533 harvested real chirps, next to the correlation the simulation predicts
over the same narrow range.

**Caveats.** Accuracy is measured on simulations (real chirp templates on
real background segments); on real recordings only the agreement on big,
long harvested chirps is checked, not small chirps. The simulator draws every
split from one template library, so test chirps reuse training templates
(and individuals): the error on chirps of new fish is not measured. The
parameters are only valid for boxes of the detector recorded in the bundle.
For population distributions, calibrated estimates are wider than the truth
(blurred by the error) and uncalibrated ones are pulled toward the training
mix; the report measures both against the truth.

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
