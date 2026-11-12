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
