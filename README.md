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
