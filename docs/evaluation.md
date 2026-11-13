# Evaluation

`chirpdetector evaluate` scores predictions against ground truth and writes
every number with the sha256 of every input.

```sh
chirpdetector evaluate PREDICTIONS... --labels LABELS... --out OUT
```

(`evaluate detections` is the same command; `evaluate params` evaluates a
decoder, see [Training the decoder](decoder-training.md#evaluation).)

**Predictions** are any of: v2 tables (CSV or Parquet with `recording`,
`t_start_s`, `t_end_s`, `f_low_hz`, `f_high_hz`, `score`, optionally
`emitter_id`, `emitter_prob`, `emitter_hz`, `model_sha256`), v1's
`chirpdetector_bboxes.csv` (or its recording directory), or v1 boxes
(`t1, f1, t2, f2, score`, with `--nms-iou 0.2` for v1's per-batch NMS).
The pipeline's chirp table needs its flagged rows removed and the emitter
column renamed first (see [Python API](api.md#reading-the-pipelines-output)).

**Labels** (repeatable) are any of:

- audian label files of hand-labelled recordings
  (`traces-grid1-editable-labels.csv`): spans of category `labeled_region`
  inside which every chirp was labelled, and `chirp` boxes. A region is
  exhaustive: a prediction inside it (and inside its frequency band) that
  matches no chirp is a false positive;
- `label-emitters` output (`<out>/labels/<recording>.csv`): spot labels
  (`chirp`, `chirp_emitter_unclear`, `not_chirp`) that say nothing about
  the rest of the recording;
- simulated scenes (`chirps.csv` or the scene directory): visible chirps
  are labels, invisible ones `ignore`.

## Matching rule

The rule of the v1 detector benchmark: a prediction and a label match if
their centre times differ by at most 100 ms (`--tolerance-s`) and their
frequency ranges overlap; matching is one-to-one with the most pairs and,
among those, the smallest total |Δt| (`--method hungarian`; `greedy` is the
alternative). In a labelled region, only predictions whose centre lies in
the region and whose frequency range overlaps the region's band count
(`--band-filter`, default on: this drops boxes on EOD harmonics above the
band). Labels are matched in three passes: positive labels, then `ignore`
labels (their predictions are not counted), then `not_chirp` labels (their
predictions are false positives).

## Metrics

From pooled counts, per recording, per dataset and over all, for thresholds
0.25-0.95 (`--thresholds`): precision, recall, F1; exact binomial
(Clopper-Pearson) and region-bootstrap confidence intervals at the
reference threshold (`--reference-threshold 0.5`, `--n-bootstrap 5000`);
PR curves and average precision. With emitters and tracks
(`--tracks`, default: the recording directories), emitter accuracy,
coverage, calibration (reliability bins, ECE) and end-to-end metrics
(chirp found *and* assigned to the right fish), each pooled and by ΔF of
the two closest live tracks (0-10, 10-25, 25-50, 50-100, 100+ Hz,
`single`, `unknown`).

## Outputs

`OUT/metrics.json` (headline numbers, config, sha256 of inputs) and CSV
tables: `summary.csv`, `intervals.csv`, `pr_curve.csv`,
`average_precision.csv`, `per_region.csv`, `per_spot.csv`,
`spot_summary.csv`, `items.csv` (the outcome of every label and
prediction), `assignment.csv`, `coverage_accuracy.csv`, `reliability.csv`,
`confusion.csv`, `end_to_end.csv`; `pr_curve.png` with `--plot`.

## Examples

Against a simulated scene (the pipeline's kept rows, renamed as in
[Python API](api.md#reading-the-pipelines-output)):

```text
$ chirpdetector evaluate kept.parquet --labels scenes/000000 --out eval_kept
threshold 0.5: precision 1.000 [0.863, 1.000], recall 1.000 [0.863, 1.000], F1 1.000 (25 labels, 1 regions)
wrote 14 files to eval_kept
```

v1's boxes on the four hand-labelled recordings, with v1's NMS:

```sh
chirpdetector evaluate detector_benchmark/results/boxes/*.csv --nms-iou 0.2 \
    --labels REC/traces-grid1-editable-labels.csv ... --out eval_v1
```

From Python:

```python
import pandas as pd
from chirpdetector.evaluation import (
    GroundTruth,
    evaluate,
    read_ground_truth,
    read_predictions,
)

truth = GroundTruth.concat(read_ground_truth(f) for f in label_files)
predictions = pd.concat(read_predictions(f) for f in prediction_files)
result = evaluate(predictions, truth, tracks={"2019-12-09-10_00": tracks})
result.write("out/")
```

## Attaching metrics to a bundle

`chirpdetector.evaluation.attach_to_bundle` writes `metrics.json` into a
bundle, and only for predictions whose `model_sha256` is the bundle's own
weight hash; loading a bundle refuses a `metrics.json` of other weights. A
reported number can therefore always be traced to the weights that made it.

## Showcase

```sh
chirpdetector showcase --snippets snippets.toml --out showcase \
    --bundle BUNDLE [--bundle OTHER] [--v1] [--params DECODER] --device cpu
```

renders models side by side on short snippets of real recordings (tracks,
boxes with scores, predicted emitter EODf, hand labels where the snippet
lies in a labelled region, decoded height and FWHM with `--params`), in a
contact sheet that grows with every model added. `--v1` adds v1's saved
detections as a column. Snippets:

```toml
[[snippet]]
name = "close-df"
recording = "/data/2019_tube_competition/2020-03-13-10_00"
t_start_s = 9400.0
t_end_s = 9410.0
reason = "two fish ~7 Hz apart, both chirping"
```

Optional keys: `f_min_hz`, `f_max_hz` (default 500-1500 Hz) and `labels`
(default: the recording's audian labels). Needs the `label` extra.
