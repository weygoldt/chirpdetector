# Human against model

`chirpdetector label-benchmark` compares a human labeller with detectors on
simulated scenes, where the truth is known: it samples items, writes a
candidates file that the human labels blind and a separate truth file, and
scores both on the same items.

## A simple benchmark

```sh
# 1. detections of the model on every test scene: <scene>.parquet from predict_span
#    (with the scene's tracks, post-processing on, score >= 0.05)
# 2. sample items
chirpdetector label-benchmark build --scenes /abs/path/scenes/test --detections DETS \
    --candidates bench/candidates/bench.parquet --truth bench/truth/bench_truth.parquet
# 3. label them blind
chirpdetector label-emitters --candidates bench/candidates/bench.parquet --out bench/labels \
    --labeler NAME --blind --serve
# 4. score
chirpdetector label-benchmark score --truth bench/truth/bench_truth.parquet \
    --labels bench/labels --out bench/report
```

Give `--scenes` as an absolute path: the candidates file stores the scene
paths as given, and the labeller resolves relative paths against the
candidates file's directory. Keep candidates and truth in different
directories; nothing in the labeller opens the truth.

A detections directory can be made with:

```python
from pathlib import Path
from gridtools.io import load_tracks, open_recording
from chirpdetector.bundle import load_bundle
from chirpdetector.inference import predict_span

bundle, scenes, out = (
    load_bundle("default"),
    Path("/abs/path/scenes/test"),
    Path("DETS"),
)
out.mkdir(parents=True, exist_ok=True)
for scene in sorted(d for d in scenes.iterdir() if (d / "chirps.parquet").exists()):
    rec = open_recording(scene)
    det = predict_span(
        bundle,
        rec,
        0.0,
        rec.duration_s,
        name=scene.name,
        score_threshold=0.05,
        tracks=load_tracks(scene),
    )
    det.attrs = {}
    det.to_parquet(out / f"{scene.name}.parquet", index=False)
```

Items (`label-benchmark build`, defaults in brackets): visible true chirps
(`--n-chirps` 150), of which `--n-ambiguous` (75) from the lowest SNR
quartile with excursion below 60 Hz; model detections without any chirp
(`--n-hard` 100); quiet baseline windows (`--n-easy` 50); invisible chirps
(`--n-invisible` 15, reported, not scored). True chirps carry weights that
reweight the mix to all visible test chirps.

The report (`report.md`, `metrics.json`, `items.csv`, `report.png`, tables
by stratum, SNR quartile and excursion) gives recall on true chirps,
false-alarm rate on negatives and emitter accuracy for human and model,
exact two-sided McNemar tests on the paired decisions and stratified
bootstrap confidence intervals.

## The edge-case benchmark (v2)

`build-v2` samples new scenes so that the report can say where a model
works and where it does not, compares several models, and repeats items
to measure the human's test-retest agreement.

```sh
B=bench
chirpdetector label-benchmark features SCENE_DIR... --out $B/features --templates TEMPLATES/templates.parquet
chirpdetector label-benchmark gaps --scenes /abs/path/scenes --features $B/features --out /abs/path/gaps
chirpdetector label-benchmark build-v2 --scenes /abs/path/scenes --features $B/features \
    --gap-root /abs/path/gaps --repeat-from $B/truth/bench_truth.parquet \
    --model dualhead=DETS_DUALHEAD --model yolo26s=DETS_YOLO --model v1=DETS_V1:v1 \
    --candidates $B/candidates/v2.parquet --truth $B/truth/v2_truth.parquet
chirpdetector label-emitters --candidates $B/candidates/v2.parquet --out $B/labels_v2 --blind --serve
chirpdetector label-benchmark score --out $B/report_v2 \
    --set v1=$B/truth/bench_truth.parquet:$B/labels \
    --set v2=$B/truth/v2_truth.parquet:$B/labels_v2 \
    --model dualhead=DETS_DUALHEAD --model yolo26s=DETS_YOLO --model v1=DETS_V1:v1 \
    --features $B/features
```

- `features` computes per-chirp features of scenes (nearest other fish
  including real background fish, overlap, bout, drift or rise, click
  nearby, few electrodes, rippled or plateau chirp, long, short) and caches
  them.
- `gaps` copies scenes with the emitter's track cut around one chirp each
  (simulated tracks have no gaps; wavetracker's often break at chirps).
- `build-v2` puts every visible chirp into one stratum, rarest feature first
  (gap, click, baseline at another fish's 2nd harmonic, rippled/plateau,
  long, short, ΔF 0-5 / 5-10 Hz, overlap, bout, drift, few electrodes,
  ΔF 10-25 / 25-50 Hz), the rest into SNR quartile x excursion cells.
  Negatives: each model's false alarms over the whole score range, harmonic
  copies (fundamental detected or not), clicks, beats, rises, plain
  baselines; `--repeat-from` adds repeats of labelled items.
- Model directories hold one `<scene>.parquet` of `predict_span`
  (tracks, post-processing on, `keep_dropped=True`, score >= 0.05), or the
  v1 runner's `<dataset>/<scene>.csv` (suffix `:v1`). A true chirp counts as
  detected if a reported detection (kept or re-attributed) at the
  threshold matches it one-to-one.
- `score` with `--set` and `--model` merges the sets and reports recall by
  SNR, excursion, ΔF and edge case, false alarms per negative type,
  harmonic outcomes, emitter accuracy by ΔF, invisible chirps, human time
  and test-retest agreement, with bootstrap CIs and McNemar tests, and a
  table of where the model is better, equal or worse than the human.

`label-benchmark gallery` shows real boxes the labeller called `not_chirp`
next to simulated true chirps with similar decoded height, FWHM and
contrast (never from the benchmark's scenes, `--exclude`).

## Result of the project's benchmark

1,010 blind items (695 from the edge-case set, 315 from the first set), one
expert labeller, models at score >= 0.5:

| | human | dual-head | YOLO26s | v1 |
|---|---|---|---|---|
| recall on true chirps | 0.924 | 0.963 | 0.974 | 0.438 |
| recall, reweighted to all visible test chirps | 0.948 | 0.968 | 0.964 | 0.489 |
| false-alarm rate, all negatives | 0.083 | 0.090 | 0.049 | 0.015 |
| chirp found with the right emitter | 0.812 | 0.868 | | |

The dual-head model was better than the human below -20 dB SNR (0.811 vs
0.609), at ΔF below 10 Hz, on scenes with few electrodes and on harmonic
copies, and worse on false alarms among other models' detections (0.160 vs
0.073). The human's test-retest agreement on 30 repeated pairs was 0.900
(Cohen's kappa 0.796); median time per item 1.06 s.

> **Verified.** `build`, blind labelling through the server's API, `score`
> (both report formats), `features`, `gaps` and `build-v2` were run on three
> 20-s simulated scenes on the CPU.
