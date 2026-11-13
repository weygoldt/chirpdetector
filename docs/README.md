# chirpdetector documentation

Detect the chirps of wave-type weakly electric fish in multi-electrode grid
recordings, assign each chirp to the fish that emitted it, and measure its
height, duration and undershoot.

```sh
uv pip install "git+https://github.com/weygoldt/chirpdetector"
chirpdetector run --out detections /path/to/recording
```

Back to the [project README](../README.md).

## How it works

```text
recording (raw, any sampling rate)          wavetracker tracks
   |                                              |
   v                                              |
click removal (>= 9 electrodes)                   |
   |                                              |
   v                                              |
3-window linear-frequency spectrogram,            |
500-1500 Hz, 3.2-s tiles                          |
   |                                              |
   v                                              v
dual-head RT-DETRv2: box + emitter EODf ---> nearest live track = emitter,
   |                                          with a confidence
   v
chirp-parameter decoder: height, FWHM, 10 % width, undershoot (calibrated)
   |
   v
post-processing: harmonic copies, duplicates (flagged, not hidden)
   |
   v
chirps_rtdetr.parquet, legacy arrays, meta.json, QC report, windows
```

The detector and the decoder are trained on simulated scenes from
[gridtools](https://github.com/weygoldt/gridtools): real chirp templates
on simulated fish, mixed into real background recordings, with exact
ground truth. They are evaluated on hand-labelled real recordings, on
held-out simulations of new fish, and against a human labeller.

## Pages

Getting started

- [Installation](installation.md): uv and pip, extras, GPU memory
- [Quickstart](quickstart.md): the pipeline on a recording
- [Pretrained models](models.md): the released detector and decoder, cache, offline use, bundles

Detecting chirps

- [The production pipeline](pipeline.md): `chirpdetector run`, run config, outputs, chirp table, legacy arrays, QC, windows, assignment
- [Python API](api.md): `predict_span`, bundles, spectrograms, reading the outputs
- [Post-processing](postprocessing.md): harmonic copies, duplicates, re-attribution
- [Chirp parameters](decoder.md): height, FWHM, 10 % width, undershoot; calibration
- [Click removal](clicks.md): the electrode-coincidence guard

Training and evaluation

- [Detector datasets](datasets.md): from simulated scenes and hand-labelled recordings
- [Training a detector](training.md): dual-head (default) and YOLO26
- [Training the decoder](decoder-training.md): samples, calibration, evaluation report
- [Evaluation](evaluation.md): matching rule, metrics, showcase

Labelling

- [Labelling chirps and emitters](labelling.md): `label-emitters`, both frontends, keys, candidates, blind mode, SSH
- [Human against model](benchmark.md): `label-benchmark`

Background and reference

- [Design notes](design.md): why linear frequency, footprint boxes, the emitter head, calibration
- [Limitations](limitations.md): what the evaluation shows and what not
- [Command line](cli.md): every command and environment variable
- [Development](development.md): tests and checks

## Performance of the released models

| test | dual-head | YOLO26s (box only) | v1 (2024) |
|---|---|---|---|
| hand-labelled real recordings, 365 chirps, precision / recall / F1 at 0.5 | 0.938 / 0.992 / 0.964 | 0.947 / 0.984 / 0.965 | 1.000 / 0.701 / 0.824 |
| simulated test scenes, after post-processing | 0.976 / 0.957 / 0.966 | 0.989 / 0.952 / 0.970 | |
| emitter correct (simulated, detected chirps) | 0.914 | | |

Decoder, simulated test chirps of unseen fish, median absolute error:
height 9.5 Hz (6.7 %), FWHM 1.16 ms (9.8 %). Blind benchmark on 1,010
simulated items: recall 0.963 (model) vs 0.924 (expert labeller).
