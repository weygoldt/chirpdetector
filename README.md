<p align="center">
  <img src="assets/logo.png" width="100" alt="chirpdetector logo">
</p>

# chirpdetector

Detect the chirps of wave-type weakly electric fish (e.g. *Apteronotus
leptorhynchus*) in multi-electrode grid recordings, assign each chirp to the
fish that emitted it, and measure its height, duration and undershoot.

A chirp is a brief excursion of a fish's electric organ discharge frequency
(EODf), from about 10 ms to several hundred ms long. In a spectrogram of a
grid recording it is a short, tall feature on top of the fish's baseline
frequency, often among several fish a few Hz apart, clicks and power-line
hum. chirpdetector finds these features with an object detector on a
linear-frequency spectrogram and assigns each detection by predicting the
emitter's baseline EODf and picking the nearest frequency track (wavetracker
format) alive at the chirp.

The package contains:

- **The production pipeline** (`chirpdetector run`): whole recordings in,
  one table of chirps per recording out (boxes, scores, emitter track and
  assignment confidence, chirp parameters, quality flags), plus legacy
  `chirp_times`/`chirp_ids` arrays, a QC report and annotated windows.
- **The detector**: a dual-head RT-DETRv2-r18 that predicts each chirp's box
  and its emitter's baseline EODf in one pass (the default); YOLO26 is
  available as a box-only detector.
- **The chirp-parameter decoder**: height, full width at half height, width
  at 10 % of the height, and undershoot of every detected chirp, calibrated,
  with uncertainties.
- **Post-processing** of harmonic copies and duplicate boxes, and **click
  removal** before the spectrogram.
- **Training and evaluation**: detector datasets from simulated scenes
  ([gridtools](https://github.com/weygoldt/gridtools)) and hand-labelled
  recordings, training of both detectors and the decoder, model bundles with
  provenance, and evaluation against hand labels and simulations.
- **A labelling tool** (`chirpdetector label-emitters`, matplotlib window or
  browser) for chirps and their emitters, and a blind human-against-model
  benchmark (`chirpdetector label-benchmark`).

## Installation

```sh
uv pip install "git+https://github.com/weygoldt/chirpdetector"
# or
pip install "git+https://github.com/weygoldt/chirpdetector"
# with the labelling tool (matplotlib):
uv pip install "chirpdetector[label] @ git+https://github.com/weygoldt/chirpdetector"
```

Python 3.13 or later. A CUDA GPU is optional; everything also runs on the
CPU, more slowly. The PyPI package named `gridtools` is unrelated to the
[gridtools](https://github.com/weygoldt/gridtools) that chirpdetector
depends on. [docs/installation.md](docs/installation.md) covers a project
setup with `uv add`, the extras, and a fallback install command.

## Pretrained models

The released dual-head detector and chirp-parameter decoder are the GitHub
release [`models-v1`](https://github.com/weygoldt/chirpdetector/releases/tag/models-v1).
chirpdetector downloads them on first use, verifies their checksums and
caches them (`$CHIRPDETECTOR_CACHE`, else `$XDG_CACHE_HOME/chirpdetector`,
else `~/.cache/chirpdetector`):

```sh
chirpdetector models list
chirpdetector models download          # both, e.g. before going offline
chirpdetector models path detector
```

In Python: `chirpdetector.pretrained.download(name)`, `path(name)` and
`list_models()`; `load_bundle("default")` and `predict_span(...,
params="default")` use the released models. `CHIRPDETECTOR_OFFLINE=1`
forbids downloads. Details: [docs/models.md](docs/models.md).

## Quickstart

You need a recording directory with the raw data (fishgrid
`traces-grid1.raw` with its `fishgrid.cfg`, or one multichannel wav file)
and the wavetracker frequency tracks of its fish (`fund_v.npy`,
`ident_v.npy`, `idx_v.npy`, `times.npy`). Then:

```sh
chirpdetector run --out detections /path/to/recording
```

This uses the released detector and decoder and every default setting, on
CUDA if available (`--device cpu` forces the CPU). The output is in
`detections/default/<recording>/`: `chirps_rtdetr.parquet` (one row per
detection), `chirp_times_rtdetr.npy` and `chirp_ids_rtdetr.npy`,
`meta.json`, `qc/qc.md` and a few annotated windows. Rows with `chirp ==
True` are the chirps at the operating threshold (0.5); `track_id` is the
emitter's track. A run config (`chirpdetector run run.toml RECORDING`)
changes any setting; [`examples/run.toml`](examples/run.toml) lists them.

[docs/quickstart.md](docs/quickstart.md) walks through this on a simulated
recording, including reading the table and re-rendering a window.

## Performance

From the evaluation reports of the chirp competition project, for the
released models (the reports are not part of this repository):

| test | released dual-head | box-only YOLO26s | chirpdetector v1 (2024) |
|---|---|---|---|
| hand-labelled real recordings, 365 chirps in 4 recordings, threshold 0.5: precision / recall / F1 | 0.938 / 0.992 / 0.964 | 0.947 / 0.984 / 0.965 | 1.000 / 0.701 / 0.824 |
| simulated test scenes (unseen fish and background recordings), after post-processing | 0.976 / 0.957 / 0.966 | 0.989 / 0.952 / 0.970 | |
| emitter correct, detected simulated test chirps | 0.914 | no emitter head | |

The real labels are the 335 hand-labelled chirps of the v1 benchmark plus 30
chirps the labeller confirmed among the detectors' unmatched boxes. The
assignment baseline without an emitter head (nearest track to the box's lower
edge plus the median offset) reaches 0.648 on the same simulated chirps. In a
blind benchmark on 1,010 simulated items, the dual-head detector found 0.963
of the true chirps and an expert labeller 0.924; with the right emitter, 0.868
and 0.812. The decoder's median absolute errors on simulated test chirps of
unseen fish: height 9.5 Hz (6.7 %), FWHM 1.16 ms (9.8 %), 10 % width 1.58 ms
(7.0 %), undershoot 1.52 Hz. Read [docs/limitations.md](docs/limitations.md)
before relying on these numbers.

## Documentation

Start at [docs/README.md](docs/README.md). The pages:

- [Installation](docs/installation.md): uv and pip, extras, GPU
- [Quickstart](docs/quickstart.md): the pipeline on a recording in five minutes
- [Pretrained models](docs/models.md): download, cache, offline use, bundles
- [The production pipeline](docs/pipeline.md): run config, outputs, chirp table, legacy arrays, QC, windows, render
- [Python API](docs/api.md): `predict_span`, bundles, reading the outputs
- [Post-processing](docs/postprocessing.md): harmonic copies, duplicates, re-attribution
- [Chirp parameters](docs/decoder.md): what the decoder estimates and how to use it
- [Click removal](docs/clicks.md): when it runs and what it protects
- [Detector datasets](docs/datasets.md): from simulated scenes and hand labels
- [Training a detector](docs/training.md): dual-head (default) and YOLO26, bundles
- [Training the decoder](docs/decoder-training.md): samples, calibration, evaluation
- [Evaluation](docs/evaluation.md): matching rule, metrics, outputs
- [Labelling](docs/labelling.md): `label-emitters`, both frontends, keys, candidates, blind mode
- [Human against model](docs/benchmark.md): `label-benchmark`
- [Design notes](docs/design.md): linear frequency, footprint boxes, emitter head, calibration
- [Limitations](docs/limitations.md): what the numbers do and do not show
- [Command line](docs/cli.md): every command
- [Development](docs/development.md): tests and checks

## Versions

This is version 2. The detector used for the chirp competition paper (2024)
is tag `paper-chirp-competition-v1`. v2 shares no models or data formats
with v1; `chirpdetector evaluate` still reads v1's `chirpdetector_bboxes.csv`.

## License

MIT. `chirpdetector.preprocessing.clicks` is ported from audian and
wavetracker (GPL); see its module docstring.
