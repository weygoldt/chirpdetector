# Training the decoder

The chirp-parameter decoder learns from simulated scenes
([gridtools](https://github.com/weygoldt/gridtools) `sim run`), on the
boxes of the detector that will feed it.

```sh
chirpdetector train params examples/train_params.toml \
    --scenes SCENES --detector DETECTOR_BUNDLE --run-dir RUN
chirpdetector evaluate params RUN/bundle --scenes SCENES
```

`SCENES` is a scene directory, a directory of scenes, or a directory with
`train/`, `val/` and `test/` scene directories (as `gridtools sim run`
writes them per split). A scene's split is the directory it lies in;
`split = "train=0:1600,val=1600:1800,test=1800:2000"` in the config assigns
by seed instead. Only scenes on real background segments are used
(`scene_background = "hybrid"`; `"any"` admits synthetic-noise scenes).

## What training does

1. Finds the scenes and their splits.
2. Extracts the sample set (`RUN/samples`, or `--samples DIR` to reuse one
   made with the same settings): the detector runs over every scene, its
   boxes are matched one-to-one to the true chirps (centres within 100 ms,
   overlapping frequency ranges), and each matched box becomes a sample with
   its crop and the chirp's per-period truth. Extraction is resumable.
3. Trains the decoder (a compact CNN with Gaussian heads, beta-NLL loss in
   log space) on the usable `train` samples, keeps the epoch with the
   smallest validation error and stops after `optim.patience` epochs without
   improvement. `test` samples are never read.
4. Fits the [calibration](decoder.md#calibration) and the validated domain
   on the `val` samples at the detector's operating threshold.
5. Writes the bundle, which records the detector's weight hash.

## Config

Every key is optional; `chirpdetector.params.training` lists the defaults.
The released decoder was trained with
[`examples/train_params.toml`](../examples/train_params.toml):

```toml
seed = 0
device = "auto"
max_vram_fraction = 0.1
workers = 5                   # sample extraction processes
scene_background = "hybrid"   # only scenes on real background segments

[boxes]
mode = "detector"             # train on the boxes of the detector that feeds it
score_threshold = 0.25        # detector boxes kept in the sample set

[crop]
size = [96, 96]
margin_rel = 0.15
canvas_s = 3.2

[optim]
epochs = 40
batch = 256
lr = 0.002
patience = 8
min_score = 0.25

[calibration]
method = "inverse-median"
min_score = 0.5               # fit on validation boxes at the detector's threshold
```

`[boxes] mode = "footprint"` trains on the scenes' footprint boxes
(jittered) instead of a detector's boxes; such a decoder records no
detector.

## Run directory

```text
RUN/
  run.json                 run id, sample set, detector, best epoch, bundle
  config.json              resolved config (config.toml: the input file)
  provenance.json          code SHAs, uv.lock hash, versions, hardware
  samples/                 sample set: manifest.json, samples.parquet, crops.npy
  curves.csv               per-epoch losses, validation errors, learning rate
  val_predictions.parquet  raw validation predictions (calibration input)
  bundle/                  the params bundle
```

## Evaluation

`chirpdetector evaluate params BUNDLE --scenes SCENES` predicts the test
split's samples (boxes of the bundle's own detector) and writes, to
`BUNDLE/eval/` (or `--out`), `report.md`, `metrics.json` (also attached to
the bundle unless `--no-attach`), `predictions.parquet`, CSV tables and
figures:

- accuracy before and after calibration: median absolute and relative
  error, p90, bias, shrinkage slope, by family, SNR, ΔF, overlap with other
  chirps, and decile of the true value;
- whether the error is coupled to the chirp's own parameters within SNR
  strata;
- template reuse between training and test chirps (none with fish-disjoint
  template splits);
- coverage of the standard deviations;
- distribution fidelity: KS and Wasserstein distances of estimated and true
  distributions, joint height x FWHM histograms, also reweighted to the
  template library's natural mix of chirp sizes;
- whether separated chirp types stay separated and a continuum stays a
  continuum;
- with `--real` (default): agreement on the harvested long chirps of the
  competition recordings (template library v4 and the recordings; give
  their locations with `--library` and `--recordings`, whose defaults are
  paths on the development machine). Use `--no-real` without them.

## Trying it on a small scale

On the CPU, with three tiny split directories of 20-s synthetic scenes
(`gridtools sim run` with `[background] prob = 0`), a config with
`scene_background = "any"`, `device = "cpu"` and `epochs = 3`:

```sh
chirpdetector train params smoke_params.toml --scenes split_scenes \
    --detector DETECTOR_BUNDLE --run-dir params_run
chirpdetector evaluate params params_run/bundle --scenes split_scenes --no-real --device cpu
```

Both finished in about 30 s and 10 s (with warnings that the cluster tests
need more chirps). The released decoder took 35 epochs on the 3,000
sim_v4 scenes.
