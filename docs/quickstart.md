# Quickstart

Detect, assign and measure the chirps of one recording with the released
models, then look at the result.

## 1. Install

```sh
uv pip install "chirpdetector[label] @ git+https://github.com/weygoldt/chirpdetector"
```

## 2. Check the recording

A recording directory holds

- the raw data: fishgrid's `traces-grid1.raw` next to its `fishgrid.cfg`,
  or a single multichannel wav file (an `electrodes.csv` with columns
  `channel, x_m, y_m` next to it gives the electrode positions of layouts
  that are not a full rectangle), and
- the frequency tracks of the fish in wavetracker's format: `fund_v.npy`,
  `ident_v.npy`, `idx_v.npy`, `times.npy` (and optionally `sign_v.npy`).
  Without tracks the chirps are detected and measured but not assigned
  (`assign_status = "no_tracks"`). Tracks elsewhere: `[assignment] tracks`
  in a run config (step 6).

The sampling rate does not matter: every setting is in seconds and Hz.

## 3. Run

```sh
chirpdetector run --out detections /path/to/recording
```

Without a run config, the pipeline uses the released detector and decoder
and every default setting (store threshold 0.05, operating threshold 0.5,
post-processing on, assignment within 20 Hz of a track), on CUDA if
available, else on the CPU (`--device cpu` forces the CPU). On first use
the two models (81 MB) are downloaded into `~/.cache/chirpdetector`
([Pretrained models](models.md)). The run is called `default`.

A 20-s simulated 9-electrode scene on the CPU (paths shortened):

```text
INFO chirpdetector.pipeline.run: run default: 1 recordings, 0 complete (skipped), 1 to do -> nocfg/default
INFO chirpdetector.pipeline.run: [1/1] 000000: start
INFO chirpdetector.pipeline.run: [1/1] 000000: done in 0h00m05s (4x real time): 25 chirps, 25 assigned
INFO chirpdetector.pipeline.candidates: validation candidates: 19 -> nocfg/default/candidates.csv
INFO chirpdetector.pipeline.run: run default: 1 done, 0 skipped, 0 failed
```

Runs are resumable: run the same command again and finished recordings are
skipped; `--list` shows what is done.

```sh
chirpdetector run --out detections /path/to/recording --list
chirpdetector run --out detections --dataset /path/to/many/recordings            # every recording directory in it
chirpdetector run --out detections --dataset /path/to/many/recordings --dry-run  # first 300 s of each
chirpdetector run --out detections /path/to/recording --start-s 1140 --limit-s 30  # a 30-s span
```

## 4. Look at the output

```text
detections/default/
  run.json  run.log  columns.md  candidates.csv  candidates_strata.csv
  <recording>/
    chirps_rtdetr.parquet     one row per detection (score >= 0.05)
    chunks_rtdetr.parquet     one row per processed chunk
    chirp_times_rtdetr.npy    legacy: chirp times (s) ...
    chirp_ids_rtdetr.npy      ... and their track ids
    meta.json                 inputs, models, code, config, counts, timings
    qc/qc.md, qc/qc.png       QC report
    windows/                  annotated 10-s windows (png + npz)
```

`qc/qc.md` summarises the recording:

```text
| quantity | value |
|---|---|
| detections stored | 36 |
| above operating threshold | 28 |
| chirps (operating, not flagged) | 25 |
| flagged as harmonic (operating) | 3 |
| chirps assigned | 25 |
| harmonic copies re-attributed to their fish | 0 |
| nearest-track distance q0.5 | 1.52 Hz |
...
```

In Python:

```python
import pandas as pd

chirps = pd.read_parquet("detections/default/REC/chirps_rtdetr.parquet")
chirps = chirps[chirps["chirp"]]  # operating threshold, not flagged
per_fish = chirps.groupby("track_id").size()
sizes = chirps[["t_center_s", "track_id", "height_hz", "fwhm_s", "params_ok"]]
```

`columns.md` in the run directory documents every column; the most used
ones are `t_center_s` (chirp time), `track_id` (emitter, wavetracker
`ident_v` value), `assign_confidence`, `height_hz` and `fwhm_s`
(calibrated decoder estimates) with `params_ok`.

## 5. Re-render a window

Any window can be drawn again from the run's outputs (the spectrogram is
recomputed from the raw data):

```sh
chirpdetector render detections/default REC 5 15
```

This writes a PNG and an NPZ with the data to redraw it to
`detections/default/REC/renders/`.

## 6. Use a run config

A run config names the run and changes any setting:

```toml
# run.toml
[run]
name = "first_run"
out = "detections"      # relative to this file

[compute]
device = "cpu"
```

```sh
chirpdetector run run.toml /path/to/recording
```

The output then goes to `detections/first_run/`, with the config copied to
`config.toml`. Every key and its default: [The production
pipeline](pipeline.md#run-config) and [`examples/run.toml`](../examples/run.toml).

## Next

- Compare the chunked pipeline with the reference path on part of a
  recording: `chirpdetector check-reference run.toml REC_DIR --start-s 0
  --duration-s 20` (see [The production pipeline](pipeline.md#checking-the-pipeline)).
- Audit the output by hand: label `candidates.csv` with
  [`label-emitters`](labelling.md).
- Score the output against labels: [Evaluation](evaluation.md).

> **Verified.** The commands on this page were run on the CPU on a 20-s
> simulated scene (`gridtools sim run`, 3 fish on a 3 x 3 grid) and on 30 s
> of a real recording, with the models downloaded into a fresh cache.
