# The production pipeline

`chirpdetector run` processes whole recordings: it detects chirps, estimates
their parameters, assigns them to the recording's frequency tracks, flags
harmonic copies and duplicates, and writes one table per recording with
everything needed to analyse and audit the result.

```sh
chirpdetector run --out OUT REC_DIR...                # no config: released models, every default
chirpdetector run CONFIG.toml REC_DIR...              # some recordings
chirpdetector run CONFIG.toml --dataset ROOT          # every recording directory in ROOT
chirpdetector run CONFIG.toml REC_DIR --dry-run       # first 300 s, into run '<name>_dryrun'
chirpdetector run CONFIG.toml --dataset ROOT --list   # what is done and what is left
```

Without a config file, `--out` is required, the run is called `default`,
and the device is CUDA if available, else the CPU.

| option | effect |
|---|---|
| `--dataset ROOT` | process every recording directory in `ROOT` |
| `--out`, `--name` | override `[run] out` and `[run] name` (`--out` is required without a config) |
| `--dry-run` | only `--limit-s` (default 300 s) of each recording, into the run `<name>_dryrun` |
| `--limit-s S`, `--start-s T` | process only `[T, T + S)` of each recording |
| `--redo` | recompute recordings that are complete |
| `--device`, `--vram` | override `[compute] device` and `max_vram_fraction` |
| `--list` | list the recordings and their status; run nothing |

A recording directory is a fishgrid directory (`traces-grid1.raw` and
`fishgrid.cfg`) or a directory with one multichannel wav file. A failing
recording is logged with its traceback, recorded in `run.json`, and the run
moves on; the exit status is 1 if any recording failed.

## Per recording

1. Open the recording (memory-mapped) and its tracks.
2. Detect and decode chunk by chunk: each chunk of `chunk_tiles` detector
   tiles (48 tiles = 105.6 s) is read with some context, cleaned of clicks
   (where the grid has enough electrodes, see [Click removal](clicks.md)),
   turned into the detector's spectrogram, cut into the global tiles of the
   recording and run through the detector down to the store threshold; the
   decoder estimates the parameters of every box. Reading and cleaning run
   ahead of the GPU in a thread pool. The result equals
   `predict_span` over the whole recording (see
   [Checking the pipeline](#checking-the-pipeline)).
3. Assign every detection to a track ([Assignment](#assignment)).
4. Flag second-harmonic copies and duplicate boxes, and re-attribute
   harmonic copies whose fundamental was missed
   ([Post-processing](postprocessing.md)). Flagged rows are kept in the
   table with their reason.
5. Write the chirp table, the chunk table, the legacy arrays, the QC report,
   sample windows and, last, `meta.json`.

Outputs go to a hidden temporary directory that is renamed when everything
is written, so a recording's directory exists only when it is complete.
On a rerun, a recording is skipped if its `meta.json` says `"status":
"complete"` and was made with the same models, detection settings and time
range; otherwise it is recomputed.

## Run config

Only `[run] name` is required. Unknown keys are refused, so a typo cannot
silently fall back to a default. Relative paths are relative to the config
file's directory. [`examples/run.toml`](../examples/run.toml) lists every
key with its default; the sections:

```toml
[run]
name = "my_run"               # outputs go to OUT/<name>/<recording>/
out = "detections"

[models]
detector = "default"          # released detector, "pretrained:NAME", or a bundle directory
decoder = "default"           # released decoder, a bundle directory, or "none"

[detection]
store_threshold = 0.05
operating_threshold = 0.5
postprocess = true

[assignment]
tracks = "."                  # relative to the recording directory
tolerance_hz = 20.0
max_gap_s = 10.0

[compute]
device = "cuda"
max_vram_fraction = 0.4

[validation]
windows = 10
```

| key | meaning |
|---|---|
| `run.out` | output root; the run goes to `OUT/<name>/` (the code default is a path of the development machine: always set it) |
| `models.detector` | `"default"` (the released detector, the default), `"pretrained:NAME"` (a published model, see [Pretrained models](models.md)), or a bundle directory |
| `models.decoder` | default: the published decoder trained on the detector's boxes (for the released detector: the released decoder), else none; `"default"`, `"pretrained:NAME"` or a bundle directory; `"none"` (or `""`) runs without the decoder |
| `models.allow_detector_mismatch` | use a decoder trained on another detector's boxes (its estimates are then not validated) |
| `detection.store_threshold` | every detection down to this score is stored, so the operating threshold can change later without rerunning |
| `detection.operating_threshold` | marked in the column `operating`; the legacy arrays, QC and windows use it |
| `detection.postprocess` | flag harmonic copies and duplicates |
| `assignment.tracks` | directory of the wavetracker arrays, relative to the recording directory |
| `assignment.tolerance_hz` | no assignment if the nearest live track is farther from the predicted emitter EODf |
| `assignment.max_gap_s` | a track is alive across gaps in its detections up to this long |
| `assignment.baseline_window_s`, `baseline_guard_s` | the track's baseline is its mean in `[onset - window, onset - guard)` |
| `compute.device` | `cuda` or `cpu` |
| `compute.max_vram_fraction` | cap on this process's share of GPU memory (`CHIRP_MAX_VRAM_FRACTION` also applies; the smaller wins) |
| `compute.chunk_tiles` | tiles per chunk (a multiple of the detector's batch of 16) |
| `compute.click_workers`, `clean_processes`, `prefetch` | threads and processes for reading and click removal, chunks read ahead |
| `validation.qc` | write the QC report |
| `validation.windows`, `window_s` | annotated sample windows per recording and their length |
| `validation.candidates_per_cell`, `seed` | size and seed of the validation sample (`candidates.csv`) |

The resolved config is written to `run.json` and every `meta.json`
(model references resolved to `pretrained:<versioned name>`, so a run made
with one release is not resumed with another); a config file is also
copied to `config.toml`.

## Outputs

```text
OUT/<name>/
  run.json                   manifest: config, code, models, recordings and their status
  run.log                    log of every invocation (appended)
  config.toml                the config file the run was started with (if any)
  columns.md                 the chirp table's columns, documented
  candidates.csv             validation sample in the labeller's format
  candidates_strata.csv      the same ids with stratum and assignment fields
  <recording>/
    chirps_<model>.parquet   one row per detection with score >= store threshold
    chunks_<model>.parquet   one row per chunk: span, clicks, timings
    chirp_times_<model>.npy  legacy export
    chirp_ids_<model>.npy    legacy export
    meta.json                inputs, models, code, config, counts, timings, QC numbers
    qc/qc.png, qc/qc.md      QC report
    windows/                 annotated sample windows (png + npz, windows.csv)
    renders/                 windows drawn later with `chirpdetector render`
```

`<model>` is the detector family: `rtdetr` for the dual-head detector,
`yolo` for YOLO26. The weights' sha256 and architecture are in `meta.json`
and in the parquet file's metadata (key `chirpdetector`).

### The chirp table

One row per detection with `score >= store_threshold`, after merging the
tiles. Times are seconds from the recording's first sample; the recording's
absolute start is `inputs.start` in `meta.json`. Rows flagged by
post-processing are kept (`kept == False`). `columns.md` documents every
column; the groups:

| group | columns |
|---|---|
| identity | `recording`, `chirp_id` (row after sorting by `t_center_s`), `tile`, `span_id`, `model_sha256` |
| box | `t_start_s`, `t_end_s`, `t_center_s` (the chirp time), `f_low_hz`, `f_high_hz` |
| decision | `score`, `operating` (score >= operating threshold), `kept` (not flagged), `chirp` (= `operating` and `kept`: counts as a chirp) |
| emitter | `emitter_hz` (predicted baseline EODf of the emitter; NaN for box-only models) |
| assignment | `track_id` (wavetracker `ident_v` value, float, NaN = unassigned), `track_hz`, `emitter_track_df_hz`, `assign_status`, `assign_confidence`, `nearest_track_id`, `nearest_track_hz`, `nearest_df_hz`, `second_track_id`, `second_track_hz`, `second_df_hz`, `assign_margin_hz`, `n_live_tracks`, `track_df_other_hz` (ΔF of the emitter's track to its closest live neighbour) |
| decoder, calibrated | `height_hz`, `fwhm_s`, `width10_s`, `undershoot_hz` and their `*_sd_*` |
| decoder, raw | `height_raw_hz`, `fwhm_raw_s`, `width10_raw_s`, `undershoot_raw_hz` |
| decoder domain | `params_ok`, `params_flag` (`edge`, `band`, `size`, `uncertain`, `harmonic`) |
| post-processing | `pp_drop_reason` (`""`, `harmonic`, `duplicate`), `pp_harmonic_of_track`, `pp_harmonic_pair_of`, `pp_duplicate_of`, `pp_reattributed`, `pp_harmonic_emitter_hz`, `pp_harmonic_f_low_hz`, `pp_harmonic_f_high_hz` |
| clicks | `click_fraction_box`, `span_click_fraction`, `span_n_click_runs` |

`assign_status` is one of `assigned`, `reattributed` (a harmonic copy given
to its fish), `beyond_tolerance`, `no_live_track`, `no_tracks` and
`no_emitter` (the detector predicts no emitter, i.e. a box-only model).

What to use: `chirps[chirps["chirp"]]` are the chirps at the operating
threshold. To use another threshold, select `score >= t` and `kept`
instead (both post-processing flags are independent of the threshold;
re-attribution was evaluated at the operating threshold). For chirp
parameters per chirp, use the calibrated columns with `params_ok`; for
distributions over many chirps, compare calibrated and raw ([Chirp
parameters](decoder.md)).

### Legacy arrays

`chirp_times_<model>.npy` and `chirp_ids_<model>.npy` have the semantics of
v1's `_rcnn` arrays, which the chirp competition analysis code reads: 1-D
float64, the box centre in seconds from the recording's first sample and
the raw `ident_v` value of the assigned track. They hold the rows with
`chirp == True` and `assign_status` `assigned` or `reattributed`, sorted by
time; unassigned chirps are not in them (no NaN, no -1).

### meta.json

`status`, `recording`, `run`, `created`, `host`, `detection_hash`,
`t_range_s`, `inputs` (recording directory, files with sizes and hashes,
sample rate, number of samples, duration, electrodes, start time),
`tracks`, `models` (paths, weight hashes, architecture), `code` (versions
and git SHAs), `config`, `click_removal`, `postprocess`, `assignment`,
`operating_threshold`, `counts` (stored, operating, chirps, dropped by
reason, assignment status, legacy, re-attributed, decoder flags, chirps per
track), `timings`, `qc`, `windows`, `outputs`, `legacy`.

### QC report

`qc/qc.png` has one panel each for: chirp rate over time per track; the
score histogram (kept vs flagged, operating threshold marked); the distance
of the predicted emitter EODf to the nearest live track (tolerance marked);
the assignment confidence of assigned chirps; assignment status and
post-processing counts; decoder flags; click removal over time. `qc/qc.md`
tabulates the counts and the quantiles of the nearest-track distance.

### Sample windows and rendering

Each recording gets `validation.windows` annotated windows of
`validation.window_s` seconds: about 40 % at random, 30 % on the densest
chirping and 30 % on uncertain chirps (unassigned, assignment confidence
below 0.8, or a score within 0.15 of the operating threshold). Every window
is a PNG and an NPZ with what is needed to redraw it: the reference-window
power in dB (`spec_db`, float16), `times_s`, `freqs_hz`, the window's
detections (`boxes`), the tracks (`track_ids`, `track_offsets`,
`track_times_s`, `track_freqs_hz`) and `meta` (JSON).

Any other window is drawn on demand; the spectrogram is recomputed from the
raw data with the detector's front end and click removal:

```sh
chirpdetector render OUT/<name> RECORDING T0 T1                  # into RECORDING/renders/
chirpdetector render OUT/<name> RECORDING T0 T1 --out figures/   # elsewhere
```

### Validation candidates

`candidates.csv` is a stratified sample of the run's detections in the
[candidates format](labelling.md#candidates-file), to audit precision and
assignment by hand with `label-emitters --candidates`: score x ΔF cells
(score bins from the store threshold up, so detections below the operating
threshold are audited too), chirps the decoder flagged, rows that
post-processing flagged, and re-attributed harmonic copies. Within a
stratum every recording is equally likely. `candidates_strata.csv` keeps
the stratum of each candidate for the evaluation.

> **Note.** A detection whose box starts before the recording (possible
> for a chirp in the first 0.1 s) gets a negative `t_start_s`, and
> `label-emitters` refuses a candidates file with such a row. Drop those rows
> before labelling.

## Assignment

The dual-head detector predicts, with every box, the emitter's baseline
EODf (`emitter_hz`). A detection is assigned to the nearest *live* track
to that frequency, compared at the chirp's onset (`t_start_s`, not the
centre, where a tracker following the excursion would be pulled away from
the baseline) with the track's baseline before the chirp: the mean of its
detections in the second before the onset, or its frequency at the onset
if it has none there. A track is live at the onset if it exists there and
is not inside a gap longer than `max_gap_s`. No assignment if the nearest
live track is farther than `tolerance_hz` (20 Hz).

`assign_confidence` is the posterior probability that the chosen track is
the emitter under a Student-t error model of `emitter_hz` (fitted to the
released detector's true positives on simulated test scenes: 1.73 degrees
of freedom, scale 1.52 Hz; median absolute error 1.29 Hz, 90 % within
4.7 Hz), with every live track equally likely a priori and a 5 % chance
that the emitter has no live track. On the simulated test chirps it is well
calibrated (expected calibration error 0.013). Simulated tracks are
noise-free; real tracks add the tracker's error, so on real data the
confidence is an upper bound.

YOLO26 bundles predict no emitter: their rows get `assign_status =
"no_emitter"`. Assign them yourself if needed (e.g. nearest track to the
box's lower edge plus an offset), knowing that this baseline was correct
for 0.648 of the simulated test chirps, against 0.914 for the emitter head.

## Checking the pipeline

The chunked pipeline is meant to equal `predict_span` on the whole
recording (one tiling of all frames, each tile normalised on its own,
cross-tile duplicates merged). `check-reference` runs both on a part of a
recording and compares the detections row by row:

```sh
chirpdetector check-reference run.toml REC_DIR --start-s 0 --duration-s 20 --device cpu --out check.json
```

On the CPU the two are bit-identical (`"identical": true`); on CUDA the
spectrograms agree to within 1e-6 (relative), which can flip a pixel's
rounding.

## Throughput

On an RTX 4080 with a 40 % memory cap and the default compute settings,
the three dry runs on two 6-h recordings took 93-213x real time per
recording (dense 9-electrode recording: 154-213x; sparse 15-electrode
recording, where click removal dominates: 93-149x). On the CPU, a 20-s
9-electrode scene took about 10 s.
