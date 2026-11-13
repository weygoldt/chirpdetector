# Python API

The pipeline is built from functions you can call directly. The module
docstrings document every function; this page shows the common uses.

## Detect chirps in a span: `predict_span`

```python
from gridtools.io import load_tracks, open_recording
from chirpdetector.inference import predict_span

rec_dir = "/data/2019-12-09-10_00"
recording = open_recording(rec_dir)  # fishgrid directory or wav, memory-mapped
tracks = load_tracks(rec_dir)  # wavetracker arrays, optional

chirps = predict_span(
    "default",  # the released detector, or a bundle directory
    recording,
    1140.0,  # span start (s from the first sample)
    1420.0,  # span end
    name="2019-12-09-10_00",  # value of the `recording` column
    tracks=tracks,  # assign emitters, find harmonic copies
    params="default",  # add the decoder's chirp parameters
    device="cuda",  # or "cpu"
)
```

`predict_span` does what the detector datasets did: click removal where the
grid allows it, the bundle's spectrogram front end, tiles of 3.2 s with
1 s overlap (each normalised on its own frames), the model, the exact map
from pixels to time and frequency, and the merge of overlapping tiles
(each tile keeps the detections centred in its own part; a chirp found in
two tiles near their shared boundary is kept once). With `tracks`, each
detection of an emitter-head model is assigned to the live track nearest to
its `emitter_hz` (`emitter_id`, within `emitter_tolerance_hz`, default
50 Hz, at the box centre). Then [post-processing](postprocessing.md) drops
harmonic copies and duplicates (`keep_dropped=True` keeps them with their
reason; `postprocess=False` switches it off).

Columns: `recording`, `tile`, `t_start_s`, `t_end_s`, `f_low_hz`,
`f_high_hz`, `score`, `model_sha256`, `emitter_hz` (NaN for box-only
models), with tracks `emitter_id` and `emitter_df_hz`, with `params` the
[decoder's columns](decoder.md#output-columns), and the post-processing
columns `drop_reason`, `harmonic_of_id`, `harmonic_pair_of`,
`duplicate_of`, `reattributed`, `harmonic_emitter_hz`, `harmonic_f_low_hz`,
`harmonic_f_high_hz`. `chirps.attrs["provenance"]` records the model hash,
threshold, tolerances and post-processing settings and counts.

`predict_span` and the pipeline differ in the assignment: the pipeline
assigns at the chirp's onset against the track's baseline in the preceding
second, within 20 Hz, with a confidence ([Assignment](pipeline.md#assignment));
`predict_span` uses the nearest live track at the box centre within 50 Hz.
For analyses, use the pipeline's table.

## Bundles

```python
from chirpdetector.bundle import load_bundle

bundle = load_bundle("default")  # or a bundle directory; verifies the sha256
bundle.model_type  # "dualhead" or "ultralytics-yolo"
bundle.spectrogram  # the front end (SpectrogramConfig)
bundle.score_threshold
```

Loading the model separately lets you reuse it across many calls:

```python
from chirpdetector.prediction import load_detector
from chirpdetector.params.inference import ParamsEstimator

detector = load_detector(bundle, device="cuda")
params = ParamsEstimator("default", device="cuda")
chirps = predict_span(
    bundle,
    recording,
    0.0,
    recording.duration_s,
    name="rec",
    detector=detector,
    params=params,
    tracks=tracks,
)
```

## Spectrograms

The front end is one function, used by datasets, training and inference
alike; its config is in seconds and Hz and stored in every bundle:

```python
from chirpdetector.spectrogram import (
    SpectrogramConfig,
    compute_spectrogram,
    normalize,
    to_image,
)

config = bundle.spectrogram  # or SpectrogramConfig()
samples = recording.read(
    0, int(10 * recording.samplerate_hz)
)  # (n_samples, n_electrodes)
spec = compute_spectrogram(samples, recording.samplerate_hz, config)
image = to_image(normalize(spec, config))  # (H, W, C) uint8, row 0 = highest frequency
```

## Reading the pipeline's output

```python
import json
import numpy as np
import pandas as pd

d = "detections/first_run/2019-12-09-10_00"
chirps = pd.read_parquet(f"{d}/chirps_rtdetr.parquet")
meta = json.load(open(f"{d}/meta.json"))

chirps = chirps[chirps["chirp"]]  # chirps at the operating threshold
assigned = chirps[chirps["assign_status"].isin(["assigned", "reattributed"])]
times = np.load(f"{d}/chirp_times_rtdetr.npy")  # = assigned.t_center_s, sorted
ids = np.load(f"{d}/chirp_ids_rtdetr.npy")  # = assigned.track_id
```

To score the pipeline's output with [`chirpdetector evaluate`](evaluation.md),
keep the unflagged rows and name the emitter column as the evaluation
expects:

```python
c = pd.read_parquet(f"{d}/chirps_rtdetr.parquet")
c[c["kept"]].rename(columns={"track_id": "emitter_id"}).to_parquet("kept.parquet")
```

## Modules

| module | what |
|---|---|
| `chirpdetector.pipeline` | `run_pipeline` and its parts: config, streaming, assignment, outputs, QC, windows, candidates |
| `chirpdetector.inference` | `predict_span`, `detect_in_spectrogram`, `merge_tiles` |
| `chirpdetector.prediction` | `load_detector`, `predict_images`, `predict_dataset` |
| `chirpdetector.bundle` | `load_bundle`, bundle format |
| `chirpdetector.pretrained` | `download`, `path`, `list_models` of the released models |
| `chirpdetector.spectrogram` | the front end, framing for chunked computation, pixel <-> time/frequency |
| `chirpdetector.preprocessing` | click removal |
| `chirpdetector.postprocess` | harmonic copies, duplicates, re-attribution |
| `chirpdetector.params` | the chirp-parameter decoder: truth, crops, model, training, calibration, inference, evaluation |
| `chirpdetector.data` | detector dataset contract and builders |
| `chirpdetector.models` | detector backends (`dualhead`, `ultralytics-yolo`) |
| `chirpdetector.training` | detector training, grouped folds, provenance |
| `chirpdetector.evaluation` | matching, metrics, reports, showcase |
| `chirpdetector.labeling` | the labeller and the human benchmark |
