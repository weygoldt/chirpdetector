# Command line

`chirpdetector --help` and `chirpdetector COMMAND --help` list every option
with its default. `-v` before the command turns on debug logging.

| command | what | page |
|---|---|---|
| `run [CONFIG] [REC_DIR...] [--dataset ROOT] [--out OUT]` | detect, decode and assign chirps in whole recordings (resumable); without a config, `--out` and the released models | [Pipeline](pipeline.md) |
| `render RUN_DIR RECORDING T0 T1` | re-render a window of a run (PNG + NPZ) | [Pipeline](pipeline.md#sample-windows-and-rendering) |
| `check-reference CONFIG REC_DIR` | compare the chunked pipeline with `predict_span` on part of a recording | [Pipeline](pipeline.md#checking-the-pipeline) |
| `models list` / `download [NAME...]` / `path NAME` | the released models: list, download, locate in the cache | [Pretrained models](models.md) |
| `bundle inspect BUNDLE` | verify a bundle's weight hash and print what it holds | [Pretrained models](models.md#bundles) |
| `data build-scenes [SCENES]` | tile simulated scenes into a detector dataset | [Detector datasets](datasets.md) |
| `data build-real` | tile the hand-labelled regions of real recordings into a test set | [Detector datasets](datasets.md#from-hand-labelled-recordings) |
| `dataset validate DATASET` | check a dataset against the contract | [Detector datasets](datasets.md) |
| `train detector CONFIG --dataset D --run-dir R` | train a detector (dual-head by default, or YOLO26) | [Training a detector](training.md) |
| `train params CONFIG --scenes S --detector B --run-dir R` | train the chirp-parameter decoder | [Training the decoder](decoder-training.md) |
| `evaluate PREDICTIONS... --labels L... --out O` | score detections (also `evaluate detections`) | [Evaluation](evaluation.md) |
| `evaluate params BUNDLE --scenes S` | evaluate a decoder | [Training the decoder](decoder-training.md#evaluation) |
| `showcase --snippets S --out O --bundle B` | models side by side on snippets of real recordings | [Evaluation](evaluation.md#showcase) |
| `label-emitters` | label chirps and their emitters (matplotlib or browser) | [Labelling](labelling.md) |
| `label-summary [OUT]` | count labels by decision, source and ΔF bin | [Labelling](labelling.md) |
| `label-benchmark build` / `score` | human against model: sample items, score labels | [Human against model](benchmark.md) |
| `label-benchmark features` / `gaps` / `build-v2` | the edge-case benchmark | [Human against model](benchmark.md#the-edge-case-benchmark-v2) |
| `label-benchmark gallery` | real `not_chirp` boxes next to similar simulated chirps | [Human against model](benchmark.md) |

## Environment variables

| variable | effect |
|---|---|
| `CHIRP_MAX_VRAM_FRACTION` | cap on this process's share of GPU memory (training and inference; the smaller of it and the config's value applies) |
| `CHIRPDETECTOR_CACHE` | cache directory of the released models (default `$XDG_CACHE_HOME/chirpdetector`, else `~/.cache/chirpdetector`) |
| `CHIRPDETECTOR_OFFLINE` | `1`: never download a model |
| `CHIRPDETECTOR_MODELS_URL` | download the released models from this base URL instead of the GitHub release (mirrors, tests) |
| `HF_HOME` | Hugging Face cache, where the COCO weights for training the dual-head model are stored |
