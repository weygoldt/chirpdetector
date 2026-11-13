# Detector datasets

A detector dataset is a directory of spectrogram tiles with their chirp
boxes, in one format that every trainer and the evaluation read
(`chirpdetector.data.contract`):

```text
DATASET/
  dataset.json        manifest: front end, tiling, sources, groups and splits,
                      per-image geometry, content hashes, provenance
  boxes.parquet       one row per chirp box and per ignore region
  images/<id>.png     spectrogram tiles, H x W x 3 uint8 (channel c = analysis window c)
  labels/<id>.txt     YOLO labels of the chirps, "0 xc yc w h"
  diagnostics/        contact sheet and counts (not part of the contract)
```

- **Images** are tiles of the front end's normalised spectrogram: 3.2 s x
  500-1500 Hz, 640 x 205 pixels (5 ms hop, 4.88 Hz bins), channel 0 the
  0.2048-s window, channels 1 and 2 the 0.0512-s and 0.0128-s windows. Row 0
  is the highest frequency. Every image has the same size, so a detector
  never rescales a spectrogram.
- **Geometry**: every image stores the exact map between its pixels and
  absolute time and frequency; boxes are stored in seconds and Hz and, for
  chirps, also as YOLO labels derived from them.
- **Boxes** carry `emitter_hz` (the emitter's baseline EODf at the chirp's
  centre), `emitter_id`, `live_tracks_hz` (all fish alive at the chirp),
  and per-chirp covariates (SNR, family, ΔF to the nearest fish, overlap,
  template id) for stratified evaluation.
- **Ignore regions** (`kind == "ignore"`, reason `invisible`, `edge`,
  `harmonic`, `unlabelled`) count neither as chirps nor as background: the
  losses of both detectors are masked there.
- **Splits** are assigned to groups (a simulated scene, a real recording),
  never to images, so no source is on both sides of a split; identical
  images in two splits are refused.
- **Hashes**: `content_sha256` identifies the dataset in training runs and
  bundles.

```sh
chirpdetector dataset validate DATASET     # check the contract, hash and decode every file
```

## From simulated scenes

```sh
chirpdetector data build-scenes SCENES_DIR --out DATASET --split train=0:1600,val=1600:1800,test=1800:2000
chirpdetector data build-scenes --out DATASET \
    --split-dir train=SCENES/train --split-dir val=SCENES/val --split-dir test=SCENES/test \
    --workers 8 --threads 2 --device cuda
```

`SCENES_DIR` is a directory of scenes or a quoted glob. `--split` assigns
splits by seed range (or fractions with `--split-seed`); `--split-dir`
assigns the scenes of a directory to a split (the released detector's
dataset `sim_v4` was built this way from `train_v4/{train,val,test}`).
`--limit N` builds from the first N scenes only.

For each scene: the recording is cleaned of clicks as inference will clean
it (or not at all where the grid has too few electrodes), its spectrogram is
computed once and cut into tiles that cover the scene end to end, each tile
normalised on its own. The boxes are the chirps'
[footprints](design.md#footprint-boxes): visible chirps are positives,
clipped to the tile if at least 70 % of the box and its centre lie in it
(else ignore regions, `edge`); invisible chirps and second-harmonic
footprints are ignore regions.

A small run, on two 20-s scenes on the CPU:

```text
$ chirpdetector data build-scenes scenes --out datasets/tiny --split train=0:1,val=1:2,test=2:3 --workers 2
wrote 18 images of 2 scenes to datasets/tiny ({'train': 9, 'val': 9, 'test': 0}); content sha256 9575497bce8e
$ chirpdetector dataset validate datasets/tiny
valid: 18 images ({'train': 9, 'val': 9, 'test': 0}), 2 groups, content sha256 9575497bce8ee6a10746472c0c1174dddce25fcfea5a7068a665df7374a4c2d3
```

## From hand-labelled recordings

```sh
chirpdetector data build-real --out DATASET --recording REC_DIR [--recording REC_DIR ...]
```

Tiles the labelled regions (audian label files,
`traces-grid1-editable-labels.csv`, category `labeled_region`) of real
recordings into the test split, with the same front end, tiling and click
removal; each recording is one group. Hand-drawn chirp boxes are kept as
drawn; parts of a tile outside the region's labelled band become ignore
regions (`unlabelled`). `--recording` is required (the chirp competition's
real test set is its four hand-labelled recordings). `emitter_hz` is only
a proxy here (nearest track to
the box's lower edge); the real test set scores detection.

## In Python

```python
from chirpdetector.data import load_dataset

dataset = load_dataset("DATASET")  # validates first
dataset.manifest.image_ids("test")
```

Builders write the format with `chirpdetector.data.DatasetWriter`;
`chirpdetector.data.synthetic` makes tiny datasets of boxes on noise for
tests.
