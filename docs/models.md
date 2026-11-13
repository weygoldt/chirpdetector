# Pretrained models

The released models are the assets of the GitHub release
[`models-v1`](https://github.com/weygoldt/chirpdetector/releases/tag/models-v1):

| name | versioned name | file | what |
|---|---|---|---|
| `detector` | `dualhead-rtdetrv2-r18-sim_v4` | `chirpdetector-dualhead-rtdetrv2-r18-sim_v4.zip` (78 MB) | dual-head RT-DETRv2-r18: chirp boxes and the emitter's baseline EODf (weights sha256 `9fcdef3c4371…`) |
| `decoder` | `decoder-dualhead-sim_v4` | `chirpdetector-decoder-dualhead-sim_v4.zip` (3 MB) | chirp-parameter decoder trained on that detector's boxes (weights sha256 `c0801c565a4b…`), with its evaluation report |

`SHA256SUMS` lists the checksums of both files. Each zip holds one
[bundle](#bundles).

## Download and cache

chirpdetector downloads a model the first time it is used and checks it
against the checksums. The cache directory is `$CHIRPDETECTOR_CACHE` if set,
else `$XDG_CACHE_HOME/chirpdetector`, else `~/.cache/chirpdetector`.

```sh
chirpdetector models list                 # names, versions, cached or not
chirpdetector models download             # all models
chirpdetector models download detector    # one
chirpdetector models path decoder         # the bundle directory in the cache
```

From Python:

```python
from chirpdetector import pretrained

pretrained.list_models()
bundle_dir = pretrained.download("detector")  # returns the cached bundle directory
pretrained.path("decoder")
```

`"default"` stands for the released model wherever a bundle is expected:

```python
from chirpdetector.bundle import load_bundle
from chirpdetector.inference import predict_span

detector = load_bundle("default")
chirps = predict_span(
    detector, recording, t0, t1, name="rec", tracks=tracks, params="default"
)
```

In a run config, `[models] detector` and `decoder` default to `"default"`.
Give a bundle directory instead to use another model, or `decoder = ""` to
run without the decoder.

## Offline use

Download once with `chirpdetector models download`, then set
`CHIRPDETECTOR_OFFLINE=1`: chirpdetector then never touches the network and
fails with an error if a model is not cached. Copying the cache directory to
another machine (and pointing `CHIRPDETECTOR_CACHE` at it) works too.

Without the automatic download, unzip the release files yourself and give
the bundle directories by path:

```toml
[models]
detector = "/models/chirpdetector-dualhead-rtdetrv2-r18-sim_v4"
decoder = "/models/chirpdetector-decoder-dualhead-sim_v4"
```

## Bundles

A bundle is a directory with everything needed to use and trust a set of
weights:

```text
BUNDLE/
  bundle.json     model type and description, class map, spectrogram front end,
                  tiling, preprocessing (click removal), default threshold,
                  training provenance, sha256 of the weights
  weights.pt      the weights
  metrics.json    evaluation of exactly these weights (optional)
```

Loading a bundle recomputes the sha256 of `weights.pt` and refuses a bundle
whose weights differ from the recorded hash, or whose `metrics.json` names
other weights. A decoder bundle records the weight hash of the detector it
was trained on, and the pipeline and `predict_span` refuse to combine it
with another detector (`allow_detector_mismatch` overrides this).

```sh
chirpdetector bundle inspect BUNDLE
```

prints and verifies a bundle. For the released detector:

```text
model       dualhead {"architecture": "rtdetr_v2_r18vd", "input_shape": [224, 640], "image_shape": [205, 640], ...}
weights     weights.pt sha256 9fcdef3c4371bbaee0eca75f5614aaf254bb920e42b72896d32c1c11c9bc0754 (verified)
classes     ['chirp']
threshold   0.5
front end   windows [0.2048, 0.0512, 0.0128] s, hop 0.005 s, 500-1500 Hz (205 bins), sum over electrodes
tiling      640 x 205 px, 3.2 s per tile, overlap 1 s
...
metrics     precision 0.868, recall 1.000, f1 0.929 at threshold 0.5
```

The attached `metrics.json` holds the evaluation on the 335 original hand
labels; with the 30 chirps the labeller confirmed later, the same detector
scores precision 0.938, recall 0.992 (see [Limitations](limitations.md)).

## How the released models were made

- **Detector**: fine-tuned from the COCO checkpoint `PekingU/rtdetr_v2_r18vd`
  on simulated scenes only: 3,000 one-minute scenes (gridtools, `sim_v4`)
  with simulated fish carrying real chirp templates, mixed into real
  multi-electrode background recordings taken from the fish's inactive
  period and screened for chirps. Template sets and background recordings
  are disjoint between training, validation and test. Selected on
  validation fitness (AP x assignment accuracy): epoch 16 of 22 (early
  stopping). Config: `examples/train_dualhead.toml`.
- **Decoder**: trained on that detector's boxes on the same scenes; ground
  truth is the noise-free instantaneous frequency of the emitting fish, one
  value per EOD period. Config: `examples/train_params.toml`.
