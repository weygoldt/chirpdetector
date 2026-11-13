# Training a detector

```sh
chirpdetector train detector CONFIG.toml --dataset DATASET --run-dir RUN
```

The config selects the model, optimisation, augmentation and
cross-validation (`chirpdetector.training.config`). Two models are built
in:

| `[model] type` | model | predicts | assignment in the pipeline |
|---|---|---|---|
| `dualhead` (default) | RT-DETRv2 (ResNet18-vd by default, COCO weights) with an emitter head | boxes and the emitter's baseline EODf | nearest live track to the predicted EODf |
| `ultralytics-yolo` | YOLO26 (`yolo26s` by default), NMS-free one-to-one head | boxes only | none (`assign_status = "no_emitter"`) |

Start from the example configs; they hold the settings of the released
models:

```sh
chirpdetector train detector examples/train_dualhead.toml --dataset DATASET --run-dir runs/dualhead
chirpdetector train detector examples/train_yolo26s.toml  --dataset DATASET --run-dir runs/yolo26s
```

> **Optimiser defaults.** A config without `[model] type` trains the
> dual-head model, but the `[optim]` defaults (lr0 1e-3, 100 epochs, warm-up
> 3 epochs, patience 20) were tuned for YOLO26. The dual-head model was
> trained with lr0 1e-4, lrf 0.05, warm-up 0.5 epochs, weight decay 1e-4,
> 30 epochs and patience 6, as in
> [`examples/train_dualhead.toml`](../examples/train_dualhead.toml).

## Dual-head config

[`examples/train_dualhead.toml`](../examples/train_dualhead.toml), the
settings of the released detector, abridged:

```toml
seed = 0
deterministic = false        # deformable attention has no deterministic CUDA backward
device = "cuda"
max_vram_fraction = 0.35
score_threshold = 0.5

[model]
type = "dualhead"
architecture = "rtdetr_v2_r18vd"   # COCO weights from the Hugging Face hub
pretrained = true
max_det = 50
emitter_loss_weight = 10.0
time_shift_px = 32
val_max_images = 2000

[optim]
epochs = 30
batch = 16
lr0 = 0.0001
lrf = 0.05
warmup_epochs = 0.5
weight_decay = 0.0001
patience = 6

[cv]
n_folds = 0                  # one fit on the train groups,
final_fit = true             # validated on the val groups
```

`[model]` keys of the dual-head model (`chirpdetector.models.dualhead`):
`architecture` (`rtdetr_v2_r18vd`, `rtdetr_v2_r34vd`, `rtdetr_v2_r50vd`;
`tiny` for tests), `pretrained` (COCO weights from the Hugging Face hub),
`weights` (continue from a dual-head weights file), `resume` (continue an
interrupted fit from its `dualhead/state.pt`, in a new run directory),
`num_queries`, `num_denoising`, `max_det`, `emitter_loss_weight`,
`backbone_lr_scale`, `grad_clip`, `ema_decay`, `time_shift_px` (random time
shifts in training) and `val_max_images`.

After every epoch the moving average of the weights is validated with the
evaluation's matching rule on the validation images: precision, recall,
F1, AP, the emitter error of the true positives and the assignment
accuracy. The fitness is AP x assignment accuracy; the best epoch is
shipped. A fit that is stopped (Ctrl-C, or stopping its systemd scope)
ships its best validated epoch as the bundle.

## YOLO26 config

[`examples/train_yolo26s.toml`](../examples/train_yolo26s.toml), the box-only
reference, abridged:

```toml
[model]
type = "ultralytics-yolo"
architecture = "yolo26s"     # ultralytics downloads the COCO weights
pretrained = true

[optim]
epochs = 80
batch = 32
lr0 = 0.001
patience = 15

[cv]
n_folds = 0
final_fit = true
```

`[model] architecture` is any YOLO26 size (`yolo26n` ... `yolo26x`); the
ultralytics trainer is used with our dataset, augmentation and loss
masking. Validation (early stopping, `best.pt`) uses ultralytics' metrics
with the NMS-free head as deployed; the matching-rule metrics come from the
evaluation afterwards.

## What both share

- **No resampling.** Images are fed at their native size, zero-padded at
  the bottom and right to the model stride (205 x 640 -> 224 x 640): one
  pixel is one frame by one frequency bin, in training and at inference.
- **Augmentation**: optional time reversal (`time_flip`, off by default)
  and, for the dual-head model, time shifts. No vertical flips, colour
  jitter, rotation, scaling, translation or mosaic: they break the time and
  frequency axes of a spectrogram. Variety comes from the simulator.
- **Ignore regions**: the classification loss of an unmatched prediction
  inside an ignore region is zero, so invisible chirps and harmonic copies
  neither train the model to fire nor to stay silent.
- **Splits**: with `cv.n_folds = k >= 2`, grouped k-fold cross-validation
  over the `train` groups, each fold from scratch; with `cv.final_fit`, one
  model on all `train` groups, validated on the `val` groups. That model is
  the run's bundle. `test` groups are never read. `--fold k` or `--fold
  final` trains one fit, e.g. folds in parallel processes into one run
  directory.
- **GPU memory**: `max_vram_fraction` and `CHIRP_MAX_VRAM_FRACTION` cap the
  process's share; batch sizes are fixed.

## Run directory

```text
RUN/
  run.json             run id, dataset hash, fits, CV summary, bundle
  config.json          resolved config (config.toml: the input file)
  provenance.json      git SHAs of chirpdetector and gridtools (+ dirty flags),
                       uv.lock hash, versions, hardware
  dataset.json         copy of the dataset manifest
  folds.json           group -> fold
  fold-0/ ... final/   one directory per fit:
    fit.json           groups, image counts, best epoch, validation metrics
    curves.csv         per-epoch losses, metrics, learning rates
    bundle/            weights.pt + bundle.json
    dualhead/ or ultralytics/   the backend's own files (checkpoints)
```

A run directory refuses fits made with a different config or dataset.
`chirpdetector bundle inspect RUN/final/bundle` verifies and prints the
bundle; [Evaluation](evaluation.md) scores it.

## Trying it on a small scale

Both example configs train on the CPU with `device = "cpu"`, `epochs = 1`,
a small `batch` and `workers = 0`; on the 18-image dataset from
[Detector datasets](datasets.md) each took about 10 s:

```text
$ chirpdetector train detector smoke_dualhead.toml --dataset datasets/tiny --run-dir train_dh
...
final: 9 train / 4 val images, batch 2, 4 steps per epoch, device cpu, VRAM cap 0.35
final epoch 1: loss 139.802 | P 0.028 R 0.625 F1 0.053 AP 0.060 | emitter MAE 127.6 Hz, assignment 0.200 | fitness 0.0120 (best)
bundle: train_dh/final/bundle
```

The released dual-head model trained on the 64,800 training tiles of
sim_v4 at batch 16 (bf16 on an RTX 4080, peak 3.1 GiB of GPU memory); early
stopping ended it after 22 of 30 epochs, and epoch 16 was shipped.

## Adding a model

A backend implements three methods (`resolve_model_config`, `fit`, `load`)
and registers itself with `chirpdetector.models.register_backend`;
datasets, folds, run directories, bundles, the pixel-to-time/frequency
map and the evaluation are shared. A backend with an emitter head reports
the emitter's row as an extra pixel column (`emitter_y`), which becomes
`emitter_hz`.
