# Installation

chirpdetector needs Python 3.13 or later. It runs on the CPU; a CUDA GPU
makes the pipeline and training much faster (the full pipeline ran at
93-213x real time per recording on an RTX 4080 with a 40 % memory cap).

## Into an environment

```sh
uv pip install "git+https://github.com/weygoldt/chirpdetector"
# or
pip install "git+https://github.com/weygoldt/chirpdetector"
```

This installs [gridtools](https://github.com/weygoldt/gridtools) (recording
I/O and the simulator) from GitHub as well. Do not install `gridtools` from
PyPI: that name belongs to an unrelated package.

Versions of chirpdetector up to `main` 341fee5 found gridtools through a
relative path (`../gridtools`) and fail to install from GitHub. To install
one of those, name both packages and tell uv to ignore that path:

```sh
uv pip install --no-sources \
    "gridtools @ git+https://github.com/weygoldt/gridtools" \
    "chirpdetector[dualhead,label] @ git+https://github.com/weygoldt/chirpdetector@341fee5"
```

## Into a uv project

```sh
uv add "chirpdetector @ git+https://github.com/weygoldt/chirpdetector"
uv add "chirpdetector[label] @ git+https://github.com/weygoldt/chirpdetector"   # with the labeller
```

## Extras

| extra | adds | needed for |
|---|---|---|
| `label` | matplotlib | `label-emitters` (both frontends render with it), `showcase`, the figures of `evaluate params` and `label-benchmark` |
| `dualhead` | nothing new | transformers (the dual-head detector) is a core dependency; the extra is kept so that `chirpdetector[dualhead]` installs keep working |

## GPU memory

On a shared GPU, cap this process's share of GPU memory with the
environment variable `CHIRP_MAX_VRAM_FRACTION` (e.g. `0.3`), or with
`max_vram_fraction` in the training and run configs, or `--vram` of
`chirpdetector run`. The smallest of them applies
(`torch.cuda.set_per_process_memory_fraction`). Batch sizes are fixed in the
configs; there is no automatic batch sizing.

## From source

```sh
git clone https://github.com/weygoldt/chirpdetector
cd chirpdetector
uv sync --all-extras          # gridtools comes from its GitHub main
uv run chirpdetector --help
```

To change gridtools and chirpdetector together, work in the
chirp_competition_upgrade workspace (private), which holds both as git
submodules and installs both editable; see [Development](development.md).
