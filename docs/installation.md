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

!!! note "If the plain install fails"
    Up to `main` 341fee5, chirpdetector found gridtools through a relative
    path (`../gridtools`), which only works in a checkout of both
    repositories side by side. With such a version, name both packages and
    tell uv to ignore that path:

    ```sh
    uv pip install --no-sources \
        "gridtools @ git+https://github.com/weygoldt/gridtools" \
        "chirpdetector[dualhead,label] @ git+https://github.com/weygoldt/chirpdetector"
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
| `dualhead` | transformers | the dual-head detector; kept as an alias now that transformers is a core dependency |

## GPU memory

On a shared GPU, cap this process's share of GPU memory with the
environment variable `CHIRP_MAX_VRAM_FRACTION` (e.g. `0.3`), or with
`max_vram_fraction` in the training and run configs, or `--vram` of
`chirpdetector run`. The smallest of them applies
(`torch.cuda.set_per_process_memory_fraction`). Batch sizes are fixed in the
configs; there is no automatic batch sizing.

## From source

For development, clone both repositories side by side, or the
[chirp_competition_upgrade](https://github.com/weygoldt/chirp_competition_upgrade)
workspace that holds both as submodules (private), and see
[Development](development.md).

```sh
git clone https://github.com/weygoldt/gridtools
git clone https://github.com/weygoldt/chirpdetector
cd chirpdetector
uv sync --all-extras
uv run chirpdetector --help
```
