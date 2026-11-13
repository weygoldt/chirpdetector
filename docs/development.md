# Development

chirpdetector is developed together with
[gridtools](https://github.com/weygoldt/gridtools), in the
chirp_competition_upgrade workspace (private) that holds both as git
submodules and pins one environment (`uv.lock`) for both.

```sh
git clone --recurse-submodules git@github.com:weygoldt/chirp_competition_upgrade.git
cd chirp_competition_upgrade
uv sync
make check            # ruff, ruff format --check, pyright and pytest of both packages
```

In a standalone checkout:

```sh
uv sync --all-extras
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest
```

Test markers: `slow` (long-running), `gpu` (needs CUDA; skipped without
it), `data` (needs the competition recordings; skipped without them). With
`CUDA_VISIBLE_DEVICES=""` the suite runs on the CPU only. `tests/test_examples.py`
loads every config in `examples/`, so the configs the docs embed stay valid.

## Documentation

The pages in `docs/` are plain Markdown that render on GitHub and build as
a site with MkDocs (Material theme; config in `mkdocs.yml`):

```sh
uvx --from mkdocs==1.6.1 --with mkdocs-material==9.8.1 mkdocs build --strict   # into site/
uvx --from mkdocs==1.6.1 --with mkdocs-material==9.8.1 mkdocs serve            # live preview
```

`--8<--` lines include files from the repository (the example configs), so
the site always shows the configs the tests load. CI builds the site on
every push. To publish it on the repository's GitHub Pages:

```sh
uvx --from mkdocs==1.6.1 --with mkdocs-material==9.8.1 mkdocs gh-deploy
```

## Conventions

- One definition per transform: the spectrogram front end, the chirp box
  geometry and the decoder input exist once and are used by dataset
  building, training and inference.
- Physical units in every config and field name (`_s`, `_hz`); the sampling
  rate comes with the data.
- Every random draw is seeded; run directories record configs, git SHAs (with
  dirty flags), the `uv.lock` hash, versions and hardware.
- Splits by source (scene, recording), never by image.
- Tests check physics and geometry, not only shapes.
