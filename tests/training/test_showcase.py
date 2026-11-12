"""The showcase: panels per snippet and model, a contact sheet that grows."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image
from typer.testing import CliRunner

from chirpdetector.bundle import Bundle, load_bundle, write_bundle
from chirpdetector.cli import app
from chirpdetector.data import Tiling
from chirpdetector.evaluation.showcase import (
    Snippet,
    bundle_column,
    read_snippets,
    render_showcase,
    v1_column,
)
from chirpdetector.spectrogram import SpectrogramConfig

from .conftest import STUB, StubBackend

CONFIG = SpectrogramConfig()
TILING = Tiling(
    tile_duration_s=0.64, overlap_s=0.2, width_px=128, height_px=CONFIG.n_freqs
)


def _bundle(directory: Path, weights: bytes) -> Bundle:
    source = directory.parent / f"{directory.name}.bin"
    source.write_bytes(weights)
    write_bundle(
        directory,
        weights=source,
        model_type=STUB,
        model={"architecture": "stub"},
        spectrogram=CONFIG,
        tiling=TILING,
        preprocessing={"click_removal": None},
        score_threshold=0.5,
        training={"run_id": "test"},
    )
    return load_bundle(directory)


@pytest.fixture
def snippets_file(tmp_path: Path, make_synthetic_trial: Callable[[Path], Path]) -> Path:
    trial = make_synthetic_trial(tmp_path / "2000-01-01-10_00")
    path = tmp_path / "snippets.toml"
    path.write_text(
        f"""
[[snippet]]
name = "chirp"
recording = "{trial}"
t_start_s = 3.0
t_end_s = 5.0
reason = "fish 0 chirps at 4 s"

[[snippet]]
name = "quiet"
recording = "{trial}"
t_start_s = 5.5
t_end_s = 7.5
f_min_hz = 600.0
f_max_hz = 900.0
reason = "no chirp"
"""
    )
    return path


def test_snippets_are_read_and_checked(snippets_file: Path, tmp_path: Path) -> None:
    snippets = read_snippets(snippets_file)
    assert [s.name for s in snippets] == ["chirp", "quiet"]
    assert snippets[1].f_max_hz == 900.0
    with pytest.raises(ValueError, match="t_end_s"):
        Snippet(name="x", recording="r", t_start_s=2.0, t_end_s=1.0)
    bad = tmp_path / "bad.toml"
    bad.write_text(snippets_file.read_text().replace('"quiet"', '"chirp"'))
    with pytest.raises(ValueError, match="unique"):
        read_snippets(bad)


def test_showcase_grows_by_columns(
    snippets_file: Path, tmp_path: Path, stub_backend: StubBackend
) -> None:
    stub_backend.output = pd.DataFrame(
        {
            "x_min": [60.0],
            "y_min": [100.0],
            "x_max": [70.0],
            "y_max": [140.0],
            "score": [0.9],
            "emitter_y": [140.0],
        }
    )
    snippets = read_snippets(snippets_file)
    first = _bundle(tmp_path / "b1", b"one")
    out = tmp_path / "showcase"
    sheet = render_showcase(
        [bundle_column(first), v1_column()], snippets, out, device="cpu"
    )
    index = json.loads((out / "showcase.json").read_text())
    assert [c["id"] for c in index["columns"]] == ["v1", first.weights_sha256[:12]]
    panels = sorted((out / "panels").glob("*.png"))
    assert len(panels) == 4
    width, height = Image.open(panels[0]).size
    assert Image.open(sheet).size == (2 * width, 2 * height)
    stamp = panels[0].stat().st_mtime_ns

    second = _bundle(tmp_path / "b2", b"two")
    render_showcase([bundle_column(second, threshold=0.3)], snippets, out, device="cpu")
    index = json.loads((out / "showcase.json").read_text())
    assert [c["id"] for c in index["columns"]] == [
        "v1",
        first.weights_sha256[:12],
        second.weights_sha256[:12],
    ]
    assert index["columns"][2]["threshold"] == 0.3
    assert len(list((out / "panels").glob("*.png"))) == 6
    assert panels[0].stat().st_mtime_ns == stamp  # existing panels are kept
    assert Image.open(sheet).size == (3 * width, 2 * height)
    assert "fish 0 chirps at 4 s" in (out / "index.md").read_text()


def test_showcase_cli(
    snippets_file: Path, tmp_path: Path, stub_backend: StubBackend
) -> None:
    bundle = _bundle(tmp_path / "b", b"cli")
    out = tmp_path / "cli"
    result = CliRunner().invoke(
        app,
        [
            "showcase",
            "--bundle",
            str(bundle.path),
            "--v1",
            "--snippets",
            str(snippets_file),
            "--out",
            str(out),
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "contact_sheet.png").is_file()
    assert (out / "index.md").is_file()
