"""Synthetic detector datasets and helpers to break them."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from chirpdetector.data.contract import BOXES_FILE, MANIFEST_FILE, sha256_file
from chirpdetector.data.synthetic import make_synthetic_dataset


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A valid synthetic dataset (6 groups: 4 train, 1 val, 1 test)."""
    directory = tmp_path_factory.mktemp("datasets") / "synthetic"
    make_synthetic_dataset(directory, seed=3)
    return directory


@pytest.fixture
def broken(synthetic_dir: Path, tmp_path: Path) -> Callable[..., Path]:
    """Copy the synthetic dataset and edit its manifest and/or boxes.

    ``broken(manifest=f, boxes=g)``: ``f(dict)`` edits the parsed
    ``dataset.json`` in place, ``g(DataFrame) -> DataFrame`` the box
    table (its sha256 in the manifest is updated). The stored
    ``content_sha256`` is dropped, so that only the edit is wrong.
    """

    def make(
        manifest: Callable[[dict[str, Any]], None] | None = None,
        boxes: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> Path:
        directory = tmp_path / f"broken-{len(list(tmp_path.iterdir()))}"
        shutil.copytree(synthetic_dir, directory)
        data = json.loads((directory / MANIFEST_FILE).read_text())
        data.pop("content_sha256")
        if boxes is not None:
            table = pd.read_parquet(directory / BOXES_FILE)
            boxes(table).to_parquet(directory / BOXES_FILE, index=False)
            data["boxes_sha256"] = sha256_file(directory / BOXES_FILE)
        if manifest is not None:
            manifest(data)
        (directory / MANIFEST_FILE).write_text(json.dumps(data))
        return directory

    return make
