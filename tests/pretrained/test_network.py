"""The published models, downloaded from GitHub (opt-in: CHIRPDETECTOR_TEST_NETWORK=1)."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest

from chirpdetector import pretrained
from chirpdetector.bundle import load_bundle
from chirpdetector.data.contract import sha256_file
from chirpdetector.params import ParamsEstimator, load_params_bundle
from chirpdetector.prediction import load_detector


@pytest.mark.network
@pytest.mark.slow
def test_the_published_models_download_and_load_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(pretrained.ENV_CACHE, str(tmp_path / "cache"))
    monkeypatch.delenv(pretrained.ENV_MODELS_URL)
    for model in pretrained.list_models():
        directory = pretrained.download(model.name)  # checks size and sha256
        assert directory == tmp_path / "cache" / model.name
        assert sha256_file(directory / "weights.pt") == model.weights_sha256
    detector = load_bundle("default")
    decoder = load_params_bundle("default")
    assert decoder.detector is not None
    assert decoder.detector["weights_sha256"] == detector.weights_sha256

    # from here on, any network access (e.g. the Hugging Face hub) fails
    def refuse(*_: object, **__: object) -> NoReturn:
        msg = "network access"
        raise AssertionError(msg)

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf-home"))
    monkeypatch.setenv(pretrained.ENV_OFFLINE, "1")
    model = load_detector(load_bundle("default"), device="cpu")
    tile = np.zeros(
        (detector.tiling.height_px, detector.tiling.width_px, 3), dtype=np.uint8
    )
    assert len(model.predict([tile], score_threshold=0.5)) == 1
    ParamsEstimator("default", device="cpu")
