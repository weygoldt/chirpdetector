"""``load_bundle("default")`` and friends resolve the pretrained models."""

from __future__ import annotations

from pathlib import Path

import pytest

from chirpdetector.bundle import Bundle, load_bundle, write_bundle
from chirpdetector.data.tiles import TileLayout
from chirpdetector.params.bundle import load_params_bundle
from chirpdetector.spectrogram import SpectrogramConfig

from ..release import Release


def _detector_bundle(directory: Path) -> Bundle:
    weights = directory / "w.bin"
    directory.mkdir(parents=True)
    weights.write_bytes(b"pretrained stub detector")
    config = SpectrogramConfig()
    return write_bundle(
        directory / "bundle",
        weights=weights,
        model_type="pretrained-stub",
        model={},
        spectrogram=config,
        tiling=TileLayout().tiling(config),
        preprocessing={"click_removal": None},
        score_threshold=0.5,
        training={"run_id": "stub"},
    )


def test_load_bundle_default_is_the_pretrained_detector(
    release: Release, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _detector_bundle(tmp_path / "source")
    release.publish(source.path, "stub-detector-v1")
    decoder = release.publish(
        release.fake_bundle("decoder"), "stub-decoder-v1", "decoder"
    )
    bundle = load_bundle("default")
    assert bundle.path == release.cache / "stub-detector-v1"
    assert bundle.weights_sha256 == source.weights_sha256
    assert bundle.to_dict() == source.to_dict()
    for reference in ("pretrained:detector", "pretrained:stub-detector-v1"):
        assert load_bundle(reference).path == bundle.path
    assert len(release.server.requests) == 1
    with pytest.raises(ValueError, match="is a decoder, not a detector"):
        load_bundle(f"pretrained:{decoder.name}")
    # a Path is always a directory, even one called "default"
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_bundle(Path("default"))


def test_load_params_bundle_default_needs_a_decoder(release: Release) -> None:
    release.publish(release.fake_bundle("detector"), "stub-detector-v1")
    with pytest.raises(KeyError, match="unknown pretrained model 'decoder'"):
        load_params_bundle("default")
    with pytest.raises(ValueError, match="is a detector, not a decoder"):
        load_params_bundle("pretrained:stub-detector-v1")
