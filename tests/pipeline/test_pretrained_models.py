"""Run configs without model paths: the pretrained models and the default config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from typer.testing import CliRunner

from chirpdetector.bundle import Bundle
from chirpdetector.cli import app
from chirpdetector.pipeline import run as run_module
from chirpdetector.pipeline.config import ConfigError, config_from_dict, default_config
from chirpdetector.pipeline.run import RunSummary, load_models

from ..release import Release

DETECTOR = "pretrained:dualhead-rtdetrv2-r18-sim_v4"
DECODER = "pretrained:decoder-dualhead-sim_v4"


def _config(**models: object) -> Any:
    data: dict[str, Any] = {"run": {"name": "r"}}
    if models:
        data["models"] = models
    return config_from_dict(data, base_dir=Path("/base"))


def test_models_default_to_the_pretrained_ones() -> None:
    cfg = _config()
    assert (cfg.models.detector, cfg.models.decoder) == (DETECTOR, DECODER)
    explicit = _config(detector="default", decoder="pretrained:decoder")
    assert explicit.models == cfg.models
    assert explicit.detection_hash() == cfg.detection_hash()
    assert _config(decoder="none").models.decoder is None
    # a detector of one's own gets no decoder unless one is named
    own = _config(detector="runs/r1/bundle")
    assert own.models.detector == Path("/base/runs/r1/bundle")
    assert own.models.decoder is None
    assert (
        _config(detector="runs/r1/bundle", decoder="default").models.decoder == DECODER
    )
    with pytest.raises(ConfigError, match="is a decoder, not a detector"):
        _config(detector="pretrained:decoder")
    with pytest.raises(ConfigError, match="unknown pretrained model"):
        _config(detector="pretrained:yolo")


def test_another_default_model_is_another_run(
    release: Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = release.publish(release.fake_bundle("a"), "fake-detector-v1")
    old = _config()
    assert old.models.detector == f"pretrained:{first.name}"
    release.publish(release.fake_bundle("b"), "fake-detector-v2")
    new = _config()
    assert new.models.detector == "pretrained:fake-detector-v2"
    assert new.detection_hash() != old.detection_hash()
    assert release.server.requests == []  # configs never download


def test_default_config() -> None:
    cfg = default_config(Path("/out"))
    assert cfg.run_dir == Path("/out/default")
    assert (cfg.models.detector, cfg.models.decoder) == (DETECTOR, DECODER)
    assert cfg.compute.device == ("cuda" if torch.cuda.is_available() else "cpu")


def test_load_models_downloads_the_pretrained_detector(
    release: Release, stub_bundle: Bundle
) -> None:
    model = release.publish(stub_bundle.path, "stub-detector-v1")
    cfg = config_from_dict({"run": {"name": "r"}, "compute": {"device": "cpu"}})
    assert cfg.models.decoder is None  # no published decoder for this detector
    models = load_models(cfg)
    assert models.bundle.path == release.cache / model.name
    assert models.bundle.weights_sha256 == stub_bundle.weights_sha256
    assert models.info["detector"]["path"] == str(release.cache / model.name)
    assert models.estimator is None
    assert release.server.requests == [model.asset]


def test_cli_run_without_a_config_uses_the_default_config(
    tmp_path: Path, trial_dirs: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(config: Any, paths: list[Path], **kwargs: Any) -> RunSummary:
        seen.update(config=config, paths=paths, **kwargs)
        return RunSummary(run_dir=config.run_dir, done=[p.name for p in paths])

    monkeypatch.setattr(run_module, "run_pipeline", fake_run)
    runner = CliRunner()
    result = runner.invoke(
        app, ["run", "--out", str(tmp_path / "out"), *map(str, trial_dirs)]
    )
    assert result.exit_code == 0, result.output
    assert seen["config"].run_dir == tmp_path / "out" / "default"
    assert seen["config"].models.detector == DETECTOR
    assert seen["config"].models.decoder == DECODER
    assert seen["paths"] == [p.resolve() for p in trial_dirs]
    assert seen["config_file"] is None
    # --list works without a config and downloads nothing
    listed = runner.invoke(
        app, ["run", "--out", str(tmp_path / "out"), str(trial_dirs[0]), "--list"]
    )
    assert listed.exit_code == 0, listed.output
    assert listed.output.startswith("todo")
    # neither a config nor --out
    bare = runner.invoke(app, ["run", str(trial_dirs[0])])
    assert bare.exit_code == 2
    assert "--out OUT for the default config" in bare.output
    missing = runner.invoke(app, ["run", str(tmp_path / "nope.toml")])
    assert missing.exit_code == 2
    assert "does not exist" in missing.output
