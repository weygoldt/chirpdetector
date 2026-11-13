"""``chirpdetector models list | download | path``."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from chirpdetector import pretrained
from chirpdetector.cli import app

from ..release import Release

runner = CliRunner()


def test_list_shows_the_models_and_their_cache_state(release: Release) -> None:
    release.publish(release.fake_bundle("a"), "fake-detector-v1")
    release.publish(release.fake_bundle("b"), "fake-decoder-v1", "decoder")
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines[0].split() == ["NAME", "KIND", "ALIAS", "SIZE", "STATUS"]
    assert lines[1].split()[:3] == ["fake-detector-v1", "detector", "detector"]
    assert lines[1].endswith("not downloaded")
    assert lines[-1] == f"cache: {release.cache}"
    assert release.server.requests == []
    pretrained.path("fake-decoder-v1")
    result = runner.invoke(app, ["models", "list"])
    row = next(x for x in result.stdout.splitlines() if x.startswith("fake-decoder"))
    assert row.split()[-1] == "cached"


def test_download_fetches_every_model_or_the_named_ones(release: Release) -> None:
    det = release.publish(release.fake_bundle("a"), "fake-detector-v1")
    dec = release.publish(release.fake_bundle("b"), "fake-decoder-v1", "decoder")
    result = runner.invoke(app, ["models", "download"])
    assert result.exit_code == 0, result.output
    assert f"fake-detector-v1: {release.cache / 'fake-detector-v1'}" in result.stdout
    assert f"fake-decoder-v1: {release.cache / 'fake-decoder-v1'}" in result.stdout
    assert sorted(release.server.requests) == sorted([det.asset, dec.asset])
    # cached: nothing to do; --force downloads again
    assert runner.invoke(app, ["models", "download", "decoder"]).exit_code == 0
    assert len(release.server.requests) == 2
    result = runner.invoke(app, ["models", "download", "decoder", "--force"])
    assert result.exit_code == 0, result.output
    assert release.server.requests[-1] == dec.asset


def test_path_prints_the_bundle_directory(release: Release) -> None:
    release.publish(release.fake_bundle("a"), "fake-detector-v1")
    result = runner.invoke(app, ["models", "path", "detector"])
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{release.cache / 'fake-detector-v1'}\n"
    assert "downloading fake-detector-v1" in result.stderr


def test_errors_exit_with_a_message(
    release: Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    release.publish(release.fake_bundle("a"), "fake-detector-v1")
    unknown = runner.invoke(app, ["models", "path", "yolo"])
    assert unknown.exit_code == 2
    assert "unknown pretrained model 'yolo'" in unknown.stderr
    monkeypatch.setenv(pretrained.ENV_OFFLINE, "1")
    for args in (["path", "detector"], ["download"]):
        offline = runner.invoke(app, ["models", *args])
        assert offline.exit_code == 1
        assert "CHIRPDETECTOR_OFFLINE is set" in offline.stderr
    assert release.server.requests == []
