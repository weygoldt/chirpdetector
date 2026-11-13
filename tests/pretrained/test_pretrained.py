"""Pretrained models: registry, cache, download checks, offline mode, concurrency."""

from __future__ import annotations

import fcntl
import json
import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import chirpdetector
from chirpdetector import pretrained
from chirpdetector.pretrained import PretrainedError

from ..release import Release


def _contents(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


# ---------------------------------------------------------------- registry


def test_registry_lists_the_published_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(pretrained.ENV_MODELS_URL)
    names = [m.name for m in pretrained.list_models()]
    assert names == ["dualhead-rtdetrv2-r18-sim_v4", "decoder-dualhead-sim_v4"]
    detector = pretrained.get_model("detector")
    decoder = pretrained.get_model("decoder")
    assert detector is pretrained.get_model("dualhead-rtdetrv2-r18-sim_v4")
    assert (detector.kind, decoder.kind) == ("detector", "decoder")
    assert decoder.detector == detector.name
    assert detector.url == (
        "https://github.com/weygoldt/chirpdetector/releases/download/models-v1/"
        "chirpdetector-dualhead-rtdetrv2-r18-sim_v4.zip"
    )
    assert detector.size == 77_619_777
    assert detector.sha256.startswith("e005d9362414")
    assert detector.weights_sha256.startswith("9fcdef3c4371")
    assert decoder.size == 3_345_930
    assert decoder.sha256.startswith("62b22a8846e4")
    assert decoder.weights_sha256.startswith("c0801c565a4b")
    with pytest.raises(KeyError, match="unknown pretrained model"):
        pretrained.get_model("yolo26s")


def test_references_name_models_and_paths_stay_paths() -> None:
    detector = pretrained.get_model("detector").name
    decoder = pretrained.get_model("decoder").name
    assert pretrained.reference("default", "detector") == detector
    assert pretrained.reference("default", "decoder") == decoder
    assert pretrained.reference("pretrained:detector", "detector") == detector
    assert pretrained.reference(f"pretrained:{decoder}", "decoder") == decoder
    assert pretrained.reference(Path("default"), "detector") is None
    assert pretrained.reference("runs/r1/bundle", "detector") is None
    with pytest.raises(ValueError, match="is a decoder, not a detector"):
        pretrained.reference("pretrained:decoder", "detector")
    with pytest.raises(KeyError):
        pretrained.reference("pretrained:nothing", "detector")


def test_cache_dir_follows_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(pretrained.ENV_CACHE, str(tmp_path / "explicit"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert pretrained.cache_dir() == tmp_path / "explicit"
    monkeypatch.delenv(pretrained.ENV_CACHE)
    assert pretrained.cache_dir() == tmp_path / "xdg" / "chirpdetector"
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert pretrained.cache_dir() == tmp_path / "home" / ".cache" / "chirpdetector"


# --------------------------------------------------------------- downloads


def test_download_verifies_unpacks_and_reuses_the_cache(release: Release) -> None:
    model = release.publish(release.fake_bundle("a", b"weights a"), "fake-v1")
    assert not pretrained.is_cached("fake-v1")
    directory = pretrained.path("fake-v1")
    assert directory == release.cache / "fake-v1"
    assert (directory / "weights.pt").read_bytes() == b"weights a"
    assert release.server.requests == [model.asset]
    # cached: aliases resolve to it, nothing is downloaded again
    assert pretrained.path("detector") == directory
    assert pretrained.download("fake-v1") == directory
    assert pretrained.is_cached("fake-v1")
    assert release.server.requests == [model.asset]
    # no temporary files are left; one lock file per model
    assert _contents(release.cache) == [".fake-v1.lock", "fake-v1"]
    # force: downloaded and installed again
    assert pretrained.download("fake-v1", force=True) == directory
    assert release.server.requests == [model.asset] * 2
    assert pretrained.is_cached("fake-v1")
    assert _contents(release.cache) == [".fake-v1.lock", "fake-v1"]


def test_a_download_with_another_checksum_is_rejected(release: Release) -> None:
    model = release.publish(release.fake_bundle("a"), "fake-v1")
    archive = release.assets / model.asset
    data = bytearray(archive.read_bytes())
    data[-1] ^= 0xFF  # same size, other content
    archive.write_bytes(bytes(data))
    with pytest.raises(PretrainedError, match=r"has sha256 .* expected"):
        pretrained.path("fake-v1")
    assert _contents(release.cache) == [".fake-v1.lock"]


@pytest.mark.parametrize(("delta", "match"), [(1, "only"), (-1, "more than")])
def test_a_download_of_another_size_is_rejected(
    release: Release, delta: int, match: str
) -> None:
    model = release.publish(release.fake_bundle("a"), "fake-v1", install=False)
    release.install(replace(model, size=model.size + delta))
    with pytest.raises(PretrainedError, match=rf"sent {match} .* bytes, expected"):
        pretrained.path("fake-v1")
    assert _contents(release.cache) == [".fake-v1.lock"]


def test_the_weights_are_checked_after_unpacking(release: Release) -> None:
    model = release.publish(release.fake_bundle("a"), "fake-v1", install=False)
    release.install(replace(model, weights_sha256="0" * 64))
    with pytest.raises(PretrainedError, match=r"names weights .*, not 0000"):
        pretrained.path("fake-v1")
    assert _contents(release.cache) == [".fake-v1.lock"]


def test_a_zip_with_another_layout_is_rejected(release: Release) -> None:
    model = release.publish(release.fake_bundle("a"), "fake-v1", install=False)
    release.install(replace(model, directory="something-else"))
    with pytest.raises(PretrainedError, match="must hold only the directory"):
        pretrained.path("fake-v1")
    assert _contents(release.cache) == [".fake-v1.lock"]


def test_offline_mode_never_downloads(
    release: Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = release.publish(release.fake_bundle("a"), "fake-v1")
    monkeypatch.setenv(pretrained.ENV_OFFLINE, "1")
    with pytest.raises(PretrainedError) as info:
        pretrained.path("detector")
    text = str(info.value)
    assert "is not in the cache" in text
    assert "CHIRPDETECTOR_OFFLINE is set" in text
    assert "chirpdetector models download fake-v1" in text
    assert str(release.cache / "fake-v1") in text
    assert model.url in text
    assert release.server.requests == []
    # once cached, offline use works
    monkeypatch.delenv(pretrained.ENV_OFFLINE)
    directory = pretrained.path("fake-v1")
    monkeypatch.setenv(pretrained.ENV_OFFLINE, "true")
    assert pretrained.path("fake-v1") == directory
    assert release.server.requests == [model.asset]


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_network_failures_say_how_to_get_the_model(
    release: Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    release.publish(release.fake_bundle("a"), "fake-v1")
    monkeypatch.setenv(pretrained.ENV_MODELS_URL, f"http://127.0.0.1:{_closed_port()}")
    with pytest.raises(PretrainedError) as refused:
        pretrained.path("fake-v1")
    monkeypatch.setenv(pretrained.ENV_MODELS_URL, f"{release.server.url}/elsewhere")
    with pytest.raises(PretrainedError, match="HTTP Error 404") as missing:
        pretrained.path("fake-v1")
    for err in (refused, missing):
        text = str(err.value)
        assert "is not in the cache" in text
        assert "failed" in text
        assert "chirpdetector models download fake-v1" in text
    assert _contents(release.cache) == [".fake-v1.lock"]


def test_a_damaged_cache_is_detected_and_downloaded_again(
    release: Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = release.publish(release.fake_bundle("a", b"good weights"), "fake-v1")
    directory = pretrained.path("fake-v1")
    (directory / "weights.pt").write_bytes(b"bad weights")
    assert not pretrained.is_cached("fake-v1")
    assert pretrained.path("fake-v1") == directory
    assert (directory / "weights.pt").read_bytes() == b"good weights"
    assert release.server.requests == [model.asset] * 2
    # offline, a damaged copy is reported, not used
    (directory / "bundle.json").write_text("{")
    monkeypatch.setenv(pretrained.ENV_OFFLINE, "1")
    with pytest.raises(PretrainedError, match=r"damaged in the cache .*bundle.json"):
        pretrained.path("fake-v1")


def test_concurrent_processes_download_once(release: Release) -> None:
    weights = os.urandom(1 << 20)
    model = release.publish(release.fake_bundle("a", weights), "fake-v1")
    release.server.chunk_delay_s = 0.02  # 16 chunks: about 0.3 s per download
    script = textwrap.dedent(
        """
        import json, sys
        from chirpdetector import pretrained
        model = pretrained.PretrainedModel(**json.loads(sys.argv[1]))
        pretrained.REGISTRY[model.name] = model
        print(pretrained.path(model.name))
        """
    )
    src = str(Path(chirpdetector.__file__).parents[1])
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]),
    }
    release.cache.mkdir(parents=True)
    # hold the lock until every process waits for it, then let them race
    with (release.cache / ".fake-v1.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", script, json.dumps(asdict(model))],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
            )
            for _ in range(3)
        ]
        errors: list[list[str]] = [[] for _ in procs]

        def drain(k: int) -> None:
            stream = procs[k].stderr
            assert stream is not None
            for line in stream:
                errors[k].append(line)

        readers = [threading.Thread(target=drain, args=(k,)) for k in range(3)]
        for reader in readers:
            reader.start()
        deadline = time.monotonic() + 60
        while not all(any("waiting" in line for line in e) for e in errors):
            assert time.monotonic() < deadline, errors
            assert all(p.poll() is None for p in procs), errors
            time.sleep(0.05)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    outputs = []
    for proc in procs:
        assert proc.stdout is not None
        outputs.append(proc.stdout.read().strip())
        proc.wait(timeout=60)
    for reader in readers:
        reader.join(timeout=10)
    assert [p.returncode for p in procs] == [0, 0, 0], errors
    assert outputs == [str(release.cache / "fake-v1")] * 3
    assert release.server.requests == [model.asset]
    assert (release.cache / "fake-v1" / "weights.pt").read_bytes() == weights
    assert _contents(release.cache) == [".fake-v1.lock", "fake-v1"]
