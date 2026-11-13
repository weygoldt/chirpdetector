"""A local stand-in for the model release: fake bundles served over HTTP (stdlib)."""

from __future__ import annotations

import hashlib
import http.server
import json
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from chirpdetector import pretrained
from chirpdetector.pretrained import Kind, PretrainedModel


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ModelServer:
    """Serves the files of a directory; records every GET (asset name)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.requests: list[str] = []
        self.chunk_delay_s = 0.0
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                name = self.path.lstrip("/")
                server.requests.append(name)
                file = server.root / name
                if not file.is_file():
                    self.send_error(404)
                    return
                data = file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                for start in range(0, len(data), 1 << 16):
                    self.wfile.write(data[start : start + (1 << 16)])
                    if server.chunk_delay_s:
                        time.sleep(server.chunk_delay_s)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@dataclass
class Release:
    """Assets directory, its server and the cache the tests download into."""

    assets: Path
    cache: Path
    server: ModelServer
    monkeypatch: pytest.MonkeyPatch
    published: list[PretrainedModel] = field(default_factory=list)

    def fake_bundle(self, name: str, weights: bytes = b"fake weights") -> Path:
        """A directory with ``bundle.json`` and ``weights.pt`` (not loadable)."""
        directory = self.assets.parent / "bundles" / name
        directory.mkdir(parents=True)
        (directory / "weights.pt").write_bytes(weights)
        record = {"weights_file": "weights.pt", "weights_sha256": sha256_bytes(weights)}
        (directory / "bundle.json").write_text(json.dumps(record))
        return directory

    def publish(
        self,
        bundle: Path,
        name: str,
        kind: Kind = "detector",
        *,
        detector: str | None = None,
        install: bool = True,
    ) -> PretrainedModel:
        """Zip a bundle directory as an asset and register it (default of its kind)."""
        directory = f"chirpdetector-{name}"
        asset = f"{directory}.zip"
        with zipfile.ZipFile(self.assets / asset, "w") as zf:
            for file in sorted(bundle.rglob("*")):
                zf.write(file, f"{directory}/{file.relative_to(bundle).as_posix()}")
        data = (self.assets / asset).read_bytes()
        record = json.loads((bundle / "bundle.json").read_text())
        model = PretrainedModel(
            name=name,
            kind=kind,
            description=f"test {kind}",
            asset=asset,
            size=len(data),
            sha256=sha256_bytes(data),
            directory=directory,
            weights_sha256=record["weights_sha256"],
            detector=detector,
        )
        if install:
            self.published.append(model)
            self.install(*self.published)
        return model

    def install(self, *models: PretrainedModel) -> None:
        """Make these the registry; the last of each kind is its default."""
        self.monkeypatch.setattr(pretrained, "REGISTRY", {m.name: m for m in models})
        self.monkeypatch.setattr(
            pretrained, "ALIASES", {m.kind: m.name for m in models}
        )
