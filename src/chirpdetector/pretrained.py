"""Pretrained models: downloaded on first use, cached per user.

The published models are assets of a GitHub release of chirpdetector
(`RELEASE`). Each is a zip holding one bundle directory
(`chirpdetector.bundle` for the detector, `chirpdetector.params.bundle`
for the chirp-parameter decoder). `REGISTRY` lists them under stable,
versioned names; `ALIASES` maps ``"detector"`` and ``"decoder"`` to the
default ones::

    from chirpdetector import pretrained

    pretrained.list_models()            # what is published
    pretrained.path("detector")         # bundle directory, downloaded if missing
    pretrained.download("decoder-dualhead-sim_v4")

The loaders accept references to them: ``load_bundle("default")`` is the
default detector, ``load_params_bundle("default")`` (and so
``predict_span(..., params="default")``) the default decoder, and
``"pretrained:NAME"`` names any registry model (name or alias). Only the
*string* ``"default"`` is a reference; ``Path("default")`` is a directory.

Cache
-----
``$CHIRPDETECTOR_CACHE``, else ``$XDG_CACHE_HOME/chirpdetector``, else
``~/.cache/chirpdetector``; one bundle directory per model name
(``CACHE/dualhead-rtdetrv2-r18-sim_v4/bundle.json``, ...). A download goes
to a temporary file in the cache, its size and sha256 are checked against
the registry, it is unpacked into a temporary directory, the bundle's
weights are checked against the registry's hash, and the directory is
renamed into place (atomic). A lock file per model (``CACHE/.NAME.lock``)
makes concurrent processes wait for one download instead of repeating it;
without file locks (Windows) the atomic rename still keeps the cache
consistent. Every use re-checks the weights' sha256, so a damaged cache is
noticed: it is downloaded again (online) or reported (offline).

Environment
-----------
``CHIRPDETECTOR_CACHE``
    Cache directory (see above).
``CHIRPDETECTOR_OFFLINE``
    ``1`` (or ``true``, ``yes``, ``on``): never download. A model that is
    not in the cache raises `PretrainedError`, which says where to put it.
``CHIRPDETECTOR_MODELS_URL``
    Base URL of the assets instead of the GitHub release, for tests and
    mirrors: the asset ``A`` is fetched from ``$CHIRPDETECTOR_MODELS_URL/A``.

Downloads report progress on stderr. Nothing here touches the network
unless a model is missing from the cache; loading and running the models
never does (the dual-head weights file holds its full RT-DETRv2 config).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

__all__ = [
    "ALIASES",
    "DEFAULT",
    "ENV_CACHE",
    "ENV_MODELS_URL",
    "ENV_OFFLINE",
    "PREFIX",
    "REGISTRY",
    "RELEASE",
    "Kind",
    "PretrainedError",
    "PretrainedModel",
    "cache_dir",
    "download",
    "get_model",
    "is_cached",
    "is_offline",
    "list_models",
    "path",
    "reference",
    "resolve",
]

log = logging.getLogger(__name__)

RELEASE = "models-v1"
"""GitHub release that holds the assets of `REGISTRY`."""
RELEASES_URL = "https://github.com/weygoldt/chirpdetector/releases/download"
ENV_CACHE = "CHIRPDETECTOR_CACHE"
ENV_OFFLINE = "CHIRPDETECTOR_OFFLINE"
ENV_MODELS_URL = "CHIRPDETECTOR_MODELS_URL"
DEFAULT = "default"
"""Reference to the default model of a kind (`reference`)."""
PREFIX = "pretrained:"
"""Prefix of a reference to a registry model by name or alias."""

TIMEOUT_S = 60.0
"""Socket timeout of a download (connect and every read)."""
_CHUNK = 1 << 20
_TRUE = frozenset({"1", "true", "yes", "on"})

Kind = Literal["detector", "decoder"]


class PretrainedError(RuntimeError):
    """A pretrained model is not available, or its download is not the published file."""


@dataclass(frozen=True)
class PretrainedModel:
    """A published model.

    Attributes
    ----------
    name
        Stable, versioned name; also the bundle's directory in the cache.
    kind
        ``"detector"`` (a detector bundle) or ``"decoder"`` (a
        chirp-parameter bundle).
    description
        One line.
    asset
        File name of the zip in the release.
    size
        Size of the zip in bytes.
    sha256
        sha256 of the zip.
    directory
        The one top-level directory of the zip (the bundle).
    weights_sha256
        sha256 of the bundle's weights (what the loaders verify).
    release
        Release tag that holds the asset.
    detector
        For a decoder: the name of the detector whose boxes it was trained
        on (it is only valid with that detector).
    """

    name: str
    kind: Kind
    description: str
    asset: str
    size: int
    sha256: str
    directory: str
    weights_sha256: str
    release: str = RELEASE
    detector: str | None = None

    @property
    def url(self) -> str:
        """Where the asset is downloaded from (honours ``CHIRPDETECTOR_MODELS_URL``)."""
        base = os.environ.get(ENV_MODELS_URL, "").strip().rstrip("/")
        return f"{base or f'{RELEASES_URL}/{self.release}'}/{self.asset}"


_DETECTOR = PretrainedModel(
    name="dualhead-rtdetrv2-r18-sim_v4",
    kind="detector",
    description=(
        "dual-head RT-DETRv2-r18: chirp boxes and the emitter's EOD frequency "
        "(trained on sim_v4)"
    ),
    asset="chirpdetector-dualhead-rtdetrv2-r18-sim_v4.zip",
    size=77_619_777,
    sha256="e005d9362414bd6222eae069ba955acb2c2bd4aaa7d0a29caa26eac0c2370d64",
    directory="chirpdetector-dualhead-rtdetrv2-r18-sim_v4",
    weights_sha256="9fcdef3c4371bbaee0eca75f5614aaf254bb920e42b72896d32c1c11c9bc0754",
)
_DECODER = PretrainedModel(
    name="decoder-dualhead-sim_v4",
    kind="decoder",
    description=(
        "chirp-parameter decoder (height, FWHM, width at 10 %, undershoot) "
        "for the boxes of dualhead-rtdetrv2-r18-sim_v4"
    ),
    asset="chirpdetector-decoder-dualhead-sim_v4.zip",
    size=3_345_930,
    sha256="62b22a8846e4b1511190d0a791e11631ec0792d8c2529224460c8d397e9b263b",
    directory="chirpdetector-decoder-dualhead-sim_v4",
    weights_sha256="c0801c565a4b4803075d825e3cf29653e0176db07f243fa54a7cbde996e9414c",
    detector=_DETECTOR.name,
)

REGISTRY: dict[str, PretrainedModel] = {m.name: m for m in (_DETECTOR, _DECODER)}
"""Published models by name."""
ALIASES: dict[str, str] = {"detector": _DETECTOR.name, "decoder": _DECODER.name}
"""The default model of each kind; ``"default"`` refers to these."""


# ---------------------------------------------------------------- registry


def get_model(name: str) -> PretrainedModel:
    """The registry entry of a name or alias.

    Raises
    ------
    KeyError
        If no model has that name or alias.
    """
    model = REGISTRY.get(ALIASES.get(name, name))
    if model is None:
        known = sorted({*REGISTRY, *ALIASES})
        msg = f"unknown pretrained model {name!r}; known: {known}"
        raise KeyError(msg)
    return model


def list_models() -> list[PretrainedModel]:
    """Every published model."""
    return list(REGISTRY.values())


def reference(value: object, kind: Kind) -> str | None:
    """The model name a reference stands for, or None if ``value`` is no reference.

    Parameters
    ----------
    value
        ``"default"`` (the default model of ``kind``), ``"pretrained:NAME"``
        (a registry name or alias), or anything else (a path: None).
    kind
        What the caller loads.

    Raises
    ------
    KeyError
        If ``NAME`` is not in the registry.
    ValueError
        If ``NAME`` is a model of another kind.
    """
    if not isinstance(value, str):
        return None
    if value == DEFAULT:
        return get_model(kind).name
    if not value.startswith(PREFIX):
        return None
    model = get_model(value.removeprefix(PREFIX))
    if model.kind != kind:
        msg = f"{value!r} is a {model.kind}, not a {kind}"
        raise ValueError(msg)
    return model.name


def resolve(value: object, kind: Kind) -> Path | None:
    """The bundle directory of a reference (downloaded if missing), or None.

    See `reference` for what counts as a reference.
    """
    name = reference(value, kind)
    return None if name is None else path(name)


# ------------------------------------------------------------------- cache


def cache_dir() -> Path:
    """``$CHIRPDETECTOR_CACHE``, ``$XDG_CACHE_HOME/chirpdetector`` or ``~/.cache/chirpdetector``."""
    explicit = os.environ.get(ENV_CACHE, "").strip()
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg and Path(xdg).is_absolute():
        return Path(xdg) / "chirpdetector"
    return Path.home() / ".cache" / "chirpdetector"


def is_offline() -> bool:
    """Whether ``CHIRPDETECTOR_OFFLINE`` forbids downloads."""
    return os.environ.get(ENV_OFFLINE, "").strip().lower() in _TRUE


def is_cached(name: str) -> bool:
    """Whether a model is in the cache and intact (no download)."""
    model = get_model(name)
    return _problem(model, cache_dir() / model.name) is None


def path(name: str) -> Path:
    """The bundle directory of a model, downloaded first if it is missing or damaged.

    Parameters
    ----------
    name
        Registry name or alias (`get_model`).

    Raises
    ------
    PretrainedError
        If the model is not in the cache and cannot be downloaded (offline,
        network failure), or the download is not the published file.
    """
    model = get_model(name)
    target = cache_dir() / model.name
    problem = _problem(model, target)
    if problem is None:
        return target
    return _fetch(model, problem=problem)


def download(name: str, *, force: bool = False) -> Path:
    """Download a model into the cache (if missing, damaged, or ``force``).

    Returns
    -------
    Path
        Its bundle directory.

    Raises
    ------
    PretrainedError
        As `path`.
    """
    if not force:
        return path(name)
    return _fetch(get_model(name), problem=_FORCED, force=True)


# -------------------------------------------------------------- internals


def _sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


_MISSING = "not downloaded"
_FORCED = "download forced"


def _problem(model: PretrainedModel, directory: Path) -> str | None:
    """What is wrong with a model's bundle directory (None: intact)."""
    record = directory / "bundle.json"
    if not directory.exists():
        return _MISSING
    try:
        data = json.loads(record.read_text())
        recorded = data["weights_sha256"]
        weights = directory / str(data["weights_file"])
    except (OSError, ValueError, KeyError, TypeError) as err:
        return f"unreadable bundle.json ({err})"
    if recorded != model.weights_sha256:
        return f"bundle.json names weights {recorded}, not {model.weights_sha256}"
    if not weights.is_file():
        return f"{weights.name} is missing"
    actual = _sha256(weights)
    if actual != model.weights_sha256:
        return f"{weights.name} has sha256 {actual}, not {model.weights_sha256}"
    return None


def _unavailable(model: PretrainedModel, target: Path, problem: str, why: str) -> str:
    """The error message for a model that is needed but cannot be fetched."""
    if problem == _MISSING:
        state = "is not in the cache"
    elif problem == _FORCED:
        state = "was to be downloaded again"
    else:
        state = f"is damaged in the cache ({problem})"
    return (
        f"pretrained model {model.name!r} {state} ({target}) and {why}. "
        f"Download it where there is network access with `chirpdetector models "
        f"download {model.name}` and copy {target} to the same place here (or set "
        f"{ENV_CACHE}); or download {model.url}, check its sha256 "
        f"({model.sha256}) and unzip it into {target.parent}, renaming "
        f"{model.directory}/ to {model.name}/."
    )


def _fetch(model: PretrainedModel, *, problem: str, force: bool = False) -> Path:
    """Download, verify and install a model (module docstring)."""
    root = cache_dir()
    target = root / model.name
    if is_offline():
        raise PretrainedError(
            _unavailable(model, target, problem, f"{ENV_OFFLINE} is set")
        )
    root.mkdir(parents=True, exist_ok=True)
    with _lock(root / f".{model.name}.lock", model.name):
        if not force:
            current = _problem(model, target)
            if current is None:  # another process installed it meanwhile
                return target
            if current != _MISSING:
                log.warning("%s: %s; downloading it again", target, current)
            problem = current
        with tempfile.TemporaryDirectory(dir=root, prefix=f".{model.name}-") as tmp:
            work = Path(tmp)
            archive = work / model.asset
            _download(model, archive, target, problem)
            bundle = _extract(model, archive, work / "unpacked")
            issue = _problem(model, bundle)
            if issue is not None:
                msg = f"{model.asset} from {model.url}: {issue}; nothing was installed"
                raise PretrainedError(msg)
            _install(model, bundle, target, work / "old", replace=force)
    return target


@contextlib.contextmanager
def _lock(file: Path, name: str) -> Iterator[None]:
    """An exclusive lock on ``file`` (a no-op without `fcntl`)."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows: rely on the atomic rename
        yield
        return
    with file.open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _say(f"chirpdetector: waiting for another process to download {name}")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _mb(n: float) -> str:
    return f"{n / 1e6:.1f} MB"


class _Progress:
    """Download progress on stderr: a bar on a terminal, else start and end lines."""

    def __init__(self, label: str, total: int) -> None:
        self.label, self.total = label, total
        self.tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self.last = 0.0

    def update(self, done: int) -> None:
        now = time.monotonic()
        if not self.tty or (now - self.last < 0.1 and done < self.total):
            return
        self.last = now
        fraction = min(1.0, done / self.total) if self.total else 1.0
        bar = "#" * int(30 * fraction)
        sys.stderr.write(
            f"\r  {self.label} [{bar:<30}] {_mb(done)} / {_mb(self.total)} "
            f"{100 * fraction:3.0f}%"
        )
        sys.stderr.flush()

    def close(self) -> None:
        if self.tty:
            sys.stderr.write("\n")
            sys.stderr.flush()


def _download(
    model: PretrainedModel, archive: Path, target: Path, problem: str
) -> None:
    """Fetch the asset into ``archive``; check its size and sha256."""
    from chirpdetector import __version__

    url = model.url
    _say(f"chirpdetector: downloading {model.name} ({_mb(model.size)}) from {url}")
    request = urllib.request.Request(
        url, headers={"User-Agent": f"chirpdetector/{__version__}"}
    )
    digest = hashlib.sha256()
    size = 0
    progress = _Progress(model.name, model.size)
    try:
        with (
            urllib.request.urlopen(request, timeout=TIMEOUT_S) as response,
            archive.open("wb") as out,
        ):
            while chunk := response.read(_CHUNK):
                size += len(chunk)
                if size > model.size:
                    break
                digest.update(chunk)
                out.write(chunk)
                progress.update(size)
    except (OSError, ValueError) as err:  # URLError, HTTPError, timeouts, bad URLs
        progress.close()
        reason = (
            err
            if isinstance(err, urllib.error.HTTPError)
            else getattr(err, "reason", None) or err
        )
        why = f"downloading {url} failed: {reason}"
        raise PretrainedError(_unavailable(model, target, problem, why)) from err
    progress.close()
    if size != model.size:
        more = "more than" if size > model.size else "only"
        msg = (
            f"{url} sent {more} {size:,} bytes, expected {model.size:,}: not the "
            f"published {model.asset}; nothing was installed"
        )
        raise PretrainedError(msg)
    actual = digest.hexdigest()
    if actual != model.sha256:
        msg = (
            f"{url} has sha256 {actual}, expected {model.sha256}: not the "
            f"published {model.asset}; nothing was installed"
        )
        raise PretrainedError(msg)
    _say(f"chirpdetector: {model.name} verified (sha256 {actual[:12]})")


def _extract(model: PretrainedModel, archive: Path, into: Path) -> Path:
    """Unpack the verified zip; it must hold exactly ``model.directory``."""
    prefix = f"{model.directory}/"
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        foreign = [
            n
            for n in names
            if not n.startswith(prefix) or ".." in n.split("/") or n.startswith("/")
        ]
        if foreign or not names:
            msg = (
                f"{model.asset} must hold only the directory {prefix}, found "
                f"{foreign[:3] or 'nothing'}"
            )
            raise PretrainedError(msg)
        zf.extractall(into)
    return into / model.directory


def _install(
    model: PretrainedModel, bundle: Path, target: Path, old: Path, *, replace: bool
) -> None:
    """Move ``bundle`` to ``target`` (atomic rename).

    A damaged ``target`` (or any, with ``replace``) is first moved aside
    into ``old``, which the caller's temporary directory removes.
    """
    if target.exists() and (replace or _problem(model, target) is not None):
        target.rename(old)
    try:
        bundle.rename(target)
    except OSError:
        # without a lock (no fcntl), another process may have won the race
        if _problem(model, target) is not None:
            raise
        log.debug("%s was installed by another process", target)
