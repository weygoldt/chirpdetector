"""What a run was made with: code, lock file, hardware, versions."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from chirpdetector.data.contract import sha256_file

__all__ = ["code_state", "collect", "git_state", "hardware", "uv_lock", "versions"]

_PACKAGES = (
    "chirpdetector",
    "gridtools",
    "ultralytics",
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "scipy",
    "opencv-python",
    "pyarrow",
)


def _git(directory: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_state(directory: str | Path) -> dict[str, Any]:
    """Commit and dirty flag of the git checkout containing ``directory``.

    Returns
    -------
    dict
        ``sha`` (None outside a git checkout), ``dirty`` (tracked files
        modified; None if unknown) and ``path``.
    """
    directory = Path(directory)
    sha = _git(directory, "rev-parse", "HEAD")
    status = (
        _git(directory, "status", "--porcelain", "--untracked-files=no")
        if sha
        else None
    )
    return {
        "path": str(directory),
        "sha": sha,
        "dirty": None if status is None else bool(status),
    }


def code_state() -> dict[str, dict[str, Any]]:
    """`git_state` of the chirpdetector and gridtools sources in use."""
    import gridtools

    import chirpdetector

    out = {}
    for name, module in (("chirpdetector", chirpdetector), ("gridtools", gridtools)):
        files = getattr(module, "__file__", None)
        out[name] = (
            git_state(Path(files).parent)
            if files
            else {"path": None, "sha": None, "dirty": None}
        )
    return out


def uv_lock() -> dict[str, Any] | None:
    """Path and sha256 of the ``uv.lock`` nearest above the chirpdetector sources."""
    import chirpdetector

    start = Path(chirpdetector.__file__).resolve().parent
    for directory in (start, *start.parents):
        candidate = directory / "uv.lock"
        if candidate.is_file():
            return {"path": str(candidate), "sha256": sha256_file(candidate)}
    return None


def hardware() -> dict[str, Any]:
    """CPU, memory and GPU of this machine."""
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    info: dict[str, Any] = {
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpus": [],
    }
    try:
        import psutil

        info["memory_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:  # pragma: no cover
        info["memory_gb"] = None
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            info["gpus"].append(
                {
                    "index": index,
                    "name": props.name,
                    "memory_gb": round(props.total_memory / 2**30, 1),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    return info


def versions() -> dict[str, str]:
    """Python and package versions."""
    out = {"python": platform.python_version()}
    for name in _PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
    return out


def collect(**extra: Any) -> dict[str, Any]:
    """All provenance of a run, plus ``extra`` entries."""
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": list(sys.argv),
        "code": code_state(),
        "uv_lock": uv_lock(),
        "versions": versions(),
        "hardware": hardware(),
        "environment": {
            key: os.environ.get(key)
            for key in ("CHIRP_MAX_VRAM_FRACTION", "CUDA_VISIBLE_DEVICES")
        },
        **extra,
    }
