"""train_detector: folds by group, fresh fits, run directory, CLI (stub backend)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from chirpdetector.bundle import load_bundle
from chirpdetector.cli import app
from chirpdetector.data import Dataset
from chirpdetector.training import DetectorTrainingConfig, train_detector

from .conftest import STUB, StubBackend


def _config(**cv: object) -> DetectorTrainingConfig:
    return DetectorTrainingConfig.from_dict(
        {"model": {"type": STUB}, "device": "cpu", "cv": {"n_folds": 3, **cv}}
    )


def test_cv_and_final_fit(
    tmp_path: Path, dataset: Dataset, dataset_dir: Path, stub_backend: StubBackend
) -> None:
    result = train_detector(dataset_dir, _config(), tmp_path / "run")
    manifest = dataset.manifest
    train_groups = set(manifest.groups("train"))
    assert [f.name for f in result.fits] == ["fold-0", "fold-1", "fold-2", "final"]
    assert len(stub_backend.fits) == 4  # one fresh fit per fold, plus the final one
    validated: list[str] = []
    for fit in result.fits[:3]:
        assert not set(fit.train_groups) & set(fit.val_groups)
        assert set(fit.train_groups) | set(fit.val_groups) == train_groups
        validated += fit.val_groups
    assert sorted(validated) == sorted(train_groups)
    for request in stub_backend.fits:
        groups_of = {manifest.image(i).group for i in request.train_ids}
        val_of = {manifest.image(i).group for i in request.val_ids}
        assert not groups_of & val_of
        assert "test" not in {manifest.splits[g] for g in groups_of | val_of}
    final = result.fits[-1]
    assert set(final.train_groups) == train_groups
    assert set(final.val_groups) == set(manifest.groups("val"))
    # every fit has its own weights and bundle
    assert len({f.weights_sha256 for f in result.fits}) == 4
    assert result.bundle == final.bundle
    bundle = load_bundle(final.bundle)
    assert bundle.training["dataset_sha256"] == manifest.content_sha256
    assert bundle.training["fit"] == "final"
    assert bundle.preprocessing == manifest.preprocessing
    assert set(bundle.training["code"]) == {"chirpdetector", "gridtools"}
    assert result.cv["n_folds"] == 3


def test_run_directory_contents(
    tmp_path: Path, dataset: Dataset, dataset_dir: Path, stub_backend: StubBackend
) -> None:
    config_path = tmp_path / "train.toml"
    config_path.write_text(
        f'device = "cpu"\n[model]\ntype = "{STUB}"\n[cv]\nn_folds = 2\n'
    )
    run = tmp_path / "run"
    train_detector(dataset_dir, config_path, run)
    for name in (
        "run.json",
        "config.json",
        "config.toml",
        "provenance.json",
        "dataset.json",
        "folds.json",
    ):
        assert (run / name).is_file(), name
    provenance = json.loads((run / "provenance.json").read_text())
    for key in ("code", "uv_lock", "versions", "hardware"):
        assert key in provenance
    assert provenance["code"]["chirpdetector"]["sha"] is not None
    assert "dirty" in provenance["code"]["chirpdetector"]
    assert provenance["uv_lock"]["sha256"]
    assert provenance["dataset"]["content_sha256"] == dataset.manifest.content_sha256
    assert "torch" in provenance["versions"]
    assert "ultralytics" in provenance["versions"]
    folds = json.loads((run / "folds.json").read_text())
    assert set(folds) == set(dataset.manifest.groups("train"))
    for fit in ("fold-0", "fold-1", "final"):
        for name in ("fit.json", "provenance.json", "curves.csv", "bundle/bundle.json"):
            assert (run / fit / name).is_file(), f"{fit}/{name}"


def test_single_folds_share_a_run_dir(
    tmp_path: Path, dataset_dir: Path, stub_backend: StubBackend
) -> None:
    run = tmp_path / "run"
    first = train_detector(dataset_dir, _config(), run, fold=1)
    assert [f.name for f in first.fits] == ["fold-1"]
    both = train_detector(dataset_dir, _config(), run, fold="final")
    assert [f.name for f in both.fits] == ["fold-1", "final"]
    assert both.run_id == first.run_id
    with pytest.raises(FileExistsError, match="already trained"):
        train_detector(dataset_dir, _config(), run, fold=1)
    with pytest.raises(ValueError, match="different training config"):
        train_detector(dataset_dir, _config(final_fit=False), run, fold=0)
    with pytest.raises(ValueError, match="fold must be"):
        train_detector(dataset_dir, _config(), tmp_path / "other", fold=3)


def test_final_fit_needs_val_groups(tmp_path: Path, stub_backend: StubBackend) -> None:
    from chirpdetector.data.synthetic import make_synthetic_dataset

    no_val = tmp_path / "no-val"
    make_synthetic_dataset(no_val, splits=("train", "train", "test"))
    with pytest.raises(ValueError, match="needs 'val' groups"):
        train_detector(no_val, _config(n_folds=0), tmp_path / "run")


def test_train_cli(
    tmp_path: Path, dataset_dir: Path, stub_backend: StubBackend
) -> None:
    config = tmp_path / "train.toml"
    config.write_text(f'device = "cpu"\n[model]\ntype = "{STUB}"\n[cv]\nn_folds = 0\n')
    run = tmp_path / "run"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "train",
            "detector",
            str(config),
            "--dataset",
            str(dataset_dir),
            "--run-dir",
            str(run),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "bundle:" in result.output
    assert (run / "final" / "bundle" / "bundle.json").is_file()
    result = runner.invoke(
        app,
        [
            "train",
            "detector",
            str(config),
            "--dataset",
            str(dataset_dir),
            "--run-dir",
            str(tmp_path / "r2"),
            "--fold",
            "x",
        ],
    )
    assert result.exit_code != 0
    result = runner.invoke(app, ["dataset", "validate", str(dataset_dir)])
    assert result.exit_code == 0, result.output
    assert "valid: 24 images" in result.output
