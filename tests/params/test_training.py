"""CPU smoke tests: training, bundle, evaluation, predict_span columns, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gridtools.io import open_recording
from typer.testing import CliRunner

from chirpdetector.bundle import Bundle, BundleError
from chirpdetector.cli import app
from chirpdetector.inference import predict_span
from chirpdetector.params import PARAM_COLUMNS, ParamsEstimator, load_params_bundle
from chirpdetector.params.bundle import ParamsBundle
from chirpdetector.params.evaluation import agreement_summary
from chirpdetector.params.samples import discover_scenes
from chirpdetector.params.training import ParamsTrainingConfig, train_params

from .conftest import TINY_CONFIG, footprint_detections

runner = CliRunner()


def _config(**changes: object) -> ParamsTrainingConfig:
    data = json.loads(json.dumps(TINY_CONFIG))
    data["scene_background"] = "any"  # the test scenes have no real background
    data.update(changes)
    return ParamsTrainingConfig.from_dict(data)


@pytest.fixture(scope="session")
def footprint_run(
    params_scenes: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """A decoder trained on jittered footprint boxes (no detector)."""
    run = tmp_path_factory.mktemp("footprint_run")
    config = _config(boxes={"mode": "footprint"})
    train_params(config, [params_scenes], run)
    return run


@pytest.fixture(scope="session")
def detector_run(
    params_scenes: Path,
    stub_detector: Bundle,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """A decoder trained on the (stand-in) detector's boxes."""
    import chirpdetector.inference as inference

    run = tmp_path_factory.mktemp("detector_run")
    refs = discover_scenes([params_scenes])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            inference,
            "detect_in_spectrogram",
            footprint_detections({r.source_id: r.directory for r in refs}),
        )
        train_params(_config(), [params_scenes], run, detector=stub_detector)
    return run


def test_config_round_trips_and_rejects_unknown_keys() -> None:
    config = _config()
    assert ParamsTrainingConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="unknown"):
        ParamsTrainingConfig.from_dict({"optim": {"epoch": 3}})
    with pytest.raises(ValueError, match="scene_background"):
        ParamsTrainingConfig(scene_background="real")
    assert ParamsTrainingConfig().scene_background == "hybrid"


def test_training_writes_a_loadable_bundle(footprint_run: Path) -> None:
    run = json.loads((footprint_run / "run.json").read_text())
    bundle = load_params_bundle(run["bundle"])
    assert bundle.model_type == "chirp-params"
    assert bundle.detector is None
    assert bundle.crop.size == (32, 32)
    assert set(bundle.calibration.maps) == {
        "height_hz",
        "fwhm_s",
        "width10_s",
        "undershoot_hz",
    }
    assert set(bundle.domain["sigma_threshold"]) == set(bundle.calibration.maps)
    assert bundle.training["n_train"] > 0
    assert (footprint_run / "curves.csv").is_file()
    assert (footprint_run / "val_predictions.parquet").is_file()
    # a new run in the same directory is refused
    with pytest.raises(FileExistsError):
        train_params(_config(boxes={"mode": "footprint"}), [], footprint_run)
    # tampered weights are refused
    copy = footprint_run.parent / "tampered"
    import shutil

    shutil.copytree(bundle.path, copy)
    (copy / "weights.pt").write_bytes(b"x")
    with pytest.raises(BundleError, match="sha256"):
        load_params_bundle(copy)


def test_estimator_gives_calibrated_columns(footprint_run: Path) -> None:
    from chirpdetector.params.samples import load_samples

    bundle = load_params_bundle(footprint_run / "bundle")
    samples = load_samples(footprint_run / "samples")
    rows = samples.select("test", usable=True)
    estimator = ParamsEstimator(bundle, device="cpu")
    out = estimator.estimate_inputs(samples.inputs(rows))
    assert list(out.columns) == list(PARAM_COLUMNS)
    assert len(out) == len(rows)
    for name in ("height_hz", "fwhm_s", "width10_s"):
        assert (out[name].to_numpy(float) > 0).all()
    assert (out["undershoot_hz"].to_numpy(float) >= 0).all()
    assert (out["height_sd_hz"].to_numpy(float) > 0).all()
    assert out["params_ok"].dtype == bool
    # band flag: only boxes that reach the band's edge, not crop margins
    edge = estimator.at_band_edge(
        [[1.0, 1.1, 600.0, 700.0], [1.0, 1.1, 1300.0, 1450.0], [1.0, 1.1, 1400, 1500.0]]
    )
    np.testing.assert_array_equal(edge, [False, False, True])
    assert set(",".join(out["params_flag"]).split(",")) <= {
        "",
        "edge",
        "band",
        "size",
        "uncertain",
    }


def test_detector_run_records_the_detector(
    detector_run: Path, stub_detector: Bundle
) -> None:
    bundle = load_params_bundle(detector_run / "bundle")
    assert bundle.detector is not None
    assert bundle.detector["weights_sha256"] == stub_detector.weights_sha256
    assert bundle.spectrogram == stub_detector.spectrogram
    assert dict(bundle.preprocessing) == dict(stub_detector.preprocessing)


def test_predict_span_adds_parameter_columns(
    detector_run: Path, stub_detector: Bundle, params_scenes: Path
) -> None:
    recording = open_recording(params_scenes / "test" / "001800" / "recording.wav")
    plain = predict_span(stub_detector, recording, 1.0, 7.0, name="s", device="cpu")
    params = load_params_bundle(detector_run / "bundle")
    out = predict_span(
        stub_detector, recording, 1.0, 7.0, name="s", device="cpu", params=params.path
    )
    pd.testing.assert_frame_equal(out.loc[:, list(plain.columns)], plain)
    assert list(out.columns[len(plain.columns) :]) == list(PARAM_COLUMNS)
    assert len(out) > 0
    assert np.isfinite(out["height_hz"]).all()
    # no detections: the columns are there, the table is empty
    none = predict_span(
        stub_detector,
        recording,
        1.0,
        7.0,
        name="s",
        device="cpu",
        params=params.path,
        score_threshold=0.99,
    )
    assert len(none) == 0
    assert set(PARAM_COLUMNS) <= set(none.columns)


def test_predict_span_refuses_another_detectors_decoder(
    footprint_run: Path,
    detector_run: Path,
    stub_detector: Bundle,
    params_scenes: Path,
    tmp_path: Path,
) -> None:
    import shutil

    recording = open_recording(params_scenes / "test" / "001800" / "recording.wav")
    # same front end, but trained on another detector's boxes
    other = tmp_path / "other"
    shutil.copytree(detector_run / "bundle", other)
    record = json.loads((other / "bundle.json").read_text())
    record["detector"]["weights_sha256"] = "0" * 64
    (other / "bundle.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="only valid with that detector"):
        predict_span(
            stub_detector, recording, 1.0, 4.0, name="s", device="cpu", params=other
        )
    out = predict_span(
        stub_detector,
        recording,
        1.0,
        4.0,
        name="s",
        device="cpu",
        params=other,
        allow_detector_mismatch=True,
    )
    assert "height_hz" in out.columns
    # another preprocessing (click removal) is refused even with the override
    decoder = load_params_bundle(footprint_run / "bundle")
    with pytest.raises(ValueError, match="preprocessing"):
        predict_span(
            stub_detector,
            recording,
            1.0,
            4.0,
            name="s",
            device="cpu",
            params=decoder.path,
            allow_detector_mismatch=True,
        )


def test_evaluation_writes_report_and_metrics(
    detector_run: Path, params_scenes: Path, tmp_path: Path
) -> None:
    from chirpdetector.params.evaluation import run_evaluation

    bundle_dir = detector_run / "bundle"
    report = run_evaluation(
        bundle_dir, [params_scenes], out=tmp_path / "eval", device="cpu"
    )
    text = report.read_text()
    for heading in (
        "## Accuracy before and after calibration",
        "## Uncertainty calibration",
        "## Distribution fidelity",
        "## Cluster structure",
    ):
        assert heading in text
    assert "## Error vs the chirp's own parameters" in text
    assert "**Template reuse**" in text
    metrics = json.loads((tmp_path / "eval" / "metrics.json").read_text())
    assert metrics["split"] == "test"
    assert set(metrics["accuracy"]) == {"raw", "cal"}
    assert metrics["natural_prior"]["pool_size"] > 0
    # the bundle records its training templates; test chirps are checked against them
    bundle = load_params_bundle(bundle_dir)
    assert len(bundle.training["train_template_ids"]) > 0
    reuse = metrics["template_reuse"]
    assert reuse["known"]
    assert 0 <= reuse["fraction_template_seen"] <= reuse["fraction_group_seen"] <= 1
    assert (tmp_path / "eval" / "coupling_bins.csv").is_file()
    for name in ("fig_pred_vs_true.png", "fig_marginals.png", "fig_joint.png"):
        assert (tmp_path / "eval" / name).is_file()
    bundle: ParamsBundle = load_params_bundle(bundle_dir)  # metrics attached
    assert bundle.metrics is not None
    assert bundle.metrics["weights_sha256"] == bundle.weights_sha256


def test_cli_inspect_and_help(footprint_run: Path) -> None:
    result = runner.invoke(app, ["bundle", "inspect", str(footprint_run / "bundle")])
    assert result.exit_code == 0, result.output
    assert "chirp-params" in result.output
    assert "detector    none" in result.output
    result = runner.invoke(app, ["train", "params", "--help"])
    assert result.exit_code == 0
    assert "--detector" in result.output
    result = runner.invoke(app, ["evaluate", "params", "--help"])
    assert result.exit_code == 0
    assert "--scenes" in result.output
    result = runner.invoke(app, ["evaluate", "--help"])
    assert "params" in result.output
    assert "detections" in result.output


def test_cli_trains(params_scenes: Path, tmp_path: Path) -> None:
    config = tmp_path / "params.toml"
    config.write_text(
        'device = "cpu"\nworkers = 0\nscene_background = "any"\n'
        '[boxes]\nmode = "footprint"\n[crop]\nsize = [32, 32]\n'
        "[model]\nstem = 4\nwidths = [4, 4, 4]\nbottleneck = 4\nhidden = 8\n"
        "[optim]\nepochs = 1\nbatch = 16\namp = false\n"
        "[calibration]\nn_bins = 4\nmin_count = 5\n"
    )
    result = runner.invoke(
        app,
        [
            "train",
            "params",
            str(config),
            "--scenes",
            str(params_scenes),
            "--run-dir",
            str(tmp_path / "run"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "bundle:" in result.output
    assert (tmp_path / "run" / "config.toml").is_file()


def test_real_agreement_compares_with_the_trace(
    detector_run: Path,
    stub_detector: Bundle,
    params_scenes: Path,
    params_library: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real-data check, on a simulated scene posing as a recording.

    Three chirps of a test scene are entered in a copy of the template
    library as harvested ("competition") chirps of recording ``001800``;
    their traces are the templates, so the reference equals the truth.
    """
    import shutil

    import chirpdetector.inference.spans as spans
    from chirpdetector.params.evaluation import agreement_summary, real_agreement

    scene = params_scenes / "test" / "001800"
    chirps = pd.read_csv(scene / "chirps.csv")
    chirps = chirps.loc[
        chirps["visible"].to_numpy(bool) & ~chirps["truncated"].to_numpy(bool)
    ]
    chirps = chirps.drop_duplicates("template_id").iloc[:3]
    library = tmp_path / "library"
    shutil.copytree(params_library, library)
    meta = pd.read_parquet(library / "templates.parquet")
    meta["source"] = "chirp_chamber"
    meta["recording"] = ""
    meta["align_abs_s"] = np.nan
    meta["baseline_hz"] = np.nan
    for row in chirps.itertuples():
        k = int(np.flatnonzero(meta["template_id"].to_numpy() == row.template_id)[0])
        meta.loc[k, ["source", "recording"]] = ["competition", "001800"]
        meta.loc[k, ["align_abs_s", "baseline_hz"]] = [row.t_center_s, row.baseline_hz]
    meta.to_parquet(library / "templates.parquet", index=False)
    monkeypatch.setattr(
        spans, "detect_in_spectrogram", footprint_detections({"001800": scene})
    )
    estimator = ParamsEstimator(detector_run / "bundle", device="cpu")
    real = real_agreement(
        estimator, stub_detector, library, params_scenes / "test", device="cpu"
    )
    assert len(real) == len(chirps)
    assert real["detected"].to_numpy(bool).all()
    excursion = chirps.set_index("template_id").loc[real["template_id"], "excursion_hz"]
    np.testing.assert_allclose(real["ref_height_hz"], excursion.to_numpy(), rtol=0.03)
    assert np.isfinite(real["height_hz"].to_numpy(float)).all()
    summary = agreement_summary(real)
    assert summary["detected"] == len(chirps)
    assert summary["height_hz"]["n"] == len(chirps)
    assert "individual" in real.columns


def test_agreement_summary_gives_the_simulated_correlation_over_the_same_range() -> (
    None
):
    """A narrow reference range caps r; the summary says what simulation predicts."""
    rng = np.random.default_rng(0)
    truth = rng.uniform(40, 500, 4000)  # simulated test chirps, wide range
    sim = pd.DataFrame(
        {
            "true_height_hz": truth,
            "cal_height_hz": truth * np.exp(rng.normal(0, 0.1, truth.size)),
            "family": np.where(truth > 200, "big", "small"),
            "f_high_hz": 800 + truth + 25,
            "baseline_hz": 800.0,
        }
    )
    ref = rng.uniform(380, 460, 300)  # harvested chirps: narrow range
    real = pd.DataFrame(
        {
            "detected": True,
            "individual": ["a"] * 250 + ["b"] * 50,
            "ref_height_hz": ref,
            "height_hz": ref * np.exp(rng.normal(0, 0.1, ref.size)),
            "f_high_hz": 800 + ref + 60,
            "baseline_hz": 800.0,
        }
    )
    summary = agreement_summary(real, sim=sim)
    h = summary["height_hz"]
    assert np.corrcoef(sim["true_height_hz"], sim["cal_height_hz"])[0, 1] > 0.9
    assert h["sim_pearson_r"] < 0.6  # the same error over the narrow range
    assert h["pearson_r"] == pytest.approx(h["sim_pearson_r"], abs=0.15)
    assert h["sim_n"] < 1000
    assert summary["largest_individual_share"] == pytest.approx(250 / 300)
    assert summary["box_top_offset_hz"]["real"] == pytest.approx(60.0)
    assert summary["box_top_offset_hz"]["sim_big"] == pytest.approx(25.0)


def test_predict_span_gives_the_decoder_the_training_crop(
    params_scenes: Path,
    stub_detector: Bundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The crop a detection gets at inference equals the training crop of the
    same box, with click removal on (it runs on the whole scene in training,
    on the widened span at inference)."""
    from dataclasses import replace

    import chirpdetector.inference.spans as spans
    from chirpdetector.params.crops import CropConfig, decoder_inputs
    from chirpdetector.params.samples import _prepare_scene
    from chirpdetector.preprocessing import ClickRemoval, can_remove_clicks
    from chirpdetector.spectrogram import compute_spectrogram

    detector = replace(
        stub_detector, preprocessing={"click_removal": ClickRemoval().to_dict()}
    )
    ref = next(r for r in discover_scenes([params_scenes]) if r.split == "test")
    recording = open_recording(ref.directory / "recording.wav")
    assert can_remove_clicks(
        recording.n_channels, samplerate_hz=recording.samplerate_hz
    )
    monkeypatch.setattr(
        spans,
        "detect_in_spectrogram",
        footprint_detections({ref.source_id: ref.directory}),
    )
    crop = CropConfig(size=(48, 48))
    captured: dict[str, object] = {}

    class Capture(ParamsEstimator):
        def __init__(self) -> None:  # no bundle: only the crop geometry
            self.bundle = type("B", (), {"crop": crop})()  # type: ignore[assignment]

        def check_detector(
            self, detector: Bundle, *, allow_mismatch: bool = False
        ) -> None:
            pass

        def estimate(self, spec, boxes):  # type: ignore[no-untyped-def,override]
            captured["spec"], captured["boxes"] = spec, boxes
            return pd.DataFrame(index=boxes.index, columns=list(PARAM_COLUMNS))

    t0, t1 = 2.0, 5.5
    predict_span(
        detector, recording, t0, t1, name=ref.source_id, device="cpu", params=Capture()
    )
    boxes = captured["boxes"]
    assert isinstance(boxes, pd.DataFrame)
    centre = 0.5 * (boxes["t_start_s"] + boxes["t_end_s"])
    boxes = boxes.loc[(centre > t0) & (centre < t1)]
    assert len(boxes) >= 2
    edges = boxes[["t_start_s", "t_end_s", "f_low_hz", "f_high_hz"]].to_numpy()
    at_inference = decoder_inputs(captured["spec"], edges, crop)  # type: ignore[arg-type]
    # the training path: the whole scene, cleaned as a whole
    prepared = _prepare_scene(ref, detector.preprocessing["click_removal"])
    raw = recording.read(0, recording.n_samples)
    assert not np.allclose(raw, prepared.samples)  # clicks were removed
    whole = compute_spectrogram(
        prepared.samples, prepared.samplerate_hz, detector.spectrogram, device="cpu"
    )
    in_training = decoder_inputs(whole, edges, crop)
    assert not at_inference.canvas_shifted.any()
    np.testing.assert_array_equal(at_inference.crops, in_training.crops)
    np.testing.assert_array_equal(at_inference.scalars, in_training.scalars)
