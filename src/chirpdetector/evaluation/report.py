"""Run a complete evaluation and write its results.

:func:`evaluate` scores a prediction table against ground truth and returns
an :class:`EvaluationResult`: frozen, with every table as a DataFrame and
the headline numbers in :meth:`EvaluationResult.metrics`.
:meth:`EvaluationResult.write` saves ``metrics.json`` and the tables as
CSV; :func:`attach_to_bundle` writes ``metrics.json`` into a model bundle
together with the sha256 of the files that were evaluated.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from gridtools.io import Tracks

from chirpdetector.evaluation.assignment import (
    ALL,
    DEFAULT_BIN_EDGES_HZ,
    UNKNOWN_BIN,
    assignment_summary,
    assignment_table,
    bin_names,
    confusion,
    coverage_accuracy,
    delta_f_at,
    df_bins,
    end_to_end,
    reliability,
)
from chirpdetector.evaluation.detection import (
    average_precision,
    confidence_intervals,
    pr_curve,
    summarize,
)
from chirpdetector.evaluation.matching import COUNT_COLUMNS, MatchConfig, Scope
from chirpdetector.evaluation.schema import (
    LABEL_COLUMNS,
    PREDICTION_COLUMNS,
    REGION_COLUMNS,
    GroundTruth,
    as_series,
    floats,
    grouped,
    ints,
    pick,
    select,
    strings,
    validate_predictions,
)

__all__ = [
    "DEFAULT_THRESHOLDS",
    "METRICS_FORMAT",
    "EvaluationConfig",
    "EvaluationResult",
    "attach_to_bundle",
    "evaluate",
    "parse_thresholds",
    "sha256_file",
]

log = logging.getLogger(__name__)

METRICS_FORMAT = 1
"""Version of the ``metrics.json`` layout."""


def parse_thresholds(spec: str) -> tuple[float, ...]:
    """Parse ``START:STOP:STEP`` (inclusive) or a comma-separated list.

    Values are computed in decimal arithmetic, so ``0.25:0.95:0.05`` gives
    exactly the floats nearest to 0.25, 0.30, ..., 0.95 (as the benchmark's
    ``np.round(np.arange(0.25, 0.951, 0.05), 2)``).

    Raises
    ------
    ValueError
        On malformed input or a non-positive step.
    """
    try:
        if ":" in spec:
            start, stop, step = (Decimal(v.strip()) for v in spec.split(":"))
            if step <= 0:
                msg = f"step must be positive in {spec!r}"
                raise ValueError(msg)
            n = int((stop - start) / step)
            values = [start + k * step for k in range(n + 1)]
        else:
            values = [Decimal(v.strip()) for v in spec.split(",")]
    except (InvalidOperation, TypeError) as err:
        msg = f"thresholds must look like 0.25:0.95:0.05 or 0.3,0.5, got {spec!r}"
        raise ValueError(msg) from err
    if not values:
        msg = f"no thresholds in {spec!r}"
        raise ValueError(msg)
    return tuple(float(v) for v in values)


DEFAULT_THRESHOLDS = parse_thresholds("0.25:0.95:0.05")
"""Score thresholds of the sweep (those of ``detector_benchmark``)."""


@dataclass(frozen=True)
class EvaluationConfig:
    """Settings of an evaluation.

    Parameters
    ----------
    match
        Matching rule (default: the benchmark's).
    thresholds
        Score thresholds of the sweep (predictions with ``score >=
        threshold`` are kept).
    reference_threshold
        Threshold of the confidence intervals, item table, assignment and
        end-to-end metrics (v1 ran at 0.5). Always added to the sweep.
    alpha
        1 - confidence level of all intervals.
    n_bootstrap
        Bootstrap resamples over regions.
    bootstrap_seed
        Seed of the bootstrap generators.
    calibration_bins
        Equal-width reliability bins on [0, 1].
    df_bin_edges_hz
        Lower edges of the ΔF bins; the last bin is open.
    track_max_gap_s
        A track is live where its detections are at most this far apart.
    """

    match: MatchConfig = field(default_factory=MatchConfig)
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    reference_threshold: float = 0.5
    alpha: float = 0.05
    n_bootstrap: int = 5000
    bootstrap_seed: int = 0
    calibration_bins: int = 10
    df_bin_edges_hz: tuple[float, ...] = DEFAULT_BIN_EDGES_HZ
    track_max_gap_s: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            msg = f"alpha must lie in (0, 1), got {self.alpha}"
            raise ValueError(msg)
        if self.n_bootstrap < 0 or self.calibration_bins < 1:
            msg = "n_bootstrap must be >= 0 and calibration_bins >= 1"
            raise ValueError(msg)
        values = (*self.thresholds, self.reference_threshold)
        if not all(math.isfinite(t) for t in values):
            msg = "thresholds must be finite"
            raise ValueError(msg)
        object.__setattr__(
            self, "thresholds", tuple(sorted({float(t) for t in self.thresholds}))
        )
        object.__setattr__(
            self, "df_bin_edges_hz", tuple(float(e) for e in self.df_bin_edges_hz)
        )

    @property
    def all_thresholds(self) -> tuple[float, ...]:
        """The sweep plus the reference threshold, sorted."""
        return tuple(sorted({*self.thresholds, float(self.reference_threshold)}))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    """All tables of an evaluation (see :func:`evaluate`).

    Attributes
    ----------
    config
        The settings used.
    per_region
        Counts per region and threshold (``COUNT_COLUMNS``).
    summary
        Precision, recall and F1 per recording, dataset and pooled
        (``level``), per threshold: the layout of the benchmark's
        ``summary.csv``.
    intervals
        Exact binomial and bootstrap CIs at the reference threshold, pooled
        and per dataset.
    pr_curve, average_precision
        Precision-recall curves at every distinct score and their AP, per
        level.
    per_spot, spot_summary
        Counts and rates of spot labels (non-exhaustive labels outside
        regions) per recording and threshold.
    items
        At the reference threshold, one row per scored label or prediction:
        ``outcome`` is ``tp``, ``fn``, ``fp``, ``ignored`` or ``tn`` (an
        unmatched ``not_chirp`` label), with emitters, scores, centre times
        and the ΔF bin.
    assignment, coverage_accuracy, reliability, confusion
        Assignment metrics, pooled and per ΔF bin.
    end_to_end
        Chirps detected with the correct emitter.
    provenance
        Versions, time and content hashes of the inputs.
    """

    config: EvaluationConfig
    per_region: pd.DataFrame
    summary: pd.DataFrame
    intervals: pd.DataFrame
    pr_curve: pd.DataFrame
    average_precision: pd.DataFrame
    per_spot: pd.DataFrame
    spot_summary: pd.DataFrame
    items: pd.DataFrame
    assignment: pd.DataFrame
    coverage_accuracy: pd.DataFrame
    reliability: pd.DataFrame
    confusion: pd.DataFrame
    end_to_end: pd.DataFrame
    provenance: dict[str, Any]

    TABLES = (
        "summary",
        "per_region",
        "intervals",
        "pr_curve",
        "average_precision",
        "spot_summary",
        "per_spot",
        "items",
        "assignment",
        "coverage_accuracy",
        "reliability",
        "confusion",
        "end_to_end",
    )

    def metrics(self) -> dict[str, Any]:
        """Headline numbers at the reference threshold (JSON-ready)."""
        ref = self.config.reference_threshold
        keys = [*COUNT_COLUMNS, "fn", "fp", "precision", "recall", "f1"]
        ap = {
            (r["level"], _name(r["dataset"]), _name(r["recording"])): r[
                "average_precision"
            ]
            for r in _records(self.average_precision)
        }
        ci = {str(k): v for k, v in self.intervals.to_dict(orient="index").items()}

        def entry(r: dict[str, Any], *, with_ci: bool) -> dict[str, Any]:
            out = {k: r[k] for k in keys if k in r}
            key = (r["level"], _name(r["dataset"]), _name(r["recording"]))
            if key in ap:
                out["average_precision"] = ap[key]
            group = "all" if r["level"] == "all" else _name(r["dataset"])
            if with_ci and group in ci:
                c = ci[group]
                out["precision_ci"] = [c["precision_lo"], c["precision_hi"]]
                out["recall_ci"] = [c["recall_lo"], c["recall_hi"]]
                out["precision_boot_ci"] = [
                    c["boot_precision_lo"],
                    c["boot_precision_hi"],
                ]
                out["recall_boot_ci"] = [c["boot_recall_lo"], c["boot_recall_hi"]]
                out["f1_boot_ci"] = [c["boot_f1_lo"], c["boot_f1_hi"]]
                out["n_regions"] = c["n_regions"]
            return out

        def levels(table: pd.DataFrame, *, with_ci: bool) -> dict[str, Any]:
            at = select(table, floats(table["threshold"]) == ref)
            out: dict[str, Any] = {"datasets": {}, "recordings": {}}
            for r in _records(at):
                if r["level"] == "all":
                    out["all"] = entry(r, with_ci=with_ci)
                elif r["level"] == "dataset":
                    out["datasets"][r["dataset"]] = entry(r, with_ci=with_ci)
                else:
                    out["recordings"][r["recording"]] = entry(r, with_ci=False)
            return out

        out: dict[str, Any] = {
            "detection": {
                "reference_threshold": ref,
                **levels(self.summary, with_ci=True),
            }
        }
        if len(self.spot_summary):
            out["spot"] = levels(self.spot_summary, with_ci=False)
        if len(self.assignment):
            rows = {r["df_bin"]: r for r in _records(self.assignment)}
            out["assignment"] = {
                "all": rows[ALL],
                "by_df_bin": {b: r for b, r in rows.items() if b != ALL},
            }
        if len(self.end_to_end):
            e2e: dict[str, Any] = {}
            fields = ["labels", "detections", "tp", "fn", "fp"]
            fields += ["precision", "recall", "f1", "n_regions"]
            for r in _records(self.end_to_end):
                if r["level"] != "all":
                    continue
                scope = e2e.setdefault(r["scope"], {"by_df_bin": {}})
                values = {k: r[k] for k in fields}
                if r["df_bin"] == ALL:
                    scope["all"] = values
                else:
                    scope["by_df_bin"][r["df_bin"]] = values
            out["end_to_end"] = e2e
        return _jsonable(out)

    def to_dict(self) -> dict[str, Any]:
        """``metrics.json`` content: format, config, provenance, metrics."""
        return _jsonable(
            {
                "format": METRICS_FORMAT,
                "config": self.config.to_dict(),
                "provenance": self.provenance,
                "metrics": self.metrics(),
            }
        )

    def to_json(self, indent: int = 1) -> str:
        """``metrics.json`` as a string (NaN written as null)."""
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def write(self, directory: str | Path, *, plot: bool = False) -> list[Path]:
        """Write ``metrics.json`` and every table as CSV into ``directory``.

        Parameters
        ----------
        directory
            Output directory (created if needed).
        plot
            Also draw ``pr_curve.png`` (needs matplotlib).

        Returns
        -------
        The written files.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written = [directory / "metrics.json"]
        _write_text_atomic(self.to_json() + "\n", written[0])
        for name in self.TABLES:
            path = directory / f"{name}.csv"
            table: pd.DataFrame = getattr(self, name)
            table.to_csv(path, index=name == "intervals")
            written.append(path)
        if plot:
            written.append(self.plot_pr_curve(directory / "pr_curve.png"))
        return written

    def plot_pr_curve(self, path: str | Path) -> Path:
        """Draw pooled and per-dataset PR curves, marking the reference threshold.

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        """
        from matplotlib.figure import Figure  # lazy: optional dependency

        path = Path(path)
        fig = Figure(figsize=(4.5, 4.0), layout="constrained")
        ax = fig.add_subplot()
        ref = self.config.reference_threshold
        points = {
            (r["level"], _name(r["dataset"])): (r["recall"], r["precision"])
            for r in _records(
                select(self.summary, floats(self.summary["threshold"]) == ref)
            )
        }
        level = np.asarray(strings(self.pr_curve["level"]), dtype=object)
        for lvl, keys in (("all", []), ("dataset", ["dataset"])):
            for named, g in grouped(select(self.pr_curve, level == lvl), keys):
                g = select(g, ints(g["detections"]) > 0)
                name = named.get("dataset", "pooled")
                (line,) = ax.step(
                    floats(g["recall"]),
                    floats(g["precision"]),
                    where="post",
                    label=name,
                )
                point = points.get((lvl, named.get("dataset")))
                if point is not None:
                    ax.plot(*point, "o", color=line.get_color())
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"dots: threshold {ref:g}", fontsize="small")
        ax.legend(fontsize="small", loc="lower left")
        fig.savefig(path, dpi=150)
        return path


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): v for k, v in r.items()} for r in df.to_dict(orient="records")]


def _name(value: Any) -> str | None:
    """A dataset or recording name; None where pooled (NaN)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def _jsonable(obj: Any) -> Any:
    """Convert numpy/pandas scalars to Python; NaN and NA to None."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if obj is None or obj is pd.NA:
        return None
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, bool | int | float | str):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def _write_text_atomic(text: str, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's content."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _table_sha256(df: pd.DataFrame, columns: Sequence[str]) -> str:
    text = df[list(columns)].to_csv(index=False, float_format="%.17g")
    return hashlib.sha256(text.encode()).hexdigest()


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in ("chirpdetector", "gridtools", "numpy", "scipy", "pandas"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "unknown"
    return out


_ITEM_COLUMNS = (
    "recording",
    "dataset",
    "scope",
    "region",
    "outcome",
    "label_row",
    "pred_row",
    "label",
    "true_emitter_id",
    "pred_emitter_id",
    "emitter_prob",
    "score",
    "label_t_center_s",
    "pred_t_center_s",
    "dt_s",
    "delta_f_hz",
    "n_live_tracks",
    "df_bin",
)


def _model_hashes(preds: pd.DataFrame) -> list[str]:
    """Distinct ``model_sha256`` values of the predictions (empty if absent)."""
    if "model_sha256" not in preds.columns:
        return []
    values = as_series(preds["model_sha256"]).dropna()
    return sorted({str(v) for v in values})


def _rows(index: np.ndarray, local: np.ndarray) -> pd.Series:
    """Map scope-local indices (-1 = none) to table rows (NA = none)."""
    rows = np.full(len(local), -1, dtype=np.int64)
    rows[local >= 0] = index[local[local >= 0]]
    return pd.Series(rows).where(local >= 0).astype("Int64")


def _scope_items(
    scope: Scope, threshold: float, key: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Item rows of one scope at one threshold, and its counts."""
    m = scope.match(threshold)
    rows: list[tuple[str, int, int]] = []  # outcome, label, prediction (local)
    rows += [("tp", int(a), int(b)) for a, b in zip(*m.tp, strict=True)]
    matched = np.zeros(len(scope.kind), dtype=np.bool_)
    matched[m.tp[0]] = True
    matched[m.negative[0]] = True
    rows += [("fn", int(a), -1) for a in np.flatnonzero(scope.positive & ~matched)]
    rows += [("ignored", int(a), int(b)) for a, b in zip(*m.ignored, strict=True)]
    rows += [("fp", int(a), int(b)) for a, b in zip(*m.negative, strict=True)]
    rows += [("tn", int(a), -1) for a in np.flatnonzero(scope.negative & ~matched)]
    if scope.is_region:
        explained = np.zeros(len(m.counted), dtype=np.bool_)
        explained[m.tp[1]] = True
        explained[m.negative[1]] = True
        rows += [("fp", -1, int(b)) for b in np.flatnonzero(m.counted & ~explained)]
    li = np.array([r[1] for r in rows], dtype=np.int64)
    pj = np.array([r[2] for r in rows], dtype=np.int64)
    df = pd.DataFrame(
        {
            **{k: [v] * len(rows) for k, v in key.items()},
            "outcome": [r[0] for r in rows],
            "label_row": _rows(scope.label_index, li),
            "pred_row": _rows(scope.pred_index, pj),
        }
    )
    return df, m.counts(scope)


def _take(
    values: npt.NDArray[Any], rows: npt.NDArray[np.int64], fill: Any
) -> npt.NDArray[np.object_]:
    """``values[rows]`` with ``fill`` where ``rows`` is -1."""
    out = np.full(len(rows), fill, dtype=object)
    ok = rows >= 0
    out[ok] = values[rows[ok]]
    return out


def _attach_item_details(
    items: pd.DataFrame,
    gt: GroundTruth,
    predictions: pd.DataFrame,
    tracks: Mapping[str, Tracks],
    config: EvaluationConfig,
) -> pd.DataFrame:
    labels = gt.labels
    lr = as_series(items["label_row"]).astype("Int64").fillna(-1).to_numpy(np.int64)
    pr = as_series(items["pred_row"]).astype("Int64").fillna(-1).to_numpy(np.int64)

    def label_column(column: str, fill: Any) -> npt.NDArray[np.object_]:
        return _take(as_series(labels[column]).to_numpy(dtype=object), lr, fill)

    def pred_column(column: str, fill: Any) -> npt.NDArray[np.object_]:
        return _take(as_series(predictions[column]).to_numpy(dtype=object), pr, fill)

    p_center = 0.5 * (floats(predictions["t_start_s"]) + floats(predictions["t_end_s"]))
    label_t = label_column("t_center_s", np.nan).astype(np.float64)
    pred_t = _take(p_center, pr, np.nan).astype(np.float64)
    items = items.copy()
    items["label"] = label_column("label", None)
    items["true_emitter_id"] = pd.array(
        label_column("emitter_id", pd.NA), dtype="Int64"
    )
    items["pred_emitter_id"] = pd.array(pred_column("emitter_id", pd.NA), dtype="Int64")
    items["emitter_prob"] = pred_column("emitter_prob", np.nan).astype(np.float64)
    items["score"] = pred_column("score", np.nan).astype(np.float64)
    items["label_t_center_s"] = label_t
    items["pred_t_center_s"] = pred_t
    items["dt_s"] = np.abs(label_t - pred_t)
    t = np.where(lr >= 0, label_t, pred_t)
    delta = np.full(len(items), np.nan)
    live = np.full(len(items), -1, dtype=np.int64)
    bins = np.full(len(items), UNKNOWN_BIN, dtype=object)
    recording = np.asarray(strings(items["recording"]), dtype=object)
    for rec in sorted(set(recording.tolist()) & set(tracks)):
        idx = np.flatnonzero(recording == rec)
        d, k = delta_f_at(tracks[rec], t[idx], config.track_max_gap_s)
        delta[idx] = d
        live[idx] = k
        bins[idx] = df_bins(d, k, config.df_bin_edges_hz)
    items["delta_f_hz"] = delta
    items["n_live_tracks"] = pd.Series(np.where(live >= 0, live, np.nan)).astype(
        "Int64"
    )
    items["df_bin"] = bins
    return pick(items, _ITEM_COLUMNS)


def evaluate(
    predictions: pd.DataFrame,
    ground_truth: GroundTruth,
    *,
    config: EvaluationConfig | None = None,
    tracks: Mapping[str, Tracks] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> EvaluationResult:
    """Score predictions against ground truth.

    Parameters
    ----------
    predictions
        Prediction table with a ``recording`` column (see
        :func:`~chirpdetector.evaluation.schema.validate_predictions`).
        Predictions of recordings without ground truth are ignored (with a
        warning). For v1's pre-NMS boxes, apply
        :func:`~chirpdetector.evaluation.predictions.non_max_suppression`
        first.
    ground_truth
        Regions and labels.
    config
        Settings; default :class:`EvaluationConfig`.
    tracks
        Tracks per recording, for ΔF; recordings without tracks get the ΔF
        bin ``unknown``.
    provenance
        Extra provenance (e.g. input files and their sha256), stored in the
        result.

    Returns
    -------
    The result. Counts at every threshold come from one pass per scope
    (:meth:`~chirpdetector.evaluation.matching.Scope.count_events`); at the
    reference threshold they are checked against a direct match.
    """
    config = config or EvaluationConfig()
    tracks = dict(tracks or {})
    preds = validate_predictions(predictions)
    gt = ground_truth
    recordings = set(gt.recordings)
    extra = sorted(set(strings(preds["recording"])) - recordings)
    if extra:
        log.warning(
            "ignoring predictions of recordings without ground truth: %s", extra
        )
    missing = sorted(recordings - set(strings(preds["recording"])))
    if missing:
        log.warning("no predictions for %s: every label is missed", missing)
    pred_recs = np.asarray(strings(preds["recording"]), dtype=object)
    by_rec = {rec: select(preds, pred_recs == rec) for rec in gt.recordings}
    thresholds = np.asarray(config.all_thresholds, dtype=np.float64)
    ref = float(config.reference_threshold)
    datasets = gt.datasets

    plan: list[tuple[str, int | None, tuple[float, float, float, float] | None]] = []
    region_recs = strings(gt.regions["recording"])
    region_ids = ints(gt.regions["region"])
    spans = pick(gt.regions, ["t_start_s", "t_end_s", "f_low_hz", "f_high_hz"])
    for rec, rid, box in zip(
        region_recs, region_ids.tolist(), spans.to_numpy(np.float64), strict=True
    ):
        plan.append((rec, rid, (box[0], box[1], box[2], box[3])))
    label_recs = np.asarray(strings(gt.labels["recording"]), dtype=object)
    label_region = floats(gt.labels["region"])
    spot = np.isnan(label_region)
    plan += [(rec, None, None) for rec in sorted(set(label_recs[spot].tolist()))]

    scope_counts, events, scope_labels, item_parts = [], [], [], []
    for rec, rid, box in plan:
        in_scope = (label_recs == rec) & (spot if rid is None else label_region == rid)
        scope = Scope(select(gt.labels, in_scope), by_rec[rec], config.match, box)
        key = {
            "recording": rec,
            "dataset": datasets[rec],
            "scope": "spot" if rid is None else "region",
            "region": pd.NA if rid is None else rid,
        }
        events_of_scope = scope.count_events()
        counts = scope.counts_at(thresholds, events_of_scope)
        span = {
            "t_start_s": np.nan if box is None else box[0],
            "t_end_s": np.nan if box is None else box[1],
        }
        scope_counts.append(counts.assign(**key, **span))
        items, direct = _scope_items(scope, ref, key)
        at_ref = _records(select(counts, floats(counts["threshold"]) == ref))[0]
        if any(int(at_ref[k]) != direct[k] for k in COUNT_COLUMNS):
            msg = f"internal error: count events disagree with matching in {key}"
            raise RuntimeError(msg)
        item_parts.append(items)
        if rid is not None:
            events.append(events_of_scope.assign(dataset=key["dataset"], recording=rec))
            scope_labels.append(
                {
                    "dataset": key["dataset"],
                    "recording": rec,
                    "labels": int(at_ref["labels"]),
                    "negatives": int(at_ref["negatives"]),
                }
            )

    count_cols = ["threshold", *COUNT_COLUMNS]
    table_cols = ["scope", "dataset", "recording", "region", "t_start_s", "t_end_s"]
    per_scope = (
        pick(pd.concat(scope_counts, ignore_index=True), [*table_cols, *count_cols])
        if scope_counts
        else pd.DataFrame(columns=[*table_cols, *count_cols])
    )
    per_scope["region"] = as_series(per_scope["region"]).astype("Int64")
    scope_kind = np.asarray(strings(per_scope["scope"]), dtype=object)
    per_region = pick(
        select(per_scope, scope_kind == "region"), table_cols[1:] + count_cols
    ).reset_index(drop=True)
    per_spot = pick(
        select(per_scope, scope_kind == "spot"), ["dataset", "recording", *count_cols]
    ).reset_index(drop=True)
    summary = summarize(per_region)
    spot_summary = summarize(per_spot)
    intervals = confidence_intervals(
        select(per_region, floats(per_region["threshold"]) == ref),
        alpha=config.alpha,
        n_boot=config.n_bootstrap,
        seed=config.bootstrap_seed,
    )
    event_cols = ["score", "detections", "tp", "ignored", "fp_negative"]
    all_events = (
        pd.concat(events, ignore_index=True)
        if events
        else pd.DataFrame(columns=[*event_cols, "dataset", "recording"])
    )
    curve = pr_curve(
        all_events,
        pd.DataFrame(
            scope_labels, columns=["dataset", "recording", "labels", "negatives"]
        ),
    )
    ap = average_precision(curve)

    item_cols = ["recording", "dataset", "scope", "region", "outcome"]
    items = (
        pd.concat(item_parts, ignore_index=True)
        if item_parts
        else pd.DataFrame(columns=[*item_cols, "label_row", "pred_row"])
    )
    items = _attach_item_details(items, gt, preds, tracks, config)
    names = bin_names(config.df_bin_edges_hz)
    table = assignment_table(items)
    scored = int(np.isin(pred_recs, list(recordings)).sum())
    result = EvaluationResult(
        config=config,
        per_region=per_region,
        summary=summary,
        intervals=intervals,
        pr_curve=curve,
        average_precision=ap,
        per_spot=per_spot,
        spot_summary=spot_summary,
        items=items,
        assignment=assignment_summary(table, names, config.calibration_bins),
        coverage_accuracy=coverage_accuracy(table, names),
        reliability=reliability(table, names, config.calibration_bins),
        confusion=confusion(table),
        end_to_end=end_to_end(items, names),
        provenance=_jsonable(
            {
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "versions": _versions(),
                "predictions_sha256": _table_sha256(
                    preds,
                    ["recording", *PREDICTION_COLUMNS, "emitter_id", "emitter_prob"],
                ),
                "ground_truth_sha256": hashlib.sha256(
                    (
                        _table_sha256(gt.regions, REGION_COLUMNS)
                        + _table_sha256(gt.labels, LABEL_COLUMNS)
                    ).encode()
                ).hexdigest(),
                "model_sha256": _model_hashes(preds),
                "n_predictions": len(preds),
                "n_predictions_scored_recordings": scored,
                "recordings": list(gt.recordings),
                "recordings_with_tracks": sorted(set(tracks) & recordings),
                **dict(provenance or {}),
            }
        ),
    )
    return result


def attach_to_bundle(
    result: EvaluationResult,
    bundle_dir: str | Path,
    evaluated: str | Path | Sequence[str | Path],
) -> Path:
    """Write ``metrics.json`` into a model bundle.

    For a chirpdetector bundle (a directory with ``bundle.json``, see
    :mod:`chirpdetector.bundle`) the bundle is loaded and its weights are
    verified first, and the predictions must have been made by exactly
    these weights: ``result.provenance["model_sha256"]`` (the
    ``model_sha256`` column of the predictions, which
    :func:`chirpdetector.prediction.predict_images` writes) must be the
    bundle's weight hash and nothing else. The hash is stored as
    ``weights_sha256`` in ``metrics.json``, and
    :func:`chirpdetector.bundle.load_bundle` refuses a ``metrics.json`` of
    other weights.

    Parameters
    ----------
    result
        The evaluation.
    bundle_dir
        Existing bundle directory (next to the weights).
    evaluated
        The predictions file(s) or weights the result was computed from;
        their sha256 is stored with the metrics, so the numbers can be
        traced to exactly these bytes.

    Returns
    -------
    Path of the written ``metrics.json``.

    Raises
    ------
    FileNotFoundError
        If the bundle directory or an evaluated file does not exist.
    ValueError
        If the predictions of a chirpdetector bundle were not made by its
        weights (or carry no ``model_sha256``), or the bundle is invalid.
    """
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise FileNotFoundError(bundle_dir)
    weights_sha256 = None
    if (bundle_dir / "bundle.json").is_file():
        from chirpdetector.bundle import load_bundle

        weights_sha256 = load_bundle(bundle_dir).weights_sha256
        hashes = list(result.provenance.get("model_sha256") or [])
        if hashes != [weights_sha256]:
            msg = (
                f"metrics of predictions by weights {hashes or 'of unknown origin'} "
                f"cannot be attached to bundle {bundle_dir} (weights "
                f"{weights_sha256}); evaluate predictions made by this bundle "
                "(chirpdetector.prediction adds their model_sha256)"
            )
            raise ValueError(msg)
    files = [evaluated] if isinstance(evaluated, str | Path) else list(evaluated)
    entries = []
    for f in map(Path, files):
        if not f.is_file():
            raise FileNotFoundError(f)
        resolved = f.resolve()
        name = (
            str(resolved.relative_to(bundle_dir.resolve()))
            if resolved.is_relative_to(bundle_dir.resolve())
            else str(resolved)
        )
        entries.append({"path": name, "sha256": sha256_file(f)})
    data = result.to_dict()
    data["evaluated"] = entries
    if weights_sha256 is not None:
        data["weights_sha256"] = weights_sha256
    path = bundle_dir / "metrics.json"
    _write_text_atomic(json.dumps(data, indent=1, allow_nan=False) + "\n", path)
    return path
