"""The tables the evaluation works on: predictions and ground truth.

Predictions
-----------
One row per detected chirp: the box ``t_start_s, t_end_s, f_low_hz,
f_high_hz``, a detector ``score`` and the ``recording`` it belongs to;
optionally the assigned emitter ``emitter_id`` (a track id, nullable
integer; NA = no emitter assigned) and its probability ``emitter_prob``,
and the emitter's EOD frequency ``emitter_hz`` predicted by models with an
emitter head (NaN = none; turned into ``emitter_id`` by
:func:`~chirpdetector.evaluation.assignment.assign_by_frequency`), and the
``model_sha256`` of the weights that made the predictions (see
:func:`~chirpdetector.evaluation.report.attach_to_bundle`). Other columns
(e.g. v1's ``batch``) are kept.

Ground truth
------------
:class:`GroundTruth` holds labelled *regions* and *labels* of one or more
recordings, each recording belonging to one *dataset* (used for pooling).

regions
    Time spans with a frequency band inside which every chirp was labelled
    (audian category ``labeled_region``). A region's labels are exhaustive,
    so a prediction inside a region that matches no chirp is a false
    positive.
labels
    Boxes with a ``label``:

    ``chirp``
        A chirp; ``emitter_id`` is its emitter if known (NA otherwise).
    ``chirp_emitter_unclear``
        A chirp whose emitter could not be told. Counts for detection,
        never for assignment.
    ``not_chirp``
        A box known not to be a chirp (e.g. a v1 detection the labeller
        rejected). A prediction matching it is a false positive.
    ``ignore``
        A chirp that is neither required nor penalised (e.g. a simulated
        chirp below the visibility threshold). A prediction matching it is
        not counted.

    A label whose centre time lies in a region's span belongs to that region
    (column ``region``). All other labels are *spot labels* (``region`` is
    NA): the labeller judged single boxes, e.g. with ``chirpdetector
    label-emitters``, and said nothing about the rest of the recording.

How each kind is scored is described in :mod:`chirpdetector.evaluation.matching`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from gridtools.io import read_labels

__all__ = [
    "LABEL_COLUMNS",
    "LABEL_VALUES",
    "POSITIVE_LABELS",
    "PREDICTION_COLUMNS",
    "REGION_COLUMNS",
    "UNKNOWN_DATASET",
    "Column",
    "GroundTruth",
    "as_series",
    "floats",
    "grouped",
    "infer_dataset",
    "ints",
    "pick",
    "read_ground_truth",
    "select",
    "simulation_ground_truth",
    "strings",
    "validate_predictions",
]

log = logging.getLogger(__name__)

PREDICTION_COLUMNS = ("t_start_s", "t_end_s", "f_low_hz", "f_high_hz", "score")
"""Required columns of a prediction table (besides ``recording``)."""

LABEL_VALUES = ("chirp", "chirp_emitter_unclear", "not_chirp", "ignore")
"""Allowed values of the ``label`` column of the ground truth."""

POSITIVE_LABELS = ("chirp", "chirp_emitter_unclear")
"""Labels that a detector should find."""

REGION_COLUMNS = (
    "recording",
    "dataset",
    "region",
    "t_start_s",
    "t_end_s",
    "f_low_hz",
    "f_high_hz",
)
"""Columns of :attr:`GroundTruth.regions`."""

LABEL_COLUMNS = (
    "recording",
    "dataset",
    "region",
    "t_start_s",
    "t_end_s",
    "f_low_hz",
    "f_high_hz",
    "t_center_s",
    "label",
    "emitter_id",
)
"""Columns of :attr:`GroundTruth.labels` (extra columns are kept after these)."""

UNKNOWN_DATASET = "unknown"
"""Dataset of recordings whose dataset is not known."""

_AUDIAN_LABEL_FILE_CATEGORIES = ("chirp", "not_chirp")
_SIM_COLUMNS = (
    "t_onset_s",
    "t_center_s",
    "t_offset_s",
    "f_low_hz",
    "f_high_hz",
    "emitter_id",
)
_DATASET_SKIP = frozenset({"raw"})
_BOX = ("t_start_s", "t_end_s", "f_low_hz", "f_high_hz")


FloatArray = npt.NDArray[np.float64]
Column = pd.Series | pd.DataFrame
"""What pandas' annotations give for ``df[name]`` (a DataFrame if the name
is duplicated)."""


def as_series(values: Column) -> pd.Series:
    """Narrow ``df[name]`` to a Series.

    Raises
    ------
    ValueError
        If ``values`` is a DataFrame (a duplicated column name).
    """
    if isinstance(values, pd.DataFrame):
        msg = f"duplicate columns {list(values.columns)}"
        raise ValueError(msg)
    return values


def floats(values: Column) -> FloatArray:
    """A column as float64 (NA becomes NaN); raises on non-numeric values."""
    return as_series(values).to_numpy(dtype=np.float64, na_value=np.nan)


def strings(values: Column) -> list[str]:
    """A column as a list of str."""
    return [str(v) for v in as_series(values).to_numpy()]


def ints(values: Column) -> npt.NDArray[np.int64]:
    """A column of integers (no NA) as int64."""
    return as_series(values).to_numpy(dtype=np.int64)


def select(df: pd.DataFrame, mask: npt.ArrayLike) -> pd.DataFrame:
    """Rows of ``df`` where the boolean ``mask`` is true."""
    return cast("pd.DataFrame", df.loc[np.asarray(mask, dtype=np.bool_)])


def pick(df: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    """Columns ``names`` of ``df``, in that order."""
    return cast("pd.DataFrame", df.loc[:, list(names)])


def grouped(
    df: pd.DataFrame, keys: Sequence[str]
) -> list[tuple[dict[str, str], pd.DataFrame]]:
    """Groups of ``df`` by ``keys`` (sorted), with the key values by name."""
    if not keys:
        return [({}, df)]
    values = [strings(df[k]) for k in keys]
    combos = sorted(set(zip(*values, strict=True)))
    rows = list(zip(*values, strict=True))
    out = []
    for combo in combos:
        mask = np.array([r == combo for r in rows], dtype=np.bool_)
        out.append((dict(zip(keys, combo, strict=True)), select(df, mask)))
    return out


def _as_int64(values: Column, name: str) -> pd.Series:
    """Convert to a nullable integer column; reject non-integral numbers."""
    values = as_series(values)
    if isinstance(values.dtype, pd.Int64Dtype):
        return values
    numbers = floats(values)
    finite = np.isfinite(numbers)
    if np.any(np.isinf(numbers)) or np.any(
        numbers[finite] != np.round(numbers[finite])
    ):
        msg = f"{name} must hold integers"
        raise ValueError(msg)
    ints = pd.array(np.where(finite, numbers, 0.0).astype(np.int64), dtype="Int64")
    ints[~finite] = pd.NA
    return pd.Series(ints, index=values.index)


def _na_int64(n: int) -> pd.arrays.IntegerArray:
    return pd.array([pd.NA] * n, dtype="Int64")  # type: ignore[return-value]


def _check_spans(df: pd.DataFrame, what: str, *, unbounded: bool = False) -> None:
    times = df.loc[:, ["t_start_s", "t_end_s"]].to_numpy(dtype=np.float64)
    if np.isnan(times).any() or not (unbounded or np.isfinite(times).all()):
        msg = (
            f"{what}: t_start_s and t_end_s must be {'set' if unbounded else 'finite'}"
        )
        raise ValueError(msg)
    if np.any(times[:, 1] < times[:, 0]):
        msg = f"{what}: t_end_s < t_start_s"
        raise ValueError(msg)
    bands = df.loc[:, ["f_low_hz", "f_high_hz"]].to_numpy(dtype=np.float64)
    if np.isnan(bands).any():
        msg = f"{what}: f_low_hz and f_high_hz must not be NaN"
        raise ValueError(msg)
    if np.any(bands[:, 1] < bands[:, 0]):
        msg = f"{what}: f_high_hz < f_low_hz"
        raise ValueError(msg)


def validate_predictions(
    predictions: pd.DataFrame, *, recording: str | None = None
) -> pd.DataFrame:
    """Check a prediction table and return a normalised copy.

    Parameters
    ----------
    predictions
        One row per detection with the columns in :data:`PREDICTION_COLUMNS`
        and, optionally, ``recording``, ``emitter_id`` and ``emitter_prob``.
    recording
        Recording of every row; required if there is no ``recording``
        column, and must agree with it otherwise.

    Returns
    -------
    A copy with a fresh index; box columns and ``score`` as float64,
    ``recording`` as str, ``emitter_id`` as Int64 (all NA if absent) and
    ``emitter_prob`` as float64 (all NaN if absent); ``emitter_hz``, if
    present, as float64.

    Raises
    ------
    ValueError
        On a missing column, a non-finite box or score, an inverted box, a
        non-integral emitter id, an emitter probability outside [0, 1] or
        an emitter frequency that is not positive.
    """
    df = predictions.copy().reset_index(drop=True)
    missing = [c for c in PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        msg = f"predictions: missing columns {missing}"
        raise ValueError(msg)
    if "recording" not in df.columns:
        if recording is None:
            msg = "predictions need a 'recording' column or recording=..."
            raise ValueError(msg)
        df["recording"] = recording
    elif recording is not None and any(
        r != recording for r in strings(df["recording"])
    ):
        msg = f"predictions: 'recording' column disagrees with recording={recording!r}"
        raise ValueError(msg)
    df["recording"] = strings(df["recording"])
    for column in PREDICTION_COLUMNS:
        df[column] = floats(df[column])
    values = df.loc[:, list(PREDICTION_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        msg = "predictions: boxes and scores must be finite"
        raise ValueError(msg)
    _check_spans(df, "predictions")
    if "emitter_id" in df.columns:
        df["emitter_id"] = _as_int64(df["emitter_id"], "predictions: emitter_id")
    else:
        df["emitter_id"] = _na_int64(len(df))
    if "emitter_prob" in df.columns:
        prob = floats(df["emitter_prob"])
        defined = ~np.isnan(prob)
        if np.any((prob[defined] < 0) | (prob[defined] > 1)):
            msg = "predictions: emitter_prob must lie in [0, 1]"
            raise ValueError(msg)
        df["emitter_prob"] = prob
    else:
        df["emitter_prob"] = np.full(len(df), np.nan)
    if "emitter_hz" in df.columns:
        emitter = floats(df["emitter_hz"])
        if np.any(np.isinf(emitter)) or np.any(emitter[np.isfinite(emitter)] <= 0):
            msg = "predictions: emitter_hz must be positive (or NaN)"
            raise ValueError(msg)
        df["emitter_hz"] = emitter
    return df


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    dtypes = {"region": "Int64", "emitter_id": "Int64"}
    text = ("recording", "dataset", "label")
    return pd.DataFrame(
        {
            c: pd.Series(dtype=dtypes.get(c, "object" if c in text else "float64"))
            for c in columns
        }
    )


def _fill_bands(df: pd.DataFrame) -> None:
    """A missing band edge means the box is unbounded on that side."""
    low, high = floats(df["f_low_hz"]), floats(df["f_high_hz"])
    df["f_low_hz"] = np.where(np.isnan(low), -np.inf, low)
    df["f_high_hz"] = np.where(np.isnan(high), np.inf, high)


def _assign_regions(labels: pd.DataFrame, regions: pd.DataFrame) -> pd.Series:
    """Region of every label: the region whose span contains its centre."""
    region = pd.Series(_na_int64(len(labels)), index=labels.index)
    label_rec = np.asarray(strings(labels["recording"]), dtype=object)
    tc = floats(labels["t_center_s"])
    rec = strings(regions["recording"])
    t0, t1 = floats(regions["t_start_s"]), floats(regions["t_end_s"])
    ids = regions["region"].to_numpy()
    for k in range(len(regions)):
        inside = (label_rec == rec[k]) & (tc >= t0[k]) & (tc <= t1[k])
        region.iloc[np.flatnonzero(inside)] = int(ids[k])
    return region


@dataclass(frozen=True)
class GroundTruth:
    """Labelled regions and labels of one or more recordings.

    Parameters
    ----------
    regions
        Columns :data:`REGION_COLUMNS`; ``region`` is an integer id, unique
        within a recording. A NaN band edge means unbounded.
    labels
        Columns :data:`LABEL_COLUMNS` (``region``, ``t_center_s``,
        ``emitter_id`` and ``dataset`` may be omitted: ``t_center_s``
        defaults to the box midpoint, ``region`` to the region containing
        that centre, ``emitter_id`` to NA, ``dataset`` to the dataset of the
        recording's regions). A NaN band edge means unbounded.

    Both tables are copied, normalised and validated on construction, and
    sorted by dataset, recording and time.

    Raises
    ------
    ValueError
        See :meth:`validate`.
    """

    regions: pd.DataFrame
    labels: pd.DataFrame

    def __post_init__(self) -> None:
        regions = self.regions.copy()
        labels = self.labels.copy()
        if "dataset" not in labels.columns:
            datasets = dict(
                zip(
                    strings(regions["recording"]),
                    strings(regions["dataset"]),
                    strict=True,
                )
            )
            labels["dataset"] = [
                datasets.get(r, UNKNOWN_DATASET) for r in strings(labels["recording"])
            ]
        if "t_center_s" not in labels.columns:
            labels["t_center_s"] = 0.5 * (
                floats(labels["t_start_s"]) + floats(labels["t_end_s"])
            )
        if "emitter_id" not in labels.columns:
            labels["emitter_id"] = _na_int64(len(labels))
        for df in (regions, labels):
            for column in ("recording", "dataset"):
                df[column] = pd.Series(
                    strings(df[column]), index=df.index, dtype=object
                )
            for column in ("t_start_s", "t_end_s"):
                df[column] = floats(df[column])
            _fill_bands(df)
        regions["region"] = _as_int64(regions["region"], "regions: region")
        labels["t_center_s"] = floats(labels["t_center_s"])
        labels["label"] = pd.Series(
            strings(labels["label"]), index=labels.index, dtype=object
        )
        labels["emitter_id"] = _as_int64(labels["emitter_id"], "labels: emitter_id")
        if "region" not in labels.columns:
            labels["region"] = _assign_regions(labels, regions)
        labels["region"] = _as_int64(labels["region"], "labels: region")
        regions = regions.loc[:, list(REGION_COLUMNS)].sort_values(
            by=["dataset", "recording", "region"], kind="stable"
        )
        extra = [c for c in labels.columns if c not in LABEL_COLUMNS]
        labels = labels.loc[:, [*LABEL_COLUMNS, *extra]].sort_values(
            by=["dataset", "recording", "t_center_s"], kind="stable"
        )
        object.__setattr__(self, "regions", regions.reset_index(drop=True))
        object.__setattr__(self, "labels", labels.reset_index(drop=True))
        self.validate()

    def validate(self) -> None:
        """Check the tables.

        Raises
        ------
        ValueError
            On NaN or inverted spans, non-finite label times, unknown label
            values, duplicate region ids, regions of one recording that
            overlap in time (a label would count twice), labels pointing to
            a missing region, or a recording assigned to two datasets.
        """
        _check_spans(self.regions, "regions", unbounded=True)
        _check_spans(self.labels, "labels")
        if not np.isfinite(floats(self.labels["t_center_s"])).all():
            msg = "labels: t_center_s must be finite"
            raise ValueError(msg)
        bad = sorted(set(strings(self.labels["label"])) - set(LABEL_VALUES))
        if bad:
            msg = f"labels: unknown label values {bad}; allowed: {LABEL_VALUES}"
            raise ValueError(msg)
        region_ids = floats(self.regions["region"])
        if np.isnan(region_ids).any():
            msg = "regions: region ids must not be NA"
            raise ValueError(msg)
        region_recs = strings(self.regions["recording"])
        known = set(zip(region_recs, region_ids.astype(np.int64).tolist(), strict=True))
        if len(known) != len(region_recs):
            msg = "regions: duplicate region id within a recording"
            raise ValueError(msg)
        t0, t1 = floats(self.regions["t_start_s"]), floats(self.regions["t_end_s"])
        for rec in set(region_recs):
            rows = np.flatnonzero(np.asarray(region_recs, dtype=object) == rec)
            rows = rows[np.argsort(t0[rows], kind="stable")]
            if np.any(t0[rows[1:]] <= t1[rows[:-1]]):
                msg = f"regions of {rec} overlap in time"
                raise ValueError(msg)
        label_region = floats(self.labels["region"])
        has_region = ~np.isnan(label_region)
        label_recs = np.asarray(strings(self.labels["recording"]), dtype=object)
        refs = zip(
            label_recs[has_region].tolist(),
            label_region[has_region].astype(np.int64).tolist(),
            strict=True,
        )
        missing = {ref for ref in refs if ref not in known}
        if missing:
            msg = f"labels refer to missing regions {sorted(missing)}"
            raise ValueError(msg)
        pairs = set(
            zip(
                [*region_recs, *label_recs.tolist()],
                [*strings(self.regions["dataset"]), *strings(self.labels["dataset"])],
                strict=True,
            )
        )
        if len({rec for rec, _ in pairs}) != len(pairs):
            msg = "a recording belongs to more than one dataset"
            raise ValueError(msg)

    @property
    def recordings(self) -> tuple[str, ...]:
        """All recordings, sorted."""
        names = set(strings(self.regions["recording"]))
        names |= set(strings(self.labels["recording"]))
        return tuple(sorted(names))

    @property
    def datasets(self) -> dict[str, str]:
        """Dataset of every recording."""
        out = {}
        for df in (self.regions, self.labels):
            out.update(
                zip(strings(df["recording"]), strings(df["dataset"]), strict=True)
            )
        return out

    @classmethod
    def empty(cls) -> GroundTruth:
        """Ground truth without regions or labels."""
        return cls(_empty(REGION_COLUMNS), _empty(LABEL_COLUMNS))

    @classmethod
    def concat(cls, parts: Iterable[GroundTruth]) -> GroundTruth:
        """Combine ground truths, e.g. of several recordings.

        Region ids of a recording that appears in several parts are
        renumbered after the first part's. A recording whose dataset is
        :data:`UNKNOWN_DATASET` in one part takes the dataset it has in
        another.
        """
        parts = list(parts)
        if not parts:
            return cls.empty()
        regions, labels = [], []
        next_id: dict[str, int] = {}
        for part in parts:
            reg = part.regions.copy()
            lab = part.labels.copy()
            reg_recs = strings(reg["recording"])
            offsets = {rec: next_id.get(rec, 0) for rec in set(reg_recs)}
            reg_ids = floats(reg["region"]) + np.array(
                [offsets[r] for r in reg_recs], dtype=np.float64
            )
            reg["region"] = _as_int64(pd.Series(reg_ids, index=reg.index), "region")
            shift = np.array(
                [offsets.get(r, 0) for r in strings(lab["recording"])],
                dtype=np.float64,
            )
            lab_ids = floats(lab["region"]) + shift
            lab["region"] = _as_int64(pd.Series(lab_ids, index=lab.index), "region")
            for rec, rid in zip(reg_recs, reg_ids.tolist(), strict=True):
                next_id[rec] = max(next_id.get(rec, 0), int(rid) + 1)
            regions.append(reg)
            labels.append(lab)
        reg = pd.concat(regions, ignore_index=True)
        lab = pd.concat(labels, ignore_index=True)
        known: dict[str, str] = {}
        for df in (reg, lab):
            for rec, ds in zip(
                strings(df["recording"]), strings(df["dataset"]), strict=True
            ):
                if ds != UNKNOWN_DATASET:
                    known.setdefault(rec, ds)
        for df in (reg, lab):
            df["dataset"] = [
                known.get(rec, ds) if ds == UNKNOWN_DATASET else ds
                for rec, ds in zip(
                    strings(df["recording"]), strings(df["dataset"]), strict=True
                )
            ]
        return cls(reg, lab)


def infer_dataset(recording_dir: str | Path) -> str:
    """Dataset name of a recording directory: the name of its parent.

    Parent directories named ``raw`` are skipped, so both
    ``2019_tube_competition/2019-12-09-10_00`` and
    ``2022_tube_competition/raw/2022-05-12-10_00`` give
    ``*_tube_competition``.
    """
    for parent in Path(recording_dir).resolve().parents:
        if parent.name and parent.name not in _DATASET_SKIP:
            return parent.name
    return UNKNOWN_DATASET


def read_ground_truth(
    path: str | Path, *, recording: str | None = None, dataset: str | None = None
) -> GroundTruth:
    """Read an audian label file as ground truth.

    Two layouts are understood:

    - hand labels inside a recording directory
      (``traces-grid1-editable-labels.csv``): categories ``labeled_region``
      and ``chirp``; the recording is the directory's name and the dataset
      its parent's (:func:`infer_dataset`);
    - files written by ``chirpdetector label-emitters``
      (``<out>/labels/<trial>.csv``, recognised by their ``item_id``
      column): one spot label per labelled detection, with ``label`` and
      ``emitter_id``; the recording is the trial in ``item_id`` and the
      dataset is unknown (:meth:`GroundTruth.concat` fills it in from
      another part of the same recording).

    Rows of other categories are skipped (and logged).

    Parameters
    ----------
    path
        The label file.
    recording, dataset
        Override the inferred names.

    Returns
    -------
    The ground truth. Label centres are box midpoints, as in
    ``detector_benchmark``.
    """
    path = Path(path)
    table = read_labels(path)
    tool_file = "item_id" in table.columns
    if recording is None:
        if tool_file and len(table):
            trials = table["item_id"].astype(str).str.rsplit("#", n=1).str[0].unique()
            if len(trials) != 1:
                msg = f"{path}: labels of several trials {sorted(trials)}"
                raise ValueError(msg)
            recording = str(trials[0])
        elif tool_file:
            recording = path.stem
        else:
            recording = path.parent.name
    if dataset is None:
        dataset = UNKNOWN_DATASET if tool_file else infer_dataset(path.parent)
    category = np.asarray(strings(table["category"]), dtype=object)
    regions = table.loc[category == "labeled_region"]
    is_label = np.isin(category, _AUDIAN_LABEL_FILE_CATEGORIES)
    chirps = table.loc[is_label]
    skipped = sorted(set(category[~is_label & (category != "labeled_region")]))
    if skipped:
        log.info("%s: skipping rows of categories %s", path, skipped)
    label = np.asarray(strings(chirps["category"]), dtype=object)
    if "label" in chirps.columns:
        given = chirps["label"].to_numpy(dtype=object)
        named = np.array([isinstance(v, str) and v != "" for v in given], dtype=bool)
        label[named] = given[named]
    region_df = pd.DataFrame(
        {
            "recording": recording,
            "dataset": dataset,
            "region": np.arange(len(regions)),
            **{c: floats(regions[c]) for c in _BOX},
        }
    )
    label_df = pd.DataFrame(
        {
            "recording": recording,
            "dataset": dataset,
            **{c: floats(chirps[c]) for c in _BOX},
            "label": label,
        }
    )
    if "emitter_id" in chirps.columns:
        ids = pd.Series(floats(chirps["emitter_id"]), index=label_df.index)
        label_df["emitter_id"] = _as_int64(ids, f"{path}: emitter_id")
    return GroundTruth(region_df, label_df)


def _scene_duration_s(scene_dir: Path) -> float | None:
    path = scene_dir / "scene.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    for container in (data, data.get("config", {}), data.get("scene", {})):
        if isinstance(container, dict) and "duration_s" in container:
            return float(container["duration_s"])
    return None


def simulation_ground_truth(
    path: str | Path,
    *,
    recording: str | None = None,
    dataset: str = "simulation",
    duration_s: float | None = None,
    band_hz: tuple[float, float] | None = None,
    visible_column: str | None = None,
) -> GroundTruth:
    """Ground truth of a simulated scene from its ``chirps.csv``.

    The whole scene is one labelled region. Every simulated chirp becomes a
    ``chirp`` label with a known ``emitter_id``; its box spans
    ``t_onset_s``-``t_offset_s`` and ``f_low_hz``-``f_high_hz``, and its
    centre is the scene's ``t_center_s`` (not the box midpoint).

    Parameters
    ----------
    path
        ``chirps.csv`` or the scene directory containing it.
    recording
        Name of the scene; default: the scene directory's name.
    dataset
        Dataset used for pooling.
    duration_s
        Length of the scene. Default: ``duration_s`` from the scene's
        ``scene.json`` (top level, ``config`` or ``scene``) if present,
        otherwise unbounded. The region spans ``[0, duration_s]``.
    band_hz
        Frequency band of the region; predictions outside it are dropped as
        harmonics (see :class:`~chirpdetector.evaluation.matching.MatchConfig`).
        Default: unbounded, i.e. nothing is dropped.
    visible_column
        Optional boolean column; chirps where it is false become ``ignore``
        labels (neither required nor penalised).

    Raises
    ------
    ValueError
        If a required column is missing.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "chirps.csv"
    scene_dir = path.parent
    table = pd.read_csv(path)
    missing = [c for c in _SIM_COLUMNS if c not in table.columns]
    if visible_column is not None and visible_column not in table.columns:
        missing.append(visible_column)
    if missing:
        msg = f"{path}: missing columns {missing}"
        raise ValueError(msg)
    recording = recording or scene_dir.name
    if duration_s is None:
        duration_s = _scene_duration_s(scene_dir)
    t_end = np.inf if duration_s is None else float(duration_s)
    if not t_end > 0:
        msg = f"{path}: duration_s must be positive, got {duration_s}"
        raise ValueError(msg)
    f_low, f_high = (-np.inf, np.inf) if band_hz is None else band_hz
    regions = pd.DataFrame(
        {
            "recording": [recording],
            "dataset": [dataset],
            "region": [0],
            "t_start_s": [0.0],
            "t_end_s": [t_end],
            "f_low_hz": [float(f_low)],
            "f_high_hz": [float(f_high)],
        }
    )
    label = np.full(len(table), "chirp", dtype=object)
    if visible_column is not None:
        visible = table[visible_column].to_numpy(dtype=bool)
        label[~visible] = "ignore"
    tc = floats(table["t_center_s"])
    outside = (tc < 0) | (tc > t_end)
    if outside.any():
        log.warning(
            "%s: %d chirps centred outside the scene; they become spot labels",
            path,
            int(outside.sum()),
        )
    labels = pd.DataFrame(
        {
            "recording": recording,
            "dataset": dataset,
            "region": pd.Series(np.where(outside, np.nan, 0.0)).astype("Int64"),
            "t_start_s": floats(table["t_onset_s"]),
            "t_end_s": floats(table["t_offset_s"]),
            "f_low_hz": floats(table["f_low_hz"]),
            "f_high_hz": floats(table["f_high_hz"]),
            "t_center_s": tc,
            "label": label,
            "emitter_id": _as_int64(table["emitter_id"], f"{path}: emitter_id"),
        }
    )
    return GroundTruth(regions, labels)
