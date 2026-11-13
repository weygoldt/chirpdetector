"""Lookalike gallery: real boxes next to simulated true chirps like them.

For every real candidate the labeller called ``not_chirp``, the gallery
shows the real cutout beside ``n_matches`` simulated true chirps (visible,
matched by the detector) with similar decoded parameters and similar
contrast, all rendered with the labeller's single view
(:mod:`chirpdetector.labeling.view`: same span, same percentile colour
scaling), the simulated truth annotated.

Similarity
----------
- *Parameters*: the chirp-parameter decoder's calibrated ``height_hz`` and
  ``fwhm_s`` for the box (the real candidate's own box; the simulated
  chirp's matched detection box), compared on a log scale.
- *Contrast* (an SNR proxy; the decoder estimates no SNR, and a real box
  has no counterfactual footprint): :func:`box_contrast_db`, computed the
  same way on real and simulated recordings. The simulated chirps' true
  ``snr_db`` is annotated next to it.

Distance: ``(ln(h/h0) / ln(1.5))^2 + (ln(w/w0) / ln(1.5))^2 + ((c - c0) /
3 dB)^2``; the matches come from distinct scenes. Scenes listed in
``exclude`` (the human benchmark's) are never shown, so the gallery cannot
reveal benchmark answers.
"""

from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from gridtools.io import Recording, Tracks, load_tracks, open_recording

from chirpdetector.labeling.queue import QueueItem
from chirpdetector.labeling.render import band_spectrogram
from chirpdetector.labeling.view import ViewConfig, colour_lut, colourize, render_view

__all__ = [
    "GalleryConfig",
    "box_contrast_db",
    "estimate_boxes",
    "match_lookalikes",
    "real_cases",
    "sim_pool",
    "write_gallery",
]

_BOX = ("t_start_s", "t_end_s", "f_low_hz", "f_high_hz")
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class GalleryConfig:
    """Matching and rendering."""

    n_matches: int = 3
    height_scale: float = math.log(1.5)
    fwhm_scale: float = math.log(1.5)
    contrast_scale_db: float = 3.0
    contrast_window_s: float = 0.2048
    contrast_margin_s: float = 0.5
    tolerance_s: float = 0.1


def box_contrast_db(
    rec: Recording,
    box: Sequence[float],
    *,
    window_s: float = 0.2048,
    margin_s: float = 0.5,
) -> float:
    """How much a box stands out from the same band around it, in dB.

    Electrode-summed spectrogram of the box's band from ``margin_s`` before
    to ``margin_s`` after the box: per frequency row, the maximum inside the
    box minus the median outside it; the 90th percentile over rows (the
    rows the chirp's excursion crosses). NaN if a side is missing.
    """
    t0, t1, f0, f1 = (float(v) for v in box)
    lo = max(0.0, t0 - margin_s)
    hi = min(rec.duration_s, t1 + margin_s)
    spec = band_spectrogram(
        rec, (lo, hi, f0, f1), window_s=window_s, hop_s=0.0016, fft_pad=2
    )
    n_time = spec.db.shape[1]
    e = spec.extent
    times = e[0] + (np.arange(n_time) + 0.5) * (e[1] - e[0]) / n_time
    inside = (times >= t0) & (times <= t1)
    if inside.all() or not inside.any():
        return math.nan
    diff = spec.db[:, inside].max(axis=1) - np.median(spec.db[:, ~inside], axis=1)
    return float(np.percentile(diff, 90))


def real_cases(
    labels: pd.DataFrame,
    candidates: pd.DataFrame,
    label: str = "not_chirp",
    labeler: str | None = None,
) -> pd.DataFrame:
    """Candidates (``read_candidates`` table) a labeller gave ``label``.

    Returns the candidate rows plus ``item_id``, ``labeler`` and
    ``labeled_at`` of the label.
    """

    def text(df: pd.DataFrame, column: str) -> list[str]:
        if column not in df.columns:
            return [""] * len(df)
        return [
            "" if v is None or v != v else str(v)
            for v in df[column].to_numpy(dtype=object)
        ]

    keep = [
        lab == label and (labeler is None or who == labeler)
        for lab, who in zip(text(labels, "label"), text(labels, "labeler"), strict=True)
    ]
    chosen = pd.DataFrame(
        {
            column: [v for v, k in zip(text(labels, column), keep, strict=True) if k]
            for column in ("item_id", "labeler", "labeled_at")
        }
    )
    ids = [
        f"{n}#{c}"
        for n, c in zip(
            text(candidates, "_name"), text(candidates, "candidate_id"), strict=True
        )
    ]
    out = candidates.assign(item_id=ids).merge(chosen, on="item_id", how="inner")
    return out.sort_values(["_name", "t_start_s"]).reset_index(drop=True)


def estimate_boxes(
    boxes: pd.DataFrame,
    recordings: Mapping[str, Recording],
    detector: str | Path,
    params: str | Path,
    *,
    device: str | None = None,
    name_column: str = "_name",
) -> pd.DataFrame:
    """Decoded chirp parameters of given boxes (the production decoder).

    The decoder (`chirpdetector.params.ParamsEstimator`) runs on the
    detector bundle's spectrogram of each box's surroundings, exactly as in
    `chirpdetector.inference.predict_span`, but for these boxes rather than
    fresh detections.
    """
    from chirpdetector.bundle import load_bundle
    from chirpdetector.inference.spans import span_spectrogram
    from chirpdetector.params.inference import ESTIMATE_COLUMNS, ParamsEstimator

    bundle = load_bundle(detector)
    estimator = ParamsEstimator(params, device=device)
    estimator.check_detector(bundle, allow_mismatch=False)
    rows = []
    for i in range(len(boxes)):
        row = boxes.iloc[[i]]
        rec = recordings[str(row[name_column].iloc[0])]
        fs = rec.samplerate_hz
        t0, t1 = float(row["t_start_s"].iloc[0]), float(row["t_end_s"].iloc[0])
        pad = estimator.context_s + 0.5
        start = max(0, int((t0 - pad) * fs))
        stop = min(rec.n_samples, int((t1 + pad) * fs))
        spec = span_spectrogram(bundle, rec, start, stop, device=device)
        rows.append(estimator.estimate(spec, row.loc[:, list(_BOX)]))
    if not rows:
        return pd.DataFrame(columns=list(ESTIMATE_COLUMNS))
    out = pd.concat(rows)
    out.index = boxes.index
    return out


def sim_pool(
    scene_dirs: Sequence[str | Path],
    detections: Mapping[str, pd.DataFrame],
    *,
    exclude: set[str] | None = None,
    tolerance_s: float = 0.1,
) -> pd.DataFrame:
    """Visible simulated chirps matched by a detection with decoded parameters.

    One row per chirp: scene, recording, the chirp's truth (``emitter_id``,
    ``excursion_hz``, ``duration_s``, ``snr_db``, ``dip``, ``baseline_hz``),
    the matched detection's box, ``score`` and decoder columns. Scenes in
    ``exclude`` are skipped.
    """
    from chirpdetector.labeling.benchmark import _kept, _load_chirps
    from chirpdetector.params.samples import match_predictions

    rows = []
    for directory in sorted(Path(d) for d in scene_dirs):
        if exclude and directory.name in exclude:
            continue
        det = detections.get(directory.name)
        if det is None or not len(det) or "height_hz" not in det.columns:
            continue
        det = _kept(det)
        chirps = _load_chirps(directory)
        index, _ = match_predictions(det, chirps, tolerance_s)
        for d, c in enumerate(index):
            if (
                c < 0
                or not bool(chirps["visible"].iloc[c])
                or bool(chirps["truncated"].iloc[c])
            ):
                continue
            chirp, hit = chirps.iloc[c], det.iloc[d]
            rows.append(
                {
                    "scene": directory.name,
                    "recording": str(directory),
                    "chirp_id": int(chirp["chirp_id"])
                    if "chirp_id" in chirp
                    else int(c),
                    "emitter_id": int(chirp["emitter_id"]),
                    "baseline_hz": float(chirp["baseline_hz"]),
                    "excursion_hz": float(chirp["excursion_hz"]),
                    "duration_s": float(chirp["duration_s"]),
                    "snr_db": float(chirp["snr_db"]),
                    "dip": float(chirp["dip"]),
                    **{c2: float(hit[c2]) for c2 in _BOX},
                    "score": float(hit["score"]),
                    "height_hz": float(hit["height_hz"]),
                    "fwhm_s": float(hit["fwhm_s"]),
                    "params_ok": bool(hit["params_ok"]) if "params_ok" in hit else True,
                }
            )
    return pd.DataFrame(rows)


def _distance(
    pool: pd.DataFrame,
    height: float,
    fwhm: float,
    contrast: float,
    config: GalleryConfig,
) -> FloatArray:
    h = pool["height_hz"].to_numpy(dtype=np.float64)
    w = pool["fwhm_s"].to_numpy(dtype=np.float64)
    c = pool["contrast_db"].to_numpy(dtype=np.float64)
    d = (np.log(np.maximum(h, 1e-3) / max(height, 1e-3)) / config.height_scale) ** 2
    d += (np.log(np.maximum(w, 1e-6) / max(fwhm, 1e-6)) / config.fwhm_scale) ** 2
    if math.isfinite(contrast):
        d += np.where(
            np.isfinite(c), ((c - contrast) / config.contrast_scale_db) ** 2, np.inf
        )
    return np.where(np.isfinite(d), d, np.inf)


def match_lookalikes(
    case: Mapping[str, Any], pool: pd.DataFrame, config: GalleryConfig
) -> pd.DataFrame:
    """The ``n_matches`` nearest pool chirps of a case, from distinct scenes."""
    d = _distance(
        pool,
        float(case["height_hz"]),
        float(case["fwhm_s"]),
        float(case["contrast_db"]),
        config,
    )
    order = np.argsort(d, kind="stable")
    chosen: list[int] = []
    scenes: set[str] = set()
    for i in order:
        if not np.isfinite(d[i]):
            break
        scene = str(pool["scene"].iloc[i])
        if scene in scenes:
            continue
        chosen.append(int(i))
        scenes.add(scene)
        if len(chosen) == config.n_matches:
            break
    out = pool.iloc[chosen].copy()
    out["distance"] = d[chosen]
    return out.reset_index(drop=True)


def _item(box: Sequence[float], name: str) -> QueueItem:
    return QueueItem(
        item_id=f"{name}#gallery",
        trial=name,
        det_index=0,
        t_start_s=float(box[0]),
        t_end_s=float(box[1]),
        f_low_hz=float(box[2]),
        f_high_hz=float(box[3]),
        n_candidates=0,
        delta_f_hz=None,
        df_bin="",
    )


def _panels(
    ax_top: Any,
    ax_bottom: Any,
    rec: Recording,
    tracks: Tracks,
    box: Sequence[float],
    title: str,
    lines: Sequence[str],
    view_config: ViewConfig,
    lut: npt.NDArray[np.uint8],
) -> None:
    from matplotlib.patches import Rectangle

    view = render_view(rec, tracks, _item(box, title), view_config)
    for ax, panel in ((ax_top, view.overview), (ax_bottom, view.zoom)):
        ax.imshow(
            colourize(panel, lut),
            origin="lower",
            aspect="auto",
            extent=panel.extent,
            interpolation="antialiased",
        )
        ax.add_patch(
            Rectangle(
                (box[0], box[2]),
                box[1] - box[0],
                box[3] - box[2],
                fill=False,
                edgecolor="white",
                lw=0.9,
                ls="--",
            )
        )
        for track in view.tracks:
            ax.plot(
                track.times_s,
                track.freqs_hz,
                color="#d0d0d0",
                lw=0.6,
                ls=":",
                alpha=0.7,
            )
        ax.set_xlim(panel.extent[0], panel.extent[1])
        ax.set_ylim(panel.extent[2], panel.extent[3])
        ax.tick_params(labelsize=7)
    ax_top.set_title(title, fontsize=9, loc="left")
    ax_bottom.text(
        0.01,
        0.02,
        "\n".join(lines),
        transform=ax_bottom.transAxes,
        fontsize=7.5,
        family="monospace",
        color="white",
        va="bottom",
        bbox={"facecolor": "black", "alpha": 0.6, "edgecolor": "none"},
    )


def _fmt(value: object, spec: str) -> str:
    try:
        number = float(value)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        return "-"
    return "-" if not math.isfinite(number) else format(number, spec)


def write_gallery(
    cases: pd.DataFrame,
    pool: pd.DataFrame,
    out_dir: str | Path,
    *,
    config: GalleryConfig | None = None,
    view_config: ViewConfig | None = None,
) -> Path:
    """One PNG per case plus ``index.html`` and ``index.csv``.

    Parameters
    ----------
    cases
        Real cases (:func:`real_cases`) with decoder columns
        (:func:`estimate_boxes`) and ``contrast_db``.
    pool
        Simulated chirps (:func:`sim_pool`) with ``contrast_db``.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    config = config or GalleryConfig()
    view_config = view_config or ViewConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lut = colour_lut()
    recordings: dict[str, Recording] = {}
    tracks: dict[str, Tracks] = {}

    def source(rec_dir: str, track_dir: str) -> tuple[Recording, Tracks]:
        if rec_dir not in recordings:
            recordings[rec_dir] = open_recording(rec_dir)
            tracks[rec_dir] = load_tracks(track_dir)
        return recordings[rec_dir], tracks[rec_dir]

    index_rows = []
    for k, case in enumerate(cases.to_dict("records"), start=1):
        matches = match_lookalikes(case, pool, config)
        n = 1 + len(matches)
        fig, axes = plt.subplots(
            2,
            n,
            figsize=(5.2 * n, 7.2),
            squeeze=False,
            gridspec_kw={"height_ratios": [1.0, 1.0]},
        )
        box = [float(case[c]) for c in _BOX]
        rec, trk = source(str(case["recording"]), str(case["tracks"]))
        _panels(
            axes[0, 0],
            axes[1, 0],
            rec,
            trk,
            box,
            f"REAL {case['item_id']}",
            [
                f"labelled: not_chirp ({case.get('labeler', '')})",
                f"t {box[0]:.2f}-{box[1]:.2f} s  model score {_fmt(case.get('score'), '.2f')}",
                f"source {str(case.get('source', ''))[:38]}",
                f"decoder: height {_fmt(case['height_hz'], '.0f')} Hz, FWHM {_fmt(1e3 * float(case['fwhm_s']), '.1f')} ms",
                f"contrast {_fmt(case['contrast_db'], '.1f')} dB  params_ok {case.get('params_ok')}",
            ],
            view_config,
            lut,
        )
        sims = []
        for j, sim in enumerate(matches.to_dict("records"), start=1):
            sbox = [float(sim[c]) for c in _BOX]
            srec, strk = source(str(sim["recording"]), str(sim["recording"]))
            _panels(
                axes[0, j],
                axes[1, j],
                srec,
                strk,
                sbox,
                f"SIMULATED true chirp {sim['scene']} #{sim['chirp_id']}",
                [
                    f"TRUTH: chirp of fish {sim['emitter_id']} ({sim['baseline_hz']:.0f} Hz)",
                    f"excursion {sim['excursion_hz']:.0f} Hz, duration {1e3 * sim['duration_s']:.0f} ms",
                    f"SNR {sim['snr_db']:.1f} dB, amplitude dip {sim['dip']:.2f}",
                    f"decoder: height {sim['height_hz']:.0f} Hz, FWHM {1e3 * sim['fwhm_s']:.1f} ms",
                    f"contrast {_fmt(sim['contrast_db'], '.1f')} dB  model score {sim['score']:.2f}",
                ],
                view_config,
                lut,
            )
            sims.append(sim)
        fig.suptitle(
            f"case {k}: real 'not_chirp' (left) next to the {len(matches)} most similar "
            "simulated true chirps (decoded height and FWHM, contrast); same view and colour scaling",
            fontsize=10,
        )
        fig.tight_layout()
        safe = str(case["item_id"]).replace("#", "_")
        name = f"case_{k:02d}_{safe}.png"
        fig.savefig(out / name, dpi=80)
        plt.close(fig)
        row: dict[str, Any] = {
            "case": k,
            "png": name,
            "item_id": case["item_id"],
            "recording": case["recording"],
            **{c: case[c] for c in _BOX},
            "score": case.get("score"),
            "source": case.get("source"),
            "height_hz": case["height_hz"],
            "fwhm_s": case["fwhm_s"],
            "contrast_db": case["contrast_db"],
            "params_ok": case.get("params_ok"),
        }
        for j, sim in enumerate(sims, start=1):
            for key in (
                "scene",
                "chirp_id",
                "excursion_hz",
                "duration_s",
                "snr_db",
                "dip",
                "height_hz",
                "fwhm_s",
                "contrast_db",
                "distance",
            ):
                row[f"sim{j}_{key}"] = sim[key]
        index_rows.append(row)
    index = pd.DataFrame(index_rows)
    index.to_csv(out / "index.csv", index=False)
    parts = [
        "<!doctype html><meta charset='utf-8'><title>lookalikes</title>",
        "<style>body{font:14px sans-serif;margin:16px}td{vertical-align:top;padding:6px}"
        "img{width:100%;max-width:1600px}</style>",
        "<h1>Real 'not_chirp' labels next to similar simulated true chirps</h1>",
        "<p>Each case: the real box (left, labelled not_chirp) and the simulated true "
        "chirps (visible, found by the detector) nearest in decoded height and FWHM "
        "and in box contrast (an SNR proxy, computed the same way on both). Same "
        "labeller view and percentile colour scaling. No scene of the human benchmark "
        "is shown.</p>",
    ]
    for row in index_rows:
        sims_text = "; ".join(
            f"{row.get(f'sim{j}_scene')}: exc {_fmt(row.get(f'sim{j}_excursion_hz'), '.0f')} Hz, "
            f"SNR {_fmt(row.get(f'sim{j}_snr_db'), '.1f')} dB"
            for j in range(1, config.n_matches + 1)
            if f"sim{j}_scene" in row
        )
        parts.append(
            f"<h2>case {row['case']}: {html.escape(str(row['item_id']))}</h2>"
            f"<p>decoder height {_fmt(row['height_hz'], '.0f')} Hz, FWHM "
            f"{_fmt(1e3 * float(row['fwhm_s']), '.1f')} ms, contrast "
            f"{_fmt(row['contrast_db'], '.1f')} dB; model score {_fmt(row['score'], '.2f')}. "
            f"Simulated: {html.escape(sims_text)}</p>"
            f"<a href='{row['png']}'><img src='{row['png']}' alt='case {row['case']}'></a>"
        )
    (out / "index.html").write_text("\n".join(parts))
    return out / "index.html"
