# Labelling chirps and their emitters

`chirpdetector label-emitters` shows one candidate box at a time and
records whether it is a chirp and which fish emitted it. It runs as a
maximised matplotlib window or as a small web server for a browser (for
labelling over SSH). Both frontends share the queue, the snapping of clicks
to tracks, the stored fields and undo, and a session started in one resumes
in the other. Needs the `label` extra.

```sh
# matplotlib window on this machine
chirpdetector label-emitters --candidates candidates.csv --out labels --labeler NAME
# the same in a browser
chirpdetector label-emitters --candidates candidates.csv --out labels --labeler NAME --serve --port 8765
# counts by decision, candidate source and ΔF bin (also while a session runs)
chirpdetector label-summary labels
```

`--out` is required (the same command with the same `--out` resumes);
`--labeler` defaults to your login name.

## The view

The top panel shows the whole chirp (2 s around the box, 500 Hz around the
box and the baselines near it); `w` switches its analysis window between
204.8 ms (separates baselines a few Hz apart) and 51.2 ms (shows the chirp's
shape). The bottom panel zooms on the baselines of the fish near the box,
always with the 204.8-ms window and zero-padded so that a few Hz span many
pixels. Every live track is marked at both edges with its number, 1, 2, ...
from the lowest baseline; tracks are bridged across wavetracker gaps of up to
`--track-gap-s` (10 s) and the gap is shown. Numbers depend only on the
tracks, never on a model's prediction, and the model's emitter is hidden
unless you press `e` (recorded with each label as `prediction_shown`).

Click on the baseline the chirp comes from: the click snaps to the
highlighted track (the nearest one at the clicked time within
`--tolerance-hz`, 15 Hz); a click farther from every track is rejected,
not guessed.

## Keys

| input | action |
|---|---|
| left click on a baseline, or `1`-`9` | chirp from that track |
| `0` or middle click | chirp, emitter unclear |
| `x` or right click | not a chirp |
| `s` or space | skip (the item comes back after the open ones) |
| `u`, backspace (browser also ctrl+z) | undo the last action (jumps to its item) |
| left / right arrow | previous / next item |
| `o` or end | next open item |
| `w` | top panel window 204.8 / 51.2 ms |
| `e` | show / hide the model's predicted emitter |
| `f`, `h` | full screen, help |
| `q` | quit (matplotlib; every action is already saved) |

## Labelling from another computer

The server binds to 127.0.0.1 only. Forward its port over SSH and open the
page locally:

```sh
# on the machine with the data
chirpdetector label-emitters --candidates candidates.csv --out labels --labeler NAME --serve --port 8765
# on your laptop
ssh -N -L 8765:localhost:8765 USER@HOST
# then open http://localhost:8765/
```

The server prints the matching `ssh` line when it starts. Clicks are sent
as positions in the panel and converted to time and frequency on the
server, so a label never depends on the browser's arithmetic. A tab left
open on a stale queue reloads itself.

## Candidates file

Any list of boxes, as CSV or Parquet, one row per box. The format is stable;
the production pipeline writes its validation sample (`candidates.csv`) in
it.

| column | | meaning |
|---|---|---|
| `recording` | required | recording directory (relative paths: relative to the candidates file) |
| `t_start_s`, `t_end_s` | required | box in seconds from the recording's start (not negative) |
| `f_low_hz`, `f_high_hz` | required | box in Hz |
| `candidate_id` | optional | id of the box within its recording (no `#`); the same id in another file is the same item and reuses its label. Default: a hash of recording name and box |
| `score` | optional | detector score |
| `emitter_hz` | optional | the model's predicted emitter EODf |
| `track_id` | optional | the model's predicted emitter track |
| `source` | optional | free text: model, run, stratum |
| `tracks` | optional | wavetracker directory, if not the recording directory |

Other columns are stored with each label as `candidate_<name>`. `score`,
`emitter_hz`, `track_id` and `source` stay hidden unless the labeller shows
the prediction. The queue keeps every row, shuffled with `--seed`
(`--keep-order` keeps the file's order), and is saved in
`<out>/queues/<queue_id>.json`; the same command resumes the same queue.

Instead of `--candidates`, trial directories with v1 detections
(`chirpdetector_bboxes.csv`) can be given; the queue then samples
`--per-bin` detections per trial and ΔF bin (`--bins`). `--view grid` shows
pages of small cutouts instead (matplotlib only).

## Output

Every action is appended to `<out>/events.jsonl` and flushed to disk before
the call returns; undo appends an event, the log is never edited. One label
file per recording, `<out>/labels/<recording>.csv`, is derived from it after
every action: one row per labelled item and labeller, readable by
`chirpdetector evaluate --labels`. Columns: the audian box columns, then

| column | meaning |
|---|---|
| `label` | `chirp`, `not_chirp` or `chirp_emitter_unclear` |
| `emitter_id` | track id of the emitter (NA unless a track was picked) |
| `clicked_hz`, `clicked_t_s` | where you clicked (NA for keyboard picks) |
| `labeler`, `labeled_at`, `tool_version`, `queue_id`, `item_id`, `det_index`, `df_bin` | provenance |
| `emitter_baseline_hz` | the emitter track's frequency at the box start, from the track |
| `click_offset_hz` | click minus the emitter track's frequency at the click time |
| `via`, `frontend` | `mouse`/`key`; `mpl`, `mpl-grid` or `browser` |
| `window_s`, `prediction_shown` | window shown, whether the model's emitter was visible |
| `n_candidates`, `delta_f_hz` | live tracks at the box centre, ΔF of the closest two |
| `recording_dir`, `tracks_dir` | where the data were read |
| `elapsed_s`, `blind` | seconds to the decision, blind session or not |
| `candidate_id`, `candidate_score`, `candidate_emitter_hz`, `candidate_track_id`, `candidate_source`, `candidate_*` | the candidates file's columns |

Only one session per output directory runs at a time (a lock on the log).

## Blind mode

`--blind` hides everything that could bias the labeller: no box, only a time
marker at its centre (with `--blind-hint`, the default, a faint fixed-size
box of 0.3 s x 200 Hz at the candidate, showing where to look but not its
shape), no score, source, recording name or predicted emitter (`e` is
disabled, and the server never sends them). The time per item is stored.
The [human benchmark](benchmark.md) labels this way.

> **Verified.** The browser server was run on the pipeline's
> `candidates.csv` of a simulated scene (port 8781), in normal and blind
> mode; labels were posted through its HTTP API (`/api/action`,
> `/api/undo`) and `label-summary` counted them.
