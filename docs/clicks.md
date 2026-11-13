# Click removal

Electrodes pick up clicks: impulses of a few milliseconds, up to four orders
of magnitude above the noise, on one or several electrodes. In a
spectrogram a click is a vertical stripe across all frequencies, and
chirpdetector v1 often took them for chirps.
`chirpdetector.preprocessing.clicks` repairs them before the spectrogram,
in the pipeline, in `predict_span` and in the dataset builders alike (the
bundle records whether and how).

It is a port of audian's click removal (itself a port of wavetracker's),
with the same detection and repair:

- clicks are searched above 4 kHz (`high_band_hz`, at any sampling rate),
  where a click reaches and a fish's narrowband EOD does not; loud events
  longer than 0.5 s are signals and stay;
- what stands out of the spectral floor (EODs, harmonics, hum) is set
  aside, the clicks are bridged by straight lines in the rest, and the tonal
  part is added back, scaled to what the samples around each hole hold.

Samples outside a click are returned unchanged.

## The electrode-coincidence guard

On grid recordings, the higher EOD harmonics of a fish next to an electrode
reach the click band during a big chirp, so per-channel removal repairs
chirps (24 of the 335 hand-labelled chirps were touched, by up to -11 dB).
Real clicks reach most electrodes (8-14 of 15), chirp-triggered events at
most 4. chirpdetector therefore repairs an event on all electrodes only
when it reaches at least half of them (`coincidence = 0.5`), and otherwise
only where it is very strong (`isolated_threshold`).

This needs enough electrodes: below 9 (`ClickRemoval().min_electrodes`), a
loud chirp could pass the coincidence test, and on one channel clicks and
chirps cannot be told apart. Click removal therefore refuses such
recordings, and the pipeline, `predict_span` and the dataset builders skip
it with a warning ("too few electrodes for click removal; skipped"). The
simulated training scenes have 1-16 electrodes, so the detector learned to
reject the clicks it sees where removal is not possible.

## Use

```python
from chirpdetector.preprocessing import ClickRemoval, can_remove_clicks, remove_clicks

config = ClickRemoval()
# samples: (n_samples, n_electrodes) float array
if can_remove_clicks(samples.shape[1], config, samplerate_hz=samplerate_hz):
    cleaned, mask = remove_clicks(
        samples, samplerate_hz, config
    )  # mask: repaired samples
```

Recordings are cleaned in blocks fixed in absolute time (2 s plus 0.5 s
margins), so a sample comes out the same however the recording is chunked;
`read_clean` and `plan_span` do this for spans of long recordings. The
pipeline reports the repaired fraction per chunk (`chunks_<model>.parquet`),
per box (`click_fraction_box`) and over time (QC report).
