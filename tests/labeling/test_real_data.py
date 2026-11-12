"""Smoke test on a real recording (needs /mnt/data2; marked ``data``)."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from chirpdetector.labeling.gui import Labeller
from chirpdetector.labeling.queue import QueueConfig, load_or_build_queue
from chirpdetector.labeling.store import LabelStore

TRIAL = Path("/mnt/data2/2019_tube_competition/2019-12-09-10_00")


@pytest.mark.data
def test_real_trial_queue_and_prefetched_page_switch(tmp_path: Path) -> None:
    queue = load_or_build_queue([TRIAL], QueueConfig(per_bin=20, seed=0), tmp_path)
    counts = queue.counts_per_bin()
    assert counts["0-10"] == 20
    assert counts["single"] == 20
    with LabelStore(tmp_path, "test", "2") as store:
        fig = plt.figure(figsize=(18, 9.6), dpi=100)
        labeller = Labeller(queue, {TRIAL.name: TRIAL}, store, figure=fig)
        fig.canvas.draw()  # warm-up (fonts, caches)
        switch_s = []
        for page in (1, 2, 3):  # the machine is shared: judge the fastest switch
            labeller._page_future(page).result()  # wait for the prefetch
            tic = time.perf_counter()
            labeller.next_page()  # with Agg, draw_idle draws synchronously
            switch_s.append(time.perf_counter() - tic)
        for slot, cut in zip(labeller.slots, labeller.cutouts, strict=False):
            assert cut is not None
            n_freq, n_time = cut.db.shape
            extent = slot.image.get_extent()
            dt, df = (extent[1] - extent[0]) / n_time, (extent[3] - extent[2]) / n_freq
            np.testing.assert_allclose(extent[:2], cut.window[:2], atol=dt)
            np.testing.assert_allclose(extent[2:], cut.window[2:], atol=df)
        labeller.shutdown()
        plt.close(fig)
    # Target is < 0.3 s; the bound is loose because the test machine is shared
    # and often saturated. It still catches losing the prefetch (~1 s+).
    assert min(switch_s) < 1.0, f"page switches took {switch_s} s"
