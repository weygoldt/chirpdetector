"""The browser labeller's HTTP server, on an ephemeral port."""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Iterator
from importlib import resources
from pathlib import Path

import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import pytest
from gridtools.io import read_labels

from chirpdetector.labeling.queue import LabelQueue
from chirpdetector.labeling.server import LabelServer, encode_png
from chirpdetector.labeling.session import LabelSession
from chirpdetector.labeling.single import SingleLabeller
from chirpdetector.labeling.store import LabelStore
from chirpdetector.labeling.view import (
    TrackView,
    colour_lut,
    colourize,
    data_to_pixel,
    pixel_to_data,
    snap,
)


class Client:
    def __init__(self, server: LabelServer) -> None:
        self.base = f"http://127.0.0.1:{server.port}"

    def request(
        self, path: str, body: object = None, host: str | None = None
    ) -> tuple[int, bytes, str]:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if host is not None:
            req.add_header("Host", host)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read(), r.headers["Content-Type"]
        except urllib.error.HTTPError as err:
            return err.code, err.read(), err.headers["Content-Type"]

    def json(self, path: str, body: object = None) -> tuple[int, dict]:
        status, raw, _ = self.request(path, body)
        return status, json.loads(raw)


def _start(queue: LabelQueue, out: Path) -> tuple[LabelStore, LabelServer, Client]:
    store = LabelStore(out, "tester", "2.0.0")
    session = LabelSession(queue, store, frontend="browser")
    server = LabelServer(session, port=0)
    server.start()
    return store, server, Client(server)


@pytest.fixture
def served(candidate_queue: LabelQueue, tmp_path: Path) -> Iterator[tuple]:
    store, server, client = _start(candidate_queue, tmp_path / "out")
    yield server, client
    server.shutdown()
    store.close()


def test_page_and_state(served) -> None:
    server, client = served
    status, html, kind = client.request("/")
    assert (status, kind) == (200, "text/html; charset=utf-8")
    assert b"BEGIN shared-math" in html
    status, state = client.json("/api/state")
    assert status == 200
    assert state["n"] == 4
    assert state["windows"] == [0.2048, 0.0512]
    assert state["start_index"] == 0
    assert state["tolerance_hz"] == 15.0
    assert state["progress"]["open"] == 4
    assert "chirp from the track" in state["help"]
    assert server.url == f"http://localhost:{server.port}/"


def test_view_payload_matches_the_session(served) -> None:
    server, client = served
    status, view = client.json("/api/view?index=0&window=0.2048")
    assert status == 200
    rendered = server.session.view(0, 0.2048)
    for name in ("overview", "zoom"):
        panel = rendered.panel(name)
        assert view["panels"][name]["extent"] == list(panel.extent)
        assert view["panels"][name]["shape"] == list(panel.db.shape)
    assert [(t["key"], t["track_id"]) for t in view["tracks"]] == [(1, 0), (2, 1)]
    assert view["tracks"][0]["t"] == rendered.tracks[0].times_s.tolist()
    assert view["prediction"]["emitter_hz"] == pytest.approx(700.4)
    assert view["decision"] is None
    assert view["next_index"] == 1
    assert view["item"]["item_id"] == "2000-01-01-10_00#chirp"
    # a tab left open across a server restart compares these with its state
    assert view["queue_id"] == server.session.queue.queue_id
    assert view["n"] == 4


def test_action_results_name_their_queue(served) -> None:
    server, client = served
    _, result = client.json(
        "/api/action", {"index": 0, "window_s": 0.2048, "action": "not_chirp"}
    )
    assert result["ok"]
    assert result["queue_id"] == server.session.queue.queue_id
    _, undone = client.json("/api/undo", {})
    assert undone["queue_id"] == server.session.queue.queue_id


def test_page_reloads_on_another_queue_and_jumps_to_open_items() -> None:
    from importlib import resources

    html = (
        resources.files("chirpdetector.labeling").joinpath("labeller.html").read_text()
    )
    assert "r.queue_id !== S.state.queue_id" in html
    assert "location.reload()" in html
    assert html.count("staleQueue(") >= 4  # defined, view, action, undo
    assert "(S.progress && S.progress.total) || S.state.n" in html  # live length
    assert 'k === "o" || k === "End"' in html
    assert '"LABELLED"' in html
    assert '"OPEN"' in html


def test_image_is_the_colour_mapped_panel(served) -> None:
    server, client = served
    _, view = client.json("/api/view?index=0&window=0.2048")
    status, png, kind = client.request(view["panels"]["zoom"]["src"])
    assert (status, kind) == (200, "image/png")
    decoded = matplotlib.image.imread(io.BytesIO(png))
    expected = colourize(
        server.session.view(0, 0.2048).zoom, colour_lut(), top_down=True
    )
    np.testing.assert_array_equal(
        np.round(decoded[..., :3] * 255).astype(np.uint8), expected
    )


def test_click_is_mapped_on_the_server(served) -> None:
    server, client = served
    _, view = client.json("/api/view?index=0&window=0.2048")
    extent = view["panels"]["zoom"]["extent"]
    width, height = 1531.5, 612.0  # CSS pixels of the panel rectangle
    x, y = data_to_pixel(3.86, 701.25, width, height, extent)
    status, result = client.json(
        "/api/action",
        {
            "index": 0,
            "window_s": 0.2048,
            "prediction_shown": False,
            "action": "click",
            "panel": "zoom",
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        },
    )
    assert status == 200
    assert result["ok"]
    assert (result["t_s"], result["f_hz"]) == pytest.approx((3.86, 701.25), abs=1e-9)
    assert (result["t_s"], result["f_hz"]) == pixel_to_data(x, y, width, height, extent)
    assert result["decision"]["emitter_id"] == 0
    assert result["decision"]["emitter_baseline_hz"] == 700.0
    assert result["next_index"] == 1
    assert result["progress"]["labelled"] == 1
    decision = server.session.decision(0)
    assert decision.frontend == "browser"
    assert decision.click_offset_hz == pytest.approx(1.25, abs=1e-9)


def test_actions_undo_and_errors(served) -> None:
    server, client = served
    s = server.session

    def act(**body):
        return client.json("/api/action", {"window_s": 0.2048, **body})

    assert act(index=0, action="pick", key=2)[1]["decision"]["emitter_id"] == 1
    assert act(index=1, action="not_chirp", via="mouse")[1]["ok"]
    assert s.decision(1).via == "mouse"
    assert act(index=2, action="unclear")[1]["decision"]["label"] == (
        "chirp_emitter_unclear"
    )
    _, skipped = act(index=3, action="skip")
    assert skipped["next_index"] == 3
    assert s.status(3) == "skipped"
    _, far = act(
        index=3, action="click", panel="overview", x=1, y=1, width=10, height=10
    )
    assert not far["ok"]  # one track at 700 Hz; the top corner is ~820 Hz
    _, undone = client.json("/api/undo", {})
    assert (undone["ok"], undone["index"]) == (True, 3)
    assert s.status(3) == "open"
    _, view = client.json("/api/view?index=0&window=0.0512")
    assert view["decision"]["text"] == "chirp -> 2 (id 1)"
    assert view["window_s"] == 0.0512

    for body, message in [
        ({"index": 9, "action": "skip"}, "outside"),
        ({"index": 0, "action": "dance"}, "unknown action"),
        ({"index": 0, "action": "skip", "window_s": 0.3}, "window must be"),
        ({"index": 0, "action": "pick"}, "integer key"),
        (
            {
                "index": 0,
                "action": "click",
                "panel": "zoom",
                "x": 20,
                "y": 5,
                "width": 10,
                "height": 10,
            },
            "outside the panel",
        ),
        ({"index": 0, "action": "click", "panel": "top"}, "panel must be"),
    ]:
        status, result = client.json("/api/action", {"window_s": 0.2048, **body})
        assert status == 400, body
        assert message in result["message"]
    assert client.json("/api/view?index=x")[0] == 400
    assert client.request("/api/image?index=0&window=0.2048&panel=x")[0] == 400
    assert client.request("/nothing")[0] == 404


def test_foreign_host_headers_are_refused(served) -> None:
    _, client = served
    status, _, _ = client.request("/api/state", host="evil.example:8765")
    assert status == 403
    assert client.request("/api/state", host="localhost:8765")[0] == 200


def test_frontends_share_queue_and_labels(
    candidate_queue: LabelQueue, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    with LabelStore(out, "tester", "2") as store:  # matplotlib first
        lab = SingleLabeller(LabelSession(candidate_queue, store, background=False))
        lab.on_key(type("E", (), {"key": "1"})())
        plt.close(lab.fig)
    store, server, client = _start(candidate_queue, out)  # then the browser
    try:
        _, state = client.json("/api/state")
        assert state["start_index"] == 1
        assert state["progress"]["labelled"] == 1
        client.json(
            "/api/action", {"index": 1, "window_s": 0.2048, "action": "pick", "key": 1}
        )
    finally:
        server.shutdown()
        store.close()
    with LabelStore(out, "tester", "2") as store:  # and matplotlib again
        lab = SingleLabeller(LabelSession(candidate_queue, store, background=False))
        assert lab.index == 2
        lab.on_key(type("E", (), {"key": "u"})())  # undo reaches the browser's label
        assert lab.index == 1
        plt.close(lab.fig)
    labels = read_labels(out / "labels" / "2000-01-01-10_00.csv")
    assert labels["frontend"].tolist() == ["mpl"]


def test_encode_png_round_trip() -> None:
    rgb = np.random.default_rng(0).integers(0, 256, (7, 13, 3), dtype=np.uint8)
    decoded = matplotlib.image.imread(io.BytesIO(encode_png(rgb)))
    np.testing.assert_array_equal(np.round(decoded * 255).astype(np.uint8), rgb)
    with pytest.raises(ValueError, match="uint8"):
        encode_png(np.zeros((2, 2, 3)))  # pyright: ignore[reportArgumentType]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_page_math_matches_python() -> None:
    """The page's pixel maps, interpolation and snapping equal the server's."""
    html = (
        resources.files("chirpdetector.labeling").joinpath("labeller.html").read_text()
    )
    match = re.search(r"// BEGIN shared-math.*?// END shared-math", html, re.S)
    assert match is not None
    rng = np.random.default_rng(1)
    tracks = [
        TrackView(
            1, 4, 700.0, np.array([3.7, 3.9, 4.3]), np.array([700.0, 701.0, 699.0])
        ),
        TrackView(2, 9, 703.0, np.array([3.7, 4.3]), np.array([703.0, 706.0])),
        TrackView(3, 2, 760.0, np.array([4.0]), np.array([760.0])),
    ]
    extent = (3.7, 4.3, 690.0, 770.0)
    cases = [
        (float(rng.uniform(0, 1500)), float(rng.uniform(0, 600)), 1500.0, 600.0)
        for _ in range(300)
    ]
    payload = {
        "tracks": [
            {"key": t.key, "t": t.times_s.tolist(), "f": t.freqs_hz.tolist()}
            for t in tracks
        ],
        "extent": list(extent),
        "cases": cases,
    }
    script = (
        match.group(0)
        + """
const P = JSON.parse(require("fs").readFileSync(0, "utf8"));
const out = P.cases.map(([x, y, w, h]) => {
  const [t, f] = pixelToData(x, y, w, h, P.extent);
  const [xb, yb] = dataToPixel(t, f, w, h, P.extent);
  const s = snapTrack(P.tracks, t, f, 2.0);
  return [t, f, xb, yb, s ? s.track.key : null, s ? s.offset : null];
});
process.stdout.write(JSON.stringify(out));
"""
    )
    result = subprocess.run(
        ["node", "-e", script],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    snapped = 0
    for (x, y, w, h), (t, f, xb, yb, key, offset) in zip(
        cases, json.loads(result.stdout), strict=True
    ):
        assert (t, f) == pixel_to_data(x, y, w, h, extent)  # bit-identical
        assert (xb, yb) == data_to_pixel(t, f, w, h, extent)
        expected = snap(tuple(tracks), t, f, 2.0)
        assert key == (expected.track.key if expected else None)
        if expected is not None:
            snapped += 1
            assert offset == pytest.approx(expected.offset_hz, abs=1e-12)
    assert snapped > 10
