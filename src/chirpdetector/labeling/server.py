"""Browser labeller: a small local HTTP server (``label-emitters --serve``).

The server binds to 127.0.0.1 by default and is meant to be reached
through an SSH port-forward (``ssh -N -L 8765:localhost:8765 host``). It
uses only the standard library (:mod:`http.server`); the page is one HTML
file without a build step (``labeller.html``).

The page draws the spectrogram panels (PNG, one image pixel per spectrogram
bin) stretched into rectangles on a canvas, and the track overlays from the
view's JSON. A left click is sent as its position inside the panel
rectangle and the rectangle's size, in CSS pixels; the *server* maps it to
time and frequency (:func:`chirpdetector.labeling.view.pixel_to_data`, the
image's outer pixel edges being the rectangle's edges) and snaps it, so the
stored label never depends on the page's arithmetic. The page repeats the
same arithmetic only to highlight the track under the mouse.

Endpoints
---------
``GET /``
    The page.
``GET /api/state``
    Queue, labeller, tolerance, windows, start item, progress, help text.
``GET /api/view?index=I&window=W``
    One item's view: box, panel extents and image URLs, numbered tracks
    (polylines), the model's prediction, the current decision, the next
    item. The server starts rendering the next item at once.
``GET /api/image?index=I&window=W&panel=P``
    A panel as PNG (row 0 = highest frequency).
``POST /api/action``
    ``{"index", "window_s", "prediction_shown", "elapsed_s", "action", ...}`` with
    ``action`` one of ``click`` (``panel, x, y, width, height``), ``pick``
    (``key``), ``not_chirp``, ``unclear`` (optional ``via``; in a window item
    with the position as for ``click``), ``skip``, and for window items
    ``done`` and ``clear``.
``POST /api/undo``
    Revert the latest action.

Every response that changes state carries the new progress and the index
to show next.

Window items (several marks per item, see
:mod:`chirpdetector.labeling.session`) send their zone (``item.zone``, the
box's time span, which is not a model output) and their current marks
(``marks``); they get no hint box.

Blind sessions (``--blind``) send neither the box edges (only its centre
time, ``t_center_s``) nor the candidate's prediction, score or source, so
not even the page's data reveal them; the page draws a time marker
instead of the box and ignores ``e``.
"""

from __future__ import annotations

import getpass
import json
import logging
import math
import socket
import struct
import threading
import zlib
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np
import numpy.typing as npt

from chirpdetector.labeling.session import (
    ActionResult,
    LabelSession,
    Progress,
    describe,
)
from chirpdetector.labeling.store import Decision
from chirpdetector.labeling.view import (
    PANELS,
    CandidateView,
    colour_lut,
    colourize,
    hint_box,
    pixel_to_data,
    prediction_of,
)

__all__ = ["HELP", "LabelServer", "encode_png", "serve_until_interrupted"]

log = logging.getLogger(__name__)

HELP = """\
left click on a baseline  chirp from that fish: snaps to the highlighted
                          track (nearest within the tolerance)
1-9                       chirp from the track with that number
0 / middle click          chirp, emitter unclear
x / right click           not a chirp
s / space                 skip (comes back after the open items)
u / backspace / ctrl+z    undo the last action (jumps to its item)
left / right arrow        previous / next item (labelled or not)
o / end                   next open item (skips everything labelled)
w                         overview window 204.8 / 51.2 ms (zoom: 204.8)
e                         show / hide the model's prediction
f                         full screen
h / ?                     toggle this help
Window items (shaded zone): mark EVERY chirp centred in the zone
  left click on the baseline at the chirp   a chirp of that fish (stays)
  middle click on the chirp (or 0)          a chirp, emitter unclear
  Enter / right click / x                   done: all chirps marked (also none)
  c                                         clear this window's marks
Every action is saved at once; closing the tab loses nothing."""

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})
_MAX_BODY = 1 << 16


def encode_png(rgb: npt.NDArray[np.uint8], level: int = 6) -> bytes:
    """Encode an ``(height, width, 3)`` uint8 image as PNG (no filtering)."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        msg = f"expected (h, w, 3) uint8, got {rgb.shape} {rgb.dtype}"
        raise ValueError(msg)
    height, width, _ = rgb.shape
    rows = np.ascontiguousarray(rgb).reshape(height, width * 3)
    raw = np.concatenate([np.zeros((height, 1), dtype=np.uint8), rows], axis=1)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw.tobytes(), level))
        + chunk(b"IEND", b"")
    )


def _num(value: float | None) -> float | None:
    """JSON-safe float (None for NaN)."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _progress(p: Progress) -> dict[str, Any]:
    return {
        "total": p.total,
        "labelled": p.labelled,
        "skipped": p.skipped,
        "open": p.open,
        "per_bin": {k: list(v) for k, v in p.per_bin.items()},
    }


def _decision(
    decision: Decision | None, view_tracks: tuple = (), *, ids: bool = True
) -> Any:
    if decision is None:
        return None
    return {
        "label": decision.label,
        "emitter_id": decision.emitter_id if ids else None,
        "clicked_hz": _num(decision.clicked_hz),
        "clicked_t_s": _num(decision.clicked_t_s),
        "emitter_baseline_hz": _num(decision.emitter_baseline_hz),
        "via": decision.via,
        "text": describe(decision, view_tracks, ids=ids),
    }


def _mark(decision: Decision, view_tracks: tuple = (), *, ids: bool = True) -> Any:
    """JSON of one mark of a window item."""
    key = next((t.key for t in view_tracks if t.track_id == decision.emitter_id), None)
    return {
        "t_s": _num(decision.clicked_t_s),
        "f_hz": _num(decision.clicked_hz),
        "label": decision.label,
        "key": key if decision.emitter_id is not None else None,
        "emitter_baseline_hz": _num(decision.emitter_baseline_hz),
        "text": describe(decision, view_tracks, ids=ids),
    }


class _BadRequest(Exception):
    pass


def _elapsed(value: object) -> float | None:
    """The page's time on an item (None if absent or not a sane number)."""
    try:
        elapsed = float(value)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        return None
    return elapsed if math.isfinite(elapsed) and elapsed >= 0 else None


class LabelServer:
    """HTTP frontend of a :class:`LabelSession`.

    Parameters
    ----------
    session
        The session (its ``frontend`` should be ``browser``).
    host
        Interface to bind to; 127.0.0.1 (only this machine; reach it via an
        SSH port-forward) by default.
    port
        Port; 0 picks a free one (see :attr:`port`).
    """

    def __init__(
        self, session: LabelSession, host: str = "127.0.0.1", port: int = 8765
    ) -> None:
        self.session = session
        self.host = host
        self._lut = colour_lut()
        self._page = (
            resources.files("chirpdetector.labeling")
            .joinpath("labeller.html")
            .read_bytes()
        )
        self._allowed_hosts = _LOCAL_HOSTS | {host}
        self.httpd = ThreadingHTTPServer((host, port), self._handler_class())
        self.httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """The port the server listens on."""
        return int(self.httpd.server_address[1])

    @property
    def url(self) -> str:
        """Local URL of the page."""
        host = "localhost" if self.host in ("127.0.0.1", "::1") else self.host
        return f"http://{host}:{self.port}/"

    def serve_forever(self) -> None:
        """Serve until :meth:`shutdown` (or KeyboardInterrupt)."""
        self.httpd.serve_forever(poll_interval=0.2)

    def start(self) -> None:
        """Serve in a daemon thread (for tests)."""
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        """Stop serving and close the socket."""
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.session.shutdown()

    # payloads --------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """``/api/state``."""
        s = self.session
        return {
            "queue_id": s.queue.queue_id,
            "n": len(s),
            "labeler": s.store.labeler,
            "tolerance_hz": s.tolerance_hz,
            "windows": list(s.windows),
            "window_s": s.config.window_s,
            "start_index": s.first_index(),
            "progress": _progress(s.progress()),
            "origin": s.queue.origin,
            "help": HELP,
            "blind": s.blind,
        }

    def _window(self, value: object) -> float:
        try:
            window = float(value)  # pyright: ignore[reportArgumentType]
        except (TypeError, ValueError):
            raise _BadRequest(f"bad window {value!r}") from None
        if window not in self.session.windows:
            msg = f"window must be one of {list(self.session.windows)}, got {window}"
            raise _BadRequest(msg)
        return window

    def _index(self, value: object) -> int:
        try:
            index = int(value)  # pyright: ignore[reportArgumentType]
        except (TypeError, ValueError):
            raise _BadRequest(f"bad index {value!r}") from None
        if not 0 <= index < len(self.session):
            msg = f"index {index} outside 0..{len(self.session) - 1}"
            raise _BadRequest(msg)
        return index

    def view(self, index: int, window_s: float) -> dict[str, Any]:
        """``/api/view``."""
        s = self.session
        item = s.item(index)
        tracks = s.tracks(index)
        next_index = s.next_index(index)
        described: dict[str, Any] = {
            # blind: no recording name or item id either (repeats of earlier
            # items, or a recording's name, would hint at the answer)
            "item_id": "" if s.blind else item.item_id,
            "trial": "" if s.blind else item.trial,
            "t_center_s": item.t_center_s,
            "df_bin": item.df_bin,
            "delta_f_hz": _num(item.delta_f_hz),
            "n_candidates": item.n_candidates,
        }
        if item.is_window:  # the zone is the item, not a model output
            described["kind"] = "window"
            described["zone"] = [item.t_start_s, item.t_end_s]
        elif s.blind and s.config.blind_hint:  # fixed-size, never the box itself
            described["hint_box"] = list(hint_box(item, s.config))
        if not s.blind:  # blind: the page never learns the box
            described.update(
                t_start_s=item.t_start_s,
                t_end_s=item.t_end_s,
                f_low_hz=item.f_low_hz,
                f_high_hz=item.f_high_hz,
            )
        payload: dict[str, Any] = {
            "index": index,
            "item": described,
            "window_s": window_s,
            "tracks": [
                {
                    "key": t.key,
                    "track_id": None if s.blind else t.track_id,
                    "baseline_hz": t.baseline_hz,
                    "t": t.times_s.tolist(),
                    "f": t.freqs_hz.tolist(),
                    "gap_s": 0.0 if s.blind else t.gap_s,
                }
                for t in tracks
            ],
            "prediction": None,
            "marks": [_mark(d, tracks, ids=not s.blind) for d in s.marks(index)]
            if item.is_window
            else [],
            "status": s.status(index),
            "decision": _decision(s.decision(index), tracks, ids=not s.blind),
            "next_index": next_index,
            "queue_id": s.queue.queue_id,
            "n": len(s.queue),
            "panels": None,
            "error": None,
        }
        prediction = None if s.blind else prediction_of(item)
        if prediction is not None:
            payload["prediction"] = {
                "emitter_hz": _num(prediction.emitter_hz),
                "track_id": prediction.track_id,
                "score": _num(prediction.score),
                "source": prediction.source,
            }
        try:
            view = s.view(index, window_s)
        except Exception as err:
            log.exception("could not render item %d", index)
            payload["error"] = f"render error: {err}"
        else:
            q = s.queue.queue_id
            payload["panels"] = {
                name: {
                    "extent": list(view.panel(name).extent),
                    "shape": list(view.panel(name).db.shape),
                    "src": f"/api/image?q={q}&index={index}&window={window_s}&panel={name}",
                }
                for name in PANELS
            }
            payload["render_ms"] = view.render_ms
        s.prefetch(next_index, window_s)
        return payload

    def image(self, index: int, window_s: float, panel: str) -> bytes:
        """``/api/image``: a panel as PNG."""
        if panel not in PANELS:
            raise _BadRequest(f"panel must be one of {PANELS}")
        view: CandidateView = self.session.view(index, window_s)
        return encode_png(colourize(view.panel(panel), self._lut, top_down=True))

    def action(self, body: dict[str, Any]) -> tuple[ActionResult, dict[str, Any]]:
        """``/api/action``."""
        s = self.session
        index = self._index(body.get("index"))
        window_s = self._window(body.get("window_s", s.config.window_s))
        shown = bool(body.get("prediction_shown", False)) and not s.blind
        elapsed = _elapsed(body.get("elapsed_s"))
        action = body.get("action")
        extra: dict[str, Any] = {}
        if action == "click":
            panel = body.get("panel")
            if panel not in PANELS:
                raise _BadRequest(f"panel must be one of {PANELS}")
            try:
                x, y, w, h = (float(body[k]) for k in ("x", "y", "width", "height"))
            except (KeyError, TypeError, ValueError):
                raise _BadRequest("click needs numeric x, y, width, height") from None
            try:
                extent = s.view(index, window_s).panel(str(panel)).extent
                t_s, f_hz = pixel_to_data(x, y, w, h, extent)
            except ValueError as err:
                raise _BadRequest(str(err)) from None
            extra = {"t_s": t_s, "f_hz": f_hz}
            result = s.click(
                index,
                t_s,
                f_hz,
                window_s=window_s,
                prediction_shown=shown,
                elapsed_s=elapsed,
            )
        elif action == "pick":
            try:
                key = int(body["key"])
            except (KeyError, TypeError, ValueError):
                raise _BadRequest("pick needs an integer key") from None
            result = s.pick(
                index, key, window_s=window_s, prediction_shown=shown, elapsed_s=elapsed
            )
        elif action in ("not_chirp", "unclear"):
            label = "not_chirp" if action == "not_chirp" else "chirp_emitter_unclear"
            via = "mouse" if body.get("via") == "mouse" else "key"
            at: tuple[float, float] | None = None
            if action == "unclear" and s.is_window(index) and "panel" in body:
                at = self._position(body, index, window_s)
                extra = {"t_s": at[0], "f_hz": at[1]}
            result = s.mark(
                index,
                label,
                window_s=window_s,
                prediction_shown=shown,
                via=via,
                elapsed_s=elapsed,
                t_s=None if at is None else at[0],
                f_hz=None if at is None else at[1],
            )
        elif action == "skip":
            result = s.skip(index, elapsed_s=elapsed)
        elif action == "done":
            result = s.done(index, window_s=window_s, elapsed_s=elapsed)
        elif action == "clear":
            result = s.clear(index)
        else:
            raise _BadRequest(f"unknown action {action!r}")
        return result, extra

    def _position(
        self, body: dict[str, Any], index: int, window_s: float
    ) -> tuple[float, float]:
        """Time and frequency of a click position in a request body."""
        panel = body.get("panel")
        if panel not in PANELS:
            raise _BadRequest(f"panel must be one of {PANELS}")
        try:
            x, y, w, h = (float(body[k]) for k in ("x", "y", "width", "height"))
        except (KeyError, TypeError, ValueError):
            raise _BadRequest("a position needs numeric x, y, width, height") from None
        try:
            extent = self.session.view(index, window_s).panel(str(panel)).extent
            return pixel_to_data(x, y, w, h, extent)
        except ValueError as err:
            raise _BadRequest(str(err)) from None

    def result_payload(
        self, result: ActionResult, extra: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """JSON of an action or undo result."""
        s = self.session
        tracks = s.tracks(result.index) if result.index is not None else ()
        return {
            "ok": result.ok,
            "message": result.message,
            "index": result.index,
            "next_index": result.next_index,
            "decision": _decision(result.decision, tracks, ids=not s.blind),
            "progress": _progress(s.progress()),
            "queue_id": s.queue.queue_id,
            **(extra or {}),
        }

    # HTTP ------------------------------------------------------------------

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        app = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "chirpdetector-label/1"

            def log_message(self, format: str, *args: Any) -> None:
                log.debug("%s %s", self.address_string(), format % args)

            def _send(
                self,
                status: int,
                body: bytes,
                content_type: str,
                cache: bool = False,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header(
                    "Cache-Control", "private, max-age=86400" if cache else "no-store"
                )
                self.end_headers()
                self.wfile.write(body)

            def _json(self, status: int, payload: Any) -> None:
                body = json.dumps(payload, allow_nan=False).encode()
                self._send(status, body, "application/json")

            def _host_ok(self) -> bool:
                if app.host in ("0.0.0.0", "::"):
                    return True
                host = (self.headers.get("Host") or "").strip()
                name = (
                    host.rsplit(":", 1)[0]
                    if not host.startswith("[")
                    else (host.split("]")[0] + "]")
                )
                if name in app._allowed_hosts:
                    return True
                self._json(
                    HTTPStatus.FORBIDDEN, {"ok": False, "message": f"bad Host {host!r}"}
                )
                return False

            def do_GET(self) -> None:
                if not self._host_ok():
                    return
                url = urlsplit(self.path)
                query = {k: v[-1] for k, v in parse_qs(url.query).items()}
                try:
                    if url.path in ("/", "/index.html"):
                        self._send(HTTPStatus.OK, app._page, "text/html; charset=utf-8")
                    elif url.path == "/api/state":
                        self._json(HTTPStatus.OK, app.state())
                    elif url.path == "/api/view":
                        index = app._index(query.get("index"))
                        window = app._window(
                            query.get("window", app.session.config.window_s)
                        )
                        self._json(HTTPStatus.OK, app.view(index, window))
                    elif url.path == "/api/image":
                        index = app._index(query.get("index"))
                        window = app._window(
                            query.get("window", app.session.config.window_s)
                        )
                        png = app.image(index, window, query.get("panel", ""))
                        self._send(HTTPStatus.OK, png, "image/png", cache=True)
                    elif url.path == "/favicon.ico":
                        self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
                    else:
                        self._json(
                            HTTPStatus.NOT_FOUND, {"ok": False, "message": "not found"}
                        )
                except _BadRequest as err:
                    self._json(
                        HTTPStatus.BAD_REQUEST, {"ok": False, "message": str(err)}
                    )
                except Exception as err:
                    log.exception("GET %s failed", self.path)
                    self._json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"ok": False, "message": str(err)},
                    )

            def do_POST(self) -> None:
                if not self._host_ok():
                    return
                url = urlsplit(self.path)
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    if length > _MAX_BODY:
                        raise _BadRequest("request too large")
                    raw = self.rfile.read(length) if length else b"{}"
                    try:
                        body = json.loads(raw or b"{}")
                    except json.JSONDecodeError:
                        raise _BadRequest("body is not JSON") from None
                    if not isinstance(body, dict):
                        raise _BadRequest("body must be a JSON object")
                    if url.path == "/api/action":
                        result, extra = app.action(body)
                        self._json(HTTPStatus.OK, app.result_payload(result, extra))
                    elif url.path == "/api/undo":
                        self._json(
                            HTTPStatus.OK, app.result_payload(app.session.undo())
                        )
                    else:
                        self._json(
                            HTTPStatus.NOT_FOUND, {"ok": False, "message": "not found"}
                        )
                except _BadRequest as err:
                    self._json(
                        HTTPStatus.BAD_REQUEST, {"ok": False, "message": str(err)}
                    )
                except Exception as err:
                    log.exception("POST %s failed", self.path)
                    self._json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"ok": False, "message": str(err)},
                    )

        return Handler


def serve_until_interrupted(
    session: LabelSession,
    host: str,
    port: int,
    out_dir: str | Path,
    echo: Callable[[str], object] = print,
) -> None:
    """Serve a session until Ctrl-C, printing the URL and the SSH command.

    Raises
    ------
    RuntimeError
        If the port cannot be bound.
    """
    try:
        server = LabelServer(session, host=host, port=port)
    except OSError as err:
        msg = f"cannot listen on {host}:{port} ({err}); try another --port"
        raise RuntimeError(msg) from err
    p = session.progress()
    echo(
        f"label-emitters: {p.total} items (queue {session.queue.queue_id}), "
        f"{p.labelled} labelled, labeller {session.store.labeler}; labels in {out_dir}"
    )
    echo(f"serving {server.url} (bound to {host}:{server.port}); Ctrl-C stops")
    echo("From your computer, forward the port and open the page:")
    echo(
        f"    ssh -N -L {server.port}:localhost:{server.port} "
        f"{getpass.getuser()}@{socket.gethostname()}"
    )
    echo(f"    http://localhost:{server.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        echo("stopped; every label is saved")
    finally:
        server.shutdown()
