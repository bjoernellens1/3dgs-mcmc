"""
Live web viewer for 3DGS-MCMC training.

Optional FastAPI + WebSocket server that streams training metrics
and a live rendered camera view to a browser dashboard.

Usage:
    --web_viewer_port 6010    # Enable viewer on port 6010
    --web_viewer_image_interval 10  # Push render image every N iters

The viewer is off by default (port=0). Requires fastapi, uvicorn, websockets.
"""

from __future__ import annotations

import asyncio
import os
import socket
import time
from queue import Empty, Full, Queue
from threading import Thread
from typing import Any

import cv2
import numpy as np

# FastAPI / uvicorn — optional; WebViewer init handles graceful fallback
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    import uvicorn
    _HAS_WEB_DEPS = True
except ImportError:
    _HAS_WEB_DEPS = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_STATIC = os.path.join(_HERE, "web_viewer")

# Singleton
_WEB_VIEWER: WebViewer | None = None


def _load_static(name: str) -> str:
    """Read a static file from web_viewer/ dir."""
    with open(os.path.join(_STATIC, name), "r") as f:
        return f.read()


# ---------------------------------------------------------------------------
# WebViewer — background FastAPI + WebSocket server
# ---------------------------------------------------------------------------

class WebViewer:
    """FastAPI + WebSocket server that lives in a background daemon thread."""

    def __init__(self, port: int = 6010, image_interval: int = 10):
        if not _HAS_WEB_DEPS:
            raise ImportError("fastapi, uvicorn, and websockets are required for the web viewer")

        self.port = port
        self.image_interval = image_interval
        self._connections: set[Any] = set()
        self._queue: Queue = Queue(maxsize=5)
        self._loop: asyncio.AbstractEventLoop = None  # set during startup
        self._thread: Thread | None = None
        self._ready = False
        self.viewer_cam_idx: int = 0    # which camera to render (set by client or CLI)
        self.total_cams: int = 1        # total training cameras (set by train.py)

        app = FastAPI(title="3DGS-MCMC Live Viewer")

        # ------------------------------------------------------------------
        @app.websocket("/ws")
        async def ws_endpoint(ws: WebSocket):
            await ws.accept()
            self._connections.add(ws)
            self._log(f"WebSocket client connected ({len(self._connections)} total)")
            # Send camera info on connect
            await ws.send_json({
                "type": "camera_info",
                "cam_idx": self.viewer_cam_idx,
                "total_cams": self.total_cams,
            })
            try:
                while True:
                    msg = await ws.receive_text()
                    if msg == "ping":
                        await ws.send_text("pong")
                    else:
                        # Try to parse as JSON command
                        try:
                            import json
                            cmd = json.loads(msg)
                            if cmd.get("type") == "set_camera":
                                idx = int(cmd.get("index", 0))
                                if 0 <= idx < self.total_cams:
                                    self.viewer_cam_idx = idx
                                    self._log(f"Camera set to {idx}")
                                    await ws.send_json({
                                        "type": "camera_info",
                                        "cam_idx": idx,
                                        "total_cams": self.total_cams,
                                    })
                        except (json.JSONDecodeError, ValueError):
                            pass
            except WebSocketDisconnect:
                self._log("WebSocket client disconnected")
            except Exception as e:
                self._log(f"WebSocket error: {type(e).__name__}: {e}")
            finally:
                self._connections.discard(ws)

        # ------------------------------------------------------------------
        @app.on_event("startup")
        async def _startup():
            self._loop = asyncio.get_running_loop()
            asyncio.create_task(self._broadcast_loop())

        # ------------------------------------------------------------------
        @app.get("/")
        async def _index():
            html = _load_static("index.html")
            return HTMLResponse(html)

        # ------------------------------------------------------------------
        @app.get("/health")
        async def _health():
            return {"status": "ok", "connections": len(self._connections)}

        self._app = app

    # -- public API --------------------------------------------------------

    @staticmethod
    def _log(msg: str) -> None:
        print(f"[web-viewer] {msg}", flush=True)

    def is_ready(self) -> bool:
        """Check if the server is listening on its port."""
        if not self._ready:
            return False
        try:
            s = socket.create_connection(("127.0.0.1", self.port), timeout=0.5)
            s.close()
            return True
        except (OSError, socket.error):
            return False

    def push_metrics(self, metrics: dict) -> None:
        """Enqueue a metrics dict for broadcast to all connected clients."""
        try:
            self._queue.put_nowait(("metrics", metrics))
        except Full:
            pass

    def push_image(self, image_np: np.ndarray, iteration: int) -> None:
        """
        Enqueue a rendered image for broadcast.

        Args:
            image_np: uint8 numpy array (H, W, 3) in RGB order, values 0–255.
            iteration: current training iteration.
        """
        try:
            self._queue.put_nowait(("image", (image_np, iteration)))
        except Full:
            pass

    def run_in_thread(self) -> None:
        """Start the uvicorn server in a background daemon thread."""
        self._error: Exception | None = None

        def _run():
            try:
                config = uvicorn.Config(
                    self._app,
                    host="0.0.0.0",
                    port=self.port,
                    log_level="warning",
                    log_config=None,
                    ws_ping_interval=10,
                    ws_ping_timeout=30,
                )
                server = uvicorn.Server(config)
                self._ready = True
                server.run()
            except Exception as e:
                self._error = e
                print(f"[web-viewer] Server error: {type(e).__name__}: {e}", flush=True)

        self._thread = Thread(target=_run, daemon=True, name="web-viewer")
        self._thread.start()

    # -- internal -----------------------------------------------------------

    async def _broadcast_loop(self) -> None:
        """Background coroutine: drain queue and broadcast to websockets."""
        while True:
            try:
                msg_type, data = self._queue.get_nowait()
            except Empty:
                await asyncio.sleep(0.033)  # ~30 Hz
                continue

            dead: set[Any] = set()
            connections = list(self._connections)
            for ws in connections:
                try:
                    if msg_type == "metrics":
                        await ws.send_json(data)
                    elif msg_type == "image":
                        arr, iteration = data
                        # RGB → BGR for OpenCV
                        img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                        success, buf = cv2.imencode(
                            ".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85]
                        )
                        if success:
                            # Send JSON metadata, then binary JPEG
                            await ws.send_json({
                                "type": "image",
                                "iteration": iteration,
                                "width": arr.shape[1],
                                "height": arr.shape[0],
                            })
                            await ws.send_bytes(buf.tobytes())
                except Exception:
                    dead.add(ws)

            if dead:
                self._connections -= dead

            await asyncio.sleep(0.001)  # yield


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def start_web_viewer(
    port: int = 6010,
    image_interval: int = 10,
    total_cams: int = 1,
    viewer_cam_idx: int = 0,
) -> WebViewer | None:
    """Start the web viewer in a background thread (idempotent).

    Returns the WebViewer singleton, or None if fastapi/uvicorn aren't
    installed (caller should handle gracefully).
    """
    global _WEB_VIEWER
    if _WEB_VIEWER is not None:
        return _WEB_VIEWER

    try:
        viewer = WebViewer(port=port, image_interval=image_interval)
        viewer.total_cams = total_cams
        viewer.viewer_cam_idx = viewer_cam_idx
        viewer.run_in_thread()
        _WEB_VIEWER = viewer

        # Wait briefly for the server to start listening
        for _ in range(50):
            if viewer.is_ready():
                break
            time.sleep(0.1)

        if viewer.is_ready():
            print(
                f"[web-viewer] Live dashboard at http://127.0.0.1:{port} "
                f"(image every {image_interval} iters)",
                flush=True,
            )
        else:
            print(
                f"[web-viewer] Server may not be ready on port {port} yet. "
                f"Training continues.",
                flush=True,
            )
        return viewer
    except ImportError as e:
        print(
            f"[web-viewer] Disabled — missing dependency: {e}. "
            "Install fastapi, uvicorn, websockets to enable.",
            flush=True,
        )
        return None
    except Exception as e:
        print(
            f"[web-viewer] Failed to start on port {port}: {e}. "
            "Training continues without web viewer.",
            flush=True,
        )
        return None


def get_web_viewer() -> WebViewer | None:
    """Return the active WebViewer singleton, or None."""
    return _WEB_VIEWER


def encode_render_image(image_tensor) -> np.ndarray | None:
    """Convert a GPU float render tensor (3,H,W) to CPU uint8 RGB numpy array.

    Returns (H, W, 3) uint8 RGB, or None on failure.
    """
    try:
        arr = (
            (image_tensor.detach().clamp(0, 1) * 255)
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        return arr
    except Exception:
        return None
