"""
Persistent web viewer for 3DGS-MCMC training.

The module serves two roles:

* training client API: ``start_web_viewer()``, ``push_metrics()``,
  ``push_image()``, and cache event helpers.
* standalone server: ``python gaussian_renderer/web_viewer.py --serve ...``

The default training path uses a separate viewer process so the dashboard and
cached media remain available after training exits. The legacy in-process
thread backend is still available through ``--web_viewer_backend thread``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import Any

import cv2
import numpy as np
import torch

try:
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    import uvicorn

    _HAS_WEB_DEPS = True
except ImportError:
    _HAS_WEB_DEPS = False

_HERE = Path(__file__).resolve().parent
_STATIC = _HERE / "web_viewer"
_VENDOR = _STATIC / "vendor"

_WEB_VIEWER: Any = None


def _now() -> float:
    return time.time()


def _load_static(name: str) -> str:
    with open(_STATIC / name, "r", encoding="utf-8") as f:
        return f.read()


def _safe_child_path(root: Path, rel_path: str) -> Path:
    root = root.resolve()
    child = (root / rel_path).resolve()
    if root != child and root not in child.parents:
        raise ValueError("path escapes cache root")
    return child


def _http_json(method: str, url: str, payload: dict | None = None, timeout: float = 0.5) -> dict | None:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def _http_bytes(method: str, url: str, data: bytes, timeout: float = 1.0) -> None:
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/octet-stream"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()


class WebViewerCache:
    """Atomic manifest helper shared by training and the standalone server."""

    def __init__(self, cache_dir: str | os.PathLike[str], model_path: str = ""):
        self.cache_dir = Path(cache_dir).resolve()
        self.model_path = str(model_path)
        self.manifest_path = self.cache_dir / "manifest.json"
        self.scenes_dir = self.cache_dir / "scenes"
        self.videos_dir = self.cache_dir / "videos"
        self._lock = Lock()

    def ensure(self) -> None:
        self.scenes_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self.write_manifest(self.default_manifest())

    def default_manifest(self) -> dict:
        return {
            "version": 1,
            "model_path": self.model_path,
            "updated_at": _now(),
            "status": {"state": "starting", "message": ""},
            "camera": {"cam_idx": 0, "total_cams": 1},
            "latest_metrics": {},
            "scenes": [],
            "videos": [],
        }

    def read_manifest(self) -> dict:
        self.ensure()
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = self.default_manifest()
        data.setdefault("version", 1)
        data.setdefault("model_path", self.model_path)
        data.setdefault("status", {"state": "unknown", "message": ""})
        data.setdefault("camera", {"cam_idx": 0, "total_cams": 1})
        data.setdefault("latest_metrics", {})
        data.setdefault("scenes", [])
        data.setdefault("videos", [])
        return data

    def write_manifest(self, data: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        data["updated_at"] = _now()
        with self._lock:
            fd, tmp_name = tempfile.mkstemp(
                prefix="manifest.",
                suffix=".json.tmp",
                dir=str(self.cache_dir),
                text=True,
            )
            tmp = Path(tmp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp, self.manifest_path)

    def update(self, mutator) -> dict:
        data = self.read_manifest()
        mutator(data)
        self.write_manifest(data)
        return data

    def set_camera(self, cam_idx: int, total_cams: int | None = None) -> dict:
        def _mut(data: dict) -> None:
            camera = data.setdefault("camera", {})
            camera["cam_idx"] = int(cam_idx)
            if total_cams is not None:
                camera["total_cams"] = int(total_cams)

        return self.update(_mut)

    def set_status(self, state: str, message: str = "") -> dict:
        def _mut(data: dict) -> None:
            data["status"] = {"state": state, "message": message, "time": _now()}

        return self.update(_mut)

    def update_metrics(self, metrics: dict) -> dict:
        def _mut(data: dict) -> None:
            data["latest_metrics"] = dict(metrics)

        return self.update(_mut)

    def add_scene(self, path: str, iteration: int, final: bool = False, keep: int = 3) -> dict:
        rel = os.path.relpath(path, self.cache_dir)
        item = {
            "iteration": int(iteration),
            "final": bool(final),
            "path": rel.replace(os.sep, "/"),
            "url": "/cache/" + rel.replace(os.sep, "/"),
            "name": ("final" if final else f"iter {iteration}"),
            "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0,
            "time": _now(),
        }

        def _mut(data: dict) -> None:
            scenes = [s for s in data.get("scenes", []) if s.get("path") != item["path"]]
            scenes.append(item)
            scenes.sort(key=lambda s: (int(s.get("iteration", -1)), bool(s.get("final", False))))
            if keep > 0:
                finals = [s for s in scenes if s.get("final")]
                regular = [s for s in scenes if not s.get("final")]
                regular = regular[-keep:]
                scenes = sorted(regular + finals, key=lambda s: (int(s.get("iteration", -1)), bool(s.get("final", False))))
            data["scenes"] = scenes

        data = self.update(_mut)
        self._prune_scene_files(data)
        return data

    def add_video(self, path: str, camera_index: int, label: str, frame_count: int) -> dict:
        rel = os.path.relpath(path, self.cache_dir)
        item = {
            "camera_index": int(camera_index),
            "label": str(label),
            "frame_count": int(frame_count),
            "path": rel.replace(os.sep, "/"),
            "url": "/cache/" + rel.replace(os.sep, "/"),
            "name": str(label),
            "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0,
            "time": _now(),
        }

        def _mut(data: dict) -> None:
            videos = [v for v in data.get("videos", []) if v.get("path") != item["path"]]
            videos.append(item)
            videos.sort(key=lambda v: int(v.get("camera_index", 0)))
            data["videos"] = videos

        return self.update(_mut)

    def _prune_scene_files(self, manifest: dict) -> None:
        keep_paths = {str(s.get("path", "")) for s in manifest.get("scenes", [])}
        for path in self.scenes_dir.glob("*.ply"):
            rel = os.path.relpath(path, self.cache_dir).replace(os.sep, "/")
            if rel not in keep_paths:
                try:
                    path.unlink()
                except OSError:
                    pass


class _BaseViewerClient:
    port: int
    image_interval: int
    total_cams: int

    @property
    def viewer_cam_idx(self) -> int:
        return 0

    def is_ready(self) -> bool:
        return False

    def push_metrics(self, metrics: dict) -> None:
        pass

    def push_image(self, image_np: np.ndarray, iteration: int) -> None:
        pass

    def register_scene(self, path: str, iteration: int, final: bool = False, keep: int = 3) -> None:
        pass

    def register_video(self, path: str, camera_index: int, label: str, frame_count: int) -> None:
        pass

    def set_status(self, state: str, message: str = "") -> None:
        pass

    def close(self) -> None:
        pass


class WebViewer(_BaseViewerClient):
    """FastAPI + WebSocket server, optionally embedded in training."""

    def __init__(
        self,
        port: int = 6010,
        host: str = "0.0.0.0",
        image_interval: int = 100,
        cache_dir: str | None = None,
        model_path: str = "",
    ):
        if not _HAS_WEB_DEPS:
            raise ImportError("fastapi, uvicorn, and websockets are required for the web viewer")

        self.port = int(port)
        self.host = host
        self.image_interval = int(image_interval)
        self.cache = WebViewerCache(cache_dir or os.path.join(model_path or ".", "web_viewer_cache"), model_path)
        self.cache.ensure()
        self._connections: set[Any] = set()
        self._queue: Queue = Queue(maxsize=20)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: Thread | None = None
        self._ready = False
        self._error: Exception | None = None
        camera = self.cache.read_manifest().get("camera", {})
        self._viewer_cam_idx = int(camera.get("cam_idx", 0))
        self.total_cams = int(camera.get("total_cams", 1))

        app = FastAPI(title="3DGS-MCMC Web Viewer")

        @app.websocket("/ws")
        async def ws_endpoint(ws: WebSocket):
            await ws.accept()
            self._connections.add(ws)
            await self._send_camera_info(ws)
            await ws.send_json({"type": "state", "state": self._state_payload()})
            try:
                while True:
                    msg = await ws.receive_text()
                    if msg == "ping":
                        await ws.send_json({"type": "pong"})
                        continue
                    try:
                        cmd = json.loads(msg)
                    except json.JSONDecodeError:
                        continue
                    if cmd.get("type") == "set_camera":
                        idx = int(cmd.get("index", 0))
                        if 0 <= idx < max(1, self.total_cams):
                            self._viewer_cam_idx = idx
                            self.cache.set_camera(idx, self.total_cams)
                            await self._broadcast_json({
                                "type": "camera_info",
                                "cam_idx": idx,
                                "total_cams": self.total_cams,
                            })
            except WebSocketDisconnect:
                pass
            except Exception as e:
                self._log(f"WebSocket error: {type(e).__name__}: {e}")
            finally:
                self._connections.discard(ws)

        @app.on_event("startup")
        async def _startup():
            self._loop = asyncio.get_running_loop()
            asyncio.create_task(self._broadcast_loop())
            self._ready = True

        @app.get("/")
        async def _index():
            return HTMLResponse(_load_static("index.html"))

        @app.get("/vendor/{path:path}")
        async def _vendor(path: str):
            try:
                file_path = _safe_child_path(_VENDOR, path)
            except ValueError:
                return JSONResponse({"error": "invalid path"}, status_code=400)
            if not file_path.exists() or not file_path.is_file():
                return JSONResponse({"error": "not found"}, status_code=404)
            return FileResponse(file_path)

        @app.get("/cache/{path:path}")
        async def _cache_file(path: str):
            try:
                file_path = _safe_child_path(self.cache.cache_dir, path)
            except ValueError:
                return JSONResponse({"error": "invalid path"}, status_code=400)
            if not file_path.exists() or not file_path.is_file():
                return JSONResponse({"error": "not found"}, status_code=404)
            return FileResponse(file_path)

        @app.get("/api/health")
        @app.get("/health")
        async def _health():
            return {"status": "ok", "connections": len(self._connections)}

        @app.get("/api/state")
        async def _state():
            return self._state_payload()

        @app.post("/api/metrics")
        async def _metrics(request: Request):
            metrics = await request.json()
            self.cache.update_metrics(metrics)
            self.push_metrics(metrics)
            return {"ok": True}

        @app.post("/api/event")
        async def _event(request: Request):
            payload = await request.json()
            event_type = payload.get("type")
            if event_type == "scene":
                self.cache.add_scene(
                    payload["path"],
                    int(payload["iteration"]),
                    bool(payload.get("final", False)),
                    int(payload.get("keep", 3)),
                )
            elif event_type == "video":
                self.cache.add_video(
                    payload["path"],
                    int(payload.get("camera_index", 0)),
                    str(payload.get("label", "camera")),
                    int(payload.get("frame_count", 0)),
                )
            elif event_type == "status":
                self.cache.set_status(str(payload.get("state", "unknown")), str(payload.get("message", "")))
            elif event_type == "camera_info":
                self.total_cams = int(payload.get("total_cams", self.total_cams))
                self.cache.set_camera(self._viewer_cam_idx, self.total_cams)
            await self._broadcast_json({"type": "state", "state": self._state_payload()})
            return {"ok": True}

        @app.post("/api/image")
        async def _image(request: Request):
            body = await request.body()
            headers = request.headers
            payload = {
                "iteration": int(headers.get("x-iteration", "0")),
                "width": int(headers.get("x-width", "0")),
                "height": int(headers.get("x-height", "0")),
                "jpeg": body,
            }
            try:
                self._queue.put_nowait(("image", payload))
            except Full:
                pass
            return {"ok": True}

        self._app = app

    @staticmethod
    def _log(msg: str) -> None:
        print(f"[web-viewer] {msg}", flush=True)

    @property
    def viewer_cam_idx(self) -> int:
        return self._viewer_cam_idx

    def is_ready(self) -> bool:
        if not self._ready:
            return False
        return _check_port_ready(self.port)

    def push_metrics(self, metrics: dict) -> None:
        payload = dict(metrics)
        payload["type"] = "metrics"
        payload["connections"] = len(self._connections)
        try:
            self._queue.put_nowait(("metrics", payload))
        except Full:
            pass

    def push_image(self, image_np: np.ndarray, iteration: int) -> None:
        try:
            img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            success, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                return
            payload = {
                "iteration": int(iteration),
                "width": int(image_np.shape[1]),
                "height": int(image_np.shape[0]),
                "jpeg": buf.tobytes(),
            }
            self._queue.put_nowait(("image", payload))
        except Full:
            pass

    def register_scene(self, path: str, iteration: int, final: bool = False, keep: int = 3) -> None:
        self.cache.add_scene(path, iteration, final=final, keep=keep)
        self._enqueue_state()

    def register_video(self, path: str, camera_index: int, label: str, frame_count: int) -> None:
        self.cache.add_video(path, camera_index, label, frame_count)
        self._enqueue_state()

    def set_status(self, state: str, message: str = "") -> None:
        self.cache.set_status(state, message)
        self._enqueue_state()

    def update_camera_info(self, total_cams: int, viewer_cam_idx: int | None = None) -> None:
        self.total_cams = int(total_cams)
        if viewer_cam_idx is not None:
            self._viewer_cam_idx = int(viewer_cam_idx)
        self.cache.set_camera(self._viewer_cam_idx, self.total_cams)
        self.cache.set_status("running", "training active")
        self._enqueue_state()

    def run_in_thread(self) -> None:
        def _run():
            try:
                config = uvicorn.Config(
                    self._app,
                    host=self.host,
                    port=self.port,
                    log_level="warning",
                    log_config=None,
                    ws_ping_interval=10,
                    ws_ping_timeout=30,
                )
                uvicorn.Server(config).run()
            except Exception as e:
                self._error = e
                self._log(f"Server error: {type(e).__name__}: {e}")

        self._thread = Thread(target=_run, daemon=True, name="web-viewer")
        self._thread.start()

    def run_forever(self) -> None:
        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.port,
            log_level="info",
            log_config=None,
            ws_ping_interval=10,
            ws_ping_timeout=30,
        )
        uvicorn.Server(config).run()

    def _state_payload(self) -> dict:
        data = self.cache.read_manifest()
        data["connections"] = len(self._connections)
        data.setdefault("camera", {})
        data["camera"]["cam_idx"] = self._viewer_cam_idx
        data["camera"]["total_cams"] = self.total_cams
        return data

    async def _send_camera_info(self, ws: WebSocket) -> None:
        await ws.send_json({
            "type": "camera_info",
            "cam_idx": self._viewer_cam_idx,
            "total_cams": self.total_cams,
        })

    def _enqueue_state(self) -> None:
        try:
            self._queue.put_nowait(("state", self._state_payload()))
        except Full:
            pass

    async def _broadcast_json(self, payload: dict) -> None:
        dead: set[Any] = set()
        for ws in list(self._connections):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.add(ws)
        if dead:
            self._connections -= dead

    async def _broadcast_loop(self) -> None:
        while True:
            try:
                msg_type, data = self._queue.get_nowait()
            except Empty:
                await asyncio.sleep(0.033)
                continue

            if msg_type == "metrics":
                await self._broadcast_json(data)
            elif msg_type == "state":
                await self._broadcast_json({"type": "state", "state": data})
            elif msg_type == "image":
                dead: set[Any] = set()
                for ws in list(self._connections):
                    try:
                        await ws.send_json({
                            "type": "image",
                            "iteration": data["iteration"],
                            "width": data["width"],
                            "height": data["height"],
                        })
                        await ws.send_bytes(data["jpeg"])
                    except Exception:
                        dead.add(ws)
                if dead:
                    self._connections -= dead

            await asyncio.sleep(0.001)


class ProcessWebViewerClient(_BaseViewerClient):
    """Best-effort HTTP client used by training when the viewer is a process."""

    def __init__(
        self,
        port: int,
        host: str,
        image_interval: int,
        total_cams: int,
        cache_dir: str,
        model_path: str,
        process: subprocess.Popen | None = None,
    ):
        self.port = int(port)
        self.host = host
        self.image_interval = int(image_interval)
        self.total_cams = int(total_cams)
        self.cache = WebViewerCache(cache_dir, model_path)
        self.cache.ensure()
        self._process = process
        self._queue: Queue = Queue(maxsize=64)
        self._stop = False
        self._last_state_fetch = 0.0
        self._viewer_cam_idx = 0
        self._base_url = f"http://127.0.0.1:{self.port}"
        self._worker = Thread(target=self._drain, daemon=True, name="web-viewer-client")
        self._worker.start()
        self._enqueue("event", {"type": "camera_info", "total_cams": self.total_cams})

    @property
    def viewer_cam_idx(self) -> int:
        now = time.perf_counter()
        if now - self._last_state_fetch > 0.25:
            self._last_state_fetch = now
            try:
                state = _http_json("GET", f"{self._base_url}/api/state", timeout=0.2) or {}
                camera = state.get("camera", {})
                self._viewer_cam_idx = int(camera.get("cam_idx", self._viewer_cam_idx))
                self.total_cams = int(camera.get("total_cams", self.total_cams))
            except Exception:
                pass
        return self._viewer_cam_idx

    def is_ready(self) -> bool:
        try:
            data = _http_json("GET", f"{self._base_url}/api/health", timeout=0.3)
            return bool(data and data.get("status") == "ok")
        except Exception:
            return False

    def push_metrics(self, metrics: dict) -> None:
        self._enqueue("metrics", dict(metrics))

    def push_image(self, image_np: np.ndarray, iteration: int) -> None:
        try:
            img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            success, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                self._enqueue("image", {
                    "iteration": int(iteration),
                    "width": int(image_np.shape[1]),
                    "height": int(image_np.shape[0]),
                    "jpeg": buf.tobytes(),
                })
        except Exception:
            pass

    def register_scene(self, path: str, iteration: int, final: bool = False, keep: int = 3) -> None:
        self.cache.add_scene(path, iteration, final=final, keep=keep)
        self._enqueue("event", {
            "type": "scene",
            "path": path,
            "iteration": int(iteration),
            "final": bool(final),
            "keep": int(keep),
        })

    def register_video(self, path: str, camera_index: int, label: str, frame_count: int) -> None:
        self.cache.add_video(path, camera_index, label, frame_count)
        self._enqueue("event", {
            "type": "video",
            "path": path,
            "camera_index": int(camera_index),
            "label": label,
            "frame_count": int(frame_count),
        })

    def set_status(self, state: str, message: str = "") -> None:
        self.cache.set_status(state, message)
        self._enqueue("event", {"type": "status", "state": state, "message": message})

    def update_camera_info(self, total_cams: int, viewer_cam_idx: int | None = None) -> None:
        self.total_cams = int(total_cams)
        if viewer_cam_idx is not None:
            self._viewer_cam_idx = int(viewer_cam_idx)
        self.cache.set_camera(self._viewer_cam_idx, self.total_cams)
        self.cache.set_status("running", "training active")
        self._enqueue("event", {
            "type": "camera_info",
            "total_cams": self.total_cams,
        })

    def close(self) -> None:
        deadline = time.perf_counter() + 2.0
        while not self._queue.empty() and time.perf_counter() < deadline:
            time.sleep(0.05)
        self._stop = True
        self._worker.join(timeout=1.0)

    def _enqueue(self, kind: str, payload: dict) -> None:
        try:
            self._queue.put_nowait((kind, payload))
        except Full:
            pass

    def _drain(self) -> None:
        while not self._stop:
            try:
                kind, payload = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                if kind == "metrics":
                    _http_json("POST", f"{self._base_url}/api/metrics", payload, timeout=0.3)
                elif kind == "event":
                    _http_json("POST", f"{self._base_url}/api/event", payload, timeout=0.5)
                elif kind == "image":
                    url = f"{self._base_url}/api/image"
                    req = urllib.request.Request(
                        url,
                        data=payload["jpeg"],
                        headers={
                            "Content-Type": "application/octet-stream",
                            "X-Iteration": str(payload["iteration"]),
                            "X-Width": str(payload["width"]),
                            "X-Height": str(payload["height"]),
                        },
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=1.0) as resp:
                        resp.read()
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            except Exception:
                pass


def _check_port_ready(port: int) -> bool:
    try:
        s = socket.create_connection(("127.0.0.1", int(port)), timeout=0.5)
        s.close()
        return True
    except (OSError, socket.error):
        return False


def _pid_file(cache_dir: str) -> Path:
    return Path(cache_dir).resolve() / "viewer.pid"


def _write_pid_file(cache_dir: str, pid: int, port: int, host: str) -> None:
    path = _pid_file(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"pid": int(pid), "port": int(port), "host": host, "time": _now()}, f)
        f.write("\n")


def _start_process_server(
    port: int,
    host: str,
    image_interval: int,
    total_cams: int,
    cache_dir: str,
    model_path: str,
) -> subprocess.Popen | None:
    cache = WebViewerCache(cache_dir, model_path)
    cache.ensure()
    if _check_port_ready(port):
        return None
    cmd = [
        sys.executable,
        str(_HERE / "web_viewer.py"),
        "--serve",
        "--host",
        host,
        "--port",
        str(port),
        "--image_interval",
        str(image_interval),
        "--cache_dir",
        cache_dir,
        "--model_path",
        model_path,
        "--total_cams",
        str(total_cams),
    ]
    log_path = Path(cache_dir).resolve() / "viewer.log"
    log_file = open(log_path, "a", buffering=1)
    proc = subprocess.Popen(
        cmd,
        cwd=str(_HERE.parent),
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )
    _write_pid_file(cache_dir, proc.pid, port, host)
    for _ in range(50):
        if _check_port_ready(port):
            break
        if proc.poll() is not None:
            return proc
        time.sleep(0.1)
    return proc


def start_web_viewer(
    port: int = 6010,
    host: str = "0.0.0.0",
    image_interval: int = 100,
    total_cams: int = 1,
    viewer_cam_idx: int = 0,
    backend: str = "process",
    cache_dir: str | None = None,
    model_path: str = "",
) -> _BaseViewerClient | None:
    """Start or connect to the web viewer.

    ``backend="process"`` leaves the viewer alive after training exits.
    ``backend="thread"`` preserves the older in-process behavior.
    """
    global _WEB_VIEWER
    if _WEB_VIEWER is not None:
        try:
            _WEB_VIEWER.total_cams = int(total_cams)
            if hasattr(_WEB_VIEWER, "cache"):
                _WEB_VIEWER.cache.set_camera(viewer_cam_idx, total_cams)
            _WEB_VIEWER.set_status("running", "training active")
        except Exception:
            pass
        return _WEB_VIEWER

    if backend == "off" or port <= 0:
        return None
    cache_dir = cache_dir or os.path.join(model_path or ".", "web_viewer_cache")
    if not _HAS_WEB_DEPS:
        print(
            "[web-viewer] Disabled - missing dependency: fastapi, uvicorn, or websockets. "
            "Install the web viewer dependencies or rebuild the container image.",
            flush=True,
        )
        return None

    try:
        cache = WebViewerCache(cache_dir, model_path)
        cache.ensure()
        cache.set_camera(viewer_cam_idx, total_cams)
        cache.set_status("running", "training active")

        if backend == "process":
            proc = _start_process_server(port, host, image_interval, total_cams, cache_dir, model_path)
            client = ProcessWebViewerClient(
                port=port,
                host=host,
                image_interval=image_interval,
                total_cams=total_cams,
                cache_dir=cache_dir,
                model_path=model_path,
                process=proc,
            )
            _WEB_VIEWER = client
            if client.is_ready():
                print(
                    f"[web-viewer] Persistent dashboard at http://127.0.0.1:{port} "
                    f"(listening on {host}:{port}, cache: {cache_dir})",
                    flush=True,
                )
            else:
                print(
                    f"[web-viewer] Process started but health check failed on port {port}. "
                    "Training continues.",
                    flush=True,
                )
            return client

        if backend == "thread":
            viewer = WebViewer(
                port=port,
                host=host,
                image_interval=image_interval,
                cache_dir=cache_dir,
                model_path=model_path,
            )
            viewer.total_cams = total_cams
            viewer._viewer_cam_idx = viewer_cam_idx
            viewer.run_in_thread()
            _WEB_VIEWER = viewer
            for _ in range(50):
                if viewer.is_ready():
                    break
                time.sleep(0.1)
            print(
                f"[web-viewer] Live dashboard at http://127.0.0.1:{port} "
                f"(listening on {host}:{port}) "
                f"(image every {image_interval} iters)",
                flush=True,
            )
            return viewer

        print(f"[web-viewer] Unknown backend '{backend}', disabling viewer.", flush=True)
        return None
    except ImportError as e:
        print(
            f"[web-viewer] Disabled - missing dependency: {e}. "
            "Install fastapi, uvicorn, websockets to enable.",
            flush=True,
        )
        return None
    except Exception as e:
        print(
            f"[web-viewer] Failed to start on port {port}: {type(e).__name__}: {e}. "
            "Training continues without web viewer.",
            flush=True,
        )
        return None


def get_web_viewer() -> _BaseViewerClient | None:
    return _WEB_VIEWER


def encode_render_image(image_tensor) -> np.ndarray | None:
    """Convert a GPU float render tensor (3,H,W) to CPU uint8 RGB numpy array."""
    try:
        return (
            (image_tensor.detach().clamp(0, 1) * 255)
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
    except Exception:
        return None


_viewer_pinned_buf: Any = None


def encode_render_image_async_start(image_tensor, stream, wait_event=None, reuse_buffer: bool = True) -> Any:
    """Start an async GPU-to-CPU copy of a render image into pinned memory."""
    global _viewer_pinned_buf

    if wait_event is not None:
        stream.wait_event(wait_event)

    with torch.cuda.stream(stream):
        processed = (
            (image_tensor.detach().clamp(0, 1) * 255)
            .byte()
            .permute(1, 2, 0)
            .contiguous()
        )
        if not reuse_buffer:
            pinned_buf = torch.empty(
                processed.shape, dtype=processed.dtype, device="cpu", pin_memory=True
            )
        elif _viewer_pinned_buf is None or _viewer_pinned_buf.shape != processed.shape:
            _viewer_pinned_buf = torch.empty(
                processed.shape, dtype=processed.dtype, device="cpu", pin_memory=True
            )
            pinned_buf = _viewer_pinned_buf
        else:
            pinned_buf = _viewer_pinned_buf
        pinned_buf.copy_(processed, non_blocking=True)

    return pinned_buf


def encode_render_image_finish(pinned_buf) -> np.ndarray:
    return pinned_buf.numpy()


def _serve(args: argparse.Namespace) -> None:
    viewer = WebViewer(
        port=args.port,
        host=args.host,
        image_interval=args.image_interval,
        cache_dir=args.cache_dir,
        model_path=args.model_path,
    )
    viewer.total_cams = args.total_cams
    viewer.cache.set_camera(0, args.total_cams)
    viewer.cache.set_status("running", "viewer active")
    try:
        viewer.run_forever()
    except KeyboardInterrupt:
        pass


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="3DGS-MCMC persistent web viewer")
    parser.add_argument("--serve", action="store_true", help="Run the standalone web viewer server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6010)
    parser.add_argument("--image_interval", type=int, default=100)
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--total_cams", type=int, default=1)
    args = parser.parse_args(argv)
    if not args.serve:
        parser.error("use --serve to run the standalone viewer")
    if not args.cache_dir:
        args.cache_dir = os.path.join(args.model_path or ".", "web_viewer_cache")
    _serve(args)


if __name__ == "__main__":
    main()
