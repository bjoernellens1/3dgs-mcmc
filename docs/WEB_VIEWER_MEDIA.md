# Persistent Web Viewer Media

The web viewer is default-on and starts as a separate process during training.
It serves `http://127.0.0.1:6010` by default and keeps running after training
finishes so cached media can still be inspected.

The server binds to `0.0.0.0` by default, so it can be reached from another
machine when the container/runtime publishes the port, for example
`docker compose run --rm -p 6010:6010 ...`. Use `--web_viewer_host 127.0.0.1`
only when you intentionally want local-only access.

Disable it with:

```bash
--no-web-viewer
```

By default the training process keeps the viewer alive after training completes.
For unattended benchmark jobs, disable that hold-open behavior with:

```bash
--no-web-viewer-keep-alive
```

## Viewer Modes

- `Live Render`: JPEG frames from the internal training renderer.
- `Videos`: optional MP4 recordings from fixed training cameras.
- `Scene PLY`: cached Gaussian PLY snapshots rendered in the browser with the
  PlayCanvas engine.

Scene PLY mode compensates for the SuperSplat/PlayCanvas orientation convention
in the browser by rotating the loaded splat entity by `x=-90, z=180`. The cached
PLY files on disk are not modified.

The frontend vendors PlayCanvas and Chart.js under
`gaussian_renderer/web_viewer/vendor/` so the dashboard does not need CDN access
on restricted HPC networks.

## Cache Policy

Viewer cache defaults to:

```text
<model_path>/web_viewer_cache
```

The PLY cache writes raw 3DGS PLY files atomically into:

```text
web_viewer_cache/scenes/
```

Default cadence:

```text
--web_viewer_scene_cache_interval 500
--web_viewer_scene_cache_keep 3
```

A final PLY snapshot is always written when training completes. Keeping only a
few snapshots matters because large scenes can produce hundreds of MB per PLY,
and `save_ply()` performs a full tensor-to-CPU export plus disk write.

Raw PLY is used for v1 because PlayCanvas can load Gaussian PLY directly.
Transport compression such as gzip/zstd is not on the training hot path; it
reduces network bytes but does not remove browser parse/upload cost.

## Video Recording

Video recording is opt-in:

```bash
--record_video
```

Defaults:

```text
--record_video_cameras ""      # first/middle/last train cameras
--record_video_interval 100
--record_video_fps 30
--record_video_crf 23
--record_video_preset veryfast
```

The implementation streams raw RGB frames to an `ffmpeg` subprocess and writes
H.264 MP4. This is intentionally used instead of PyAV for the first pass: it
avoids adding a Python binary dependency to the ROCm container, keeps direct
access to libx264 CRF/preset controls, and avoids temporary image sequences.
PyAV remains a reasonable future cleanup if tighter in-process encoder control
is needed.

## Standalone Server

The viewer can also be started later against an existing output directory:

```bash
python gaussian_renderer/web_viewer.py \
  --serve \
  --model_path output/my_run \
  --cache_dir output/my_run/web_viewer_cache \
  --host 0.0.0.0 \
  --port 6010
```

This direct script form avoids importing the training renderer package, which
requires gsplat.

## Validation Notes

Smoke run used the bicycle scene with:

```text
--iterations 1500
--web_viewer_port 6023
--web_viewer_image_interval 50
--web_viewer_scene_cache_interval 500
--web_viewer_scene_cache_keep 3
--record_video
--record_video_interval 100
--taming_score_interval 500
```

Observed results:

- The viewer started before camera loading and accepted a browser connection
  while status was `loading`.
- The viewer updated to 194 train cameras once scene loading completed.
- Cached PLY snapshots were written at iterations 500, 1000, 1500, plus a final
  snapshot.
- Three MP4 camera videos were finalized with 15 frames each.
- The current unrebuild container image lacked FastAPI/Uvicorn/WebSockets and
  system `ffmpeg`; the run installed web deps at startup and used OpenCV MP4
  fallback. The Dockerfile now installs those web deps and `ffmpeg`, so rebuilt
  images use the intended persistent viewer and H.264 path directly.
