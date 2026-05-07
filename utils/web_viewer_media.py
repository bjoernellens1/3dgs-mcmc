from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import cv2


def default_video_camera_indices(total_cams: int) -> list[int]:
    if total_cams <= 0:
        return []
    candidates = [0, total_cams // 2, total_cams - 1]
    result: list[int] = []
    for idx in candidates:
        idx = max(0, min(total_cams - 1, int(idx)))
        if idx not in result:
            result.append(idx)
    return result


def parse_video_camera_indices(spec: str, total_cams: int) -> list[int]:
    spec = (spec or "").strip()
    if not spec:
        return default_video_camera_indices(total_cams)
    result: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if 0 <= idx < total_cams and idx not in result:
            result.append(idx)
    return result


class ScenePlyCacheWriter:
    def __init__(self, cache_dir: str, keep: int, viewer=None):
        self.cache_dir = Path(cache_dir).resolve()
        self.scenes_dir = self.cache_dir / "scenes"
        self.keep = max(0, int(keep))
        self.viewer = viewer
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

    def save(self, gaussians, iteration: int, final: bool = False) -> str | None:
        name = f"scene_final_{iteration:06d}.ply" if final else f"scene_{iteration:06d}.ply"
        path = self.scenes_dir / name
        tmp_path = path.with_suffix(".ply.tmp")
        try:
            gaussians.save_ply(str(tmp_path))
            os.replace(tmp_path, path)
            if self.viewer is not None:
                self.viewer.register_scene(str(path), iteration, final=final, keep=self.keep)
            return str(path)
        except Exception as exc:
            print(f"[web-viewer-cache] Failed to save PLY cache at iter {iteration}: {exc}", flush=True)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            return None


class _CameraVideoWriter:
    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: int,
        crf: int,
        preset: str,
    ):
        self.output_path = output_path
        self.tmp_path = output_path.with_name(output_path.stem + ".tmp.mp4")
        self.frame_count = 0
        self._cv_writer = None
        self._uses_ffmpeg = shutil.which("ffmpeg") is not None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._uses_ffmpeg:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._cv_writer = cv2.VideoWriter(str(self.tmp_path), fourcc, float(fps), (width, height))
            if not self._cv_writer.isOpened():
                raise RuntimeError("ffmpeg is unavailable and OpenCV VideoWriter could not open MP4 output")
            self.proc = None
            return
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(self.tmp_path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, image_np) -> None:
        if self._cv_writer is not None:
            self._cv_writer.write(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            self.frame_count += 1
            return
        if self.proc.stdin is None:
            return
        self.proc.stdin.write(image_np.tobytes())
        self.frame_count += 1

    def close(self) -> bool:
        if self._cv_writer is not None:
            self._cv_writer.release()
            os.replace(self.tmp_path, self.output_path)
            return True
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        code = self.proc.wait(timeout=30)
        if code != 0:
            return False
        os.replace(self.tmp_path, self.output_path)
        return True


class MultiCameraVideoRecorder:
    def __init__(
        self,
        cache_dir: str,
        camera_indices: Iterable[int],
        fps: int,
        crf: int,
        preset: str,
        viewer=None,
    ):
        self.cache_dir = Path(cache_dir).resolve()
        self.videos_dir = self.cache_dir / "videos"
        self.camera_indices = list(camera_indices)
        self.fps = int(fps)
        self.crf = int(crf)
        self.preset = str(preset)
        self.viewer = viewer
        self._writers: dict[int, _CameraVideoWriter] = {}
        self.videos_dir.mkdir(parents=True, exist_ok=True)

    def write(self, camera_index: int, image_np) -> None:
        if camera_index not in self.camera_indices:
            return
        writer = self._writers.get(camera_index)
        if writer is None:
            height, width = int(image_np.shape[0]), int(image_np.shape[1])
            path = self.videos_dir / f"camera_{camera_index:03d}.mp4"
            writer = _CameraVideoWriter(
                output_path=path,
                width=width,
                height=height,
                fps=self.fps,
                crf=self.crf,
                preset=self.preset,
            )
            self._writers[camera_index] = writer
        writer.write(image_np)

    def close(self) -> None:
        for camera_index, writer in list(self._writers.items()):
            try:
                ok = writer.close()
            except Exception as exc:
                print(f"[web-viewer-video] Failed to close camera {camera_index}: {exc}", flush=True)
                ok = False
            if ok and self.viewer is not None:
                try:
                    self.viewer.register_video(
                        str(writer.output_path),
                        camera_index=camera_index,
                        label=f"camera {camera_index}",
                        frame_count=writer.frame_count,
                    )
                except Exception as exc:
                    print(f"[web-viewer-video] Failed to register camera {camera_index}: {exc}", flush=True)
        self._writers.clear()
