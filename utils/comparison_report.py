"""
Post-training comparison report.

Used by both offline (train.py) and streaming (train_streaming.py) paths to
produce a uniform set of artifacts under ``<model_path>/comparison/iter_{N}/``:

  test/{name}.png            side-by-side render | gt | abs-diff (one per test view)
  test_contact_sheet.png     grid of all test views (render row + gt row)
  trajectory.mp4             video over the train cameras in order
  report.json                summary metrics (mean/min/max PSNR, counts)

The reporter prints the mean test PSNR to stdout and logs it to TensorBoard
when a writer is supplied. Test PSNR is always reported when test views exist.
"""

from __future__ import annotations

import json
import math
import os
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch

from utils.image_utils import psnr as _psnr


# ---------------------------------------------------------------------------
# reproduce.sh generation
# ---------------------------------------------------------------------------

def _parse_docker_volumes() -> dict:
    """Parse /proc/self/mountinfo to find Docker bind-mount volumes.

    Returns {container_path: host_path} for each detected bind mount.
    Skips standard system paths and filesystem types that are not user data.
    """
    SKIP_FSTYPES = {
        "overlay", "tmpfs", "proc", "sysfs", "devpts", "cgroup", "cgroup2",
        "mqueue", "shm", "nsfs", "hugetlbfs", "fuse", "debugfs", "tracefs",
        "securityfs", "pstorefs", "bpf", "autofs", "rpc_pipefs",
    }
    SKIP_MOUNT_PREFIXES = ("/proc", "/sys", "/dev", "/etc/", "/run")

    volumes = {}
    try:
        with open("/proc/self/mountinfo") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                host_root = parts[3]    # path on host device
                mountpoint = parts[4]   # path inside container

                # Find " - " separator between optional fields and fstype
                sep = None
                for i, p in enumerate(parts):
                    if p == "-":
                        sep = i
                        break
                if sep is None or sep + 1 >= len(parts):
                    continue
                fstype = parts[sep + 1]

                if fstype in SKIP_FSTYPES:
                    continue
                if mountpoint == "/":
                    continue
                if any(mountpoint == p or mountpoint.startswith(p + "/")
                       for p in SKIP_MOUNT_PREFIXES):
                    continue
                # host_root is the path within the block device's FS.
                # For most setups (ext4, xfs, plain btrfs) it equals the host path.
                if host_root.startswith("/") and len(host_root) > 1:
                    volumes[mountpoint] = host_root
    except Exception:
        pass
    return volumes


def write_reproduce_sh(path: str, argv: list) -> None:
    """Write a reproduce.sh that includes the Docker wrapper when running in a container.

    When /.dockerenv is present (Docker/Podman), the script contains:
      - A `docker compose run --rm ...` section (for re-running from the host)
      - A plain `python ...` section (for re-running inside the container)

    When not in a container, writes a plain `python ...` command.
    """
    import shlex
    import stat

    cmd = " ".join(shlex.quote(a) for a in argv)
    in_docker = os.path.exists("/.dockerenv")

    lines = [
        "#!/bin/bash",
        "# Reproduces the training run stored in this output directory.",
        "# Auto-generated at training start.",
        "",
    ]

    if in_docker:
        volumes = _parse_docker_volumes()
        lines += [
            "# ── Run from the HOST machine (project root) ─────────────────────",
            "docker compose run --rm \\",
        ]
        for cpath in sorted(volumes):
            hpath = volumes[cpath]
            lines.append(f'  -v "{hpath}:{cpath}" \\')
        lines += [
            "  train \\",
            f"  python {cmd}",
            "",
            "# ── OR run directly inside the container ─────────────────────────",
            f"# python {cmd}",
        ]
    else:
        lines += [f"python {cmd}"]

    content = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(content)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _to_uint8_hwc(img: torch.Tensor) -> np.ndarray:
    """Convert a CHW float tensor in [0, 1] to HWC uint8."""
    arr = img.detach().clamp(0.0, 1.0).cpu().numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _abs_diff(render: torch.Tensor, gt: torch.Tensor, scale: float = 5.0) -> np.ndarray:
    """|render - gt| amplified by `scale` for visibility."""
    diff = (render - gt).abs().mean(dim=0, keepdim=True) * scale
    diff = diff.clamp(0.0, 1.0).expand(3, -1, -1)
    return _to_uint8_hwc(diff)


def _hconcat_with_gap(images: List[np.ndarray], gap_px: int = 4) -> np.ndarray:
    """Stack uint8 HWC images horizontally with a thin white gap between."""
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    h = max(im.shape[0] for im in images)
    parts = []
    for i, im in enumerate(images):
        if im.shape[0] != h:
            # Pad to common height (top-aligned).
            pad = np.full((h - im.shape[0], im.shape[1], 3), 255, dtype=np.uint8)
            im = np.concatenate([im, pad], axis=0)
        parts.append(im)
        if i < len(images) - 1:
            parts.append(np.full((h, gap_px, 3), 255, dtype=np.uint8))
    return np.concatenate(parts, axis=1)


def _label_strip(text: str, width: int, height: int = 18) -> np.ndarray:
    """Render a label as a uint8 HWC strip using cv2 if available, else blank."""
    try:
        import cv2  # type: ignore
        strip = np.full((height, width, 3), 255, dtype=np.uint8)
        cv2.putText(
            strip, text, (4, height - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA,
        )
        return strip
    except Exception:
        return np.full((height, width, 3), 255, dtype=np.uint8)


def _save_png(path: str, arr_hwc_uint8: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import cv2  # type: ignore
        bgr = arr_hwc_uint8[:, :, ::-1].copy()
        cv2.imwrite(path, bgr)
    except Exception:
        # Fallback: torchvision
        import torchvision.utils as tvu
        t = torch.from_numpy(arr_hwc_uint8).permute(2, 0, 1).float() / 255.0
        tvu.save_image(t, path)


def _render_image(cam, gaussians, render_fn, pipe, background) -> torch.Tensor:
    with torch.no_grad():
        out = render_fn(cam, gaussians, pipe, background)
        img = out["render"] if isinstance(out, dict) else out
    return img.clamp(0.0, 1.0)


def _batch_render_images(cams, gaussians, pipe, background,
                         batch_render_fn=None, chunk_size: int = 32) -> list:
    """Render a list of cameras in batches. Returns list of (3,H,W) tensors."""
    if not cams:
        return []
    if batch_render_fn is None:
        return []
    with torch.no_grad():
        return batch_render_fn(cams, gaussians, pipe, background, chunk_size=chunk_size)


def _gt_image(cam) -> torch.Tensor:
    gt = cam.original_image[:3].to("cuda", non_blocking=True)
    return gt.clamp(0.0, 1.0)


def _resize_to(arr: np.ndarray, target_w: int) -> np.ndarray:
    """Resize HWC uint8 to a target width while preserving aspect; cv2 if available."""
    if arr.shape[1] == target_w:
        return arr
    new_h = max(1, int(round(arr.shape[0] * target_w / arr.shape[1])))
    try:
        import cv2  # type: ignore
        return cv2.resize(arr, (target_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        # Nearest-neighbour fallback
        ys = (np.arange(new_h) * arr.shape[0] / new_h).astype(np.int64)
        xs = (np.arange(target_w) * arr.shape[1] / target_w).astype(np.int64)
        return arr[ys][:, xs]


def _build_contact_sheet(
    pairs: List[Tuple[str, np.ndarray, np.ndarray]],
    cols: int = 4,
    cell_width: int = 320,
) -> np.ndarray:
    """Grid of (label, render, gt). Render on top row of each cell, gt on bottom."""
    if not pairs:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    cols = max(1, min(cols, len(pairs)))
    rows = math.ceil(len(pairs) / cols)
    cell_rows: List[np.ndarray] = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            idx = r * cols + c
            if idx >= len(pairs):
                blank = np.full((1, cell_width, 3), 255, dtype=np.uint8)
                row_cells.append(blank)
                continue
            name, render, gt = pairs[idx]
            render = _resize_to(render, cell_width)
            gt = _resize_to(gt, cell_width)
            label = _label_strip(name, cell_width)
            cell = np.concatenate([label, render, gt], axis=0)
            row_cells.append(cell)
        row = _hconcat_with_gap(row_cells, gap_px=4)
        cell_rows.append(row)
    # Vertical concat with thin gap
    max_w = max(row.shape[1] for row in cell_rows)
    padded = []
    for row in cell_rows:
        if row.shape[1] < max_w:
            pad = np.full((row.shape[0], max_w - row.shape[1], 3), 255, dtype=np.uint8)
            row = np.concatenate([row, pad], axis=1)
        padded.append(row)
        padded.append(np.full((4, max_w, 3), 255, dtype=np.uint8))
    return np.concatenate(padded[:-1], axis=0)


def _write_mp4(path: str, frames_iter: Iterable[np.ndarray], fps: int = 30) -> Optional[str]:
    """Write an MP4 from an iterable of uint8 HWC RGB frames.

    Tries ffmpeg first (libx264, all CPU threads via -threads 0, piped stdin).
    Falls back to cv2.VideoWriter with mp4v if ffmpeg is unavailable.
    Returns an error string on failure, or None on success.
    """
    import shutil
    import subprocess

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if shutil.which("ffmpeg"):
        proc = None
        n = 0
        h2 = w2 = None
        try:
            for frame in frames_iter:
                h, w = frame.shape[:2]
                if proc is None:
                    h2 = h - (h % 2)
                    w2 = w - (w % 2)
                    proc = subprocess.Popen(
                        [
                            "ffmpeg", "-y",
                            "-f", "rawvideo",
                            "-pix_fmt", "rgb24",
                            "-s", f"{w2}x{h2}",
                            "-r", str(fps),
                            "-i", "pipe:0",
                            "-c:v", "libx264",
                            "-threads", "0",
                            "-crf", "23",
                            "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart",
                            path,
                        ],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                frame_cropped = frame[:h2, :w2]
                proc.stdin.write(frame_cropped.tobytes())
                n += 1
        except Exception as e:
            if proc is not None:
                proc.stdin.close()
                proc.wait()
            return f"ffmpeg pipe error: {e}"
        finally:
            if proc is not None:
                proc.stdin.close()
                proc.wait()
        if n == 0:
            return "no frames written"
        if proc.returncode != 0:
            return f"ffmpeg exited with code {proc.returncode}"
        return None

    # Fallback: cv2.VideoWriter
    try:
        import cv2  # type: ignore
    except Exception as e:
        return f"ffmpeg unavailable and cv2 unavailable: {e}"

    writer = None
    n = 0
    try:
        for frame in frames_iter:
            if writer is None:
                h, w = frame.shape[:2]
                h2 = h - (h % 2)
                w2 = w - (w % 2)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(path, fourcc, fps, (w2, h2))
                if not writer.isOpened():
                    return "cv2.VideoWriter failed to open"
                _crop_h, _crop_w = h2, w2
            if frame.shape[:2] != (_crop_h, _crop_w):
                frame = frame[:_crop_h, :_crop_w]
            writer.write(frame[:, :, ::-1])  # RGB -> BGR
            n += 1
    finally:
        if writer is not None:
            writer.release()
    if n == 0:
        return "no frames written"
    return None


def write_post_training_report(
    model_path: str,
    iteration: int,
    gaussians,
    train_cams: List,
    test_cams: List,
    render_fn,
    pipe,
    background,
    *,
    tb_writer=None,
    mp4_max_frames: int = 300,
    mp4_fps: int = 15,
    contact_sheet_cols: int = 4,
    contact_sheet_cell_width: int = 320,
    log_prefix: str = "report",
    subdir: Optional[str] = None,
    batch_render_fn=None,
    skip_train_metrics: bool = False,
    skip_trajectory: bool = False,
) -> dict:
    """Produce side-by-side PNGs, contact sheet, trajectory MP4, and report.json.

    Returns a summary dict containing:
        n_train, n_test, mean_test_psnr, min_test_psnr, max_test_psnr,
        output_dir, artifacts (paths written)
    """
    out_dir = os.path.join(model_path, "comparison", subdir or f"iter_{iteration}")
    os.makedirs(out_dir, exist_ok=True)
    artifacts: List[str] = []

    summary = {
        "iteration": iteration,
        "n_train": len(train_cams) if train_cams else 0,
        "n_test": len(test_cams) if test_cams else 0,
        "mean_test_psnr": None,
        "min_test_psnr": None,
        "max_test_psnr": None,
        "mean_test_lpips": None,
        "min_test_lpips": None,
        "max_test_lpips": None,
        "mean_train_psnr": None,
        "min_train_psnr": None,
        "max_train_psnr": None,
        "mean_train_lpips": None,
        "min_train_lpips": None,
        "max_train_lpips": None,
        "output_dir": out_dir,
        "artifacts": artifacts,
    }

    try:
        from lpipsPyTorch import lpips as _lpips
    except ImportError:
        _lpips = None

    def _compute_metrics(cams: List, split: str, pre_rendered: Optional[list] = None,
                         skip_lpips: bool = False):
        psnr_list: list = []
        lpips_list: list = []
        if pre_rendered is not None and len(pre_rendered) == len(cams):
            batch_imgs_m = pre_rendered
            use_batch_m = True
        else:
            batch_imgs_m = []
            if batch_render_fn is not None:
                try:
                    batch_imgs_m = _batch_render_images(
                        cams, gaussians, pipe, background,
                        batch_render_fn=batch_render_fn,
                    )
                except Exception:
                    batch_imgs_m = []
            use_batch_m = len(batch_imgs_m) == len(cams)
        for i, cam in enumerate(cams):
            try:
                if use_batch_m:
                    img = batch_imgs_m[i].clamp(0.0, 1.0)
                else:
                    img = _render_image(cam, gaussians, render_fn, pipe, background)
                gt = _gt_image(cam)
            except Exception as e:
                print(f"[report] render failed for {cam.image_name}: {e}", flush=True)
                continue
            psnr_list.append(float(_psnr(img, gt).mean().item()))
            if _lpips is not None and not skip_lpips:
                try:
                    lpips_list.append(float(_lpips(img.unsqueeze(0), gt.unsqueeze(0), net_type='vgg').item()))
                except Exception:
                    pass
        return psnr_list, lpips_list

    # --- Test split: side-by-side + contact sheet + metrics -----------------------------
    psnrs: List[float] = []
    lpipss: List[float] = []
    pairs_for_sheet: List[Tuple[str, np.ndarray, np.ndarray]] = []
    if test_cams:
        test_dir = os.path.join(out_dir, "test")
        # Batch-render all test cameras at once when batch_render_fn is available
        if batch_render_fn is not None:
            try:
                batch_imgs = _batch_render_images(
                    test_cams, gaussians, pipe, background,
                    batch_render_fn=batch_render_fn,
                )
            except Exception as e:
                print(f"[report] batch render failed, falling back: {e}", flush=True)
                batch_imgs = []
        else:
            batch_imgs = []
        use_batch = len(batch_imgs) == len(test_cams)
        for i, cam in enumerate(test_cams):
            try:
                if use_batch:
                    img = batch_imgs[i].clamp(0.0, 1.0)
                else:
                    img = _render_image(cam, gaussians, render_fn, pipe, background)
                gt = _gt_image(cam)
            except Exception as e:
                print(f"[report] render failed for {cam.image_name}: {e}", flush=True)
                continue
            psnrs.append(float(_psnr(img, gt).mean().item()))
            if _lpips is not None:
                try:
                    lpipss.append(float(_lpips(img.unsqueeze(0), gt.unsqueeze(0), net_type='vgg').item()))
                except Exception:
                    pass
            r_u8 = _to_uint8_hwc(img)
            g_u8 = _to_uint8_hwc(gt)
            d_u8 = _abs_diff(img, gt)
            side = _hconcat_with_gap([r_u8, g_u8, d_u8], gap_px=4)
            name = str(getattr(cam, "image_name", f"view_{len(pairs_for_sheet):04d}"))
            _save_png(os.path.join(test_dir, f"{name}.png"), side)
            pairs_for_sheet.append((name, r_u8, g_u8))
        if pairs_for_sheet:
            sheet = _build_contact_sheet(
                pairs_for_sheet,
                cols=contact_sheet_cols,
                cell_width=contact_sheet_cell_width,
            )
            sheet_path = os.path.join(out_dir, "test_contact_sheet.png")
            _save_png(sheet_path, sheet)
            artifacts.append(sheet_path)
        if psnrs:
            summary["mean_test_psnr"] = float(np.mean(psnrs))
            summary["min_test_psnr"] = float(np.min(psnrs))
            summary["max_test_psnr"] = float(np.max(psnrs))
            msg = (
                f"[report] iter={iteration} test PSNR mean={summary['mean_test_psnr']:.2f}dB "
                f"min={summary['min_test_psnr']:.2f} max={summary['max_test_psnr']:.2f} "
                f"({len(psnrs)} views)"
            )
            print(msg, flush=True)
            if tb_writer is not None:
                try:
                    tb_writer.add_scalar(f"{log_prefix}/test_psnr_mean", summary["mean_test_psnr"], iteration)
                    tb_writer.add_scalar(f"{log_prefix}/test_psnr_min", summary["min_test_psnr"], iteration)
                    tb_writer.add_scalar(f"{log_prefix}/test_psnr_max", summary["max_test_psnr"], iteration)
                except Exception:
                    pass
        if lpipss:
            summary["mean_test_lpips"] = float(np.mean(lpipss))
            summary["min_test_lpips"] = float(np.min(lpipss))
            summary["max_test_lpips"] = float(np.max(lpipss))
            msg = (
                f"[report] iter={iteration} test LPIPS mean={summary['mean_test_lpips']:.4f} "
                f"min={summary['min_test_lpips']:.4f} max={summary['max_test_lpips']:.4f}"
            )
            print(msg, flush=True)
            if tb_writer is not None:
                try:
                    tb_writer.add_scalar(f"{log_prefix}/test_lpips_mean", summary["mean_test_lpips"], iteration)
                except Exception:
                    pass
    else:
        print(f"[report] iter={iteration} no test views — skipping test PSNR/LPIPS/contact sheet", flush=True)

    # Pre-render train cams once — reused for both metrics and trajectory MP4
    # Skip if both train metrics and trajectory are disabled (async quick eval).
    _train_imgs_all: list = []
    if train_cams and not (skip_train_metrics and skip_trajectory) and batch_render_fn is not None:
        try:
            _train_imgs_all = _batch_render_images(
                train_cams, gaussians, pipe, background,
                batch_render_fn=batch_render_fn,
            )
        except Exception as _re:
            print(f"[report] train pre-render failed: {_re}", flush=True)

    # --- Train split: metrics ------------------------------------
    if train_cams and not skip_train_metrics:
        train_psnrs, train_lpipss = _compute_metrics(train_cams, "train", pre_rendered=_train_imgs_all)
        if train_psnrs:
            summary["mean_train_psnr"] = float(np.mean(train_psnrs))
            summary["min_train_psnr"] = float(np.min(train_psnrs))
            summary["max_train_psnr"] = float(np.max(train_psnrs))
            msg = (
                f"[report] iter={iteration} train PSNR mean={summary['mean_train_psnr']:.2f}dB "
                f"min={summary['min_train_psnr']:.2f} max={summary['max_train_psnr']:.2f} "
                f"({len(train_psnrs)} views)"
            )
            print(msg, flush=True)
            if tb_writer is not None:
                try:
                    tb_writer.add_scalar(f"{log_prefix}/train_psnr_mean", summary["mean_train_psnr"], iteration)
                except Exception:
                    pass
        if train_lpipss:
            summary["mean_train_lpips"] = float(np.mean(train_lpipss))
            summary["min_train_lpips"] = float(np.min(train_lpipss))
            summary["max_train_lpips"] = float(np.max(train_lpipss))
            msg = (
                f"[report] iter={iteration} train LPIPS mean={summary['mean_train_lpips']:.4f} "
                f"min={summary['min_train_lpips']:.4f} max={summary['max_train_lpips']:.4f}"
            )
            print(msg, flush=True)
            if tb_writer is not None:
                try:
                    tb_writer.add_scalar(f"{log_prefix}/train_lpips_mean", summary["mean_train_lpips"], iteration)
                except Exception:
                    pass

    # --- Trajectory MP4 over train cameras ------------------------------------
    if train_cams and not skip_trajectory:
        cams_for_video = train_cams
        traj_imgs: list = []
        if len(cams_for_video) > mp4_max_frames:
            step = max(1, len(cams_for_video) // mp4_max_frames)
            _vidx = list(range(0, len(train_cams), step))[:mp4_max_frames]
            cams_for_video = [train_cams[i] for i in _vidx]
            if len(_train_imgs_all) == len(train_cams):
                traj_imgs = [_train_imgs_all[i] for i in _vidx]
        else:
            traj_imgs = _train_imgs_all  # no subsampling: reuse directly

        if not traj_imgs and batch_render_fn is not None:
            try:
                traj_imgs = _batch_render_images(
                    cams_for_video, gaussians, pipe, background,
                    batch_render_fn=batch_render_fn, chunk_size=32,
                )
            except Exception as e:
                print(f"[report] batch trajectory render failed, falling back: {e}", flush=True)
        use_traj_batch = len(traj_imgs) == len(cams_for_video)

        def _frames():
            for idx, cam in enumerate(cams_for_video):
                try:
                    if use_traj_batch:
                        img = traj_imgs[idx].clamp(0.0, 1.0)
                    else:
                        img = _render_image(cam, gaussians, render_fn, pipe, background)
                except Exception as e:
                    print(f"[report] trajectory render failed for {cam.image_name}: {e}", flush=True)
                    continue
                yield _to_uint8_hwc(img)

        mp4_path = os.path.join(out_dir, "trajectory.mp4")
        err = _write_mp4(mp4_path, _frames(), fps=mp4_fps)
        if err is None:
            artifacts.append(mp4_path)
            print(f"[report] trajectory.mp4 written ({len(cams_for_video)} frames @ {mp4_fps}fps)", flush=True)
        else:
            print(f"[report] trajectory.mp4 SKIPPED: {err}", flush=True)
    else:
        print(f"[report] iter={iteration} no train views — skipping trajectory MP4", flush=True)

    # --- report.json ----------------------------------------------------------
    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    artifacts.append(report_path)

    return summary
