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
    """Write an MP4 from an iterable of uint8 HWC frames. Returns error string on failure."""
    try:
        import cv2  # type: ignore
    except Exception as e:
        return f"cv2 unavailable: {e}"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = None
    n = 0
    try:
        for frame in frames_iter:
            if writer is None:
                h, w = frame.shape[:2]
                # Ensure even dimensions (some encoders require it)
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
    mp4_fps: int = 30,
    contact_sheet_cols: int = 4,
    contact_sheet_cell_width: int = 320,
    log_prefix: str = "report",
    subdir: Optional[str] = None,
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
        "output_dir": out_dir,
        "artifacts": artifacts,
    }

    # --- Test split: side-by-side + contact sheet -----------------------------
    psnrs: List[float] = []
    pairs_for_sheet: List[Tuple[str, np.ndarray, np.ndarray]] = []
    if test_cams:
        test_dir = os.path.join(out_dir, "test")
        for cam in test_cams:
            try:
                img = _render_image(cam, gaussians, render_fn, pipe, background)
                gt = _gt_image(cam)
            except Exception as e:
                print(f"[report] render failed for {cam.image_name}: {e}", flush=True)
                continue
            psnrs.append(float(_psnr(img, gt).mean().item()))
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
                tb_writer.add_scalar(f"{log_prefix}/test_psnr_mean", summary["mean_test_psnr"], iteration)
                tb_writer.add_scalar(f"{log_prefix}/test_psnr_min", summary["min_test_psnr"], iteration)
                tb_writer.add_scalar(f"{log_prefix}/test_psnr_max", summary["max_test_psnr"], iteration)
    else:
        print(f"[report] iter={iteration} no test views — skipping PSNR/contact sheet", flush=True)

    # --- Trajectory MP4 over train cameras ------------------------------------
    if train_cams:
        cams_for_video = train_cams
        if len(cams_for_video) > mp4_max_frames:
            step = max(1, len(cams_for_video) // mp4_max_frames)
            cams_for_video = cams_for_video[::step][:mp4_max_frames]

        def _frames():
            for cam in cams_for_video:
                try:
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
