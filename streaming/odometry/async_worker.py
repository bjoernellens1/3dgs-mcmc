"""
Async odometry worker — produces PoseReadyFrame objects on a bounded queue.

Current state (Phase 4 scaffold):
    OrbbecRosBagFrameSource / RealsenseRosBagFrameSource already run Open3D
    odometry in a background thread via _start_live_odom_worker() with a
    condition-variable wait protocol. This module provides the shared data
    types (PoseReadyFrame, OdometryStats) so higher-level code can consume
    frames without knowing whether odometry was synchronous or async.

Next evolution (Phase 5+):
    Replace the condition-variable design with a producer→queue→consumer model:
        - OrbbecAsyncOdometryWorker: decodes frames + runs Open3D in a daemon
          thread; pushes PoseReadyFrame onto a bounded Queue.
        - StreamingScene.ingest_next_frame(): pops from the queue instead of
          calling _ensure_live_odom_until().
    This removes odometry latency from the GPU training timeline entirely.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class OdometryStats:
    """Per-edge diagnostics from an odometry estimate."""
    translation_m: float = 0.0
    rotation_deg: float = 0.0
    # Open3D fitness/residual from RGBD odometry or ICP
    fitness: Optional[float] = None
    inlier_rmse: Optional[float] = None
    info_trace: Optional[float] = None
    method: str = ""
    valid: bool = True
    reason: str = ""

    def to_tensorboard_dict(self) -> dict:
        return {
            "odom/translation_m": self.translation_m,
            "odom/rotation_deg": self.rotation_deg,
            "odom/fitness": self.fitness if self.fitness is not None else float("nan"),
            "odom/inlier_rmse": self.inlier_rmse if self.inlier_rmse is not None else float("nan"),
            "odom/info_trace": self.info_trace if self.info_trace is not None else float("nan"),
            "odom/valid": 1.0 if self.valid else 0.0,
        }


@dataclass
class PoseReadyFrame:
    """A frame whose camera-to-world pose has been determined (or failed)."""
    frame: object  # StreamingRGBDFrame
    c2w: Optional[np.ndarray]  # None if odom failed
    odom_stats: OdometryStats = field(default_factory=OdometryStats)


def log_odom_stats(tb_writer, stats: OdometryStats, step: int) -> None:
    """Write per-frame odometry diagnostics to TensorBoard."""
    if tb_writer is None or stats is None:
        return
    import math
    for key, value in stats.to_tensorboard_dict().items():
        if not math.isnan(value):
            tb_writer.add_scalar(key, value, step)


class AsyncOdometryWorker:
    """
    Producer-consumer wrapper for async odometry (Phase 5 target design).

    Instantiate with a frame source that supports sequential iteration.
    Start the worker; the training loop calls get_next() to receive
    PoseReadyFrame objects without blocking on odometry computation.
    """

    def __init__(self, base_source, odometry_fn, max_queue: int = 32) -> None:
        """
        Parameters
        ----------
        base_source:  iterable of StreamingRGBDFrame.
        odometry_fn:  callable(prev_frame, curr_frame) -> OdometryStats.
                      Must be thread-safe; called on the worker thread.
        max_queue:    bounded queue depth (blocks producer if consumer is slow).
        """
        self._source = base_source
        self._odometry_fn = odometry_fn
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="async-odom-worker")
        self._stopped = False

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopped = True
        self._queue.put(None)

    def get_next(self, timeout: float = 10.0) -> Optional[PoseReadyFrame]:
        """Return the next pose-ready frame, or None if the source is exhausted."""
        return self._queue.get(timeout=timeout)

    def _run(self) -> None:
        prev_frame = None
        for frame in self._source:
            if self._stopped:
                break
            try:
                if prev_frame is None or frame.c2w is not None:
                    stats = OdometryStats(valid=True, reason="precomputed")
                else:
                    stats = self._odometry_fn(prev_frame, frame)
                prf = PoseReadyFrame(frame=frame, c2w=frame.c2w, odom_stats=stats)
            except Exception as e:
                prf = PoseReadyFrame(
                    frame=frame,
                    c2w=None,
                    odom_stats=OdometryStats(valid=False, reason=str(e)),
                )
            self._queue.put(prf)
            prev_frame = frame
        self._queue.put(None)  # sentinel
