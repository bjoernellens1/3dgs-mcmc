"""
Frame release scheduler for streaming replay simulation.

In deterministic mode (wallclock=False) a new frame is released every
streaming_steps_per_frame training iterations — fast, reproducible, no
artificial sleep. In wallclock mode frames are released by real elapsed
time at the configured fps, simulating live sensor input and dropping
frames when training is slower than the source rate.
"""
from __future__ import annotations

import time


class FrameScheduler:
    def __init__(self, fps: float, steps_per_frame: int = 50, wallclock: bool = False):
        self.fps = float(fps)
        self.steps_per_frame = max(1, int(steps_per_frame))
        self.wallclock = bool(wallclock)
        self._start_time: float | None = None
        self._next_frame_idx: int = 0

    def should_release(self, iteration: int) -> bool:
        """Return True if a new frame should be ingested at this iteration."""
        if not self.wallclock:
            # Deterministic: release on iteration 1 and every steps_per_frame after that.
            return iteration == 1 or (iteration % self.steps_per_frame) == 0
        if self._start_time is None:
            self._start_time = time.perf_counter()
        now = time.perf_counter()
        target = self._start_time + self._next_frame_idx / max(self.fps, 1e-6)
        return now >= target

    def mark_released(self) -> None:
        """Call once per actually ingested frame."""
        self._next_frame_idx += 1
