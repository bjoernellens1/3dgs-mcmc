"""
Frame release scheduler for streaming replay simulation.

Three ingestion modes (--streaming_ingestion_mode):

  iter_based      (default) Release one frame every streaming_steps_per_frame
                  training iterations — fast, reproducible, no clock.

  dataset_fps     Advance a virtual clock by each iteration's measured wall
                  time; release the next frame when its dataset timestamp is
                  due (capped at streaming_input_fps_cap fps).  Training never
                  drops frames — it runs as many iters as it can between
                  releases.  Results are deterministic regardless of GPU speed.

  wallclock_strict Use perf_counter; late frames are dropped so the map never
                  gets ahead of sensor time.  Most faithful to real SLAM but
                  results vary with hardware.
"""
from __future__ import annotations

import time


class FrameScheduler:
    def __init__(
        self,
        fps: float = 30.0,
        steps_per_frame: int = 50,
        wallclock: bool = False,
        ingestion_mode: str = "iter_based",
        fps_cap: float = 30.0,
    ):
        self.fps = float(fps)
        self.steps_per_frame = max(1, int(steps_per_frame))
        # ingestion_mode overrides legacy wallclock bool
        if ingestion_mode == "iter_based":
            self._mode = "iter_based"
        elif ingestion_mode == "dataset_fps":
            self._mode = "dataset_fps"
        elif ingestion_mode == "wallclock_strict":
            self._mode = "wallclock_strict"
        else:
            # backwards-compat: honour old wallclock kwarg
            self._mode = "wallclock_strict" if wallclock else "iter_based"

        self._fps_cap = max(float(fps_cap), 0.1)
        self._next_frame_idx: int = 0

        # Simulated clock (dataset_fps mode)
        self._sim_time: float = 0.0          # virtual seconds elapsed
        self._last_loop_time: float | None = None

        # True wall-clock (wallclock_strict mode)
        self._start_time: float | None = None
        self._frames_dropped: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_release(self, iteration: int, dt: float | None = None) -> bool:
        """Return True if a new frame should be ingested at this iteration.

        Args:
            iteration: Current training iteration (1-based).
            dt:        Wall-clock seconds this iteration took (used only in
                       dataset_fps mode to advance the simulated clock).
        """
        if self._mode == "iter_based":
            # How many frames should have been released by this iteration?
            # iter=1 always releases the first frame; after that, one per steps_per_frame.
            expected = 1 + max(0, iteration // self.steps_per_frame)
            return self._next_frame_idx < expected

        if self._mode == "dataset_fps":
            if dt is not None:
                self._sim_time += dt
            due_time = self._next_frame_idx / self._fps_cap
            return self._sim_time >= due_time

        # wallclock_strict
        if self._start_time is None:
            self._start_time = time.perf_counter()
        now = time.perf_counter()
        target = self._start_time + self._next_frame_idx / max(self._fps_cap, 1e-6)
        return now >= target

    def mark_released(self) -> None:
        """Call once per actually ingested frame."""
        if self._mode == "wallclock_strict":
            # Detect and count dropped frames
            if self._start_time is not None:
                now = time.perf_counter()
                elapsed = now - self._start_time
                expected_idx = int(elapsed * self._fps_cap)
                skipped = max(0, expected_idx - self._next_frame_idx - 1)
                self._frames_dropped += skipped
                if skipped > 0:
                    self._next_frame_idx += skipped  # skip to current
        self._next_frame_idx += 1

    def tick(self, dt: float) -> None:
        """Advance simulated clock by dt seconds (dataset_fps mode only).
        Alternative to passing dt to should_release().
        """
        if self._mode == "dataset_fps":
            self._sim_time += dt

    @property
    def frames_dropped(self) -> int:
        return self._frames_dropped

    @property
    def simulated_time(self) -> float:
        return self._sim_time

    @property
    def wallclock(self) -> bool:
        return self._mode == "wallclock_strict"
