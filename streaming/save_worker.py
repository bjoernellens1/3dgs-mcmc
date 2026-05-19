"""Async save worker — background thread for PLY/checkpoint writes."""
from __future__ import annotations

import queue
import threading


class AsyncSaveWorker:
    """Enqueue callables to run on a background daemon thread.

    Used to write PLY snapshots and optimizer checkpoints without stalling
    the training loop. At most 4 tasks are queued; callers block if the
    worker is too far behind.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=4)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                break
            try:
                task()
            except Exception:
                import traceback
                traceback.print_exc()
            finally:
                self._queue.task_done()

    def enqueue(self, fn) -> None:
        self._queue.put(fn)

    def shutdown(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=120)


# Backward-compat alias used by train_streaming.py during transition
_AsyncSaveWorker = AsyncSaveWorker
