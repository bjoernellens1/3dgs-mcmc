"""Streaming training state dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.streaming_frames import StreamingRGBDFrame


@dataclass
class StreamingTrainState:
    """Mutable loop counters passed through the streaming training loop."""
    iteration: int = 0
    n_frames_ingested: int = 0
    n_frames_trained: int = 0
    total_inserted: int = 0
    prev_insert_frame: Optional["StreamingRGBDFrame"] = None
    pending_insertions: list = field(default_factory=list)
    pending_insertion_frames: int = 0


@dataclass
class InsertionResult:
    added: int
    stats: dict = field(default_factory=dict)


@dataclass
class OdometryResult:
    """Pose estimate returned by a live or precomputed odometry source."""
    c2w: "np.ndarray"  # noqa: F821
    valid: bool
    translation_m: float = 0.0
    rotation_deg: float = 0.0
    residual: Optional[float] = None
    reason: str = ""
