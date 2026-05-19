"""
Odometry sub-package.

Shared types (OdometryStats, PoseReadyFrame) are in async_worker and used
by both the live OrbbecRosBagFrameSource and the Phase 5+ async queue path.
"""
from streaming.odometry.async_worker import (  # noqa: F401
    OdometryStats,
    PoseReadyFrame,
    AsyncOdometryWorker,
    log_odom_stats,
)
