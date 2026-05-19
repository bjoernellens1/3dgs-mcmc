"""
Streaming RGB-D frame sources.

Re-exports all source classes and the auto-detect factory from the
underlying implementation in utils/streaming_frames.py.
"""
from utils.streaming_frames import (
    StreamingRGBDFrame,
    load_frame_rgb,
    load_frame_depth_np,
    RGBDSequenceFrameSource,
    TUMFrameSource,
    ScanNetFrameSource,
    ScanNetSensFrameSource,
    ReplicaFrameSource,
    HyperSimFrameSource,
    OrbbecExportFrameSource,
    OrbbecRosBagFrameSource,
    RealsenseRosBagFrameSource,
    make_frame_source,
)

__all__ = [
    "StreamingRGBDFrame",
    "load_frame_rgb",
    "load_frame_depth_np",
    "RGBDSequenceFrameSource",
    "TUMFrameSource",
    "ScanNetFrameSource",
    "ScanNetSensFrameSource",
    "ReplicaFrameSource",
    "HyperSimFrameSource",
    "OrbbecExportFrameSource",
    "OrbbecRosBagFrameSource",
    "RealsenseRosBagFrameSource",
    "make_frame_source",
]
