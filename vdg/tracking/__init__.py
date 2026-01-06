"""
Tracking module - Feature point tracking and stabilization.

This module provides:
- FeatureTracker: Lucas-Kanade optical flow tracking with auto-replenishment
- Stabilizer: Compute stabilization transforms from tracking data
- Track data file I/O utilities

Example:
    >>> from vdg.tracking import FeatureTracker
    >>> tracker = FeatureTracker(num_features=50)
    >>> points = tracker.initialize(first_frame)
    >>> for frame in video:
    ...     points, ids, stats = tracker.update(frame)
"""

from vdg.tracking.tracker import FeatureTracker
from vdg.tracking.stabilizer import (
    Stabilizer,
    StabilizeMode,
    read_track_data,
    read_persp_data,
    apply_jitter_filter,
)
from vdg.tracking.track_io import (
    read_crv_file,
    write_crv_file,
    parse_track_line,
)

__all__ = [
    "FeatureTracker",
    "Stabilizer",
    "StabilizeMode",
    "read_track_data",
    "read_persp_data",
    "apply_jitter_filter",
    "read_crv_file",
    "write_crv_file",
    "parse_track_line",
]
