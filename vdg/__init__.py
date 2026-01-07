"""
VDG - Video Development and Grading Toolkit
============================================

A modular Python framework for video point tracking, stabilization,
frame averaging, and post-processing.

Main modules:
- vdg.tracking: Point tracking and stabilization
- vdg.processing: Frame averaging and color correction
- vdg.postprocess: Image post-processing operations
- vdg.outputs: Output handlers (video, CSV, trackers)
- vdg.pipeline: Batch processing and job management

Hardware Acceleration:
    VDG supports hardware-accelerated video encoding/decoding via FFmpeg.
    Configure with:
    
    >>> import vdg
    >>> vdg.configure_hardware(enabled=False)  # Disable HW accel
    >>> vdg.print_hardware_status()            # Show current status

Quick start:
    >>> from vdg.tracking import FeatureTracker
    >>> tracker = FeatureTracker(num_features=50)
    >>> points = tracker.initialize(first_frame)
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Convenience imports
from vdg.tracking import FeatureTracker
from vdg.outputs import OutputManager, OutputSpec
from vdg.core.hardware import (
    hw_config,
    configure_hardware,
    print_hardware_status,
    HWAccelBackend,
)

__all__ = [
    "__version__",
    "FeatureTracker",
    "OutputManager",
    "OutputSpec",
    "hw_config",
    "configure_hardware",
    "print_hardware_status",
    "HWAccelBackend",
]
