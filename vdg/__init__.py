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

__all__ = [
    "__version__",
    "FeatureTracker",
    "OutputManager",
    "OutputSpec",
]
