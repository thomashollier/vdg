"""
Output handlers module.

This module provides extensible output handlers for video tracking and processing:
- PreviewTrackOutput: Video with tracking overlay
- CSVOutput: Tracking data as CSV
- TrackersOutput: Normalized coordinates (.crv format)
- CleanVideoOutput: Stabilized video without overlays

Example:
    >>> from vdg.outputs import OutputManager
    >>> manager = OutputManager("input.mp4")
    >>> manager.add_output("previewtrack=filename=preview.mp4")
    >>> manager.add_output("csv")
    >>> manager.add_output("trackers=type=2:filter=3")
"""

from vdg.outputs.base import OutputSpec, BaseOutput
from vdg.outputs.video import PreviewTrackOutput, CleanVideoOutput
from vdg.outputs.data import CSVOutput, TrackersOutput
from vdg.outputs.manager import OutputManager, parse_output_specs, register_output_type

__all__ = [
    "OutputSpec",
    "BaseOutput",
    "PreviewTrackOutput",
    "CleanVideoOutput", 
    "CSVOutput",
    "TrackersOutput",
    "OutputManager",
    "parse_output_specs",
    "register_output_type",
]
