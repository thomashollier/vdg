"""
Utilities module - Shared helper functions.

This module provides:
- Metadata handling (exiftool integration)
- FFmpeg wrapper utilities
- Math utilities
"""

from vdg.utils.metadata import get_metadata, set_metadata, copy_metadata

__all__ = [
    "get_metadata",
    "set_metadata",
    "copy_metadata",
]
