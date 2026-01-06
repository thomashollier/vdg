"""
Core module - Base classes, protocols, and shared abstractions.
"""

from vdg.core.base import BaseProcessor, ProcessingContext
from vdg.core.video import VideoReader, VideoWriter, VideoProperties
from vdg.core.config import Config, load_config, save_config

__all__ = [
    "BaseProcessor",
    "ProcessingContext",
    "VideoReader",
    "VideoWriter",
    "VideoProperties",
    "Config",
    "load_config",
    "save_config",
]
