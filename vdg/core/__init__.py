"""
Core module - Base classes, protocols, and shared abstractions.
"""

from vdg.core.base import BaseProcessor, ProcessingContext
from vdg.core.video import VideoReader, VideoWriter, VideoProperties
from vdg.core.config import Config, load_config, save_config
from vdg.core.hardware import (
    hw_config,
    configure_hardware,
    print_hardware_status,
    HardwareConfig,
    HWAccelBackend,
)

__all__ = [
    "BaseProcessor",
    "ProcessingContext",
    "VideoReader",
    "VideoWriter",
    "VideoProperties",
    "Config",
    "load_config",
    "save_config",
    "hw_config",
    "configure_hardware",
    "print_hardware_status",
    "HardwareConfig",
    "HWAccelBackend",
]
