"""
Processing module - Frame averaging and color correction.

This module provides:
- FrameAverager: Temporal frame averaging with alpha masking
- ColorCorrector: Gamma, exposure, and CLAHE adjustments
- Filters for frame processing
"""

from vdg.processing.frame_average import (
    FrameAverager,
    FrameAverageConfig,
    FrameAverageResult,
    CompMode,
    average_frames,
)
from vdg.processing.color import (
    ColorCorrector,
    GammaFilter,
    ContrastFilter,
    BrightnessFilter,
    ExposureFilter,
    CLAHEFilter,
    create_gamma_lut,
    create_soft_contrast_lut,
    create_brightness_lut,
    create_exposure_lut,
    create_clahe,
    apply_clahe,
    apply_lut,
    smoothstep,
    smootherstep,
)

__all__ = [
    # Frame averaging
    "FrameAverager",
    "FrameAverageConfig",
    "FrameAverageResult",
    "CompMode",
    "average_frames",
    # Color correction
    "ColorCorrector",
    "GammaFilter",
    "ContrastFilter",
    "BrightnessFilter",
    "ExposureFilter",
    "CLAHEFilter",
    "create_gamma_lut",
    "create_soft_contrast_lut",
    "create_brightness_lut",
    "create_exposure_lut",
    "create_clahe",
    "apply_clahe",
    "apply_lut",
    "smoothstep",
    "smootherstep",
]
