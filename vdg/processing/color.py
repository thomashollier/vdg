"""
Color correction utilities for frame processing.

This module provides LUT generation and color correction filters
that can be applied to frames before or after processing.
"""

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from vdg.core.base import FrameFilter


def create_gamma_lut(gamma: float) -> np.ndarray:
    """
    Create a gamma correction lookup table.

    Args:
        gamma: Gamma value (>1 darkens, <1 brightens)

    Returns:
        3-channel LUT for use with cv2.LUT()
    """
    identity = np.arange(256, dtype=np.uint8)
    fidentity = identity.astype(np.float32) / 255.0
    fidentity = np.power(fidentity, gamma) * 255
    identity = fidentity.astype(np.uint8)
    return np.dstack((identity, identity, identity))


def create_soft_contrast_lut(contrast: float) -> np.ndarray:
    """
    Create a soft contrast lookup table.

    Uses an S-curve for natural-looking contrast enhancement.

    Args:
        contrast: Contrast strength (1.0 = no change, >1 = more contrast)

    Returns:
        3-channel LUT for use with cv2.LUT()
    """
    def soft_contrast_calc(x: float, k: float) -> float:
        x1 = 0.5 * pow(2 * x, k)
        x2 = 0.5 * pow(2 * (1 - x), k)
        x3 = x1 if x < 0.5 else x2
        return x3 if x < 0.5 else (1 - x3)

    identity = np.arange(256, dtype=np.uint8)
    fidentity = identity.astype(np.float32) / 255.0
    for i, v in enumerate(fidentity):
        fidentity[i] = soft_contrast_calc(v, contrast) * 255
    identity = fidentity.astype(np.uint8)
    return np.dstack((identity, identity, identity))


def create_brightness_lut(brightness: float) -> np.ndarray:
    """
    Create a brightness adjustment lookup table.

    Args:
        brightness: Brightness multiplier (1.0 = no change)

    Returns:
        3-channel LUT for use with cv2.LUT()
    """
    identity = np.arange(256, dtype=np.float32)
    identity = np.clip(identity * brightness, 0, 255).astype(np.uint8)
    return np.dstack((identity, identity, identity))


def create_exposure_lut(stops: float) -> np.ndarray:
    """
    Create an exposure adjustment lookup table.

    Args:
        stops: Exposure adjustment in stops (+1 = double brightness)

    Returns:
        3-channel LUT for use with cv2.LUT()
    """
    multiplier = pow(2, stops)
    return create_brightness_lut(multiplier)


def create_clahe(
    clip_limit: float = 40.0,
    grid_size: int = 8,
) -> cv2.CLAHE:
    """
    Create a CLAHE (Contrast Limited Adaptive Histogram Equalization) object.

    Args:
        clip_limit: Threshold for contrast limiting
        grid_size: Size of grid for histogram equalization

    Returns:
        OpenCV CLAHE object
    """
    return cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_size, grid_size)
    )


def apply_clahe(
    frame: np.ndarray,
    clip_limit: float = 40.0,
    grid_size: int = 8,
) -> np.ndarray:
    """
    Apply CLAHE to each channel of a frame.

    Args:
        frame: Input frame (uint8, uint16, or float32)
        clip_limit: CLAHE clip limit
        grid_size: CLAHE grid size

    Returns:
        Processed frame in same format as input
    """
    clahe = create_clahe(clip_limit, grid_size)

    if frame.dtype == np.float32:
        # Convert to 16-bit for processing
        frame_16 = (frame * 65535).astype(np.uint16)
        channels = [clahe.apply(frame_16[:, :, i]) for i in range(3)]
        result = cv2.merge(channels)
        return result.astype(np.float32) / 65535.0
    else:
        channels = [clahe.apply(frame[:, :, i]) for i in range(3)]
        return cv2.merge(channels)


def apply_lut(frame: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Apply a LUT to a frame with automatic type conversion.

    Args:
        frame: Input frame (uint8 or float32)
        lut: Lookup table from create_*_lut functions

    Returns:
        Processed frame in same format as input
    """
    if frame.dtype == np.float32:
        frame_uint8 = (frame * 255).astype(np.uint8)
        result = cv2.LUT(frame_uint8, lut)
        return result.astype(np.float32) / 255.0
    else:
        return cv2.LUT(frame, lut)


def smoothstep(x: np.ndarray) -> np.ndarray:
    """
    Apply smoothstep function (cubic Hermite interpolation).

    Useful for smooth alpha transitions.

    Args:
        x: Input array (should be in 0-1 range)

    Returns:
        Smoothstepped array
    """
    return x * x * (3 - 2 * x)


def smootherstep(x: np.ndarray) -> np.ndarray:
    """
    Apply Ken Perlin's smootherstep function.

    Even smoother than smoothstep with zero second derivatives at edges.

    Args:
        x: Input array (should be in 0-1 range)

    Returns:
        Processed array
    """
    return x * x * x * (x * (x * 6 - 15) + 10)


@dataclass
class GammaFilter:
    """Frame filter that applies gamma correction."""
    gamma: float = 2.2

    def __post_init__(self):
        self._lut = create_gamma_lut(self.gamma)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return apply_lut(frame, self._lut)


@dataclass
class ContrastFilter:
    """Frame filter that applies soft contrast."""
    contrast: float = 1.5

    def __post_init__(self):
        self._lut = create_soft_contrast_lut(self.contrast)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return apply_lut(frame, self._lut)


@dataclass
class BrightnessFilter:
    """Frame filter that applies brightness adjustment."""
    brightness: float = 1.0

    def __post_init__(self):
        self._lut = create_brightness_lut(self.brightness)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return apply_lut(frame, self._lut)


@dataclass
class ExposureFilter:
    """Frame filter that applies exposure adjustment in stops."""
    stops: float = 0.0

    def __post_init__(self):
        self._lut = create_exposure_lut(self.stops)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return apply_lut(frame, self._lut)


@dataclass
class CLAHEFilter:
    """Frame filter that applies CLAHE contrast enhancement."""
    clip_limit: float = 40.0
    grid_size: int = 8

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return apply_clahe(frame, self.clip_limit, self.grid_size)


class ColorCorrector:
    """
    Composable color correction pipeline.

    Chains multiple color corrections together for efficient
    processing. LUT-based corrections are combined when possible.

    Example:
        >>> corrector = ColorCorrector()
        >>> corrector.add_gamma(2.2)
        >>> corrector.add_contrast(1.3)
        >>> corrector.add_clahe(clip_limit=40)
        >>>
        >>> corrected = corrector.apply(frame)
    """

    def __init__(self):
        self._filters: list[FrameFilter] = []
        self._combined_lut: np.ndarray | None = None

    def add_gamma(self, gamma: float) -> "ColorCorrector":
        """Add gamma correction."""
        self._filters.append(GammaFilter(gamma))
        self._combined_lut = None
        return self

    def add_contrast(self, contrast: float) -> "ColorCorrector":
        """Add soft contrast."""
        self._filters.append(ContrastFilter(contrast))
        self._combined_lut = None
        return self

    def add_brightness(self, brightness: float) -> "ColorCorrector":
        """Add brightness adjustment."""
        self._filters.append(BrightnessFilter(brightness))
        self._combined_lut = None
        return self

    def add_exposure(self, stops: float) -> "ColorCorrector":
        """Add exposure adjustment in stops."""
        self._filters.append(ExposureFilter(stops))
        self._combined_lut = None
        return self

    def add_clahe(
        self,
        clip_limit: float = 40.0,
        grid_size: int = 8,
    ) -> "ColorCorrector":
        """Add CLAHE contrast enhancement."""
        self._filters.append(CLAHEFilter(clip_limit, grid_size))
        return self

    def add_filter(self, filter_: FrameFilter) -> "ColorCorrector":
        """Add a custom filter."""
        self._filters.append(filter_)
        self._combined_lut = None
        return self

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply all corrections to a frame."""
        result = frame
        for f in self._filters:
            result = f.apply(result)
        return result

    def clear(self) -> "ColorCorrector":
        """Remove all filters."""
        self._filters.clear()
        self._combined_lut = None
        return self

    def __len__(self) -> int:
        return len(self._filters)
