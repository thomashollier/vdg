"""
Frame averaging for temporal compositing.

This module provides frame averaging functionality for creating
time-collapsed images from video sequences.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from vdg.core.base import BaseProcessor, ProcessingContext
from vdg.core.video import VideoReader, get_video_properties


class CompMode(Enum):
    """Compositing mode for alpha handling."""
    ON_BLACK = 0      # Standard averaging on black background
    ON_WHITE = 1      # Composite on white background
    UNPREMULT = 2     # Unpremultiply alpha, composite on black
    UNPREMULT_WHITE = 3  # Unpremultiply alpha, composite on white


@dataclass
class FrameAverageConfig:
    """Configuration for frame averaging."""
    # Frame range
    frame_start: int = 1
    frame_end: int | None = None

    # Output settings
    brightness: float = 1.0
    add_mode: bool = False  # If True, don't divide by frame count

    # Compositing
    comp_mode: CompMode = CompMode.ON_BLACK
    use_mask: bool = False
    mask_path: Path | str | None = None

    # Alpha adjustments
    alpha_gamma: float = 1.0
    alpha_smoothstep: bool = False

    # Color corrections (applied per-frame before averaging)
    gamma: float = 1.0
    gamma_before: bool = False  # Apply gamma before adding to stack
    soft_contrast: float = 1.0

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe: bool = False
    clahe_clip: float = 40.0
    clahe_grid: int = 8
    clahe_before: bool = False  # Apply CLAHE before adding to stack

    # Output
    write_alpha: bool = False
    output_16bit: bool = True  # Output as 16-bit PNG


@dataclass
class FrameAverageResult:
    """Result of frame averaging operation."""
    output_path: Path
    alpha_path: Path | None = None
    frame_count: int = 0
    config: FrameAverageConfig = field(default_factory=FrameAverageConfig)


def _make_gamma_lut(gamma: float) -> np.ndarray:
    """Create a gamma correction LUT."""
    identity = np.arange(256, dtype=np.uint8)
    fidentity = identity.astype(np.float32) / 255.0
    fidentity = np.power(fidentity, gamma) * 255
    identity = fidentity.astype(np.uint8)
    return np.dstack((identity, identity, identity))


def _make_soft_contrast_lut(contrast: float) -> np.ndarray:
    """Create a soft contrast LUT."""
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


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """Apply smoothstep function to array."""
    return x * x * (3 - 2 * x)


def _apply_clahe(
    frame: np.ndarray,
    clip_limit: float = 40.0,
    grid_size: int = 8,
) -> np.ndarray:
    """Apply CLAHE to each channel of a frame."""
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_size, grid_size)
    )

    if frame.dtype == np.float32:
        # Convert to 16-bit for CLAHE processing
        frame_16 = (frame * 65535).astype(np.uint16)
        r = clahe.apply(frame_16[:, :, 0])
        g = clahe.apply(frame_16[:, :, 1])
        b = clahe.apply(frame_16[:, :, 2])
        result = cv2.merge([r, g, b])
        return result.astype(np.float32) / 65535.0
    elif frame.dtype == np.uint16:
        r = clahe.apply(frame[:, :, 0])
        g = clahe.apply(frame[:, :, 1])
        b = clahe.apply(frame[:, :, 2])
        return cv2.merge([r, g, b])
    else:
        # Assume uint8
        r = clahe.apply(frame[:, :, 0])
        g = clahe.apply(frame[:, :, 1])
        b = clahe.apply(frame[:, :, 2])
        return cv2.merge([r, g, b])


def _compute_alpha(frame_data: np.ndarray) -> np.ndarray:
    """
    Compute alpha from frame luminance.

    Uses a power function to create smooth falloff from bright areas.
    """
    # Scale up and clamp to get alpha
    alpha = np.power(np.clip(frame_data * 255 / 16.0, 0, 1), 4)
    return alpha


class FrameAverager(BaseProcessor):
    """
    Temporal frame averager for creating time-collapsed images.

    Accumulates frames over time and produces an averaged result,
    with optional alpha masking and color corrections.

    Example:
        >>> config = FrameAverageConfig(
        ...     frame_start=1,
        ...     frame_end=500,
        ...     comp_mode=CompMode.ON_WHITE,
        ...     clahe=True,
        ... )
        >>> averager = FrameAverager(config)
        >>> result = averager.process("input.mp4", "output.png")
    """

    def __init__(
        self,
        config: FrameAverageConfig | None = None,
        context: ProcessingContext | None = None,
    ):
        """
        Initialize the frame averager.

        Args:
            config: Frame averaging configuration
            context: Processing context
        """
        super().__init__(context)
        self.config = config or FrameAverageConfig()

        # Buffers
        self._buffer: np.ndarray | None = None
        self._alpha_add: np.ndarray | None = None
        self._alpha_max: np.ndarray | None = None

        # LUTs
        self._contrast_lut: np.ndarray | None = None
        self._gamma_lut: np.ndarray | None = None

        # State
        self._frame_count = 0
        self._mask_reader: VideoReader | None = None
        self._video_props: dict[str, Any] | None = None

        # Pre-compute LUTs if needed
        if self.config.soft_contrast != 1.0:
            self._contrast_lut = _make_soft_contrast_lut(self.config.soft_contrast)
        if self.config.gamma != 1.0 and self.config.gamma_before:
            self._gamma_lut = _make_gamma_lut(1.0 / self.config.gamma)

    def initialize(self, video_props: dict[str, Any]) -> None:
        """Initialize buffers based on video properties."""
        self._video_props = video_props
        height, width = video_props['height'], video_props['width']

        # Initialize accumulation buffer
        self._buffer = np.zeros((height, width, 3), dtype=np.float32)

        # Initialize alpha buffers if needed
        if self.config.comp_mode != CompMode.ON_BLACK or self.config.write_alpha:
            self._alpha_add = np.zeros((height, width, 3), dtype=np.float32)
            if self.config.comp_mode == CompMode.UNPREMULT_WHITE:
                self._alpha_max = np.zeros((height, width, 3), dtype=np.float32)

        self._frame_count = 0
        self._initialized = True

    def process_frame(
        self,
        frame_num: int,
        frame: np.ndarray,
        **kwargs,
    ) -> np.ndarray | None:
        """
        Add a frame to the accumulation buffer.

        Args:
            frame_num: Current frame number
            frame: BGR frame as numpy array (uint8)
            **kwargs: May contain 'mask_frame' for external mask

        Returns:
            None (accumulation only, no output per frame)
        """
        # Convert to float
        frame_data = frame.astype(np.float32) / 255.0

        # Apply CLAHE before if configured
        if self.config.clahe and self.config.clahe_before:
            frame_data = _apply_clahe(
                frame_data,
                self.config.clahe_clip,
                self.config.clahe_grid,
            )

        # Apply gamma before if configured
        if self.config.gamma != 1.0 and self.config.gamma_before:
            frame_data = np.power(frame_data, 1.0 / self.config.gamma)

        # Apply soft contrast if configured
        if self._contrast_lut is not None:
            frame_uint8 = (frame_data * 255).astype(np.uint8)
            frame_data = cv2.LUT(frame_uint8, self._contrast_lut).astype(np.float32) / 255.0

        # Accumulate
        self._buffer += frame_data

        # Handle alpha
        if self._alpha_add is not None:
            mask_frame = kwargs.get('mask_frame')
            if mask_frame is not None:
                alpha = mask_frame.astype(np.float32) / 255.0
            else:
                alpha = _compute_alpha(frame_data)

            self._alpha_add += alpha

            if self._alpha_max is not None:
                self._alpha_max = np.maximum(alpha, self._alpha_max)

        self._frame_count += 1
        return None  # No per-frame output

    def finalize(self) -> np.ndarray:
        """
        Finalize averaging and return the result.

        Returns:
            Averaged frame as float32 array (0-1 range)
        """
        if self._frame_count == 0:
            raise RuntimeError("No frames were processed")

        result = self._buffer.copy()

        if not self.config.add_mode:
            # Divide by frame count
            result /= self._frame_count

            if self._alpha_add is not None:
                alpha = np.clip(self._alpha_add / self._frame_count, 0.000003, 1.0)

                # Apply compositing mode
                if self.config.comp_mode == CompMode.ON_WHITE:
                    if self.config.alpha_gamma != 1.0:
                        result = result / alpha
                        alpha = np.power(alpha, 1.0 / self.config.alpha_gamma)
                        result = result * alpha
                    if self.config.alpha_smoothstep:
                        alpha = _smoothstep(alpha)
                    result = (1 - alpha) + result

                elif self.config.comp_mode == CompMode.UNPREMULT:
                    result = result / alpha

                elif self.config.comp_mode == CompMode.UNPREMULT_WHITE:
                    result = result / alpha
                    if self._alpha_max is not None:
                        result = (1 - self._alpha_max) + result

        # Apply brightness
        if self.config.brightness != 1.0:
            result *= self.config.brightness

        # Apply gamma after if configured
        if self.config.gamma != 1.0 and not self.config.gamma_before:
            result = np.power(result, 1.0 / self.config.gamma)

        # Apply CLAHE after if configured
        if self.config.clahe and not self.config.clahe_before:
            result = _apply_clahe(
                result,
                self.config.clahe_clip,
                self.config.clahe_grid,
            )
            result = np.clip(result, 0, 1)

        return result

    def get_alpha(self) -> np.ndarray | None:
        """Get the accumulated alpha channel."""
        if self._alpha_add is None or self._frame_count == 0:
            return None
        return np.clip(self._alpha_add / self._frame_count, 0, 1)

    def process(
        self,
        input_path: str | Path,
        output_path: str | Path,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> FrameAverageResult:
        """
        Process a video file and output averaged image.

        Args:
            input_path: Path to input video
            output_path: Path for output image (PNG recommended)
            progress_callback: Optional callback(frame, total, percent)

        Returns:
            FrameAverageResult with paths and statistics
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Get video properties
        props = get_video_properties(input_path)

        # Determine frame range
        frame_start = self.config.frame_start
        frame_end = self.config.frame_end
        if frame_end is None or frame_end > props.frame_count:
            frame_end = props.frame_count

        # Initialize
        self.initialize(props.to_dict())

        # Set up mask reader if needed
        mask_reader = None
        if self.config.use_mask and self.config.mask_path:
            mask_path = Path(self.config.mask_path)
            if mask_path.exists():
                mask_reader = VideoReader(
                    mask_path,
                    first_frame=frame_start,
                    last_frame=frame_end,
                )
                mask_reader.open()

        total_frames = frame_end - frame_start + 1

        # Process frames
        with VideoReader(input_path, frame_start, frame_end) as reader:
            for frame_num, frame in reader:
                # Get mask frame if available
                mask_frame = None
                if mask_reader:
                    ret, mask_frame = mask_reader.read_frame()
                    if not ret:
                        mask_frame = None

                self.process_frame(frame_num, frame, mask_frame=mask_frame)

                if progress_callback:
                    current = frame_num - frame_start + 1
                    percent = 100.0 * current / total_frames
                    progress_callback(current, total_frames, percent)

        if mask_reader:
            mask_reader.close()

        # Finalize and save
        result = self.finalize()

        # Convert to 16-bit for output
        if self.config.output_16bit:
            result = np.clip(result * 65535, 0, 65535).astype(np.uint16)
        else:
            result = np.clip(result * 255, 0, 255).astype(np.uint8)

        cv2.imwrite(str(output_path), result)

        # Write alpha if requested
        alpha_path = None
        if self.config.write_alpha:
            alpha = self.get_alpha()
            if alpha is not None:
                alpha_path = output_path.with_stem(output_path.stem + "_alpha")
                if self.config.output_16bit:
                    alpha = np.clip(alpha * 65535, 0, 65535).astype(np.uint16)
                else:
                    alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)
                cv2.imwrite(str(alpha_path), alpha)

        return FrameAverageResult(
            output_path=output_path,
            alpha_path=alpha_path,
            frame_count=self._frame_count,
            config=self.config,
        )


def average_frames(
    input_path: str | Path,
    output_path: str | Path,
    frame_start: int = 1,
    frame_end: int | None = None,
    comp_mode: CompMode = CompMode.ON_BLACK,
    brightness: float = 1.0,
    gamma: float = 1.0,
    clahe: bool = False,
    clahe_clip: float = 40.0,
    verbose: bool = True,
) -> FrameAverageResult:
    """
    Convenience function for simple frame averaging.

    Args:
        input_path: Path to input video
        output_path: Path for output image
        frame_start: First frame to include
        frame_end: Last frame to include (None = all)
        comp_mode: Compositing mode
        brightness: Brightness multiplier
        gamma: Gamma correction
        clahe: Enable CLAHE contrast enhancement
        clahe_clip: CLAHE clip limit
        verbose: Print progress

    Returns:
        FrameAverageResult with output information
    """
    config = FrameAverageConfig(
        frame_start=frame_start,
        frame_end=frame_end,
        comp_mode=comp_mode,
        brightness=brightness,
        gamma=gamma,
        clahe=clahe,
        clahe_clip=clahe_clip,
    )

    averager = FrameAverager(config)

    def progress(current: int, total: int, percent: float) -> None:
        if verbose:
            import sys
            remaining = ""
            sys.stdout.write(f"\rFrame {current} of {total} -- {percent:.1f}% complete{remaining}")
            sys.stdout.flush()

    result = averager.process(input_path, output_path, progress_callback=progress)

    if verbose:
        print(f"\n\nComplete: {output_path}")

    return result
