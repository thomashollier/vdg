"""
Video stabilization based on tracking data.

This module provides stabilization transforms computed from tracking data,
supporting translation, rotation, scale, and perspective modes.
"""

import math
import re
from enum import IntEnum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


class StabilizeMode(IntEnum):
    """Stabilization modes."""
    OFF = 0
    SINGLE_POINT = 1  # Translation only
    TWO_POINT = 2     # Translation + rotation + scale
    VSTAB = 3         # Vertical alignment


def orient_coordinates(
    xy: tuple[float, float],
    movie_width: int,
    movie_height: int,
    portrait: bool,
    swap_xy: bool,
    x_flip: bool,
    y_flip: bool,
    apply_scale: bool = True,
) -> list[float]:
    """
    Transform coordinates based on orientation and flip settings.
    
    Args:
        xy: Normalized (0-1) coordinates
        movie_width: Video width in pixels
        movie_height: Video height in pixels
        portrait: True if video is portrait orientation
        swap_xy: Swap X and Y values
        x_flip: Invert X coordinate
        y_flip: Invert Y coordinate
        apply_scale: Scale to pixel coordinates
        
    Returns:
        Transformed coordinates as [x, y]
    """
    # Handle portrait/landscape orientation
    if portrait:
        # When movie loads up rotated 90deg CCW in blender
        y = xy[0]
        x = xy[1]
    else:
        # When movie shows up landscape right side up in blender
        x = xy[0]
        y = 1 - xy[1]
    
    # Swap XY if requested
    if swap_xy:
        y, x = x, y
    
    # Flip coordinates
    if x_flip:
        x = 1.0 - float(x)
    if y_flip:
        y = 1.0 - float(y)
    
    # Scale to pixel coordinates
    if apply_scale:
        x = x * movie_width
        y = y * movie_height
    
    return [x, y]


class Stabilizer:
    """
    Compute stabilization transforms from tracking data.
    
    Supports single-point (translation only), two-point (translation +
    rotation + scale), and perspective stabilization modes.
    
    Example:
        >>> stab = Stabilizer(mode=StabilizeMode.TWO_POINT)
        >>> stab.set_reference(ref_markers)
        >>> for frame, markers in tracking_data:
        ...     matrix = stab.compute_transform(markers)
        ...     stabilized = cv2.warpAffine(frame, matrix, (w, h))
    """
    
    def __init__(
        self,
        mode: StabilizeMode = StabilizeMode.TWO_POINT,
        pos_track: int = 0,
        rot0_track: int = 0,
        rot1_track: int = 1,
    ):
        """
        Initialize the stabilizer.
        
        Args:
            mode: Stabilization mode
            pos_track: Which track to use for position
            rot0_track: First track for rotation calculation
            rot1_track: Second track for rotation calculation
        """
        self.mode = mode
        self.pos_track = pos_track
        self.rot0_track = rot0_track
        self.rot1_track = rot1_track
        
        self.ref_markers: list[tuple[float, float]] | None = None
        self.movie_width = 0
        self.movie_height = 0
    
    def set_dimensions(self, width: int, height: int) -> None:
        """Set video dimensions for coordinate scaling."""
        self.movie_width = width
        self.movie_height = height
    
    def set_reference(self, markers: list[tuple[float, float]]) -> None:
        """
        Set the reference marker positions.
        
        Args:
            markers: List of (x, y) marker positions (normalized 0-1)
        """
        self.ref_markers = markers
    
    def _norm_to_real(self, pnt: tuple[float, float]) -> list[float]:
        """Convert normalized coordinates to pixel coordinates."""
        return [pnt[0] * self.movie_width, pnt[1] * self.movie_height]
    
    def compute_transform(
        self,
        markers: list[tuple[float, float]],
        x_offset: float = 0,
        y_offset: float = 0,
        r_offset: float = 0,
        scale_mult: float = 1.0,
        x_ignore: bool = False,
        y_ignore: bool = False,
    ) -> np.ndarray:
        """
        Compute the stabilization transform matrix.
        
        Args:
            markers: Current marker positions (normalized 0-1)
            x_offset: Offset in X direction (pixels)
            y_offset: Offset in Y direction (pixels)
            r_offset: Rotation offset (radians)
            scale_mult: Scale multiplier
            x_ignore: Ignore X tracking
            y_ignore: Ignore Y tracking
            
        Returns:
            2x3 affine transform matrix
        """
        if self.ref_markers is None:
            raise ValueError("Reference markers not set. Call set_reference() first.")
        
        rot = 0
        scale = scale_mult
        
        # Get position tracking point
        pnt0 = self._norm_to_real(markers[self.pos_track])
        ref0 = self._norm_to_real(self.ref_markers[self.pos_track])
        pnt0x, pnt0y = pnt0
        ref0x, ref0y = ref0
        
        # Apply ignore flags
        if x_ignore:
            pnt0x = ref0x
        if y_ignore:
            pnt0y = ref0y
        
        # Calculate rotation and scale if we have at least 2 markers
        if len(markers) >= 2 and self.mode in (StabilizeMode.TWO_POINT, StabilizeMode.VSTAB):
            pnt_r0 = self._norm_to_real(markers[self.rot0_track])
            ref_r0 = self._norm_to_real(self.ref_markers[self.rot0_track])
            pnt_r0x, pnt_r0y = pnt_r0
            ref_r0x, ref_r0y = ref_r0
            
            pnt_r1 = self._norm_to_real(markers[self.rot1_track])
            ref_r1 = self._norm_to_real(self.ref_markers[self.rot1_track])
            pnt_r1x, pnt_r1y = pnt_r1
            ref_r1x, ref_r1y = ref_r1
            
            # Calculate vectors between rotation tracking points
            ref_vector_x = ref_r1x - ref_r0x
            ref_vector_y = ref_r1y - ref_r0y
            pnt_vector_x = pnt_r1x - pnt_r0x
            pnt_vector_y = pnt_r1y - pnt_r0y
            
            # Calculate angles
            ref_angle = math.atan2(ref_vector_y, ref_vector_x) + 2 * math.pi
            pnt_angle = math.atan2(pnt_vector_y, pnt_vector_x) + 2 * math.pi
            
            # Calculate rotation and scale
            rot = (ref_angle - pnt_angle) + r_offset
            
            ref_dist = math.dist([ref_r0x, ref_r0y], [ref_r1x, ref_r1y])
            pnt_dist = math.dist([pnt_r0x, pnt_r0y], [pnt_r1x, pnt_r1y])
            if pnt_dist > 0:
                scale = (ref_dist / pnt_dist) * scale_mult
        
        # Build transformation matrices
        M_offset = np.float32([
            [1, 0, -pnt0x],
            [0, 1, -pnt0y],
            [0, 0, 1]
        ])
        
        M_place = np.float32([
            [1, 0, ref0x + x_offset],
            [0, 1, ref0y + y_offset],
            [0, 0, 1]
        ])
        
        # Rotation and scale matrix using OpenCV
        Mr = cv2.getRotationMatrix2D([0, 0], -math.degrees(rot), scale)
        Mr = np.append(Mr, [[0, 0, 1]], axis=0)
        
        # Combine transformations
        M = np.matmul(Mr, M_offset)
        M = np.matmul(M_place, M)
        
        return M[:2]
    
    def compute_perspective_transform(
        self,
        markers: list[tuple[float, float]],
        dst_offset: tuple[float, float] = (0, 0),
        dst_scale: tuple[float, float] = (1, 1),
        r_offset: float = 0,
    ) -> np.ndarray:
        """
        Compute a perspective transform matrix.
        
        Args:
            markers: Current marker positions (4 points, normalized 0-1)
            dst_offset: Destination offset (x, y) in pixels
            dst_scale: Destination scale (x, y)
            r_offset: Rotation offset (radians)
            
        Returns:
            3x3 perspective transform matrix
        """
        if self.ref_markers is None or len(self.ref_markers) < 4:
            raise ValueError("Need 4 reference markers for perspective transform")
        
        if len(markers) < 4:
            raise ValueError("Need 4 markers for perspective transform")
        
        # Scale and offset destination points
        dst_scale_arr = np.float32(dst_scale)
        dst_offset_arr = np.float32(dst_offset)
        
        dst_points = np.float32([
            self._norm_to_real(p) for p in self.ref_markers[:4]
        ]) * dst_scale_arr + dst_offset_arr
        
        src_points = np.float32([
            self._norm_to_real(p) for p in markers[:4]
        ])
        
        # Get perspective transform
        mtx = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply rotation if specified
        if r_offset != 0:
            center = [self.movie_width / 2, self.movie_height / 2]
            R = cv2.getRotationMatrix2D(center, -math.degrees(r_offset), 1)
            R = np.vstack([R, np.float32([0, 0, 1])])
            mtx = np.matmul(R, mtx)
        
        return mtx


def read_track_data(
    track_data_files: str,
    ref_frame: int,
    frame_offset: int,
    movie_width: int,
    movie_height: int,
    portrait: bool,
    swap_xy: bool,
    x_flip: bool,
    y_flip: bool,
) -> dict[str, Any]:
    """
    Read tracking data from file(s).
    
    Supports multiple tracker files separated by ':'.
    Format: FRAME [[x, y]] or FRAME x y
    
    Args:
        track_data_files: Colon-separated list of track files
        ref_frame: Reference frame number
        frame_offset: Frame number offset
        movie_width: Video width
        movie_height: Video height
        portrait: True if video is portrait
        swap_xy: Swap X and Y values
        x_flip: Invert X coordinate
        y_flip: Invert Y coordinate
        
    Returns:
        Dictionary with tracking data
    """
    trackers = track_data_files.split(':')
    tmp_dict: dict[int, dict] = {}
    
    # Pattern to match [[x, y]] format
    bracket_pattern = re.compile(r'\[\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\]')
    
    # Read first file to get frame range
    with open(trackers[0]) as f:
        lines = f.readlines()
        st = int(lines[0].split()[0])
        frames = set(range(st, st + len(lines)))
    
    # Read each tracker file
    for i, tracker_file in enumerate(trackers):
        tmp_dict[i] = {}
        with open(tracker_file) as f:
            tracker_lines = f.readlines()
        
        for line in tracker_lines:
            parts = line.strip().split()
            frame_num = int(parts[0]) + frame_offset
            
            # Try bracket format first
            match = bracket_pattern.search(line)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
            else:
                # Fall back to space-separated format
                x = float(parts[1])
                y = float(parts[2])
            
            xy = orient_coordinates(
                (x, y), movie_width, movie_height,
                portrait, swap_xy, x_flip, y_flip,
                apply_scale=False
            )
            tmp_dict[i][frame_num] = xy
    
    # Determine frame range
    ff = min(min(d.keys()) for d in tmp_dict.values())
    lf = max(max(d.keys()) for d in tmp_dict.values())
    
    # Set reference frame
    if ref_frame == -1:
        ref_frame = ff
    
    # Build output dictionary
    trackers_dict = {
        'ff': ff,
        'lf': lf,
        'refFrameInMovie': ref_frame,
        'trackerData': {}
    }
    
    for frame in range(ff, lf + 1):
        marker_data = []
        for i in range(len(trackers)):
            if frame in tmp_dict[i]:
                marker_data.append(tmp_dict[i][frame])
            else:
                # Use previous frame's data if available
                marker_data.append(marker_data[-1] if marker_data else [0.5, 0.5])
        
        trackers_dict['trackerData'][frame] = {
            'movieFrameNumber': frame,
            'markerData': marker_data
        }
    
    return trackers_dict


def read_persp_data(
    persp_file: str,
    ref_frame: int,
    frame_offset: int,
    movie_width: int,
    movie_height: int,
    portrait: bool,
    swap_xy: bool,
    x_flip: bool,
    y_flip: bool,
) -> dict[str, Any]:
    """
    Read perspective tracking data from a file.
    
    Format: FRAME [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ]
    
    Args:
        persp_file: Path to perspective track file
        ref_frame: Reference frame number
        frame_offset: Frame number offset
        movie_width: Video width
        movie_height: Video height
        portrait: True if video is portrait
        swap_xy: Swap X and Y values
        x_flip: Invert X coordinate
        y_flip: Invert Y coordinate
        
    Returns:
        Dictionary with perspective tracking data
    """
    point_pattern = re.compile(r'\[([\d.]+)\s*,\s*([\d.]+)\]')
    
    tmp_dict: dict[int, list] = {}
    
    with open(persp_file) as f:
        for line in f:
            parts = line.strip().split()
            frame_num = int(parts[0]) + frame_offset
            
            # Find all point pairs in the line
            matches = point_pattern.findall(line)
            markers = []
            for match in matches:
                x, y = float(match[0]), float(match[1])
                xy = orient_coordinates(
                    (x, y), movie_width, movie_height,
                    portrait, swap_xy, x_flip, y_flip,
                    apply_scale=False
                )
                markers.append(xy)
            
            tmp_dict[frame_num] = markers
    
    # Determine frame range
    ff = min(tmp_dict.keys())
    lf = max(tmp_dict.keys())
    
    # Set reference frame
    if ref_frame == -1:
        ref_frame = ff
    
    # Get reference data
    ref_data = tmp_dict.get(ref_frame, tmp_dict[ff])
    
    # Build output dictionary
    trackers_dict = {
        'ff': ff,
        'lf': lf,
        'refFrameInMovie': ref_frame,
        'refData': ref_data,
        'trackerData': {}
    }
    
    for frame, markers in tmp_dict.items():
        trackers_dict['trackerData'][frame] = {
            'movieFrameNumber': frame,
            'markerData': markers
        }
    
    return trackers_dict


def apply_jitter_filter(
    trackers_dict: dict[str, Any],
    sigma: float = 5.0,
) -> dict[str, Any]:
    """
    Apply Gaussian filter to tracking data to remove jitter.
    
    Args:
        trackers_dict: Dictionary containing tracking data
        sigma: Gaussian filter sigma (higher = more smoothing)
        
    Returns:
        New dictionary with filtered tracking data
    """
    tracker_data = trackers_dict['trackerData']
    frames = sorted(tracker_data.keys())
    
    if len(frames) < 3:
        return trackers_dict
    
    # Get number of markers from first frame
    first_markers = tracker_data[frames[0]]['markerData']
    num_markers = len(first_markers)
    
    # Extract x and y coordinates for each marker
    for marker_idx in range(num_markers):
        x_values = []
        y_values = []
        
        for frame in frames:
            markers = tracker_data[frame]['markerData']
            if marker_idx < len(markers):
                x_values.append(markers[marker_idx][0])
                y_values.append(markers[marker_idx][1])
            else:
                # Pad with last known value
                x_values.append(x_values[-1] if x_values else 0.5)
                y_values.append(y_values[-1] if y_values else 0.5)
        
        # Apply Gaussian filter
        x_filtered = gaussian_filter1d(x_values, sigma=sigma, mode='nearest')
        y_filtered = gaussian_filter1d(y_values, sigma=sigma, mode='nearest')
        
        # Write back filtered values
        for i, frame in enumerate(frames):
            markers = tracker_data[frame]['markerData']
            if marker_idx < len(markers):
                markers[marker_idx] = [x_filtered[i], y_filtered[i]]
    
    return trackers_dict
