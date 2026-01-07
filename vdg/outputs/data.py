"""
Data output handlers.

Provides output handlers that produce data files:
- CSVOutput: Tracking data as CSV
- TrackersOutput: Normalized coordinates (.crv format)
"""

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

import numpy as np

from vdg.outputs.base import BaseOutput, OutputSpec


class CenterMode(Protocol):
    """Protocol for center calculation modes."""
    
    def calculate(
        self,
        points: np.ndarray | None,
        roi: tuple | None,
    ) -> tuple[float, float] | None:
        """Calculate center from points and/or ROI."""
        ...


class CentroidMode:
    """Calculate center as centroid of tracked points."""
    
    def calculate(
        self,
        points: np.ndarray | None,
        roi: tuple | None,
    ) -> tuple[float, float] | None:
        if points is None or len(points) == 0:
            return None
        pts = points.reshape(-1, 2)
        return (pts[:, 0].mean(), pts[:, 1].mean())


class RoiCenterMode:
    """Calculate center as center of the ROI bounding box."""
    
    def calculate(
        self,
        points: np.ndarray | None,
        roi: tuple | None,
    ) -> tuple[float, float] | None:
        if roi is None:
            return None
        x, y, w, h = roi
        return (x + w / 2, y + h / 2)


class MedianMode:
    """Calculate center as median position of tracked points."""
    
    def calculate(
        self,
        points: np.ndarray | None,
        roi: tuple | None,
    ) -> tuple[float, float] | None:
        if points is None or len(points) == 0:
            return None
        pts = points.reshape(-1, 2)
        return (np.median(pts[:, 0]), np.median(pts[:, 1]))


# Registry of center modes
CENTER_MODES = {
    'center': CentroidMode,
    'centroid': CentroidMode,
    'roi_center': RoiCenterMode,
    'median': MedianMode,
}


class CSVOutput(BaseOutput):
    """
    Outputs tracking data as a CSV file.
    
    Columns: frame, region, point_id, x, y, track_length
    
    Options:
        filename: Output filename (default: input.csv)
    """
    
    def __init__(self, spec: OutputSpec, input_path: str):
        super().__init__(spec, input_path)
        self.file = None
        self.writer = None
    
    def _get_default_suffix(self) -> str:
        return ""
    
    def _get_default_extension(self) -> str:
        return "csv"
    
    def initialize(self, video_props: dict) -> None:
        self.file = open(self.output_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'frame', 'region', 'point_id', 'x', 'y', 'track_length'
        ])
    
    def process_frame(
        self,
        frame_num: int,
        frame: np.ndarray,
        tracking_data: dict,
    ) -> None:
        if self.writer is None:
            return
        
        points_list = tracking_data.get('points', [])
        ids_list = tracking_data.get('point_ids', [])
        lengths_list = tracking_data.get('track_lengths', [])
        
        for region, (points, point_ids, lengths) in enumerate(
            zip(points_list, ids_list, lengths_list)
        ):
            if points is None:
                continue
            
            for point, pid in zip(points, point_ids):
                x, y = point.ravel()
                length = lengths.get(pid, 0)
                self.writer.writerow([
                    frame_num, region, pid, x, y, length
                ])
    
    def finalize(self) -> None:
        if self.file:
            self.file.close()
            self.file = None
            self.writer = None


class TrackersOutput(BaseOutput):
    """
    Outputs normalized center coordinates per region (.crv format).
    
    Format: frame_num [[ x_normalized, y_normalized ]]
    
    Options:
        filename: Base filename (default: input_track01.crv)
        type: 1 for single-point, 2 for two-point (default: 1)
        mode: center, roi_center, or median (default: center)
        filter: Gaussian filter sigma (default: 0 = no filter)
    """
    
    def __init__(self, spec: OutputSpec, input_path: str):
        # Parse options BEFORE calling super().__init__()
        # because _resolve_output_path needs these values
        self.track_type = spec.get_int('type', 1)
        if self.track_type not in (1, 2):
            raise ValueError(f"Invalid tracker type: {self.track_type}. Must be 1 or 2.")
        
        mode_name = spec.get('mode', 'center').lower()
        if mode_name not in CENTER_MODES:
            raise ValueError(
                f"Invalid center mode: {mode_name}. "
                f"Available: {list(CENTER_MODES.keys())}"
            )
        self.center_mode = CENTER_MODES[mode_name]()
        
        self.filter_width = spec.get_float('filter', 0)
        
        # Generate output paths BEFORE super().__init__()
        input_path_obj = Path(input_path)
        stem = input_path_obj.stem
        
        # Check for custom filename first
        filename = spec.get('filename')
        if filename:
            self.output_paths = [Path(filename)]
        elif self.track_type == 1:
            self.output_paths = [Path(f"{stem}_track01.crv")]
        else:
            self.output_paths = [
                Path(f"{stem}_track01.crv"),
                Path(f"{stem}_track02.crv"),
            ]
        
        self.data_buffers: list[list[tuple[int, float, float]]] = []
        self.video_width = 0
        self.video_height = 0
        
        # Now call super().__init__() - it will use our _resolve_output_path
        super().__init__(spec, input_path)
    
    def _get_default_suffix(self) -> str:
        return "_track01"
    
    def _get_default_extension(self) -> str:
        return "crv"
    
    def _resolve_output_path(self) -> Path:
        """Override to return first path."""
        return self.output_paths[0] if self.output_paths else Path("track01.crv")
    
    def initialize(self, video_props: dict) -> None:
        self.video_width = video_props['width']
        self.video_height = video_props['height']
        
        # Initialize data buffers (one per region)
        num_regions = 1 if self.track_type == 1 else 2
        self.data_buffers = [[] for _ in range(num_regions)]
    
    def process_frame(
        self,
        frame_num: int,
        frame: np.ndarray,
        tracking_data: dict,
    ) -> None:
        points_list = tracking_data.get('points', [])
        rois = tracking_data.get('rois', [])
        
        if self.track_type == 1:
            # Single point tracking - use first region
            points = points_list[0] if points_list else None
            roi = rois[0] if rois else None
            center = self.center_mode.calculate(points, roi)
            
            if center and self.video_width and self.video_height:
                x_norm = center[0] / self.video_width
                y_norm = center[1] / self.video_height
                self.data_buffers[0].append((frame_num, x_norm, y_norm))
        
        else:  # type == 2
            # Two point tracking - buffer data for each region
            for i in range(2):
                points = points_list[i] if len(points_list) > i else None
                roi = rois[i] if len(rois) > i else None
                center = self.center_mode.calculate(points, roi)
                
                if center and self.video_width and self.video_height:
                    x_norm = center[0] / self.video_width
                    y_norm = center[1] / self.video_height
                    self.data_buffers[i].append((frame_num, x_norm, y_norm))
    
    def _apply_gaussian_filter(
        self,
        data: list[tuple[int, float, float]],
    ) -> list[tuple[int, float, float]]:
        """Apply Gaussian filter to smooth the trajectory data."""
        if not data or self.filter_width <= 0:
            return data
        
        try:
            from scipy.ndimage import gaussian_filter1d
        except ImportError:
            print(
                "Warning: scipy not installed, skipping Gaussian filter. "
                "Install with: pip install scipy"
            )
            return data
        
        # Extract x and y arrays
        frame_nums = [d[0] for d in data]
        x_values = np.array([d[1] for d in data])
        y_values = np.array([d[2] for d in data])
        
        # Apply Gaussian filter
        x_filtered = gaussian_filter1d(x_values, sigma=self.filter_width, mode='nearest')
        y_filtered = gaussian_filter1d(y_values, sigma=self.filter_width, mode='nearest')
        
        # Reconstruct data
        return [
            (frame_nums[i], x_filtered[i], y_filtered[i])
            for i in range(len(data))
        ]
    
    def finalize(self) -> None:
        # Open files and write filtered data
        for i, path in enumerate(self.output_paths):
            if i < len(self.data_buffers):
                # Apply filtering if enabled
                data = self.data_buffers[i]
                if self.filter_width > 0:
                    data = self._apply_gaussian_filter(data)
                
                # Write to file
                with open(path, 'w') as f:
                    for frame_num, x, y in data:
                        f.write(f"{frame_num} [[ {x}, {y}]]\n")
        
        self.data_buffers = []
    
    def get_output_paths(self) -> list[Path]:
        """Return all output paths."""
        return self.output_paths
