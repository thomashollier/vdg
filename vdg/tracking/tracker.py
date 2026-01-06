"""
Feature point tracking using Lucas-Kanade optical flow.

This module provides the FeatureTracker class which tracks features
through video frames with automatic replenishment of lost points.
"""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class TrackingStats:
    """Statistics from a tracking update."""
    frame: int
    tracked: int
    lost: int
    added: int
    total: int
    
    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for backward compatibility."""
        return {
            "frame": self.frame,
            "tracked": self.tracked,
            "lost": self.lost,
            "added": self.added,
            "total": self.total,
        }


class FeatureTracker:
    """
    Robust feature tracker with automatic replenishment.
    
    Uses Shi-Tomasi corner detection and Lucas-Kanade optical flow
    with forward-backward consistency checking.
    
    Attributes:
        num_features: Target number of features to maintain
        min_distance: Minimum distance between features in pixels
        points: Current tracked point positions (Nx1x2 array)
        point_ids: Unique IDs for each tracked point
        track_lengths: Dictionary mapping point IDs to number of frames tracked
        
    Example:
        >>> tracker = FeatureTracker(num_features=50)
        >>> points = tracker.initialize(first_frame)
        >>> for frame in video:
        ...     points, ids, stats = tracker.update(frame)
        ...     print(f"Frame {stats.frame}: {stats.tracked} tracked, {stats.lost} lost")
    """
    
    def __init__(
        self,
        num_features: int = 50,
        quality_level: float = 0.01,
        min_distance: int = 30,
        block_size: int = 7,
        min_track_quality: float = 0.3,
        initial_roi: tuple[int, int, int, int] | None = None,
        enforce_bbox: bool = False,
    ):
        """
        Initialize the feature tracker.
        
        Args:
            num_features: Target number of features to maintain
            quality_level: Shi-Tomasi corner quality threshold (0-1)
            min_distance: Minimum distance between features in pixels
            block_size: Block size for corner detection
            min_track_quality: Minimum forward-backward consistency threshold
            initial_roi: Optional (x, y, w, h) bounding box to restrict features
            enforce_bbox: If True, points outside re-centered bbox are replaced
        """
        self.num_features = num_features
        self.min_distance = min_distance
        self.min_track_quality = min_track_quality
        self.initial_roi = initial_roi
        self.use_dynamic_roi = initial_roi is not None
        self.enforce_bbox = enforce_bbox and initial_roi is not None
        
        # Store original bbox dimensions for enforcement
        if self.enforce_bbox and initial_roi is not None:
            self.bbox_width = initial_roi[2]
            self.bbox_height = initial_roi[3]
        else:
            self.bbox_width = None
            self.bbox_height = None
        
        # Shi-Tomasi corner detection parameters
        self.feature_params = {
            "maxCorners": num_features,
            "qualityLevel": quality_level,
            "minDistance": min_distance,
            "blockSize": block_size,
        }
        
        # Lucas-Kanade optical flow parameters
        self.lk_params = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        }
        
        # State
        self.prev_gray: np.ndarray | None = None
        self.points: np.ndarray | None = None
        self.point_ids: np.ndarray | None = None
        self.next_id = 0
        self.track_lengths: dict[int, int] = {}
        self.fb_errors: dict[int, float] = {}
        self.lk_errors: dict[int, float] = {}
        self.frame_count = 0
        self.current_roi = initial_roi
        self.prev_roi = initial_roi
    
    def _create_roi_mask(self, shape: tuple[int, int], roi: tuple) -> np.ndarray:
        """Create a mask for the given ROI (x, y, w, h)."""
        mask = np.zeros(shape, dtype=np.uint8)
        x, y, w, h = [int(v) for v in roi]
        # Clamp to image bounds
        x = max(0, x)
        y = max(0, y)
        h = min(h, shape[0] - y)
        w = min(w, shape[1] - x)
        mask[y:y+h, x:x+w] = 255
        return mask
    
    def _get_current_roi(self) -> tuple | None:
        """Get the current ROI based on tracked points."""
        if not self.use_dynamic_roi:
            return None
        
        if self.points is None or len(self.points) < 2:
            return self.current_roi
        
        # Compute bounding box of current points with padding
        pts = self.points.reshape(-1, 2)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        
        padding = 10
        x = x_min - padding
        y = y_min - padding
        w = (x_max - x_min) + 2 * padding
        h = (y_max - y_min) + 2 * padding
        
        self.current_roi = (x, y, w, h)
        return self.current_roi
    
    def _get_replenishment_roi(self) -> tuple | None:
        """Get the ROI for adding new features when points are dropped."""
        if not self.use_dynamic_roi:
            return None
        
        if self.points is None or len(self.points) < 2:
            return self.prev_roi
        
        # Get the maximum allowed dimensions from previous frame's ROI
        _, _, max_w, max_h = self.prev_roi
        
        # Compute bounding box of current points
        pts = self.points.reshape(-1, 2)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        
        padding = 10
        points_w = (x_max - x_min) + 2 * padding
        points_h = (y_max - y_min) + 2 * padding
        
        # Constrain to not exceed previous frame's size
        new_w = min(points_w, max_w)
        new_h = min(points_h, max_h)
        
        # Center the box on the points
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        x = center_x - new_w / 2
        y = center_y - new_h / 2
        
        return (x, y, new_w, new_h)
    
    def _get_constraint_bbox(self) -> tuple | None:
        """Get the constraint bounding box centered on current points."""
        if not self.enforce_bbox or self.bbox_width is None:
            return None
        
        if self.points is None or len(self.points) == 0:
            return self.current_roi
        
        # Compute center of current points
        pts = self.points.reshape(-1, 2)
        center_x = pts[:, 0].mean()
        center_y = pts[:, 1].mean()
        
        # Create bbox centered on points with original dimensions
        x = center_x - self.bbox_width / 2
        y = center_y - self.bbox_height / 2
        
        return (x, y, self.bbox_width, self.bbox_height)
    
    def _point_in_bbox(self, point: np.ndarray, bbox: tuple) -> bool:
        """Check if a point is inside a bounding box."""
        x, y, w, h = bbox
        px, py = point.ravel()
        return x <= px <= x + w and y <= py <= y + h
    
    def _filter_points_by_constraint(
        self,
        points: np.ndarray,
        point_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Filter out points that have drifted outside the constraint bbox."""
        if not self.enforce_bbox or len(points) == 0:
            return points, point_ids, 0
        
        constraint_bbox = self._get_constraint_bbox()
        if constraint_bbox is None:
            return points, point_ids, 0
        
        # Check each point
        in_bbox = np.array([self._point_in_bbox(p, constraint_bbox) for p in points])
        
        num_filtered = np.sum(~in_bbox)
        
        # Remove points outside bbox from tracking dictionaries
        filtered_out_ids = point_ids[~in_bbox]
        for pid in filtered_out_ids:
            self.track_lengths.pop(pid, None)
            self.fb_errors.pop(pid, None)
            self.lk_errors.pop(pid, None)
        
        return points[in_bbox], point_ids[in_bbox], num_filtered
    
    def initialize(self, frame: np.ndarray) -> np.ndarray:
        """
        Initialize tracking on the first frame.
        
        Args:
            frame: First video frame (BGR)
            
        Returns:
            Array of detected feature points (Nx1x2)
        """
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create ROI mask if specified
        mask = None
        if self.initial_roi is not None:
            mask = self._create_roi_mask(self.prev_gray.shape, self.initial_roi)
        
        self.points = cv2.goodFeaturesToTrack(
            self.prev_gray, mask=mask, **self.feature_params
        )
        
        if self.points is None:
            self.points = np.array([]).reshape(0, 1, 2)
        
        # Assign unique IDs to each point
        self.point_ids = np.arange(len(self.points))
        self.next_id = len(self.points)
        
        # Initialize track lengths and errors
        for pid in self.point_ids:
            self.track_lengths[pid] = 1
            self.fb_errors[pid] = 0.0
            self.lk_errors[pid] = 0.0
        
        self.frame_count = 1
        
        # Initialize ROIs based on detected points
        if self.use_dynamic_roi:
            self._get_current_roi()
            self.prev_roi = self.current_roi
        
        return self.points.copy()
    
    def _compute_mask(self, shape: tuple, roi: tuple = None) -> np.ndarray:
        """Create a mask excluding areas near existing points."""
        if roi is not None:
            mask = self._create_roi_mask(shape, roi)
        else:
            mask = np.ones(shape, dtype=np.uint8) * 255
        
        # Exclude areas near existing points
        if self.points is not None and len(self.points) > 0:
            for point in self.points:
                x, y = point.ravel()
                cv2.circle(mask, (int(x), int(y)), self.min_distance, 0, -1)
        
        return mask
    
    def _add_new_features(self, gray: np.ndarray, num_to_add: int) -> None:
        """Add new features to replace lost ones."""
        if num_to_add <= 0:
            return
        
        # Get ROI for replenishment
        if self.enforce_bbox:
            replenish_roi = self._get_constraint_bbox()
        else:
            replenish_roi = self._get_replenishment_roi()
        
        # Create mask to avoid detecting features near existing ones
        mask = self._compute_mask(gray.shape, roi=replenish_roi)
        
        # Detect new features
        new_feature_params = self.feature_params.copy()
        
        if self.enforce_bbox and replenish_roi is not None:
            # Detect more candidates, then select closest to center
            new_feature_params["maxCorners"] = num_to_add * 5
            
            candidate_points = cv2.goodFeaturesToTrack(
                gray, mask=mask, **new_feature_params
            )
            
            if candidate_points is not None and len(candidate_points) > 0:
                roi_x, roi_y, roi_w, roi_h = replenish_roi
                center_x = roi_x + roi_w / 2
                center_y = roi_y + roi_h / 2
                
                candidates = candidate_points.reshape(-1, 2)
                distances = np.sqrt(
                    (candidates[:, 0] - center_x)**2 +
                    (candidates[:, 1] - center_y)**2
                )
                
                sorted_indices = np.argsort(distances)
                num_to_take = min(num_to_add, len(candidates))
                selected_indices = sorted_indices[:num_to_take]
                
                new_points = candidate_points[selected_indices]
            else:
                new_points = None
        else:
            new_feature_params["maxCorners"] = num_to_add
            new_points = cv2.goodFeaturesToTrack(
                gray, mask=mask, **new_feature_params
            )
        
        if new_points is not None and len(new_points) > 0:
            # Append new points
            if self.points is None or len(self.points) == 0:
                self.points = new_points
            else:
                self.points = np.vstack([self.points, new_points])
            
            # Assign new IDs
            new_ids = np.arange(self.next_id, self.next_id + len(new_points))
            self.point_ids = np.concatenate([self.point_ids, new_ids])
            self.next_id += len(new_points)
            
            # Initialize track lengths and errors for new points
            for pid in new_ids:
                self.track_lengths[pid] = 1
                self.fb_errors[pid] = 0.0
                self.lk_errors[pid] = 0.0
    
    def _validate_tracks(
        self,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        status: np.ndarray,
        lk_error: np.ndarray,
        gray: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate tracks using forward-backward consistency check."""
        n_points = len(prev_points)
        valid = status.ravel() == 1
        fb_error = np.full(n_points, np.inf)
        lk_err = lk_error.ravel().copy()
        
        if not np.any(valid):
            return valid, fb_error, lk_err
        
        # Forward-backward consistency check
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, curr_points, None, **self.lk_params
        )
        
        # Calculate forward-backward error
        fb_error = np.linalg.norm(
            prev_points.reshape(-1, 2) - back_points.reshape(-1, 2), axis=1
        )
        
        # Points are valid if status is good and FB error is small
        valid = valid & (back_status.ravel() == 1) & (fb_error < 1.0)
        
        # Check if points are within frame bounds
        h, w = gray.shape
        in_bounds = (
            (curr_points[:, 0, 0] >= 0) &
            (curr_points[:, 0, 0] < w) &
            (curr_points[:, 0, 1] >= 0) &
            (curr_points[:, 0, 1] < h)
        )
        valid = valid & in_bounds
        
        return valid, fb_error, lk_err
    
    def update(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, TrackingStats]:
        """
        Track features to the next frame.
        
        Args:
            frame: Next video frame (BGR)
            
        Returns:
            Tuple of (current_points, point_ids, tracking_stats)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_count += 1
        
        stats = TrackingStats(
            frame=self.frame_count,
            tracked=0,
            lost=0,
            added=0,
            total=0,
        )
        
        if self.points is None or len(self.points) == 0:
            # No points to track, detect new ones
            self._add_new_features(gray, self.num_features)
            self.prev_gray = gray
            stats.added = len(self.points) if self.points is not None else 0
            stats.total = stats.added
            return self.points.copy(), self.point_ids.copy(), stats
        
        # Track points using Lucas-Kanade
        curr_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.points, None, **self.lk_params
        )
        
        # Validate tracks with forward-backward check
        valid, fb_errors, lk_errors = self._validate_tracks(
            self.points, curr_points, status, error, gray
        )
        
        # Update statistics
        stats.tracked = np.sum(valid)
        stats.lost = len(self.points) - stats.tracked
        
        # Store errors for valid points before filtering
        valid_indices = np.where(valid)[0]
        for idx in valid_indices:
            pid = self.point_ids[idx]
            self.fb_errors[pid] = fb_errors[idx]
            self.lk_errors[pid] = lk_errors[idx]
        
        # Keep only valid points
        self.points = curr_points[valid]
        old_ids = self.point_ids[valid]
        
        # Update track lengths and remove lost tracks
        lost_ids = self.point_ids[~valid]
        for pid in lost_ids:
            self.track_lengths.pop(pid, None)
            self.fb_errors.pop(pid, None)
            self.lk_errors.pop(pid, None)
        
        self.point_ids = old_ids
        for pid in self.point_ids:
            self.track_lengths[pid] += 1
        
        # Reshape points if needed
        if len(self.points) > 0:
            self.points = self.points.reshape(-1, 1, 2)
        else:
            self.points = np.array([]).reshape(0, 1, 2)
        
        # Apply constraint bbox filtering if enabled
        if self.enforce_bbox and len(self.points) > 0:
            self.points, self.point_ids, num_constrained = \
                self._filter_points_by_constraint(self.points, self.point_ids)
            stats.lost += num_constrained
            stats.tracked -= num_constrained
        
        # Replenish lost features
        num_to_add = self.num_features - len(self.points)
        if num_to_add > 0:
            self._add_new_features(gray, num_to_add)
            stats.added = num_to_add
        
        stats.total = len(self.points)
        
        # Update previous frame and ROI
        self.prev_gray = gray
        if self.use_dynamic_roi:
            self._get_current_roi()
            self.prev_roi = self.current_roi
        
        return self.points.copy(), self.point_ids.copy(), stats
    
    def get_center(self) -> tuple[float, float] | None:
        """Get the centroid of all tracked points."""
        if self.points is None or len(self.points) == 0:
            return None
        pts = self.points.reshape(-1, 2)
        return (pts[:, 0].mean(), pts[:, 1].mean())
    
    def get_roi(self) -> tuple[float, float, float, float] | None:
        """Get the current ROI bounding the tracked points."""
        return self._get_current_roi()
    
    def reset(self) -> None:
        """Reset the tracker state."""
        self.prev_gray = None
        self.points = None
        self.point_ids = None
        self.next_id = 0
        self.track_lengths.clear()
        self.fb_errors.clear()
        self.lk_errors.clear()
        self.frame_count = 0
        self.current_roi = self.initial_roi
        self.prev_roi = self.initial_roi
