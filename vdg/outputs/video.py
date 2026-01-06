"""
Video output handlers.

Provides output handlers that produce video files:
- PreviewTrackOutput: Video with tracking overlay
- CleanVideoOutput: Clean stabilized video
"""

from pathlib import Path

import cv2
import numpy as np

from vdg.outputs.base import BaseOutput, OutputSpec


class PreviewTrackOutput(BaseOutput):
    """
    Outputs video with tracking points and ROI boxes overlaid.
    
    Options:
        filename: Output filename (default: input_preview.mp4)
        mode: 'original' or 'stab' (default: original)
        overlay: 'on' or 'off' (default: on)
        showids: 'true' or 'false' (default: false)
    """
    
    def __init__(self, spec: OutputSpec, input_path: str):
        super().__init__(spec, input_path)
        self.writer: cv2.VideoWriter | None = None
        self.show_ids = spec.get_bool('showids', False)
        
        # Mode: 'original' or 'stab'
        self.mode = spec.get('mode', 'original').lower()
        if self.mode not in ('original', 'stab'):
            raise ValueError(
                f"Invalid previewtrack mode: {self.mode}. "
                "Must be 'original' or 'stab'."
            )
        
        # Overlay: 'on' or 'off'
        overlay = spec.get('overlay', 'on').lower()
        if overlay not in ('on', 'off'):
            raise ValueError(
                f"Invalid previewtrack overlay: {overlay}. "
                "Must be 'on' or 'off'."
            )
        self.overlay = overlay == 'on'
    
    def _get_default_suffix(self) -> str:
        return "_preview"
    
    def _get_default_extension(self) -> str:
        return "mp4"
    
    def initialize(self, video_props: dict) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            video_props['fps'],
            (video_props['width'], video_props['height'])
        )
    
    def _draw_visualization(
        self,
        frame: np.ndarray,
        tracking_data: dict,
        frame_num: int,
    ) -> np.ndarray:
        """Draw tracking visualization on frame."""
        vis = frame.copy()
        
        # Draw ROI boxes
        rois = tracking_data.get('rois', [])
        colors = [(0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 255, 255)]
        for i, roi in enumerate(rois):
            if roi is not None:
                x, y, w, h = [int(v) for v in roi]
                color = colors[i % len(colors)]
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    vis, str(i + 1), (x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )
        
        # Draw tracking points
        points_list = tracking_data.get('points', [])
        ids_list = tracking_data.get('point_ids', [])
        lengths_list = tracking_data.get('track_lengths', [])
        
        for points, point_ids, track_lengths in zip(points_list, ids_list, lengths_list):
            if points is None or len(points) == 0:
                continue
            
            for point, pid in zip(points, point_ids):
                x, y = point.ravel()
                x, y = int(x), int(y)
                
                # Color based on track length
                length = track_lengths.get(pid, 1)
                norm_length = min(length / 100.0, 1.0)
                
                if norm_length < 0.5:
                    r = int(255 * (norm_length * 2))
                    g = 255
                else:
                    r = 255
                    g = int(255 * (1 - (norm_length - 0.5) * 2))
                b = 0
                color = (b, g, r)
                
                cv2.circle(vis, (x, y), 4, color, -1)
                cv2.circle(vis, (x, y), 6, color, 1)
                
                if self.show_ids:
                    cv2.putText(
                        vis, str(pid), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1
                    )
        
        # Draw frame number
        cv2.putText(
            vis, f"Frame: {frame_num}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        return vis
    
    def process_frame(
        self,
        frame_num: int,
        frame: np.ndarray,
        tracking_data: dict,
    ) -> None:
        if self.writer is None:
            return
        
        if self.overlay:
            output_frame = self._draw_visualization(frame, tracking_data, frame_num)
        else:
            output_frame = frame
        
        self.writer.write(output_frame)
    
    def finalize(self) -> None:
        if self.writer:
            self.writer.release()
            self.writer = None


class CleanVideoOutput(BaseOutput):
    """
    Outputs clean stabilized video without any overlays.
    
    Options:
        filename: Output filename (default: input_stabilized.mp4)
    """
    
    def __init__(self, spec: OutputSpec, input_path: str):
        super().__init__(spec, input_path)
        self.writer: cv2.VideoWriter | None = None
    
    def _get_default_suffix(self) -> str:
        return "_stabilized"
    
    def _get_default_extension(self) -> str:
        return "mp4"
    
    def initialize(self, video_props: dict) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            video_props['fps'],
            (video_props['width'], video_props['height'])
        )
    
    def process_frame(
        self,
        frame_num: int,
        frame: np.ndarray,
        tracking_data: dict,
    ) -> None:
        if self.writer is None:
            return
        self.writer.write(frame)
    
    def finalize(self) -> None:
        if self.writer:
            self.writer.release()
            self.writer = None
