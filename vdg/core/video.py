"""
Video I/O utilities for the VDG framework.

Provides consistent interfaces for reading and writing video files,
with support for hardware acceleration where available.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import platform
import subprocess

import cv2
import numpy as np


@dataclass
class VideoProperties:
    """Properties of a video file."""
    width: int
    height: int
    fps: float
    frame_count: int
    fourcc: str = "mp4v"
    
    @classmethod
    def from_capture(cls, cap: cv2.VideoCapture) -> "VideoProperties":
        """Create VideoProperties from an OpenCV VideoCapture."""
        return cls(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for compatibility with existing code."""
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
        }
    
    @property
    def is_portrait(self) -> bool:
        """Check if video is in portrait orientation."""
        return self.height > self.width
    
    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio (width/height)."""
        return self.width / self.height if self.height > 0 else 0


class VideoReader:
    """
    High-level video reader with frame range support.
    
    Example:
        with VideoReader("input.mp4", first_frame=100, last_frame=500) as reader:
            for frame_num, frame in reader:
                process(frame)
    """
    
    def __init__(
        self,
        path: str | Path,
        first_frame: int = 1,
        last_frame: int | None = None,
    ):
        self.path = Path(path)
        self.first_frame = first_frame
        self.last_frame = last_frame
        self._cap: cv2.VideoCapture | None = None
        self._props: VideoProperties | None = None
    
    def open(self) -> "VideoReader":
        """Open the video file."""
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")
        
        self._props = VideoProperties.from_capture(self._cap)
        
        # Validate and adjust frame range
        if self.last_frame is None or self.last_frame > self._props.frame_count:
            self.last_frame = self._props.frame_count
        
        # Seek to first frame (convert to 0-indexed)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.first_frame - 1)
        
        return self
    
    def close(self) -> None:
        """Close the video file."""
        if self._cap:
            self._cap.release()
            self._cap = None
    
    @property
    def properties(self) -> VideoProperties:
        """Get video properties."""
        if self._props is None:
            raise RuntimeError("Video not opened. Call open() first.")
        return self._props
    
    @property
    def frame_range(self) -> tuple[int, int]:
        """Get the frame range being processed."""
        return (self.first_frame, self.last_frame or self.properties.frame_count)
    
    @property
    def frame_count(self) -> int:
        """Get number of frames in the processing range."""
        start, end = self.frame_range
        return end - start + 1
    
    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """Read the next frame."""
        if self._cap is None:
            raise RuntimeError("Video not opened. Call open() first.")
        ret, frame = self._cap.read()
        return ret, frame if ret else None
    
    def seek(self, frame_num: int) -> None:
        """Seek to a specific frame (1-indexed)."""
        if self._cap is None:
            raise RuntimeError("Video not opened. Call open() first.")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
    
    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over frames in the range."""
        if self._cap is None:
            self.open()
        
        current_frame = self.first_frame
        while current_frame <= (self.last_frame or self.properties.frame_count):
            ret, frame = self.read_frame()
            if not ret:
                break
            yield current_frame, frame
            current_frame += 1
    
    def __enter__(self) -> "VideoReader":
        return self.open()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False


class VideoWriter:
    """
    High-level video writer with hardware acceleration support.
    
    On macOS, uses FFmpeg with VideoToolbox for hardware encoding.
    Falls back to OpenCV VideoWriter on other platforms.
    
    Example:
        with VideoWriter("output.mp4", props) as writer:
            for frame in frames:
                writer.write(frame)
    """
    
    def __init__(
        self,
        path: str | Path,
        props: VideoProperties,
        use_hardware: bool = True,
        bitrate: str = "20M",
    ):
        self.path = Path(path)
        self.props = props
        self.use_hardware = use_hardware
        self.bitrate = bitrate
        self._writer = None
        self._ffmpeg_process = None
        self._use_ffmpeg = False
    
    def open(self) -> "VideoWriter":
        """Open the video writer."""
        # Try hardware encoding on macOS
        if self.use_hardware and platform.system() == "Darwin":
            if self._try_ffmpeg_writer():
                return self
        
        # Fall back to OpenCV
        fourcc = cv2.VideoWriter_fourcc(*self.props.fourcc)
        self._writer = cv2.VideoWriter(
            str(self.path),
            fourcc,
            self.props.fps,
            (self.props.width, self.props.height),
        )
        return self
    
    def _try_ffmpeg_writer(self) -> bool:
        """Try to set up FFmpeg with hardware encoding."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-encoders"],
                capture_output=True,
                text=True,
            )
            if "h264_videotoolbox" not in result.stdout:
                return False
            
            # Pad dimensions to multiple of 16 for hardware encoder
            out_w = ((self.props.width + 15) // 16) * 16
            out_h = ((self.props.height + 15) // 16) * 16
            
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{self.props.width}x{self.props.height}",
                "-r", str(self.props.fps),
                "-i", "-",
            ]
            
            # Add padding if needed
            if out_w != self.props.width or out_h != self.props.height:
                cmd.extend(["-vf", f"pad={out_w}:{out_h}:0:0:black"])
            
            cmd.extend([
                "-c:v", "h264_videotoolbox",
                "-allow_sw", "1",
                "-b:v", self.bitrate,
                "-pix_fmt", "yuv420p",
                str(self.path),
            ])
            
            self._ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._use_ffmpeg = True
            return True
            
        except FileNotFoundError:
            return False
    
    def write(self, frame: np.ndarray) -> None:
        """Write a frame to the video."""
        if self._use_ffmpeg and self._ffmpeg_process:
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            try:
                self._ffmpeg_process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                # Fall back to OpenCV
                self._use_ffmpeg = False
                self._ffmpeg_process = None
                self.open()
                self._writer.write(frame)
        elif self._writer:
            self._writer.write(frame)
    
    def close(self) -> None:
        """Close the video writer."""
        if self._ffmpeg_process:
            if self._ffmpeg_process.stdin:
                self._ffmpeg_process.stdin.close()
            self._ffmpeg_process.wait()
            self._ffmpeg_process = None
        if self._writer:
            self._writer.release()
            self._writer = None
    
    def __enter__(self) -> "VideoWriter":
        return self.open()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False


def get_video_properties(path: str | Path) -> VideoProperties:
    """Get properties of a video file without opening a reader."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    try:
        return VideoProperties.from_capture(cap)
    finally:
        cap.release()


def get_video_orientation(path: str | Path) -> str:
    """Determine if a video is portrait or landscape."""
    props = get_video_properties(path)
    return "portrait" if props.is_portrait else "landscape"
