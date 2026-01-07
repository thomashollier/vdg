"""
Video I/O utilities for the VDG framework.

Provides consistent interfaces for reading and writing video files,
with support for hardware acceleration where available.

Hardware acceleration is configured via vdg.core.hardware:
    from vdg.core.hardware import configure_hardware, hw_config
    
    # Disable hardware acceleration
    configure_hardware(enabled=False)
    
    # Check status
    print(hw_config.status())
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import platform
import subprocess
import shutil
import tempfile
import os

import cv2
import numpy as np

from vdg.core.hardware import hw_config


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
    
    Supports hardware-accelerated decoding via FFmpeg when available.
    Configure via vdg.core.hardware.configure_hardware().
    
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
        use_hardware: bool | None = None,
    ):
        """
        Initialize the video reader.
        
        Args:
            path: Path to video file
            first_frame: First frame to read (1-indexed)
            last_frame: Last frame to read (None = end of video)
            use_hardware: Override hardware acceleration setting (None = use global config)
        """
        self.path = Path(path)
        self.first_frame = first_frame
        self.last_frame = last_frame
        self._use_hardware = use_hardware
        
        self._cap: cv2.VideoCapture | None = None
        self._props: VideoProperties | None = None
        self._ffmpeg_process: subprocess.Popen | None = None
        self._using_ffmpeg: bool = False
        self._frame_buffer: bytes = b""
        self._frame_size: int = 0
    
    @property
    def use_hardware(self) -> bool:
        """Check if hardware decoding should be used."""
        if self._use_hardware is not None:
            return self._use_hardware
        return hw_config.enabled and hw_config.decode_enabled
    
    def open(self) -> "VideoReader":
        """Open the video file."""
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        
        # First, get video properties using OpenCV (always works)
        temp_cap = cv2.VideoCapture(str(self.path))
        if not temp_cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")
        
        self._props = VideoProperties.from_capture(temp_cap)
        temp_cap.release()
        
        # Validate and adjust frame range
        if self.last_frame is None or self.last_frame > self._props.frame_count:
            self.last_frame = self._props.frame_count
        
        # Try hardware-accelerated decoding via FFmpeg
        if self.use_hardware and self._try_ffmpeg_reader():
            return self
        
        # Fall back to OpenCV
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")
        
        # Seek to first frame (convert to 0-indexed)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.first_frame - 1)
        
        return self
    
    def _try_ffmpeg_reader(self) -> bool:
        """Try to set up FFmpeg with hardware-accelerated decoding."""
        if not shutil.which("ffmpeg"):
            return False
        
        try:
            # Build FFmpeg command for hardware decoding
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
            
            # Add hardware acceleration args
            cmd.extend(hw_config.get_ffmpeg_input_args())
            
            # Seek to start frame
            if self.first_frame > 1:
                # Calculate start time
                start_time = (self.first_frame - 1) / self._props.fps
                cmd.extend(["-ss", str(start_time)])
            
            # Input file
            cmd.extend(["-i", str(self.path)])
            
            # Limit frames if last_frame is set
            if self.last_frame:
                num_frames = self.last_frame - self.first_frame + 1
                cmd.extend(["-frames:v", str(num_frames)])
            
            # Output raw video to pipe
            cmd.extend([
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-"
            ])
            
            self._ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            # Calculate frame size
            self._frame_size = self._props.width * self._props.height * 3
            self._using_ffmpeg = True
            
            return True
            
        except (FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def close(self) -> None:
        """Close the video file."""
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._ffmpeg_process:
            self._ffmpeg_process.terminate()
            self._ffmpeg_process.wait()
            self._ffmpeg_process = None
        self._using_ffmpeg = False
    
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
        if self._using_ffmpeg and self._ffmpeg_process:
            return self._read_ffmpeg_frame()
        elif self._cap is not None:
            ret, frame = self._cap.read()
            return ret, frame if ret else None
        else:
            raise RuntimeError("Video not opened. Call open() first.")
    
    def _read_ffmpeg_frame(self) -> tuple[bool, np.ndarray | None]:
        """Read a frame from FFmpeg pipe."""
        if not self._ffmpeg_process or not self._ffmpeg_process.stdout:
            return False, None
        
        try:
            raw_frame = self._ffmpeg_process.stdout.read(self._frame_size)
            if len(raw_frame) != self._frame_size:
                return False, None
            
            frame = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = frame.reshape((self._props.height, self._props.width, 3))
            return True, frame
            
        except Exception:
            return False, None
    
    def seek(self, frame_num: int) -> None:
        """
        Seek to a specific frame (1-indexed).
        
        Note: Seeking is not supported with FFmpeg hardware decoding.
        The reader will fall back to OpenCV if seeking is needed.
        """
        if self._using_ffmpeg:
            # Close FFmpeg and reopen with OpenCV for seeking
            self.close()
            self._cap = cv2.VideoCapture(str(self.path))
            self._using_ffmpeg = False
        
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
    
    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over frames in the range."""
        if self._cap is None and self._ffmpeg_process is None:
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
    Falls back to OpenCV VideoWriter on other platforms or when disabled.
    
    Configure via vdg.core.hardware.configure_hardware().
    
    Example:
        with VideoWriter("output.mp4", props) as writer:
            for frame in frames:
                writer.write(frame)
    """
    
    def __init__(
        self,
        path: str | Path,
        props: VideoProperties,
        use_hardware: bool | None = None,
        bitrate: str | None = None,
    ):
        """
        Initialize the video writer.
        
        Args:
            path: Output video path
            props: Video properties (dimensions, fps)
            use_hardware: Override hardware acceleration setting (None = use global config)
            bitrate: Override encoding bitrate (None = use global config)
        """
        self.path = Path(path)
        self.props = props
        self._use_hardware = use_hardware
        self._bitrate = bitrate
        
        self._writer: cv2.VideoWriter | None = None
        self._ffmpeg_process: subprocess.Popen | None = None
        self._using_ffmpeg: bool = False
        self._failed: bool = False
    
    @property
    def use_hardware(self) -> bool:
        """Check if hardware encoding should be used."""
        if self._use_hardware is not None:
            return self._use_hardware
        return hw_config.enabled and hw_config.encode_enabled
    
    @property
    def bitrate(self) -> str:
        """Get the encoding bitrate."""
        return self._bitrate or hw_config.encode_bitrate
    
    def open(self) -> "VideoWriter":
        """Open the video writer."""
        # Try hardware encoding via FFmpeg
        if self.use_hardware and self._try_ffmpeg_writer():
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
        """Try to set up FFmpeg with hardware-accelerated encoding."""
        if not shutil.which("ffmpeg"):
            return False
        
        try:
            # Pad dimensions to multiple of 16 for hardware encoder
            out_w = ((self.props.width + 15) // 16) * 16
            out_h = ((self.props.height + 15) // 16) * 16
            needs_padding = (out_w != self.props.width or out_h != self.props.height)
            
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{self.props.width}x{self.props.height}",
                "-r", str(self.props.fps),
                "-i", "-",
            ]
            
            # Add padding if needed
            if needs_padding:
                cmd.extend(["-vf", f"pad={out_w}:{out_h}:0:0:black"])
            
            # Add hardware encoding args
            cmd.extend(hw_config.get_ffmpeg_output_args())
            
            # Override bitrate if specified
            if self._bitrate:
                # Remove existing bitrate arg and add new one
                cmd = [c for i, c in enumerate(cmd) if c != "-b:v" and (i == 0 or cmd[i-1] != "-b:v")]
                cmd.extend(["-b:v", self._bitrate])
            
            # Output format
            cmd.extend(["-pix_fmt", "yuv420p", str(self.path)])
            
            self._ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._using_ffmpeg = True
            
            return True
            
        except (FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def write(self, frame: np.ndarray) -> None:
        """Write a frame to the video."""
        if self._failed:
            if self._writer:
                self._writer.write(frame)
            return
        
        if self._using_ffmpeg and self._ffmpeg_process:
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            try:
                self._ffmpeg_process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                # Fall back to OpenCV
                self._failed = True
                self._using_ffmpeg = False
                if self._ffmpeg_process:
                    self._ffmpeg_process.terminate()
                    self._ffmpeg_process = None
                
                if hw_config.fallback_to_software:
                    fourcc = cv2.VideoWriter_fourcc(*self.props.fourcc)
                    self._writer = cv2.VideoWriter(
                        str(self.path),
                        fourcc,
                        self.props.fps,
                        (self.props.width, self.props.height),
                    )
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
        self._using_ffmpeg = False
    
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
