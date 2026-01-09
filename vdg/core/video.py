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

            # Add hardware acceleration args - use global config but respect local override
            if self._use_hardware is True:
                # Explicit override: always add hwaccel args if backend is configured
                if hw_config.hwaccel:
                    cmd.extend(["-hwaccel", hw_config.hwaccel])
                    if hw_config.hwaccel_device:
                        cmd.extend(["-hwaccel_device", hw_config.hwaccel_device])
                    # Don't specify hwaccel_output_format - let FFmpeg handle
                    # the conversion to system memory automatically
            else:
                # Use global config settings (may include hwaccel_output_format)
                args = hw_config.get_ffmpeg_input_args()
                # Remove hwaccel_output_format for raw pipe output compatibility
                filtered_args = []
                skip_next = False
                for arg in args:
                    if skip_next:
                        skip_next = False
                        continue
                    if arg == "-hwaccel_output_format":
                        skip_next = True
                        continue
                    filtered_args.append(arg)
                cmd.extend(filtered_args)
            
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
            # Using fallback writer after FFmpeg failure
            if self._writer:
                self._writer.write(frame)
            # If no fallback writer, frames are dropped (fallback was disabled)
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
                    if self._writer.isOpened():
                        self._writer.write(frame)
                    else:
                        import warnings
                        warnings.warn(f"Failed to initialize fallback video writer for {self.path}")
                        self._writer = None
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


class FFmpegReader:
    """
    Video reader using FFmpeg subprocess with custom filter support.

    Streams frames one at a time (O(1) memory) while supporting advanced
    FFmpeg features like HDR tonemapping, color space conversion, etc.

    Example:
        # Basic usage
        with FFmpegReader("input.mp4") as reader:
            for frame_num, frame in reader:
                process(frame)

        # HDR tonemapping
        hdr_filter = (
            "zscale=t=linear:npl=100,format=gbrpf32le,"
            "zscale=p=bt709,tonemap=hable:desat=0,"
            "zscale=t=bt709:m=bt709:r=tv,format=rgb24"
        )
        with FFmpegReader("hdr_video.mov", vf=hdr_filter) as reader:
            for frame_num, frame in reader:
                process(frame)
    """

    def __init__(
        self,
        path: str | Path,
        first_frame: int = 1,
        last_frame: int | None = None,
        vf: str | None = None,
        pix_fmt: str = "bgr24",
        hwaccel: str | None = None,
    ):
        """
        Initialize the FFmpeg reader.

        Args:
            path: Path to video file
            first_frame: First frame to read (1-indexed)
            last_frame: Last frame to read (None = end of video)
            vf: FFmpeg video filter string (e.g., "scale=1920:1080,eq=gamma=1.2")
            pix_fmt: Output pixel format (default: bgr24 for OpenCV compatibility)
            hwaccel: Hardware acceleration method (e.g., "videotoolbox", "cuda", "vaapi")
        """
        self.path = Path(path)
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.vf = vf
        self.pix_fmt = pix_fmt
        self.hwaccel = hwaccel

        self._props: VideoProperties | None = None
        self._process: subprocess.Popen | None = None
        self._frame_size: int = 0
        self._current_frame: int = 0
        self._channels: int = 3

    def _probe_video(self) -> None:
        """Get video properties using ffprobe."""
        if not shutil.which("ffprobe"):
            # Fall back to OpenCV for properties
            cap = cv2.VideoCapture(str(self.path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.path}")
            self._props = VideoProperties.from_capture(cap)
            cap.release()
            return

        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
            "-of", "csv=p=0",
            str(self.path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            parts = result.stdout.strip().split(",")

            width = int(parts[0])
            height = int(parts[1])

            # Parse frame rate (can be "30/1" or "29.97")
            fps_str = parts[2]
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)

            # nb_frames might be "N/A" for some formats
            try:
                frame_count = int(parts[3])
            except (ValueError, IndexError):
                # Estimate from duration
                frame_count = self._estimate_frame_count(fps)

            self._props = VideoProperties(
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
            )

        except (subprocess.CalledProcessError, ValueError, IndexError):
            # Fall back to OpenCV
            cap = cv2.VideoCapture(str(self.path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.path}")
            self._props = VideoProperties.from_capture(cap)
            cap.release()

    def _estimate_frame_count(self, fps: float) -> int:
        """Estimate frame count from duration when nb_frames is unavailable."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(self.path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            return int(duration * fps)
        except (subprocess.CalledProcessError, ValueError):
            return 0

    def open(self) -> "FFmpegReader":
        """Open the video file and start FFmpeg process."""
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg not found in PATH")

        # Get video properties
        self._probe_video()

        # Validate and adjust frame range
        if self.last_frame is None or self.last_frame > self._props.frame_count:
            self.last_frame = self._props.frame_count

        # Build FFmpeg command
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

        # Hardware acceleration
        if self.hwaccel:
            cmd.extend(["-hwaccel", self.hwaccel])

        # Seek to first frame (before input for faster seeking)
        if self.first_frame > 1:
            start_time = (self.first_frame - 1) / self._props.fps
            cmd.extend(["-ss", f"{start_time:.6f}"])

        # Input file
        cmd.extend(["-i", str(self.path)])

        # Limit number of frames
        num_frames = self.last_frame - self.first_frame + 1
        cmd.extend(["-frames:v", str(num_frames)])

        # Video filter
        if self.vf:
            cmd.extend(["-vf", self.vf])

        # Output format
        cmd.extend([
            "-f", "rawvideo",
            "-pix_fmt", self.pix_fmt,
            "-"
        ])

        # Determine bytes per pixel
        if self.pix_fmt in ("bgr24", "rgb24"):
            self._channels = 3
        elif self.pix_fmt in ("bgra", "rgba"):
            self._channels = 4
        elif self.pix_fmt in ("gray", "gray8"):
            self._channels = 1
        elif self.pix_fmt in ("gray16le", "gray16be"):
            self._channels = 2  # 16-bit = 2 bytes
        else:
            self._channels = 3  # Default assumption

        self._frame_size = self._props.width * self._props.height * self._channels
        self._current_frame = self.first_frame

        # Start FFmpeg process
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        return self

    def close(self) -> None:
        """Close the FFmpeg process."""
        if self._process:
            if self._process.stdout:
                self._process.stdout.close()
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

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
        """Read the next frame from FFmpeg pipe."""
        if not self._process or not self._process.stdout:
            return False, None

        if self._current_frame > self.last_frame:
            return False, None

        try:
            raw = self._process.stdout.read(self._frame_size)
            if len(raw) != self._frame_size:
                return False, None

            # Determine dtype based on pixel format
            if self.pix_fmt in ("gray16le", "gray16be"):
                dtype = np.uint16
                shape = (self._props.height, self._props.width)
            else:
                dtype = np.uint8
                if self._channels == 1:
                    shape = (self._props.height, self._props.width)
                else:
                    shape = (self._props.height, self._props.width, self._channels)

            frame = np.frombuffer(raw, dtype=dtype).reshape(shape)
            return True, frame

        except Exception:
            return False, None

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over frames in the range."""
        if self._process is None:
            self.open()

        while self._current_frame <= self.last_frame:
            ret, frame = self.read_frame()
            if not ret:
                break

            frame_num = self._current_frame
            self._current_frame += 1
            yield frame_num, frame

    def __enter__(self) -> "FFmpegReader":
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
