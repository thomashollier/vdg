"""
Hardware acceleration configuration for VDG.

This module provides package-level settings for hardware-accelerated
video encoding and decoding using platform-specific APIs:
- macOS: VideoToolbox (h264_videotoolbox)
- Linux: VAAPI, NVENC (future)
- Windows: DXVA2, NVENC (future)

Usage:
    from vdg.core.hardware import hw_config
    
    # Check current settings
    print(hw_config.enabled)
    print(hw_config.encoder)
    
    # Disable hardware acceleration
    hw_config.disable()
    
    # Re-enable
    hw_config.enable()
    
    # Or configure at module import
    import vdg
    vdg.configure_hardware(enabled=False)
"""

import platform
import subprocess
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import os


class HWAccelBackend(Enum):
    """Available hardware acceleration backends."""
    NONE = "none"
    VIDEOTOOLBOX = "videotoolbox"  # macOS
    VAAPI = "vaapi"                # Linux (Intel/AMD)
    NVENC = "nvenc"                # NVIDIA
    QSV = "qsv"                    # Intel QuickSync
    DXVA2 = "dxva2"                # Windows


@dataclass
class HardwareConfig:
    """
    Hardware acceleration configuration.
    
    This is a singleton-style configuration object that controls
    hardware acceleration across all VDG video I/O operations.
    
    Attributes:
        enabled: Master switch for hardware acceleration
        backend: Which backend to use (auto-detected if None)
        encode_enabled: Enable hardware encoding
        decode_enabled: Enable hardware decoding
        fallback_to_software: Fall back to software if hardware fails
        encoder: FFmpeg encoder name (e.g., 'h264_videotoolbox')
        decoder: FFmpeg decoder name (e.g., 'h264')
        hwaccel: FFmpeg hwaccel option (e.g., 'videotoolbox')
        encode_bitrate: Default encoding bitrate
        encode_quality: Quality preset (if supported)
    """
    enabled: bool = True
    backend: Optional[HWAccelBackend] = None
    encode_enabled: bool = True
    decode_enabled: bool = True
    fallback_to_software: bool = True
    
    # FFmpeg settings (auto-configured based on backend)
    encoder: str = ""
    decoder: str = ""
    hwaccel: str = ""
    hwaccel_device: str = ""
    
    # Encoding settings
    encode_bitrate: str = "20M"
    encode_quality: str = ""  # e.g., "medium" for NVENC
    
    # Runtime state
    _initialized: bool = field(default=False, repr=False)
    _ffmpeg_available: bool = field(default=False, repr=False)
    _hw_encode_available: bool = field(default=False, repr=False)
    _hw_decode_available: bool = field(default=False, repr=False)
    
    def __post_init__(self):
        """Auto-detect hardware capabilities on first access."""
        if not self._initialized:
            self._detect_capabilities()
    
    def _detect_capabilities(self) -> None:
        """Detect available hardware acceleration capabilities."""
        self._initialized = True
        
        # Check for FFmpeg
        self._ffmpeg_available = shutil.which("ffmpeg") is not None
        
        if not self._ffmpeg_available:
            self.enabled = False
            self.encode_enabled = False
            self.decode_enabled = False
            return
        
        # Auto-detect backend based on platform
        if self.backend is None:
            self.backend = self._detect_backend()
        
        # Configure based on backend
        self._configure_backend()
        
        # Verify encoder availability
        if self.encode_enabled and self.encoder:
            self._hw_encode_available = self._check_encoder(self.encoder)
            if not self._hw_encode_available and not self.fallback_to_software:
                self.encode_enabled = False
        
        # Verify decoder availability
        if self.decode_enabled and self.hwaccel:
            self._hw_decode_available = self._check_decoder(self.hwaccel)
            if not self._hw_decode_available and not self.fallback_to_software:
                self.decode_enabled = False
    
    def _detect_backend(self) -> HWAccelBackend:
        """Detect the best available backend for the current platform."""
        system = platform.system()
        
        if system == "Darwin":
            return HWAccelBackend.VIDEOTOOLBOX
        elif system == "Linux":
            # Check for NVIDIA GPU first
            if self._check_nvidia():
                return HWAccelBackend.NVENC
            # Fall back to VAAPI
            if self._check_vaapi():
                return HWAccelBackend.VAAPI
        elif system == "Windows":
            if self._check_nvidia():
                return HWAccelBackend.NVENC
            return HWAccelBackend.DXVA2
        
        return HWAccelBackend.NONE
    
    def _configure_backend(self) -> None:
        """Configure FFmpeg options based on selected backend."""
        if self.backend == HWAccelBackend.VIDEOTOOLBOX:
            self.encoder = "h264_videotoolbox"
            self.decoder = "h264"
            self.hwaccel = "videotoolbox"
            
        elif self.backend == HWAccelBackend.NVENC:
            self.encoder = "h264_nvenc"
            self.decoder = "h264_cuvid"
            self.hwaccel = "cuda"
            self.encode_quality = "p4"  # balanced quality/speed
            
        elif self.backend == HWAccelBackend.VAAPI:
            self.encoder = "h264_vaapi"
            self.decoder = "h264"
            self.hwaccel = "vaapi"
            self.hwaccel_device = "/dev/dri/renderD128"
            
        elif self.backend == HWAccelBackend.QSV:
            self.encoder = "h264_qsv"
            self.decoder = "h264_qsv"
            self.hwaccel = "qsv"
            
        elif self.backend == HWAccelBackend.DXVA2:
            self.encoder = ""  # DXVA2 is decode-only
            self.decoder = "h264"
            self.hwaccel = "dxva2"
            self.encode_enabled = False
            
        else:
            self.encoder = ""
            self.decoder = ""
            self.hwaccel = ""
            self.encode_enabled = False
            self.decode_enabled = False
    
    def _check_encoder(self, encoder: str) -> bool:
        """Check if a specific encoder is available in FFmpeg."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-encoders"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return encoder in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _check_decoder(self, hwaccel: str) -> bool:
        """Check if a specific hwaccel is available in FFmpeg."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-hwaccels"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return hwaccel in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _check_nvidia(self) -> bool:
        """Check for NVIDIA GPU."""
        try:
            subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                timeout=5,
            )
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _check_vaapi(self) -> bool:
        """Check for VAAPI support."""
        return os.path.exists("/dev/dri/renderD128")
    
    def enable(self) -> None:
        """Enable hardware acceleration."""
        self.enabled = True
        self.encode_enabled = self._hw_encode_available
        self.decode_enabled = self._hw_decode_available
    
    def disable(self) -> None:
        """Disable hardware acceleration."""
        self.enabled = False
        self.encode_enabled = False
        self.decode_enabled = False
    
    def enable_encoding(self, enabled: bool = True) -> None:
        """Enable or disable hardware encoding."""
        self.encode_enabled = enabled and self._hw_encode_available
    
    def enable_decoding(self, enabled: bool = True) -> None:
        """Enable or disable hardware decoding."""
        self.decode_enabled = enabled and self._hw_decode_available
    
    def get_ffmpeg_input_args(self) -> list[str]:
        """Get FFmpeg arguments for hardware-accelerated input."""
        if not self.enabled or not self.decode_enabled or not self.hwaccel:
            return []
        
        args = ["-hwaccel", self.hwaccel]
        
        if self.hwaccel_device:
            args.extend(["-hwaccel_device", self.hwaccel_device])
        
        # Output to system memory for compatibility
        args.extend(["-hwaccel_output_format", "nv12"])
        
        return args
    
    def get_ffmpeg_output_args(self) -> list[str]:
        """Get FFmpeg arguments for hardware-accelerated output."""
        if not self.enabled or not self.encode_enabled or not self.encoder:
            return ["-c:v", "libx264", "-preset", "medium"]
        
        args = ["-c:v", self.encoder]
        
        if self.backend == HWAccelBackend.VIDEOTOOLBOX:
            args.extend(["-allow_sw", "1"])  # Allow software fallback
            
        if self.encode_bitrate:
            args.extend(["-b:v", self.encode_bitrate])
            
        if self.encode_quality and self.backend == HWAccelBackend.NVENC:
            args.extend(["-preset", self.encode_quality])
        
        return args
    
    def status(self) -> dict:
        """Get current hardware acceleration status."""
        return {
            "enabled": self.enabled,
            "backend": self.backend.value if self.backend else "none",
            "platform": platform.system(),
            "ffmpeg_available": self._ffmpeg_available,
            "encode_available": self._hw_encode_available,
            "decode_available": self._hw_decode_available,
            "encode_enabled": self.encode_enabled,
            "decode_enabled": self.decode_enabled,
            "encoder": self.encoder,
            "hwaccel": self.hwaccel,
        }
    
    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        backend = self.backend.value if self.backend else "none"
        enc = "enc" if self.encode_enabled else ""
        dec = "dec" if self.decode_enabled else ""
        features = "+".join(filter(None, [enc, dec])) or "none"
        return f"HardwareConfig({status}, backend={backend}, features={features})"


# Global configuration instance
hw_config = HardwareConfig()


def configure_hardware(
    enabled: bool = True,
    backend: Optional[HWAccelBackend] = None,
    encode: bool = True,
    decode: bool = True,
    fallback: bool = True,
    bitrate: str = "20M",
) -> HardwareConfig:
    """
    Configure hardware acceleration settings.
    
    This is the primary way to configure hardware acceleration
    at the package level.
    
    Args:
        enabled: Master switch for hardware acceleration
        backend: Force a specific backend (auto-detect if None)
        encode: Enable hardware encoding
        decode: Enable hardware decoding
        fallback: Fall back to software if hardware fails
        bitrate: Default encoding bitrate
        
    Returns:
        The updated HardwareConfig instance
        
    Example:
        >>> from vdg.core.hardware import configure_hardware
        >>> configure_hardware(enabled=False)  # Disable all HW accel
        >>> configure_hardware(encode=False)   # Disable HW encoding only
    """
    global hw_config
    
    hw_config.enabled = enabled
    hw_config.fallback_to_software = fallback
    hw_config.encode_bitrate = bitrate
    
    if backend is not None:
        hw_config.backend = backend
        hw_config._configure_backend()
    
    if enabled:
        hw_config.encode_enabled = encode and hw_config._hw_encode_available
        hw_config.decode_enabled = decode and hw_config._hw_decode_available
    else:
        hw_config.encode_enabled = False
        hw_config.decode_enabled = False
    
    return hw_config


def print_hardware_status() -> None:
    """Print current hardware acceleration status."""
    status = hw_config.status()
    print("VDG Hardware Acceleration Status")
    print("=" * 40)
    print(f"  Platform:        {status['platform']}")
    print(f"  FFmpeg:          {'available' if status['ffmpeg_available'] else 'not found'}")
    print(f"  Backend:         {status['backend']}")
    print(f"  Master switch:   {'enabled' if status['enabled'] else 'disabled'}")
    print(f"  HW Encode:       {'available' if status['encode_available'] else 'unavailable'}"
          f" ({'enabled' if status['encode_enabled'] else 'disabled'})")
    print(f"  HW Decode:       {'available' if status['decode_available'] else 'unavailable'}"
          f" ({'enabled' if status['decode_enabled'] else 'disabled'})")
    if status['encoder']:
        print(f"  Encoder:         {status['encoder']}")
    if status['hwaccel']:
        print(f"  HW Accel:        {status['hwaccel']}")
