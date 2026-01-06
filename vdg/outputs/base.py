"""
Base classes for output handlers.

This module defines the OutputSpec parser and BaseOutput abstract class
that all output handlers must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class OutputSpec:
    """
    Parses output specification strings.
    
    Supports a colon-separated format similar to ffmpeg filters:
        previewtrack=filename=custom.mp4:option=value
    
    Example:
        >>> spec = OutputSpec("previewtrack=filename=test.mp4:mode=stab")
        >>> spec.output_type
        'previewtrack'
        >>> spec.get('filename')
        'test.mp4'
        >>> spec.get('mode')
        'stab'
    """
    
    def __init__(self, spec_string: str):
        """
        Parse a specification string.
        
        Args:
            spec_string: The specification string to parse
            
        Raises:
            ValueError: If the spec string is empty
        """
        self.output_type: str = ""
        self.options: dict[str, str] = {}
        
        if not spec_string:
            raise ValueError("Empty output specification")
        
        # Split by '=' for the first part to get type
        parts = spec_string.split('=', 1)
        self.output_type = parts[0].strip().lower()
        
        if len(parts) > 1:
            self._parse_options(parts[1])
    
    def _parse_options(self, options_str: str) -> None:
        """Parse colon-separated key=value pairs."""
        current_key: str | None = None
        current_value: list[str] = []
        
        tokens = options_str.split(':')
        for token in tokens:
            if '=' in token:
                # Save previous key-value if exists
                if current_key is not None:
                    self.options[current_key] = ':'.join(current_value)
                
                # Start new key-value
                key, value = token.split('=', 1)
                current_key = key.strip().lower()
                current_value = [value.strip()]
            else:
                # Continue previous value (contained ':')
                current_value.append(token)
        
        # Save last key-value
        if current_key is not None:
            self.options[current_key] = ':'.join(current_value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get an option value."""
        return self.options.get(key.lower(), default)
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get an option value as integer."""
        val = self.get(key)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError:
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get an option value as float."""
        val = self.get(key)
        if val is None:
            return default
        try:
            return float(val)
        except ValueError:
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get an option value as boolean."""
        val = self.get(key)
        if val is None:
            return default
        return val.lower() in ('true', 'yes', '1', 'on')
    
    def __repr__(self) -> str:
        return f"OutputSpec(type={self.output_type}, options={self.options})"


class BaseOutput(ABC):
    """
    Abstract base class for all output handlers.
    
    Subclasses must implement:
        - _get_default_suffix(): Default filename suffix
        - _get_default_extension(): Default file extension
        - initialize(): Set up the output (open files, etc.)
        - process_frame(): Process a single frame
        - finalize(): Clean up resources
    
    Example:
        class MyOutput(BaseOutput):
            def _get_default_suffix(self) -> str:
                return "_custom"
            
            def _get_default_extension(self) -> str:
                return "json"
            
            def initialize(self, video_props: dict) -> None:
                self.data = []
            
            def process_frame(self, frame_num, frame, tracking_data) -> None:
                self.data.append({...})
            
            def finalize(self) -> None:
                with open(self.output_path, 'w') as f:
                    json.dump(self.data, f)
    """
    
    def __init__(self, spec: OutputSpec, input_path: str):
        """
        Initialize the output handler.
        
        Args:
            spec: The parsed output specification
            input_path: Path to the input video file
        """
        self.spec = spec
        self.input_path = Path(input_path)
        self.output_path = self._resolve_output_path()
    
    @abstractmethod
    def _get_default_suffix(self) -> str:
        """Return the default suffix to add to input filename."""
        pass
    
    @abstractmethod
    def _get_default_extension(self) -> str:
        """Return the default file extension."""
        pass
    
    def _resolve_output_path(self) -> Path:
        """Resolve the output path from spec or generate default."""
        filename = self.spec.get('filename')
        if filename:
            return Path(filename)
        
        # Generate default: input_suffix.ext
        stem = self.input_path.stem
        suffix = self._get_default_suffix()
        ext = self._get_default_extension()
        return Path(f"{stem}{suffix}.{ext}")
    
    @abstractmethod
    def initialize(self, video_props: dict) -> None:
        """
        Initialize the output (open files, create writers, etc).
        
        Args:
            video_props: Dictionary with 'width', 'height', 'fps'
        """
        pass
    
    @abstractmethod
    def process_frame(
        self,
        frame_num: int,
        frame: np.ndarray,
        tracking_data: dict,
    ) -> None:
        """
        Process a single frame of data.
        
        Args:
            frame_num: Current frame number
            frame: The video frame (may be stabilized)
            tracking_data: Dictionary containing tracking information
        """
        pass
    
    @abstractmethod
    def finalize(self) -> None:
        """Finalize the output (close files, etc)."""
        pass
    
    def get_output_path(self) -> Path:
        """Return the resolved output path."""
        return self.output_path
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()
        return False
