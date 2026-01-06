"""
Base classes and protocols for the VDG framework.

This module defines the foundational abstractions that other modules build upon.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class ProcessingContext:
    """
    Context object passed through the processing pipeline.
    
    Contains shared state and configuration that processors can access.
    """
    input_path: Path
    output_dir: Path = field(default_factory=lambda: Path("."))
    frame_start: int = 1
    frame_end: int | None = None
    verbose: bool = True
    dry_run: bool = False
    extra: dict = field(default_factory=dict)
    
    @property
    def frame_range(self) -> tuple[int, int | None]:
        """Return the frame range as a tuple."""
        return (self.frame_start, self.frame_end)


class BaseProcessor(ABC):
    """
    Abstract base class for all frame processors.
    
    Processors implement a consistent interface for frame-by-frame operations.
    """
    
    def __init__(self, context: ProcessingContext | None = None):
        self.context = context
        self._initialized = False
    
    @abstractmethod
    def initialize(self, video_props: dict[str, Any]) -> None:
        """
        Initialize the processor with video properties.
        
        Args:
            video_props: Dictionary containing 'width', 'height', 'fps', 'frame_count'
        """
        pass
    
    @abstractmethod
    def process_frame(
        self, 
        frame_num: int, 
        frame: np.ndarray, 
        **kwargs
    ) -> np.ndarray | None:
        """
        Process a single frame.
        
        Args:
            frame_num: Current frame number (1-indexed)
            frame: BGR frame as numpy array
            **kwargs: Additional processing data
            
        Returns:
            Processed frame or None if frame should be skipped
        """
        pass
    
    @abstractmethod
    def finalize(self) -> Any:
        """
        Finalize processing and clean up resources.
        
        Returns:
            Any final output (path, data, etc.)
        """
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures finalize is called."""
        self.finalize()
        return False


@runtime_checkable
class FrameFilter(Protocol):
    """Protocol for simple frame filters that transform frames in-place."""
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply the filter to a frame."""
        ...


@runtime_checkable  
class TrackingDataProvider(Protocol):
    """Protocol for objects that provide tracking data."""
    
    def get_tracking_data(self, frame_num: int) -> dict[str, Any]:
        """Get tracking data for a specific frame."""
        ...


class ComposableProcessor(BaseProcessor):
    """
    A processor that can be composed with other processors.
    
    Supports chaining multiple processors together.
    """
    
    def __init__(
        self, 
        processors: list[BaseProcessor] | None = None,
        context: ProcessingContext | None = None
    ):
        super().__init__(context)
        self.processors = processors or []
    
    def add_processor(self, processor: BaseProcessor) -> "ComposableProcessor":
        """Add a processor to the chain."""
        self.processors.append(processor)
        return self
    
    def initialize(self, video_props: dict[str, Any]) -> None:
        """Initialize all chained processors."""
        for proc in self.processors:
            proc.initialize(video_props)
        self._initialized = True
    
    def process_frame(
        self, 
        frame_num: int, 
        frame: np.ndarray, 
        **kwargs
    ) -> np.ndarray | None:
        """Process frame through all chained processors."""
        result = frame
        for proc in self.processors:
            if result is None:
                break
            result = proc.process_frame(frame_num, result, **kwargs)
        return result
    
    def finalize(self) -> list[Any]:
        """Finalize all chained processors."""
        results = []
        for proc in self.processors:
            results.append(proc.finalize())
        return results


class FilterChain:
    """
    A chain of frame filters that can be applied in sequence.
    
    Example:
        chain = FilterChain()
        chain.add(GammaFilter(2.2))
        chain.add(CLAHEFilter(clip_limit=40))
        
        processed = chain.apply(frame)
    """
    
    def __init__(self, filters: list[FrameFilter] | None = None):
        self.filters = filters or []
    
    def add(self, filter_: FrameFilter) -> "FilterChain":
        """Add a filter to the chain."""
        self.filters.append(filter_)
        return self
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply all filters in sequence."""
        result = frame
        for f in self.filters:
            result = f.apply(result)
        return result
    
    def __len__(self) -> int:
        return len(self.filters)
