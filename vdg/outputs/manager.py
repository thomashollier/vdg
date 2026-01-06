"""
Output manager for coordinating multiple output handlers.
"""

from pathlib import Path
from typing import Type

import numpy as np

from vdg.outputs.base import BaseOutput, OutputSpec
from vdg.outputs.video import PreviewTrackOutput, CleanVideoOutput
from vdg.outputs.data import CSVOutput, TrackersOutput


# Registry of available output types
OUTPUT_TYPES: dict[str, Type[BaseOutput]] = {
    'previewtrack': PreviewTrackOutput,
    'preview': PreviewTrackOutput,
    'csv': CSVOutput,
    'trackers': TrackersOutput,
    'video': CleanVideoOutput,
}


def register_output_type(name: str, output_class: Type[BaseOutput]) -> None:
    """
    Register a new output type.
    
    Args:
        name: Name to use in output specifications
        output_class: Class implementing BaseOutput
        
    Example:
        >>> class MyOutput(BaseOutput):
        ...     ...
        >>> register_output_type('myoutput', MyOutput)
    """
    OUTPUT_TYPES[name.lower()] = output_class


class OutputManager:
    """
    Manages multiple output handlers.
    
    Coordinates initialization, frame processing, and finalization
    across multiple output handlers.
    
    Example:
        >>> manager = OutputManager("input.mp4")
        >>> manager.add_output("previewtrack")
        >>> manager.add_output("csv")
        >>> manager.initialize_all(video_props)
        >>> for frame_num, frame in video:
        ...     manager.process_frame(frame_num, frame, tracking_data)
        >>> manager.finalize_all()
    """
    
    def __init__(self, input_path: str):
        """
        Initialize the output manager.
        
        Args:
            input_path: Path to the input video file
        """
        self.input_path = input_path
        self.outputs: list[BaseOutput] = []
    
    def add_output(self, spec_string: str) -> BaseOutput:
        """
        Add an output from a specification string.
        
        Args:
            spec_string: Output specification (e.g., 'previewtrack=filename=test.mp4')
            
        Returns:
            The created output handler
            
        Raises:
            ValueError: If the output type is unknown
        """
        spec = OutputSpec(spec_string)
        
        if spec.output_type not in OUTPUT_TYPES:
            available = list(OUTPUT_TYPES.keys())
            raise ValueError(
                f"Unknown output type: {spec.output_type}. "
                f"Available: {available}"
            )
        
        output_class = OUTPUT_TYPES[spec.output_type]
        output = output_class(spec, self.input_path)
        self.outputs.append(output)
        return output
    
    def initialize_all(self, video_props: dict) -> None:
        """Initialize all outputs."""
        for output in self.outputs:
            output.initialize(video_props)
    
    def process_frame(
        self,
        frame_num: int,
        frame: np.ndarray,
        tracking_data: dict,
    ) -> None:
        """Process a frame through all outputs."""
        for output in self.outputs:
            output.process_frame(frame_num, frame, tracking_data)
    
    def finalize_all(self) -> None:
        """Finalize all outputs."""
        for output in self.outputs:
            output.finalize()
    
    def get_output_paths(self) -> list[Path]:
        """Get all output paths."""
        paths = []
        for output in self.outputs:
            # TrackersOutput can have multiple paths
            if hasattr(output, 'get_output_paths'):
                paths.extend(output.get_output_paths())
            else:
                paths.append(output.get_output_path())
        return paths
    
    def has_video_output(self) -> bool:
        """Check if any output requires video frames."""
        video_types = (PreviewTrackOutput, CleanVideoOutput)
        return any(isinstance(o, video_types) for o in self.outputs)
    
    def __len__(self) -> int:
        return len(self.outputs)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures finalize_all is called."""
        self.finalize_all()
        return False


def parse_output_specs(specs: list[str], input_path: str) -> OutputManager:
    """
    Parse a list of output specifications and create an OutputManager.
    
    Args:
        specs: List of output specification strings
        input_path: Path to the input video file
        
    Returns:
        Configured OutputManager
    """
    manager = OutputManager(input_path)
    for spec in specs:
        manager.add_output(spec)
    return manager
