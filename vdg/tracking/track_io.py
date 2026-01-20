"""
Track data I/O utilities.

This module provides functions for reading and writing tracking data
in the .crv format used by VDG and compatible with Blender exports.
"""

import re
from pathlib import Path
from typing import Iterator


# Pattern for parsing CRV format: FRAME [[ x, y ]]
CRV_PATTERN = re.compile(r'(\d+)\s*\[\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]\]')

# Pattern for parsing simple format: FRAME x y
SIMPLE_PATTERN = re.compile(r'(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)')


def parse_track_line(line: str) -> tuple[int, float, float] | None:
    """
    Parse a single line of tracking data.
    
    Supports two formats:
        - CRV format: FRAME [[ x, y ]]
        - Simple format: FRAME x y
    
    Args:
        line: Line of text to parse
        
    Returns:
        Tuple of (frame_number, x, y) or None if line doesn't match
    """
    line = line.strip()
    if not line:
        return None
    
    # Try CRV format first
    match = CRV_PATTERN.match(line)
    if match:
        return (
            int(match.group(1)),
            float(match.group(2)),
            float(match.group(3)),
        )
    
    # Try simple format
    match = SIMPLE_PATTERN.match(line)
    if match:
        return (
            int(match.group(1)),
            float(match.group(2)),
            float(match.group(3)),
        )
    
    return None


def read_crv_file(path: str | Path) -> dict[int, tuple[float, float]]:
    """
    Read tracking data from a .crv file.
    
    Args:
        path: Path to the .crv file
        
    Returns:
        Dictionary mapping frame numbers to (x, y) coordinates
        
    Example:
        >>> data = read_crv_file("track01.crv")
        >>> x, y = data[100]  # Get coordinates for frame 100
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Track file not found: {path}")
    
    data = {}
    with open(path, 'r') as f:
        for line in f:
            parsed = parse_track_line(line)
            if parsed:
                frame, x, y = parsed
                data[frame] = (x, y)
    
    return data


def write_crv_file(
    path: str | Path,
    data: dict[int, tuple[float, float]] | list[tuple[int, float, float]],
) -> None:
    """
    Write tracking data to a .crv file.
    
    Args:
        path: Output path for the .crv file
        data: Either a dict mapping frame -> (x, y) or a list of (frame, x, y) tuples
        
    Example:
        >>> write_crv_file("track01.crv", {100: (0.5, 0.3), 101: (0.51, 0.31)})
    """
    path = Path(path)
    
    # Convert list to sorted items if necessary
    if isinstance(data, list):
        items = sorted(data, key=lambda x: x[0])
    else:
        items = [(f, xy[0], xy[1]) for f, xy in sorted(data.items())]
    
    with open(path, 'w') as f:
        for frame, x, y in items:
            f.write(f"{frame} [[ {x}, {y}]]\n")


def iter_crv_file(path: str | Path) -> Iterator[tuple[int, float, float]]:
    """
    Iterate over tracking data from a .crv file.
    
    Useful for large files where you don't want to load everything into memory.
    
    Args:
        path: Path to the .crv file
        
    Yields:
        Tuples of (frame_number, x, y)
    """
    path = Path(path)
    with open(path, 'r') as f:
        for line in f:
            parsed = parse_track_line(line)
            if parsed:
                yield parsed


def get_crv_frame_range(path: str | Path) -> tuple[int, int]:
    """
    Get the frame range of a .crv file without loading all data.
    
    Args:
        path: Path to the .crv file
        
    Returns:
        Tuple of (first_frame, last_frame)
    """
    first_frame = None
    last_frame = None
    
    for frame, _, _ in iter_crv_file(path):
        if first_frame is None:
            first_frame = frame
        last_frame = frame
    
    if first_frame is None:
        raise ValueError(f"No valid data in {path}")
    
    return (first_frame, last_frame)


def merge_crv_files(
    paths: list[str | Path],
    output_path: str | Path,
) -> None:
    """
    Merge multiple .crv files into one.
    
    For overlapping frames, uses the average of all values.
    
    Args:
        paths: List of input .crv file paths
        output_path: Output path for merged file
    """
    # Collect all data with counts for averaging
    frame_data: dict[int, list[tuple[float, float]]] = {}
    
    for path in paths:
        for frame, x, y in iter_crv_file(path):
            if frame not in frame_data:
                frame_data[frame] = []
            frame_data[frame].append((x, y))
    
    # Average overlapping frames
    averaged_data = {}
    for frame, coords in frame_data.items():
        if len(coords) == 1:
            averaged_data[frame] = coords[0]
        else:
            avg_x = sum(c[0] for c in coords) / len(coords)
            avg_y = sum(c[1] for c in coords) / len(coords)
            averaged_data[frame] = (avg_x, avg_y)
    
    write_crv_file(output_path, averaged_data)


def interpolate_missing_frames(
    data: dict[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    """
    Interpolate missing frames in tracking data.
    
    Uses linear interpolation between known frames.
    
    Args:
        data: Dictionary mapping frame -> (x, y)
        
    Returns:
        New dictionary with interpolated frames
    """
    if not data:
        return {}
    
    frames = sorted(data.keys())
    result = dict(data)
    
    for i in range(len(frames) - 1):
        start_frame = frames[i]
        end_frame = frames[i + 1]
        
        if end_frame - start_frame > 1:
            # Need to interpolate
            start_x, start_y = data[start_frame]
            end_x, end_y = data[end_frame]
            
            for f in range(start_frame + 1, end_frame):
                t = (f - start_frame) / (end_frame - start_frame)
                x = start_x + t * (end_x - start_x)
                y = start_y + t * (end_y - start_y)
                result[f] = (x, y)
    
    return result
