"""
Metadata handling utilities.

Provides functions for reading and writing video/image metadata
using exiftool.
"""

import subprocess
from pathlib import Path
from typing import Any


def get_metadata(path: str | Path) -> dict[str, Any]:
    """
    Extract metadata from a file using exiftool.
    
    Args:
        path: Path to the file
        
    Returns:
        Dictionary of metadata key-value pairs
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        result = subprocess.run(
            ['exiftool', '-j', str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        import json
        data = json.loads(result.stdout)
        return data[0] if data else {}
    except subprocess.CalledProcessError as e:
        print(f"Warning: exiftool error: {e.stderr}")
        return {}
    except FileNotFoundError:
        print("Warning: exiftool not found. Metadata operations will be skipped.")
        return {}


def set_metadata(path: str | Path, metadata: dict[str, Any]) -> bool:
    """
    Set metadata on a file using exiftool.
    
    Args:
        path: Path to the file
        metadata: Dictionary of metadata to set
        
    Returns:
        True if successful, False otherwise
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if not metadata:
        return True
    
    # Build exiftool arguments
    args = ['exiftool', '-overwrite_original']
    for key, value in metadata.items():
        if key.startswith('Source') or key.startswith('File'):
            continue  # Skip file-specific metadata
        args.append(f'-{key}={value}')
    args.append(str(path))
    
    try:
        subprocess.run(args, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to set metadata: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Warning: exiftool not found. Metadata operations will be skipped.")
        return False


def copy_metadata(src_path: str | Path, dst_path: str | Path) -> bool:
    """
    Copy metadata from source file to destination file.
    
    Args:
        src_path: Path to source file
        dst_path: Path to destination file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        result = subprocess.run(
            ['exiftool', '-overwrite_original', '-TagsFromFile', str(src_path), str(dst_path)],
            capture_output=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to copy metadata: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Warning: exiftool not found. Metadata operations will be skipped.")
        return False


def get_gps_position(path: str | Path) -> tuple[float, float] | None:
    """
    Get GPS position from a file.
    
    Args:
        path: Path to the file
        
    Returns:
        Tuple of (latitude, longitude) or None if not found
    """
    try:
        result = subprocess.run(
            ['exiftool', '-s', '-s', '-s', '-c', '%+7f', '-GPSPosition', str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        parts = result.stdout.strip().split()
        if len(parts) >= 2:
            lat = float(parts[0].rstrip(','))
            lon = float(parts[1])
            return (lat, lon)
        return None
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


def set_gps_position(
    path: str | Path,
    latitude: float,
    longitude: float,
) -> bool:
    """
    Set GPS position on a file.
    
    Args:
        path: Path to the file
        latitude: GPS latitude
        longitude: GPS longitude
        
    Returns:
        True if successful, False otherwise
    """
    try:
        subprocess.run(
            [
                'exiftool', '-overwrite_original',
                f'-GPSLatitude={latitude}',
                f'-GPSLongitude={longitude}',
                str(path)
            ],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
