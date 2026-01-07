# VDG - Video Development and Grading Toolkit

## Project Overview
Python framework for video processing: point tracking, stabilization, frame averaging, and post-processing with hardware acceleration support.

## Architecture
- `vdg/core/` - Base abstractions, video I/O, hardware config
- `vdg/tracking/` - FeatureTracker, Stabilizer, track I/O (.crv files)
- `vdg/processing/` - Frame averaging (stub)
- `vdg/postprocess/` - OIIO operations (stub)
- `vdg/outputs/` - Extensible output handlers (video, CSV, .crv)
- `vdg/nodes/` - Web-based node editor (FastAPI + vanilla JS)
- `vdg/pipeline/` - Batch processing (stub)

## Key Components
- **HardwareConfig**: Package-level GPU acceleration (VideoToolbox/NVENC/VAAPI)
- **FeatureTracker**: Lucas-Kanade optical flow with forward-backward validation
- **Stabilizer**: Single-point, two-point, and perspective modes
- **OutputManager**: Plugin architecture for outputs

## Running
```bash
# Node editor
python -m vdg.nodes.web_editor

# Tests
pytest tests/
```

## Current State
- Core tracking and stabilization working
- Web node editor with real execution
- Stubs remain for: processing/, postprocess/, pipeline/

## Original Scripts Being Migrated
- video_tracker.py → vdg/tracking/
- stabFromTrack_v07b.py → vdg/tracking/stabilizer.py
- frameavg_v05.py → vdg/processing/ (TODO)
- postProcessMovies.py → vdg/postprocess/ (TODO)
- processMovies_v01.py → vdg/pipeline/ (TODO)
