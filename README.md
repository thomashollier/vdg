# VDG - Video Development and Grading Toolkit

A modular Python framework for video point tracking, stabilization, frame averaging, and post-processing.

## Features

- **Point Tracking**: Lucas-Kanade optical flow tracking with automatic feature replenishment
- **Blender Integration**: Export tracking data from Blender's movie clip editor
- **Stabilization**: Transform-based video stabilization (translation, rotation, scale, perspective)
- **Frame Averaging**: Temporal frame stacking with alpha compositing
- **Post-Processing**: Image manipulation using OpenImageIO
- **Batch Processing**: JSON-configured job management

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vdg.git
cd vdg

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Package Structure

```
vdg/
├── README.md
├── pyproject.toml
├── requirements.txt
├── setup.py
│
├── vdg/                          # Main package
│   ├── __init__.py               # Package exports
│   ├── __main__.py               # CLI entry point
│   │
│   ├── core/                     # Core abstractions
│   │   ├── __init__.py
│   │   ├── base.py               # Base classes and protocols
│   │   ├── video.py              # Video I/O utilities
│   │   └── config.py             # Configuration management
│   │
│   ├── tracking/                 # Point tracking module
│   │   ├── __init__.py
│   │   ├── tracker.py            # FeatureTracker class
│   │   ├── stabilizer.py         # Stabilization transforms
│   │   └── blender_export.py     # Blender tracker export
│   │
│   ├── processing/               # Video/image processing
│   │   ├── __init__.py
│   │   ├── frame_average.py      # Frame averaging/stacking
│   │   ├── color.py              # Color correction utilities
│   │   └── transforms.py         # Image transforms
│   │
│   ├── postprocess/              # Post-processing operations
│   │   ├── __init__.py
│   │   ├── composite.py          # Alpha compositing
│   │   ├── trim.py               # Image trimming/scaling
│   │   └── filters.py            # Image filters
│   │
│   ├── outputs/                  # Output handlers
│   │   ├── __init__.py
│   │   ├── base.py               # BaseOutput abstract class
│   │   ├── video.py              # Video output handlers
│   │   ├── data.py               # Data output (CSV, CRV)
│   │   └── manager.py            # OutputManager
│   │
│   ├── pipeline/                 # Batch processing
│   │   ├── __init__.py
│   │   ├── runner.py             # Job runner
│   │   └── config_gen.py         # Config file generation
│   │
│   └── utils/                    # Shared utilities
│       ├── __init__.py
│       ├── metadata.py           # Metadata handling (exiftool)
│       ├── ffmpeg.py             # FFmpeg wrapper
│       └── math.py               # Math utilities
│
├── scripts/                      # Standalone CLI scripts
│   ├── vdg-track                 # Point tracking
│   ├── vdg-stabilize             # Stabilization
│   ├── vdg-frameavg              # Frame averaging
│   └── vdg-process               # Batch processing
│
└── tests/                        # Unit tests
    ├── __init__.py
    ├── test_tracking.py
    ├── test_stabilization.py
    └── test_outputs.py
```

## Quick Start

### Command Line Usage

```bash
# Track points in a video
vdg track input.mp4 -out previewtrack -out trackers=type=2

# Stabilize using tracking data
vdg stabilize input.mp4 -td track01.crv:track02.crv -wo

# Average frames
vdg frameavg input.mp4 -fs 100 -fe 500 -wa

# Run batch processing
vdg process -c process_config.json
```

### Python API

```python
from vdg.tracking import FeatureTracker
from vdg.processing import FrameAverager
from vdg.outputs import OutputManager

# Create a tracker
tracker = FeatureTracker(num_features=50)

# Initialize on first frame
points = tracker.initialize(first_frame)

# Track through video
for frame in video_frames:
    points, ids, stats = tracker.update(frame)
```

## Module Details

### Tracking (`vdg.tracking`)

The tracking module provides feature point detection and tracking:

- `FeatureTracker`: Lucas-Kanade optical flow with forward-backward validation
- `Stabilizer`: Compute stabilization transforms from tracking data
- `BlenderExporter`: Export tracking data from Blender projects

### Processing (`vdg.processing`)

Video and image processing operations:

- `FrameAverager`: Temporal frame averaging with alpha masking
- `ColorCorrector`: Gamma, exposure, CLAHE adjustments
- `TransformEngine`: Affine and perspective transforms

### Outputs (`vdg.outputs`)

Extensible output system:

- `PreviewTrackOutput`: Video with tracking overlay
- `CSVOutput`: Tracking data as CSV
- `TrackersOutput`: Normalized coordinates (.crv format)
- `CleanVideoOutput`: Stabilized video without overlays

### Post-processing (`vdg.postprocess`)

Image post-processing using OpenImageIO:

- `Compositor`: Alpha channel manipulation and compositing
- `Trimmer`: Auto-trim and scale images
- `Filters`: Sigmoid contrast, blur, power adjustments

## Configuration

### Pipeline Configuration (JSON)

```json
{
  "global_settings": {
    "prefix": "project_name",
    "keep_temp_files": false
  },
  "movies": {
    "clip_001": {
      "passes": [
        {
          "name": "stab",
          "type": "track",
          "track_files": ["track01.crv", "track02.crv"],
          "stab_options": "-wo -wm",
          "do_stab": true,
          "do_frameavg": true
        }
      ]
    }
  }
}
```

## Extending the Framework

### Adding a New Output Type

```python
from vdg.outputs.base import BaseOutput, OutputSpec

class MyCustomOutput(BaseOutput):
    def _get_default_suffix(self) -> str:
        return "_custom"
    
    def _get_default_extension(self) -> str:
        return "json"
    
    def initialize(self, video_props: dict) -> None:
        self.data = []
    
    def process_frame(self, frame_num: int, frame, tracking_data: dict) -> None:
        self.data.append({"frame": frame_num, ...})
    
    def finalize(self) -> None:
        with open(self.output_path, 'w') as f:
            json.dump(self.data, f)

# Register the output type
from vdg.outputs import register_output_type
register_output_type('custom', MyCustomOutput)
```

### Adding a New Processing Filter

```python
from vdg.processing.base import BaseFilter

class MyFilter(BaseFilter):
    def apply(self, frame: np.ndarray) -> np.ndarray:
        # Your processing logic
        return processed_frame
```

## Dependencies

- Python 3.10+
- OpenCV (cv2)
- NumPy
- SciPy (for Gaussian filtering)
- OpenImageIO (optional, for post-processing)

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request
