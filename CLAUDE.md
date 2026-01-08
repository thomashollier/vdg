# VDG - Video Development and Grading Toolkit

## Project Overview
Python framework for video processing: point tracking, stabilization, frame averaging, and post-processing with hardware acceleration support.

## Architecture
- `vdg/core/` - Base abstractions, video I/O, hardware config
- `vdg/tracking/` - FeatureTracker, Stabilizer, track I/O (.crv files)
- `vdg/processing/` - Frame averaging, color utilities
- `vdg/postprocess/` - Alpha compositing operations with registry pattern
- `vdg/outputs/` - Extensible output handlers (video, CSV, .crv)
- `vdg/nodes/` - Web-based node editor (FastAPI + vanilla JS)
- `vdg/pipeline/` - Batch processing (stub)

## Key Components
- **HardwareConfig**: Package-level GPU acceleration (VideoToolbox/NVENC/VAAPI)
- **FeatureTracker**: Lucas-Kanade optical flow with forward-backward validation
- **Stabilizer**: Single-point, two-point, and perspective modes
- **OutputManager**: Plugin architecture for outputs
- **Post-process Operations**: Modular registry for alpha compositing operations

## Node Editor
Web-based visual graph editor at `vdg/nodes/web_editor.py`.

### Port Naming Convention
- Inputs: `video_in`, `image_in`, `mask_in`, `alpha_in`
- Outputs: `video_out`, `image_out`, `mask_out`, `alpha_out`

### Key Nodes
- **video_input**: Load video files with optional hardware decode
- **image_input**: Load existing images (PNG, TIFF, EXR, etc.)
- **feature_tracker**: Single-point tracking with ROI
- **feature_tracker_2p**: Two-point tracking in single pass
- **stabilizer**: Compute transforms from track data
- **apply_transform**: Apply stabilization with padding/offset
- **frame_average**: Accumulate frames with alpha
- **gamma**: Linear/sRGB color space conversion
- **post_process**: Alpha compositing operations with trim option

### Post-Process Operations
Located in `vdg/postprocess/operations.py` with `@register_operation` decorator:
- `comp_on_white`: Composite premultiplied RGB over white
- `comp_on_black`: Composite over black (clamp)
- `refine_alpha`: Gamma + sigmoid contrast + blur + power curve, then composite
- `divide_alpha`: Unpremultiply (divide RGB by alpha), on black
- `unpremult_on_white`: Unpremultiply then composite on white

To add new operations:
```python
@register_operation("my_op", "Description here")
def my_op(image: np.ndarray, alpha: np.ndarray, **params) -> np.ndarray:
    # Process and return RGB image
    return result
```

## Running
```bash
# Node editor
python -m vdg.nodes.web_editor

# Tests
pytest tests/
```

## Workflows
Pre-built workflows in `workflows/` directory:
- `feature_track.json` - Single point tracking
- `feature_track_2p.json` - Two-point tracking
- `stabilize_linear_composite.json` - Full stabilization pipeline
- `post_process.json` - Post-processing from existing images

## Original Scripts
- `scripts/postProcessMovies.py` - Original OIIO-based post-processing (reference)
