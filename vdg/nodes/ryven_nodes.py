"""
VDG Nodes for Ryven
====================

This package defines Ryven nodes that wrap VDG functionality.

To use:
    1. pip install ryven
    2. ryven
    3. Import this nodes package in Ryven
    
Or run programmatically:
    python -m vdg.nodes.ryven_nodes
"""

from ryven.node_env import *

# Optional: import VDG components
# These imports happen when nodes are instantiated
# from vdg.tracking import FeatureTracker, Stabilizer
# from vdg.core.video import VideoReader, VideoWriter


# =============================================================================
# INPUT NODES
# =============================================================================

class VideoInputNode(Node):
    """Load a video file."""
    
    title = 'Video Input'
    tags = ['input', 'video']
    
    init_inputs = []
    init_outputs = [
        NodeOutputType(label='video'),
        NodeOutputType(label='props'),
    ]
    
    def __init__(self, params):
        super().__init__(params)
        self.filepath = ''
        self.first_frame = 1
        self.last_frame = -1
    
    def view(self, node_gui):
        """Create the node's GUI (file picker, frame range)."""
        from qtpy.QtWidgets import QLineEdit, QSpinBox, QPushButton, QFileDialog
        
        # File path input
        self.path_edit = QLineEdit(self.filepath)
        self.path_edit.setPlaceholderText('Video file path...')
        
        browse_btn = QPushButton('Browse')
        browse_btn.clicked.connect(self._browse_file)
        
        # Frame range
        self.first_spin = QSpinBox()
        self.first_spin.setRange(1, 999999)
        self.first_spin.setValue(self.first_frame)
        
        self.last_spin = QSpinBox()
        self.last_spin.setRange(-1, 999999)
        self.last_spin.setValue(self.last_frame)
        self.last_spin.setSpecialValueText('End')
        
        node_gui.add_row('Path:', self.path_edit)
        node_gui.add_row('', browse_btn)
        node_gui.add_row('First Frame:', self.first_spin)
        node_gui.add_row('Last Frame:', self.last_spin)
    
    def _browse_file(self):
        from qtpy.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            None, 'Select Video', '', 
            'Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)'
        )
        if path:
            self.path_edit.setText(path)
            self.filepath = path
    
    def update_event(self, inp=-1):
        from vdg.core.video import VideoReader
        
        self.filepath = self.path_edit.text() if hasattr(self, 'path_edit') else self.filepath
        self.first_frame = self.first_spin.value() if hasattr(self, 'first_spin') else self.first_frame
        self.last_frame = self.last_spin.value() if hasattr(self, 'last_spin') else self.last_frame
        
        if not self.filepath:
            return
        
        last = None if self.last_frame == -1 else self.last_frame
        reader = VideoReader(self.filepath, self.first_frame, last)
        reader.open()
        
        self.set_output(0, reader)
        self.set_output(1, reader.properties)


class TrackFileInputNode(Node):
    """Load .crv track file(s)."""
    
    title = 'Track File Input'
    tags = ['input', 'tracking']
    
    init_inputs = []
    init_outputs = [
        NodeOutputType(label='track_data'),
    ]
    
    def __init__(self, params):
        super().__init__(params)
        self.filepaths = []
    
    def update_event(self, inp=-1):
        from vdg.tracking.track_io import read_crv_file
        
        tracks = []
        for path in self.filepaths:
            tracks.append(read_crv_file(path))
        
        self.set_output(0, tracks)


class ROIInputNode(Node):
    """Define a region of interest."""
    
    title = 'ROI'
    tags = ['input', 'roi']
    
    init_inputs = []
    init_outputs = [
        NodeOutputType(label='roi'),
    ]
    
    def __init__(self, params):
        super().__init__(params)
        self.x = 0
        self.y = 0
        self.width = 100
        self.height = 100
    
    def update_event(self, inp=-1):
        self.set_output(0, (self.x, self.y, self.width, self.height))


# =============================================================================
# TRACKING NODES
# =============================================================================

class FeatureTrackerNode(Node):
    """Track features in video using Lucas-Kanade optical flow."""
    
    title = 'Feature Tracker'
    tags = ['tracking', 'opencv']
    
    init_inputs = [
        NodeInputType(label='video'),
        NodeInputType(label='roi', optional=True),
    ]
    init_outputs = [
        NodeOutputType(label='points'),
        NodeOutputType(label='track_data'),
    ]
    
    def __init__(self, params):
        super().__init__(params)
        self.num_features = 30
        self.enforce_bbox = True
    
    def update_event(self, inp=-1):
        from vdg.tracking import FeatureTracker
        
        reader = self.input(0)
        roi = self.input(1)
        
        if reader is None:
            return
        
        tracker = FeatureTracker(
            num_features=self.num_features,
            initial_roi=roi,
            enforce_bbox=self.enforce_bbox,
        )
        
        all_points = []
        track_data = {}
        
        for frame_num, frame in reader:
            if frame_num == reader.first_frame:
                tracker.initialize(frame)
            else:
                tracker.update(frame)
            
            all_points.append(tracker.points.copy())
            track_data[frame_num] = {
                'points': tracker.points.copy(),
                'ids': tracker.point_ids.copy(),
            }
        
        self.set_output(0, all_points)
        self.set_output(1, track_data)


class StabilizerNode(Node):
    """Compute stabilization transforms from track data."""
    
    title = 'Stabilizer'
    tags = ['tracking', 'stabilization']
    
    init_inputs = [
        NodeInputType(label='track_files'),
        NodeInputType(label='video_props'),
    ]
    init_outputs = [
        NodeOutputType(label='transforms'),
    ]
    
    def __init__(self, params):
        super().__init__(params)
        self.mode = 'two_point'  # 'single', 'two_point', 'vstab', 'perspective'
        self.ref_frame = -1
        self.swap_xy = False
        self.x_flip = False
        self.y_flip = False
    
    def update_event(self, inp=-1):
        from vdg.tracking import Stabilizer, StabilizeMode
        from vdg.tracking.stabilizer import read_track_data
        
        track_files = self.input(0)
        props = self.input(1)
        
        if not track_files or not props:
            return
        
        # Read track data
        track_str = ':'.join(track_files) if isinstance(track_files, list) else track_files
        trackers_dict = read_track_data(
            track_str,
            ref_frame=self.ref_frame,
            movie_width=props.width,
            movie_height=props.height,
            portrait=props.is_portrait,
            swap_xy=self.swap_xy,
            x_flip=self.x_flip,
            y_flip=self.y_flip,
        )
        
        self.set_output(0, trackers_dict)


# =============================================================================
# PROCESSING NODES  
# =============================================================================

class ApplyTransformNode(Node):
    """Apply stabilization transform to video frames."""
    
    title = 'Apply Transform'
    tags = ['processing', 'stabilization']
    
    init_inputs = [
        NodeInputType(label='video'),
        NodeInputType(label='transforms'),
    ]
    init_outputs = [
        NodeOutputType(label='stabilized'),
    ]
    
    def __init__(self, params):
        super().__init__(params)
        self.x_pad = 0
        self.y_pad = 0
        self.x_offset = 0
        self.y_offset = 0


class FrameAverageNode(Node):
    """Average frames over time."""
    
    title = 'Frame Average'
    tags = ['processing', 'composite']
    
    init_inputs = [
        NodeInputType(label='video'),
    ]
    init_outputs = [
        NodeOutputType(label='image'),
        NodeOutputType(label='alpha'),
    ]
    
    def __init__(self, params):
        super().__init__(params)
        self.comp_mode = 0  # 0=black, 1=white, 2=unpremult, 3=unpremult white
        self.alpha_gamma = 1.0
        self.gamma = 1.0
        self.brightness = 1.0


class ColorCorrectionNode(Node):
    """Apply color corrections."""
    
    title = 'Color Correction'
    tags = ['processing', 'color']
    
    init_inputs = [
        NodeInputType(label='image'),
    ]
    init_outputs = [
        NodeOutputType(label='image'),
    ]
    
    def __init__(self, params):
        super().__init__(params)
        self.gamma = 1.0
        self.brightness = 1.0
        self.clahe = False
        self.clahe_clip = 40.0
        self.clahe_grid = 8


class CLAHENode(Node):
    """Apply CLAHE contrast enhancement."""
    
    title = 'CLAHE'
    tags = ['processing', 'contrast']
    
    init_inputs = [
        NodeInputType(label='image'),
    ]
    init_outputs = [
        NodeOutputType(label='image'),
    ]
    
    def __init__(self, params):
        super().__init__(params)
        self.clip_limit = 40.0
        self.grid_size = 8


# =============================================================================
# OUTPUT NODES
# =============================================================================

class VideoOutputNode(Node):
    """Write video to file."""
    
    title = 'Video Output'
    tags = ['output', 'video']
    
    init_inputs = [
        NodeInputType(label='frames'),
        NodeInputType(label='props'),
    ]
    init_outputs = []
    
    def __init__(self, params):
        super().__init__(params)
        self.filepath = 'output.mp4'
        self.use_hardware = True
        self.bitrate = '20M'


class ImageOutputNode(Node):
    """Write image to file."""
    
    title = 'Image Output'
    tags = ['output', 'image']
    
    init_inputs = [
        NodeInputType(label='image'),
    ]
    init_outputs = []
    
    def __init__(self, params):
        super().__init__(params)
        self.filepath = 'output.png'
        self.bit_depth = 16


class TrackOutputNode(Node):
    """Write tracking data to .crv file."""
    
    title = 'Track Output'
    tags = ['output', 'tracking']
    
    init_inputs = [
        NodeInputType(label='track_data'),
    ]
    init_outputs = []
    
    def __init__(self, params):
        super().__init__(params)
        self.filepath = 'track01.crv'
        self.track_type = 2
        self.center_mode = 'centroid'
        self.filter_sigma = 0


# =============================================================================
# UTILITY NODES
# =============================================================================

class MergeTracksNode(Node):
    """Merge multiple track files."""
    
    title = 'Merge Tracks'
    tags = ['utility', 'tracking']
    
    init_inputs = [
        NodeInputType(label='track1'),
        NodeInputType(label='track2'),
    ]
    init_outputs = [
        NodeOutputType(label='merged'),
    ]


class GaussianFilterNode(Node):
    """Apply Gaussian smoothing to track data."""
    
    title = 'Gaussian Filter'
    tags = ['utility', 'filter']
    
    init_inputs = [
        NodeInputType(label='data'),
    ]
    init_outputs = [
        NodeOutputType(label='filtered'),
    ]
    
    def __init__(self, params):
        super().__init__(params)
        self.sigma = 5.0


# =============================================================================
# NODE REGISTRATION
# =============================================================================

export_nodes([
    # Inputs
    VideoInputNode,
    TrackFileInputNode,
    ROIInputNode,
    
    # Tracking
    FeatureTrackerNode,
    StabilizerNode,
    
    # Processing
    ApplyTransformNode,
    FrameAverageNode,
    ColorCorrectionNode,
    CLAHENode,
    
    # Outputs
    VideoOutputNode,
    ImageOutputNode,
    TrackOutputNode,
    
    # Utilities
    MergeTracksNode,
    GaussianFilterNode,
])


if __name__ == '__main__':
    # Launch Ryven with VDG nodes
    import subprocess
    import sys
    subprocess.run([sys.executable, '-m', 'ryven'])
