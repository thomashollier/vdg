#!/usr/bin/env python3
"""
Minimal Example: VDG API Usage
==============================

Shows the essential API calls without extra boilerplate.
This is the "quick reference" version.
"""

import cv2
from pathlib import Path

from vdg.tracking import FeatureTracker, Stabilizer, StabilizeMode
from vdg.tracking.stabilizer import read_track_data
from vdg.outputs import OutputManager
from vdg.core.video import VideoReader


# =============================================================================
# STEP 1: TRACKING
# Equivalent to: python video_tracker.py input.mp4 -fs 30 -n 30 --stabilize 3 \
#                --enforce-bbox -out trackers=type=2:mode=center
# =============================================================================

input_video = "07062025165117145f_sm.mp4"
first_frame = 30

# Open video
reader = VideoReader(input_video, first_frame=first_frame)
reader.open()
props = reader.properties

# Define two ROIs for vstab mode (would normally be interactive)
roi1 = (100, 200, 150, 100)  # x, y, width, height
roi2 = (100, 400, 150, 100)

# Create trackers for each region
tracker1 = FeatureTracker(num_features=30, initial_roi=roi1, enforce_bbox=True)
tracker2 = FeatureTracker(num_features=30, initial_roi=roi2, enforce_bbox=True)

# Set up output (writes .crv files)
outputs = OutputManager(input_video)
outputs.add_output("trackers=type=2:mode=center")
outputs.initialize_all(props.to_dict())

# Process frames
for frame_num, frame in reader:
    if frame_num == first_frame:
        tracker1.initialize(frame)
        tracker2.initialize(frame)
    else:
        tracker1.update(frame)
        tracker2.update(frame)
    
    # Package tracking data for output
    tracking_data = {
        'points': [tracker1.points, tracker2.points],
        'point_ids': [tracker1.point_ids, tracker2.point_ids],
        'track_lengths': [tracker1.track_lengths, tracker2.track_lengths],
        'rois': [tracker1.get_roi(), tracker2.get_roi()],
    }
    outputs.process_frame(frame_num, frame, tracking_data)

outputs.finalize_all()
reader.close()

print("Track files created:", outputs.get_output_paths())


# =============================================================================
# STEP 2: STABILIZATION  
# Equivalent to: python stabFromTrack_v07b.py \
#   -td track01.crv:track02.crv -tk trk0102 -wo -wm -sw -yf \
#   -rf 1000 -xp 200 -yp 600 -yo -400 input.mp4
# =============================================================================

input_video_full = "07062025165117145f.mp4"
track_files = ["07062025165117145f_track01.crv", "07062025165117145f_track02.crv"]

# Get video dimensions
cap = cv2.VideoCapture(input_video_full)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
portrait = height > width

# Output dimensions with padding
x_pad, y_pad = 200, 600
out_width = width + x_pad
out_height = height + y_pad

# Read tracking data
trackers_dict = read_track_data(
    track_data_files=":".join(track_files),
    ref_frame=1000,
    frame_offset=0,
    movie_width=width,
    movie_height=height,
    portrait=portrait,
    swap_xy=True,   # -sw
    x_flip=False,
    y_flip=True,    # -yf
)

# Set up stabilizer
stabilizer = Stabilizer(mode=StabilizeMode.TWO_POINT)
stabilizer.set_dimensions(width, height)
stabilizer.set_reference(trackers_dict['trackerData'][1000]['markerData'])

# Set up output writers
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter("output_trk0102.mp4", fourcc, fps, (out_width, out_height))
mask_out = cv2.VideoWriter("output_trk0102_mask.mp4", fourcc, fps, (out_width, out_height))

# Process frames
y_offset = -400
for frame_num in range(trackers_dict['ff'], trackers_dict['lf'] + 1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get transform matrix
    markers = trackers_dict['trackerData'][frame_num]['markerData']
    matrix = stabilizer.compute_transform(
        markers,
        x_offset=x_pad / 2,
        y_offset=y_offset + y_pad / 2,
    )
    
    # Apply transform with alpha for mask
    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame_rgba[:, :, 3] = 255
    stabilized = cv2.warpAffine(frame_rgba, matrix, (out_width, out_height))
    
    # Write outputs
    video_out.write(cv2.cvtColor(stabilized, cv2.COLOR_BGRA2BGR))
    mask = cv2.cvtColor(stabilized[:, :, 3], cv2.COLOR_GRAY2BGR)
    mask_out.write(mask)

# Cleanup
cap.release()
video_out.release()
mask_out.release()

print("Stabilization complete!")
print("  Video: output_trk0102.mp4")
print("  Mask:  output_trk0102_mask.mp4")
