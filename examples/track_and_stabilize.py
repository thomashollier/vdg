#!/usr/bin/env python3
"""
Example: Track and Stabilize Pipeline
======================================

This script demonstrates using the VDG API to accomplish the equivalent of:

    python ~/bin/vdg/video_tracker.py 07062025165117145f_sm.mp4 \
        -fs 30 -n 30 --stabilize 3 --enforce-bbox \
        -out trackers=type=2:mode=center

    python ~/bin/vdg/stabFromTrack_v07b.py \
        -td 07062025165117145f_track01.crv:07062025165117145f_track02.crv \
        -tk trk0102 -wo -wm -sw -yf -rf 1000 -xp 200 -yp 600 -yo -400 \
        07062025165117145f.mp4
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from vdg.core.video import VideoReader, VideoWriter, VideoProperties
from vdg.tracking import FeatureTracker, Stabilizer, StabilizeMode
from vdg.tracking.track_io import read_crv_file, write_crv_file
from vdg.outputs import OutputManager


def select_roi_interactive(frame: np.ndarray, window_name: str = "Select ROI") -> tuple:
    """Let user select ROI interactively."""
    roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)
    return roi


def run_tracking(
    input_video: str,
    first_frame: int = 30,
    num_features: int = 30,
    stabilize_mode: int = 3,  # VSTAB
    enforce_bbox: bool = True,
    output_specs: list[str] = None,
    bboxes: list[tuple] = None,
) -> dict:
    """
    Run feature tracking on a video.
    
    Equivalent to:
        python video_tracker.py input.mp4 -fs 30 -n 30 --stabilize 3 \
            --enforce-bbox -out trackers=type=2:mode=center
    
    Args:
        input_video: Path to input video
        first_frame: Frame to start tracking from
        num_features: Number of features per region
        stabilize_mode: 0=off, 1=single, 2=two-point, 3=vstab
        enforce_bbox: Keep points within original bbox
        output_specs: List of output specifications
        bboxes: Pre-defined bounding boxes (or None for interactive)
    
    Returns:
        Dictionary with tracking results and output paths
    """
    input_path = Path(input_video)
    output_specs = output_specs or ["trackers=type=2:mode=center"]
    
    # Determine number of regions based on stabilize mode
    num_regions = 2 if stabilize_mode in (2, 3) else 1
    
    # Open video and get properties
    with VideoReader(input_video, first_frame=first_frame) as reader:
        props = reader.properties
        print(f"Video: {props.width}x{props.height} @ {props.fps}fps, {props.frame_count} frames")
        
        # Get first frame for ROI selection
        _, first_frame_img = next(iter(reader))
        reader.seek(first_frame)  # Reset to first frame
        
        # Get bounding boxes (interactive or predefined)
        if bboxes is None:
            bboxes = []
            for i in range(num_regions):
                print(f"Select ROI {i+1} of {num_regions}")
                roi = select_roi_interactive(first_frame_img, f"Select ROI {i+1}")
                bboxes.append(roi)
                print(f"  ROI {i+1}: {roi}")
        
        # Print bbox command for future use
        bbox_args = " ".join([f"--bbox {b[0]},{b[1]},{b[2]},{b[3]}" for b in bboxes])
        print(f"\nFor headless mode, use: --no-display {bbox_args}")
        
        # Create trackers for each region
        trackers = []
        for i, bbox in enumerate(bboxes):
            tracker = FeatureTracker(
                num_features=num_features,
                initial_roi=bbox,
                enforce_bbox=enforce_bbox,
            )
            trackers.append(tracker)
        
        # Set up output manager
        output_manager = OutputManager(input_video)
        for spec in output_specs:
            output_manager.add_output(spec)
        output_manager.initialize_all(props.to_dict())
        
        # Track through video
        print("\nTracking...")
        for frame_num, frame in reader:
            # Initialize or update each tracker
            all_points = []
            all_ids = []
            all_lengths = []
            all_rois = []
            
            for i, tracker in enumerate(trackers):
                if frame_num == first_frame:
                    points = tracker.initialize(frame)
                else:
                    points, ids, stats = tracker.update(frame)
                
                all_points.append(tracker.points)
                all_ids.append(tracker.point_ids)
                all_lengths.append(tracker.track_lengths)
                all_rois.append(tracker.get_roi())
            
            # Build tracking data for outputs
            tracking_data = {
                'points': all_points,
                'point_ids': all_ids,
                'track_lengths': all_lengths,
                'rois': all_rois,
            }
            
            # Process outputs
            output_manager.process_frame(frame_num, frame, tracking_data)
            
            # Progress
            if frame_num % 100 == 0:
                print(f"  Frame {frame_num}...")
        
        # Finalize outputs
        output_manager.finalize_all()
        output_paths = output_manager.get_output_paths()
        
        print(f"\nTracking complete. Outputs:")
        for p in output_paths:
            print(f"  {p}")
        
        return {
            'output_paths': output_paths,
            'bboxes': bboxes,
            'frame_range': reader.frame_range,
        }


def run_stabilization(
    input_video: str,
    track_files: list[str],
    output_token: str = "stab",
    write_output: bool = True,
    write_mask: bool = True,
    swap_xy: bool = False,
    x_flip: bool = False,
    y_flip: bool = False,
    ref_frame: int = -1,
    x_pad: float = 0,
    y_pad: float = 0,
    x_offset: float = 0,
    y_offset: float = 0,
    frame_start: int = -1,
    frame_end: int = -1,
) -> dict:
    """
    Run stabilization using tracking data.
    
    Equivalent to:
        python stabFromTrack_v07b.py \
            -td track01.crv:track02.crv -tk trk0102 \
            -wo -wm -sw -yf -rf 1000 -xp 200 -yp 600 -yo -400 \
            input.mp4
    
    Args:
        input_video: Path to input video
        track_files: List of .crv track files
        output_token: Token for output filename
        write_output: Write stabilized video
        write_mask: Write alpha mask video
        swap_xy: Swap X and Y coordinates
        x_flip: Flip X coordinate
        y_flip: Flip Y coordinate  
        ref_frame: Reference frame (-1 = first frame of track data)
        x_pad: X padding (pixels)
        y_pad: Y padding (pixels)
        x_offset: X offset (pixels)
        y_offset: Y offset (pixels)
        frame_start: First frame to process (-1 = auto)
        frame_end: Last frame to process (-1 = auto)
    
    Returns:
        Dictionary with output paths
    """
    from vdg.tracking.stabilizer import read_track_data, Stabilizer, StabilizeMode
    
    input_path = Path(input_video)
    
    # Open video to get properties
    cap = cv2.VideoCapture(input_video)
    movie_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    movie_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine orientation
    portrait = movie_height > movie_width
    print(f"Video: {movie_width}x{movie_height} ({'portrait' if portrait else 'landscape'})")
    
    # Calculate output dimensions with padding
    output_width = int(movie_width + x_pad)
    output_height = int(movie_height + y_pad)
    print(f"Output: {output_width}x{output_height}")
    
    # Read tracking data
    track_data_str = ":".join(track_files)
    trackers_dict = read_track_data(
        track_data_str,
        ref_frame=ref_frame,
        frame_offset=0,
        movie_width=movie_width,
        movie_height=movie_height,
        portrait=portrait,
        swap_xy=swap_xy,
        x_flip=x_flip,
        y_flip=y_flip,
    )
    
    # Determine frame range
    ff = trackers_dict['ff']
    lf = trackers_dict['lf']
    if frame_start == -1:
        frame_start = ff
    if frame_end == -1:
        frame_end = lf
    if ref_frame == -1:
        ref_frame = frame_start
    
    print(f"Frame range: {frame_start}-{frame_end} (ref: {ref_frame})")
    
    # Set up stabilizer
    stabilizer = Stabilizer(mode=StabilizeMode.TWO_POINT)
    stabilizer.set_dimensions(movie_width, movie_height)
    
    # Get reference markers
    ref_markers = trackers_dict['trackerData'][ref_frame]['markerData']
    stabilizer.set_reference(ref_markers)
    
    # Set up output writers
    output_path = input_path.with_name(f"{input_path.stem}_{output_token}.mp4")
    mask_path = input_path.with_name(f"{input_path.stem}_{output_token}_mask.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_writer = None
    mask_writer = None
    
    if write_output:
        output_writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (output_width, output_height)
        )
    if write_mask:
        mask_writer = cv2.VideoWriter(
            str(mask_path), fourcc, fps, (output_width, output_height)
        )
    
    # Process frames
    print("\nStabilizing...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start - 1)
    
    for frame_num in range(frame_start, frame_end + 1):
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_num}")
            break
        
        # Get markers for this frame
        if frame_num not in trackers_dict['trackerData']:
            continue
        
        markers = trackers_dict['trackerData'][frame_num]['markerData']
        
        # Compute transform
        matrix = stabilizer.compute_transform(
            markers,
            x_offset=x_offset + x_pad / 2,
            y_offset=y_offset + y_pad / 2,
        )
        
        # Apply transform
        if write_mask:
            # Add alpha channel
            frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            frame_rgba[:, :, 3] = 255
            stabilized = cv2.warpAffine(
                frame_rgba, matrix, (output_width, output_height)
            )
            
            if output_writer:
                output_writer.write(cv2.cvtColor(stabilized, cv2.COLOR_BGRA2BGR))
            
            # Extract and write mask
            mask = stabilized[:, :, 3]
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_writer.write(mask_bgr)
        else:
            stabilized = cv2.warpAffine(frame, matrix, (output_width, output_height))
            if output_writer:
                output_writer.write(stabilized)
        
        # Progress
        if frame_num % 100 == 0:
            pct = 100 * (frame_num - frame_start) / (frame_end - frame_start)
            print(f"  Frame {frame_num} ({pct:.1f}%)...")
    
    # Cleanup
    cap.release()
    if output_writer:
        output_writer.release()
    if mask_writer:
        mask_writer.release()
    
    outputs = {}
    if write_output:
        outputs['video'] = output_path
        print(f"\nOutput: {output_path}")
    if write_mask:
        outputs['mask'] = mask_path
        print(f"Mask: {mask_path}")
    
    return outputs


def main():
    """
    Main pipeline combining tracking and stabilization.
    
    This replicates the two-command workflow:
    1. Track points and export .crv files
    2. Stabilize using the .crv files
    """
    
    # === CONFIGURATION ===
    # Adjust these to match your files
    
    input_video_sm = "07062025165117145f_sm.mp4"  # Small/proxy for tracking
    input_video_full = "07062025165117145f.mp4"   # Full res for stabilization
    movie_id = "07062025165117145f"
    
    # Pre-defined bboxes (or set to None for interactive selection)
    # These would come from a previous interactive session
    bboxes = None  # Will prompt for selection
    # bboxes = [(100, 200, 150, 100), (100, 400, 150, 100)]  # Example predefined
    
    # === STEP 1: TRACKING ===
    print("=" * 60)
    print("STEP 1: Feature Tracking")
    print("=" * 60)
    
    tracking_result = run_tracking(
        input_video=input_video_sm,
        first_frame=30,
        num_features=30,
        stabilize_mode=3,  # VSTAB
        enforce_bbox=True,
        output_specs=["trackers=type=2:mode=center"],
        bboxes=bboxes,
    )
    
    # The tracker outputs .crv files named like:
    # 07062025165117145f_sm_track01.crv
    # 07062025165117145f_sm_track02.crv
    
    # === STEP 2: STABILIZATION ===
    print("\n" + "=" * 60)
    print("STEP 2: Stabilization")
    print("=" * 60)
    
    # Build track file paths (adjust naming as needed)
    track_files = [
        f"{movie_id}_track01.crv",
        f"{movie_id}_track02.crv",
    ]
    
    # Check if track files exist (they may have _sm in name from tracking step)
    for i, tf in enumerate(track_files):
        if not Path(tf).exists():
            alt = tf.replace(movie_id, f"{movie_id}_sm")
            if Path(alt).exists():
                track_files[i] = alt
                print(f"Using {alt}")
    
    stab_result = run_stabilization(
        input_video=input_video_full,
        track_files=track_files,
        output_token="trk0102",
        write_output=True,
        write_mask=True,
        swap_xy=True,   # -sw
        y_flip=True,    # -yf
        ref_frame=1000, # -rf 1000
        x_pad=200,      # -xp 200
        y_pad=600,      # -yp 600
        y_offset=-400,  # -yo -400
    )
    
    # === SUMMARY ===
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Track files: {track_files}")
    print(f"Stabilized video: {stab_result.get('video')}")
    print(f"Mask video: {stab_result.get('mask')}")


if __name__ == "__main__":
    main()
