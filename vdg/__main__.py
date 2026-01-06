"""
VDG Command Line Interface

Usage:
    vdg <command> [options]

Commands:
    track       Track features in video
    stabilize   Stabilize video using tracking data  
    frameavg    Average frames from video
    process     Run batch processing pipeline
    version     Show version information

Examples:
    vdg track input.mp4 -out previewtrack -out trackers=type=2
    vdg stabilize input.mp4 -td track01.crv:track02.crv -wo
    vdg frameavg input.mp4 -fs 100 -fe 500 -wa
    vdg process -c process_config.json
"""

import sys
import argparse

from vdg import __version__


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='vdg',
        description='Video Development and Grading Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        '-V', '--version',
        action='version',
        version=f'vdg {__version__}',
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Track command
    track_parser = subparsers.add_parser(
        'track',
        help='Track features in video',
    )
    track_parser.add_argument('input', help='Input video file')
    track_parser.add_argument(
        '-out', '--output',
        action='append',
        dest='outputs',
        metavar='SPEC',
        help='Output specification (can be used multiple times)',
    )
    track_parser.add_argument(
        '-n', '--num-features',
        type=int,
        default=50,
        help='Number of features to track (default: 50)',
    )
    track_parser.add_argument(
        '-fs', '--first-frame',
        type=int,
        default=1,
        help='First frame to process (default: 1)',
    )
    track_parser.add_argument(
        '-fe', '--frame-end',
        type=int,
        default=None,
        help='Last frame to process (default: end of video)',
    )
    track_parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable live display',
    )
    track_parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output',
    )
    
    # Stabilize command
    stab_parser = subparsers.add_parser(
        'stabilize',
        help='Stabilize video using tracking data',
    )
    stab_parser.add_argument('input', help='Input video file')
    stab_parser.add_argument(
        '-td', '--track-data',
        required=True,
        help='Track data file(s), colon-separated for multiple',
    )
    stab_parser.add_argument(
        '-wo', '--write-output',
        action='store_true',
        help='Write stabilized output',
    )
    stab_parser.add_argument(
        '-wm', '--write-mask',
        action='store_true',
        help='Write alpha mask',
    )
    stab_parser.add_argument(
        '-fs', '--frame-start',
        type=int,
        default=-1,
        help='First frame of stabilization',
    )
    stab_parser.add_argument(
        '-fe', '--frame-end',
        type=int,
        default=-1,
        help='Last frame of stabilization',
    )
    stab_parser.add_argument(
        '-jf', '--jitter-filter',
        action='store_true',
        help='Apply jitter removal filter',
    )
    stab_parser.add_argument(
        '-js', '--jitter-sigma',
        type=float,
        default=5.0,
        help='Jitter filter sigma (default: 5.0)',
    )
    
    # Frameavg command
    avg_parser = subparsers.add_parser(
        'frameavg',
        help='Average frames from video',
    )
    avg_parser.add_argument('input', help='Input video file')
    avg_parser.add_argument(
        '-fs', '--frame-start',
        type=int,
        default=-1,
        help='First frame to average',
    )
    avg_parser.add_argument(
        '-fe', '--frame-end',
        type=int,
        default=-1,
        help='Last frame to average',
    )
    avg_parser.add_argument(
        '-wa', '--write-alpha',
        action='store_true',
        help='Write alpha channel output',
    )
    avg_parser.add_argument(
        '-um', '--use-mask',
        action='store_true',
        help='Use mask movie file',
    )
    avg_parser.add_argument(
        '-tk', '--token',
        default='avg',
        help='Output filename token (default: avg)',
    )
    
    # Process command
    proc_parser = subparsers.add_parser(
        'process',
        help='Run batch processing pipeline',
    )
    proc_parser.add_argument(
        '-c', '--config',
        default='process_config.json',
        help='Configuration file (default: process_config.json)',
    )
    proc_parser.add_argument(
        '-m', '--movie',
        help='Process only specific movie ID',
    )
    proc_parser.add_argument(
        '-n', '--dry-run',
        action='store_true',
        help='Print commands without executing',
    )
    proc_parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create example configuration file',
    )
    proc_parser.add_argument(
        '--generate-config',
        action='store_true',
        help='Generate config from existing .crv files',
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Dispatch to appropriate command
    if args.command == 'track':
        return run_track(args)
    elif args.command == 'stabilize':
        return run_stabilize(args)
    elif args.command == 'frameavg':
        return run_frameavg(args)
    elif args.command == 'process':
        return run_process(args)
    else:
        parser.print_help()
        return 1


def run_track(args):
    """Run feature tracking command."""
    print(f"Tracking features in {args.input}")
    print("Note: Full tracking implementation pending migration")
    
    # This will be implemented to use the FeatureTracker
    # and OutputManager classes
    from vdg.tracking import FeatureTracker
    from vdg.outputs import OutputManager
    from vdg.core.video import VideoReader
    
    # Create tracker
    tracker = FeatureTracker(num_features=args.num_features)
    
    # Set up outputs
    output_manager = None
    if args.outputs:
        output_manager = OutputManager(args.input)
        for spec in args.outputs:
            output_manager.add_output(spec)
    
    # Process video
    with VideoReader(args.input, args.first_frame, args.frame_end) as reader:
        props = reader.properties.to_dict()
        
        if output_manager:
            output_manager.initialize_all(props)
        
        for frame_num, frame in reader:
            if frame_num == args.first_frame:
                points = tracker.initialize(frame)
            else:
                points, ids, stats = tracker.update(frame)
                if not args.quiet:
                    print(f"\rFrame {frame_num}: {stats.tracked} tracked, {stats.lost} lost", end='')
            
            if output_manager:
                tracking_data = {
                    'points': [tracker.points],
                    'point_ids': [tracker.point_ids],
                    'track_lengths': [tracker.track_lengths],
                    'rois': [tracker.get_roi()],
                }
                output_manager.process_frame(frame_num, frame, tracking_data)
        
        if output_manager:
            output_manager.finalize_all()
    
    print("\nDone!")
    return 0


def run_stabilize(args):
    """Run stabilization command."""
    print(f"Stabilizing {args.input}")
    print(f"Using track data: {args.track_data}")
    print("Note: Full stabilization implementation pending migration")
    return 0


def run_frameavg(args):
    """Run frame averaging command."""
    print(f"Averaging frames in {args.input}")
    print("Note: Full frame averaging implementation pending migration")
    return 0


def run_process(args):
    """Run batch processing command."""
    if args.create_config:
        from vdg.core.config import create_example_config
        create_example_config(args.config)
        return 0
    
    if args.generate_config:
        print("Generating config from .crv files...")
        print("Note: Full config generation pending migration")
        return 0
    
    print(f"Processing with config: {args.config}")
    print("Note: Full pipeline processing pending migration")
    return 0


if __name__ == '__main__':
    sys.exit(main())
