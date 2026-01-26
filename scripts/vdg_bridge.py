#!/usr/bin/env python3
"""
VDG Bridge - Blender addon for round-trip tracking with VDG.

This script is launched by VDG to enable tracking in Blender's Movie Clip Editor.
It loads the video, provides an export panel, and writes .crv track files.

Usage:
    blender --python vdg_bridge.py -- /path/to/session.json
"""

import bpy
import json
import sys
from pathlib import Path


# Global session data
_session = None
_session_path = None
_setup_done = False


def get_session_arg():
    """Get session.json path from command line arguments."""
    argv = sys.argv
    if "--" in argv:
        args = argv[argv.index("--") + 1:]
        if args:
            return args[0]
    return None


def load_session():
    """Load session data from JSON file."""
    global _session, _session_path

    session_path = get_session_arg()
    if not session_path:
        print("VDG Bridge: No session file provided")
        return None

    _session_path = Path(session_path)
    if not _session_path.exists():
        print(f"VDG Bridge: Session file not found: {_session_path}")
        return None

    with open(_session_path) as f:
        _session = json.load(f)

    print(f"VDG Bridge: Loaded session from {_session_path}")
    print(f"VDG Bridge: Video path: {_session.get('video_path')}")
    return _session


def setup_tracking_layout():
    """Load the VDG tracking layout template."""
    template_path = Path(__file__).parent / "vdg_tracking_layout.blend"

    if not template_path.exists():
        print(f"VDG Bridge: Template not found at {template_path}")
        return None, None

    print(f"VDG Bridge: Loading layout template from {template_path}")
    bpy.ops.wm.open_mainfile(filepath=str(template_path))

    # After loading, get fresh window/screen references from window_manager
    window = bpy.context.window_manager.windows[0]
    screen = window.screen

    # Find the clip editor area with CLIP view (main tracking view)
    clip_area = None
    for area in screen.areas:
        if area.type == 'CLIP_EDITOR':
            for space in area.spaces:
                if space.type == 'CLIP_EDITOR' and space.view == 'CLIP':
                    clip_area = area
                    break
        if clip_area:
            break

    # If no CLIP view found, use the first clip editor
    if not clip_area:
        for area in screen.areas:
            if area.type == 'CLIP_EDITOR':
                clip_area = area
                break

    if clip_area:
        return clip_area, clip_area.spaces[0]
    return None, None


def setup_clip_editor_space():
    """Find or create a clip editor space and return it."""
    # First try to set up our custom tracking layout
    try:
        return setup_tracking_layout()
    except Exception as e:
        print(f"VDG Bridge: Could not set up tracking layout: {e}")

    # Fallback: just find or create a clip editor
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'CLIP_EDITOR':
                return area, area.spaces[0]

    # No clip editor found, convert the largest area
    largest_area = None
    largest_size = 0

    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            size = area.width * area.height
            if size > largest_size:
                largest_size = size
                largest_area = area

    if largest_area:
        largest_area.type = 'CLIP_EDITOR'
        return largest_area, largest_area.spaces[0]

    return None, None


def setup_movie_clip():
    """Load video into Movie Clip Editor."""
    global _setup_done

    if _setup_done:
        return True

    if not _session:
        print("VDG Bridge: No session loaded")
        return False

    video_path = _session.get("video_path")
    if not video_path or not Path(video_path).exists():
        print(f"VDG Bridge: Video not found: {video_path}")
        return False

    print(f"VDG Bridge: Loading video: {video_path}")

    # Check if we should open an existing .blend file
    blend_file = _session.get("blend_file")
    if blend_file and Path(blend_file).exists():
        print(f"VDG Bridge: Opening existing project: {blend_file}")
        bpy.ops.wm.open_mainfile(filepath=blend_file)
        _setup_done = True
        return True

    # Set up clip editor area
    area, space = setup_clip_editor_space()
    if not area:
        print("VDG Bridge: Could not find/create clip editor area")
        return False

    # Load the movie clip
    clip_name = Path(video_path).name
    clip = bpy.data.movieclips.get(clip_name)

    if not clip:
        # Need to load the clip with proper context
        # bpy.ops.clip.open uses directory + files, not filepath
        video_dir = str(Path(video_path).parent)
        video_file = Path(video_path).name

        # Get fresh window reference for context override
        window = bpy.context.window_manager.windows[0]

        with bpy.context.temp_override(window=window, area=area, space_data=space):
            bpy.ops.clip.open(directory=video_dir, files=[{"name": video_file}])

        clip = bpy.data.movieclips.get(clip_name)
        if not clip and bpy.data.movieclips:
            clip = bpy.data.movieclips[-1]  # Get most recently added

    if not clip:
        print("VDG Bridge: Failed to load movie clip")
        return False

    # Set color space to sRGB texture
    try:
        clip.colorspace_settings.name = "srgb_texture"
        print(f"VDG Bridge: Set color space to srgb_texture")
    except Exception as e:
        print(f"VDG Bridge: Could not set color space: {e}")

    # Assign clip to ALL clip editor spaces in the layout
    # Use window_manager to get valid screen reference
    clip_count = 0
    window = bpy.context.window_manager.windows[0]
    screen = window.screen
    for area in screen.areas:
        if area.type == 'CLIP_EDITOR':
            for sp in area.spaces:
                if sp.type == 'CLIP_EDITOR':
                    sp.clip = clip
                    clip_count += 1
    print(f"VDG Bridge: Assigned clip to {clip_count} clip editor(s)")

    # Also set as active clip for the context
    bpy.context.scene.active_clip = clip

    # Set frame range
    first_frame = _session.get("first_frame", 1)
    last_frame = _session.get("last_frame", -1)

    if last_frame == -1:
        last_frame = clip.frame_duration

    bpy.context.scene.frame_start = first_frame
    bpy.context.scene.frame_end = last_frame
    bpy.context.scene.frame_current = first_frame

    # Set clip frame range too
    clip.frame_start = first_frame

    print(f"VDG Bridge: Loaded '{clip_name}', frames {first_frame}-{last_frame}")
    print(f"VDG Bridge: Clip size: {clip.size[0]}x{clip.size[1]}")

    _setup_done = True
    return True


def export_tracks():
    """Export all tracks to .crv files."""
    if not _session or not _session_path:
        return []

    # Find the clip - check space first, then all clips
    clip = None
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'CLIP_EDITOR':
                clip = area.spaces[0].clip
                break
        if clip:
            break

    if not clip and bpy.data.movieclips:
        clip = bpy.data.movieclips[0]

    if not clip:
        print("VDG Bridge: No movie clip found")
        return []

    tracking = clip.tracking
    tracks = tracking.tracks

    if not tracks:
        print("VDG Bridge: No tracks found")
        return []

    session_dir = _session_path.parent
    output_files = []

    print(f"VDG Bridge: Exporting {len(tracks)} track(s)...")

    for i, track in enumerate(tracks):
        if not track.markers:
            continue

        # Generate output filename
        track_name = track.name.replace(" ", "_").replace("/", "_")
        output_path = session_dir / f"track{i+1:02d}_{track_name}.crv"

        # Collect marker data
        track_data = []
        for marker in track.markers:
            if marker.mute:
                continue

            frame = marker.frame
            # Blender uses normalized coords (0-1) with origin at bottom-left
            # VDG uses 0-1 normalized with origin at top-left
            x = marker.co[0]
            y = 1.0 - marker.co[1]  # Flip Y axis

            track_data.append((frame, x, y))

        if not track_data:
            continue

        # Sort by frame and write
        track_data.sort(key=lambda t: t[0])

        with open(output_path, 'w') as f:
            for frame, x, y in track_data:
                f.write(f"{frame} [[ {x:.6f}, {y:.6f}]]\n")

        output_files.append(str(output_path))
        print(f"VDG Bridge: Exported {len(track_data)} frames to {output_path.name}")

    return output_files


def save_and_quit(output_files):
    """Save .blend file and quit Blender."""
    if not _session_path:
        return

    session_dir = _session_path.parent
    blend_path = session_dir / "project.blend"

    # Save .blend file
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"VDG Bridge: Saved {blend_path}")

    # Write result to session
    result = {
        "status": "complete",
        "blend_file": str(blend_path),
        "track_files": output_files,
    }

    result_path = session_dir / "result.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"VDG Bridge: Wrote result to {result_path}")

    # Quit Blender
    bpy.ops.wm.quit_blender()


# -----------------------------------------------------------------------------
# Blender UI Panel
# -----------------------------------------------------------------------------

class VDG_PT_Panel(bpy.types.Panel):
    """VDG Export Panel in the Movie Clip Editor sidebar."""
    bl_label = "VDG Export"
    bl_idname = "VDG_PT_panel"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'VDG'

    def draw(self, context):
        layout = self.layout

        clip = context.space_data.clip if context.space_data else None
        if not clip:
            layout.label(text="No clip loaded", icon='ERROR')
            return

        tracking = clip.tracking
        num_tracks = len(tracking.tracks)

        layout.label(text=f"Clip: {clip.name}")
        layout.label(text=f"Size: {clip.size[0]}x{clip.size[1]}")
        layout.label(text=f"Tracks: {num_tracks}")

        layout.separator()

        if num_tracks == 0:
            box = layout.box()
            box.label(text="To add markers:", icon='INFO')
            box.label(text="  Ctrl+Click on feature")
            box.label(text="To track:")
            box.label(text="  Ctrl+T (forward)")
            box.label(text="  Shift+Ctrl+T (backward)")
        else:
            # Show track info
            box = layout.box()
            box.label(text="Tracks:")
            for track in tracking.tracks:
                marker_count = sum(1 for m in track.markers if not m.mute)
                row = box.row()
                row.label(text=f"  {track.name}: {marker_count} frames")

        layout.separator()

        row = layout.row()
        row.scale_y = 2.0
        op = row.operator("vdg.export_tracks", text="Export to VDG", icon='EXPORT')

        if num_tracks == 0:
            row.enabled = False


class VDG_OT_ExportTracks(bpy.types.Operator):
    """Export tracks and return to VDG."""
    bl_idname = "vdg.export_tracks"
    bl_label = "Export to VDG"
    bl_description = "Export tracks as .crv files, save project, and return to VDG"

    def execute(self, context):
        clip = context.space_data.clip if context.space_data else None
        if not clip:
            self.report({'ERROR'}, "No movie clip loaded")
            return {'CANCELLED'}

        tracking = clip.tracking
        if not tracking.tracks:
            self.report({'ERROR'}, "No tracks to export. Add markers with Ctrl+Click, track with Ctrl+T")
            return {'CANCELLED'}

        output_files = export_tracks()

        if not output_files:
            self.report({'ERROR'}, "No track data exported")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Exported {len(output_files)} track(s)")

        # Save and quit
        save_and_quit(output_files)

        return {'FINISHED'}


# -----------------------------------------------------------------------------
# App Handler for delayed setup
# -----------------------------------------------------------------------------

def delayed_setup():
    """Run setup after Blender is fully loaded."""
    global _setup_done

    if _setup_done:
        return None  # Stop timer

    # Check if we have a window and screen
    if not bpy.context.window_manager.windows:
        return 0.1  # Try again in 0.1 seconds

    print("VDG Bridge: Running delayed setup...")

    if setup_movie_clip():
        print("VDG Bridge: Setup complete - ready for tracking")
    else:
        print("VDG Bridge: Setup failed")

    return None  # Stop timer


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

classes = [
    VDG_PT_Panel,
    VDG_OT_ExportTracks,
]


def register():
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            # Already registered
            pass


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    """Main entry point when run as script."""
    print("\n" + "=" * 50)
    print("VDG Bridge: Starting...")
    print("=" * 50)

    # Load session data
    session = load_session()
    if not session:
        print("VDG Bridge: No session - running in standalone mode")
        print("VDG Bridge: To use, launch via VDG's 'Edit in Blender' button")

    # Register UI
    register()

    # Schedule setup to run after Blender is fully loaded
    bpy.app.timers.register(delayed_setup, first_interval=0.5)

    print("VDG Bridge: Initialization complete")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
