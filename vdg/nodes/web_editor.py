"""
VDG Node Editor - Web Interface
================================

A web-based node editor for VDG using vanilla JavaScript.
No external CDN dependencies.

To run:
    pip install fastapi uvicorn
    python -m vdg.nodes.web_editor

Then open http://localhost:8000 in your browser.
"""

import json
from typing import Any
from dataclasses import dataclass, field

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    import uvicorn
except ImportError:
    print("Install dependencies: pip install fastapi uvicorn")
    raise

# Global abort flag for cancelling running jobs
_abort_flag = False

# File cache for autocomplete suggestions
_file_cache = {
    'video': [],
    'image': [],
    'track': [],
    'all': [],
}
_cache_directory = None

# File extension categories
_FILE_EXTENSIONS = {
    'video': {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.mxf', '.m4v'},
    'image': {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.exr', '.dpx', '.hdr'},
    'track': {'.crv', '.json'},
}


def refresh_file_cache(directory: str = None) -> dict:
    """Scan directory and update the file cache."""
    global _file_cache, _cache_directory
    from pathlib import Path
    import os

    if directory:
        _cache_directory = directory

    if not _cache_directory:
        _cache_directory = os.getcwd()

    target = Path(_cache_directory).expanduser().resolve()
    if not target.exists() or not target.is_dir():
        return _file_cache

    # Reset cache
    _file_cache = {k: [] for k in _file_cache}

    try:
        for item in target.iterdir():
            if item.is_file():
                ext = item.suffix.lower()
                name = item.name
                _file_cache['all'].append(name)

                for category, extensions in _FILE_EXTENSIONS.items():
                    if ext in extensions:
                        _file_cache[category].append(name)
                        break

        # Sort all lists
        for k in _file_cache:
            _file_cache[k].sort(key=str.lower)
    except PermissionError:
        pass

    return _file_cache


def _ensure_video_extension(path: str) -> str:
    """Ensure preview path has a video extension, not an image extension.

    FFmpeg's image2 muxer can't write multiple frames to the same file,
    so we need to ensure the path has a video extension like .mp4.
    """
    import os
    if not path:
        return path
    base, ext = os.path.splitext(path)
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
    if ext.lower() not in video_extensions:
        return base + '.mp4'
    return path


# =============================================================================
# NODE DEFINITIONS
# =============================================================================

@dataclass
class NodePort:
    name: str
    type: str
    optional: bool = False


@dataclass 
class NodeParam:
    name: str
    type: str
    default: Any = None
    min: float = None
    max: float = None
    choices: list = None


@dataclass
class NodeDefinition:
    id: str
    title: str
    category: str
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    params: list = field(default_factory=list)
    color: str = "#666"


NODE_DEFINITIONS = [
    NodeDefinition(
        id="video_input", title="Video Input", category="Input",
        inputs=[],
        outputs=[NodePort("video_out", "video"), NodePort("props", "props")],
        params=[
            NodeParam("filepath", "file", ""),
            NodeParam("first_frame", "int", 1, min=1),
            NodeParam("last_frame", "int", -1),
            NodeParam("use_hardware", "bool", False),
            NodeParam("hlg_convert", "bool", False),
        ],
        color="#4CAF50",
    ),
    NodeDefinition(
        id="image_input", title="Image Input", category="Input",
        inputs=[],
        outputs=[NodePort("image_out", "image")],
        params=[
            NodeParam("filepath", "file", ""),
        ],
        color="#4CAF50",
    ),
    NodeDefinition(
        id="track_input", title="Track File Input", category="Input",
        inputs=[],
        outputs=[NodePort("track_data", "track_data")],
        params=[NodeParam("filepath", "file", "")],
        color="#4CAF50",
    ),
    NodeDefinition(
        id="roi", title="ROI", category="Input",
        inputs=[NodePort("video_in", "video", optional=True)],
        outputs=[NodePort("roi", "roi")],
        params=[
            NodeParam("pick_roi", "button", "Pick ROI"),
            NodeParam("x", "int", 0), NodeParam("y", "int", 0),
            NodeParam("width", "int", 100), NodeParam("height", "int", 100),
        ],
        color="#4CAF50",
    ),
    NodeDefinition(
        id="feature_tracker", title="Feature Tracker", category="Tracking",
        inputs=[NodePort("video_in", "video"), NodePort("roi", "roi", optional=True)],
        outputs=[NodePort("points", "points"), NodePort("track_data", "track_data")],
        params=[
            NodeParam("num_features", "int", 30, min=1, max=200),
            NodeParam("min_distance", "int", 30, min=5, max=100),
            NodeParam("enforce_bbox", "bool", True),
            NodeParam("win_size", "int", 21, min=11, max=101),
            NodeParam("pyramid_levels", "int", 3, min=0, max=6),
            NodeParam("preview_path", "string", ""),
        ],
        color="#2196F3",
    ),
    NodeDefinition(
        id="feature_tracker_2p", title="Feature Tracker 2P", category="Tracking",
        inputs=[
            NodePort("video_in", "video"),
            NodePort("roi1", "roi", optional=True),
            NodePort("roi2", "roi", optional=True),
        ],
        outputs=[
            NodePort("track_data1", "track_data"),
            NodePort("track_data2", "track_data"),
        ],
        params=[
            NodeParam("num_features", "int", 30, min=1, max=200),
            NodeParam("min_distance", "int", 30, min=5, max=100),
            NodeParam("enforce_bbox", "bool", True),
            NodeParam("win_size", "int", 21, min=11, max=101),
            NodeParam("pyramid_levels", "int", 3, min=0, max=6),
            NodeParam("preview_path", "string", ""),
        ],
        color="#2196F3",
    ),
    NodeDefinition(
        id="stabilizer", title="Stabilizer", category="Tracking",
        inputs=[
            NodePort("track1", "track_data"),
            NodePort("track2", "track_data", optional=True),
            NodePort("props", "props"),
        ],
        outputs=[NodePort("transforms", "transforms")],
        params=[
            NodeParam("mode", "choice", "two_point", choices=["single", "two_point", "vstab", "perspective"]),
            NodeParam("ref_frame", "int", -1),
            NodeParam("swap_xy", "bool", False),
            NodeParam("x_flip", "bool", False),
            NodeParam("y_flip", "bool", False),
        ],
        color="#2196F3",
    ),
    NodeDefinition(
        id="apply_transform", title="Apply Transform", category="Processing",
        inputs=[NodePort("video_in", "video"), NodePort("transforms", "transforms")],
        outputs=[NodePort("video_out", "video"), NodePort("mask", "video")],
        params=[
            NodeParam("x_pad", "int", 0), NodeParam("y_pad", "int", 0),
            NodeParam("x_offset", "int", 0), NodeParam("y_offset", "int", 0),
        ],
        color="#FF9800",
    ),
    NodeDefinition(
        id="frame_average", title="Frame Average", category="Processing",
        inputs=[NodePort("video_in", "video"), NodePort("mask_in", "video", optional=True)],
        outputs=[NodePort("image_out", "image"), NodePort("alpha_out", "image")],
        params=[
            NodeParam("comp_mode", "choice", "on_black", choices=["on_black", "on_white", "unpremult"]),
            NodeParam("brightness", "float", 1.0, min=0.0, max=4.0),
        ],
        color="#FF9800",
    ),
    NodeDefinition(
        id="clahe", title="CLAHE", category="Processing",
        inputs=[NodePort("image_in", "image")],
        outputs=[NodePort("image_out", "image")],
        params=[
            NodeParam("clip_limit", "float", 40.0, min=1.0, max=100.0),
            NodeParam("grid_size", "int", 8, min=2, max=32),
        ],
        color="#FF9800",
    ),
    NodeDefinition(
        id="gamma", title="Gamma", category="Processing",
        inputs=[
            NodePort("video_in", "video", optional=True),
            NodePort("image_in", "image", optional=True),
            NodePort("mask_in", "video", optional=True),
        ],
        outputs=[
            NodePort("video_out", "video"),
            NodePort("image_out", "image"),
            NodePort("mask_out", "video"),
        ],
        params=[
            NodeParam("mode", "choice", "to_linear", choices=["to_linear", "to_srgb"]),
            NodeParam("gamma", "float", 2.2, min=1.0, max=3.0),
        ],
        color="#FF9800",
    ),
    NodeDefinition(
        id="post_process", title="Post Process", category="Processing",
        inputs=[
            NodePort("image_in", "image"),
            NodePort("alpha_in", "image"),
        ],
        outputs=[
            NodePort("image_out", "image"),
        ],
        params=[
            NodeParam("operation", "choice", "comp_on_white",
                      choices=["comp_on_white", "comp_on_black", "refine_alpha",
                               "divide_alpha", "unpremult_on_white"]),
            NodeParam("trim", "bool", False),
            NodeParam("gamma", "float", 2.2, min=1.0, max=4.0),
            NodeParam("contrast", "float", 60.0, min=1.0, max=200.0),
            NodeParam("threshold", "float", 0.0015, min=0.0, max=0.1),
            NodeParam("blur_size", "float", 5.0, min=0.0, max=50.0),
            NodeParam("power", "float", 8.0, min=0.1, max=20.0),
        ],
        color="#FF9800",
    ),
    NodeDefinition(
        id="video_output", title="Video Output", category="Output",
        inputs=[NodePort("video_in", "video"), NodePort("props", "props")],
        outputs=[],
        params=[
            NodeParam("filepath", "file", "output.mp4"),
            NodeParam("use_hardware", "bool", True),
            NodeParam("bitrate", "string", "20M"),
        ],
        color="#9C27B0",
    ),
    NodeDefinition(
        id="image_output", title="Image Output", category="Output",
        inputs=[NodePort("image_in", "image")],
        outputs=[],
        params=[
            NodeParam("filepath", "file", "output.png"),
            NodeParam("bit_depth", "choice", "16", choices=["8", "16"]),
        ],
        color="#9C27B0",
    ),
    NodeDefinition(
        id="track_output", title="Track Output", category="Output",
        inputs=[NodePort("track_data", "track_data"), NodePort("props", "props")],
        outputs=[],
        params=[
            NodeParam("filepath", "file", "track.crv"),
            NodeParam("center_mode", "choice", "centroid", choices=["centroid", "roi_center", "median"]),
        ],
        color="#9C27B0",
    ),
    NodeDefinition(
        id="gaussian_filter", title="Gaussian Filter", category="Utility",
        inputs=[NodePort("track_data", "track_data")],
        outputs=[NodePort("track_data", "track_data")],
        params=[NodeParam("sigma", "float", 5.0, min=0.1, max=50.0)],
        color="#607D8B",
    ),
    NodeDefinition(
        id="rgb_multiply", title="RGB Multiply", category="Processing",
        inputs=[
            NodePort("video_in", "video", optional=True),
            NodePort("image_in", "image", optional=True),
        ],
        outputs=[
            NodePort("video_out", "video"),
            NodePort("image_out", "image"),
        ],
        params=[
            NodeParam("red", "float", 1.0, min=0.0, max=4.0),
            NodeParam("green", "float", 1.0, min=0.0, max=4.0),
            NodeParam("blue", "float", 1.0, min=0.0, max=4.0),
        ],
        color="#FF9800",
    ),
]


def get_nodes_json():
    result = []
    for node in NODE_DEFINITIONS:
        result.append({
            'id': node.id, 'title': node.title, 'category': node.category, 'color': node.color,
            'inputs': [{'name': p.name, 'type': p.type, 'optional': p.optional} for p in node.inputs],
            'outputs': [{'name': p.name, 'type': p.type, 'optional': p.optional} for p in node.outputs],
            'params': [{'name': p.name, 'type': p.type, 'default': p.default, 
                       'min': p.min, 'max': p.max, 'choices': p.choices} for p in node.params],
        })
    return result


app = FastAPI(title="VDG Node Editor")

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VDG Node Editor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: system-ui, -apple-system, sans-serif; background: #1a1a2e; color: #fff; overflow: hidden; }
        .container { display: flex; height: calc(100vh - 40px); }
        .sidebar { width: 200px; background: #16213e; padding: 12px; overflow-y: auto; border-right: 1px solid #333; }
        .sidebar h2 { font-size: 14px; margin-bottom: 12px; color: #4fc3f7; }
        .sidebar h3 { font-size: 10px; margin: 12px 0 6px; text-transform: uppercase; color: #666; }
        .node-btn { display: block; width: 100%; padding: 6px 10px; margin: 3px 0; background: #252545; color: #fff; border: 1px solid #333; border-radius: 4px; cursor: grab; text-align: left; font-size: 12px; }
        .node-btn:hover { background: #2d2d5a; border-color: #4fc3f7; }
        .canvas-area { flex: 1; position: relative; overflow: hidden; }
        #canvas { position: absolute; left: -5000px; top: -5000px; width: 15000px; height: 15000px; background-image: radial-gradient(#333 1px, transparent 1px); background-size: 20px 20px; z-index: 1; transform-origin: 0 0; }
        #svg-layer { position: absolute; left: 0; top: 0; width: 100%; height: 100%; z-index: 2; pointer-events: none; }
        #svg-layer path { pointer-events: stroke; }
        .toolbar { position: absolute; top: 10px; right: 10px; display: flex; gap: 6px; z-index: 100; }
        .toolbar button { padding: 8px 14px; background: #4CAF50; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }
        .toolbar button:hover { filter: brightness(1.1); }
        .toolbar button.sec { background: #555; }
        .toolbar button.abort { background: #c62828; }
        .toolbar button.small { padding: 8px 10px; min-width: 28px; }
        .zoom-controls { display: flex; align-items: center; gap: 4px; margin-left: 8px; padding-left: 8px; border-left: 1px solid #444; }
        #zoom-level { font-size: 11px; min-width: 40px; text-align: center; color: #aaa; }
        .node { position: absolute; min-width: 140px; background: #252545; border: 2px solid #444; border-radius: 6px; cursor: move; user-select: none; font-size: 11px; z-index: 3; }
        .node.selected { border-color: #4fc3f7; box-shadow: 0 0 15px rgba(79,195,247,0.3); }
        .node-hdr { padding: 6px 10px; border-radius: 4px 4px 0 0; font-weight: 600; font-size: 11px; }
        .node-body { padding: 6px 10px; }
        .port { display: flex; align-items: center; margin: 4px 0; color: #aaa; position: relative; }
        .port.in { padding-left: 12px; }
        .port.out { padding-right: 12px; justify-content: flex-end; }
        .dot { width: 8px; height: 8px; background: #666; border: 2px solid #888; border-radius: 50%; position: absolute; cursor: crosshair; pointer-events: auto; }
        .dot:hover { background: #4fc3f7; }
        .port.in .dot { left: -5px; }
        .port.out .dot { right: -5px; }
        .port.opt { opacity: 0.5; }
        .props { width: 220px; background: #16213e; padding: 12px; border-left: 1px solid #333; overflow-y: auto; }
        .props h3 { font-size: 13px; margin-bottom: 10px; color: #4fc3f7; }
        .props-empty { color: #666; font-size: 12px; text-align: center; margin-top: 40px; }
        .prop { margin: 8px 0; }
        .prop label { display: block; font-size: 10px; color: #888; margin-bottom: 3px; text-transform: uppercase; }
        .prop input, .prop select { width: 100%; padding: 6px; background: #252545; border: 1px solid #333; border-radius: 3px; color: #fff; font-size: 12px; }
        .prop input:focus { border-color: #4fc3f7; outline: none; }
        .file-input-wrap { display: flex; gap: 4px; }
        .file-input-wrap input { flex: 1; border-radius: 3px 0 0 3px; }
        .file-browse-btn { padding: 6px 8px; background: #444; border: 1px solid #333; border-left: none; border-radius: 0 3px 3px 0; color: #aaa; cursor: pointer; font-size: 10px; }
        .file-browse-btn:hover { background: #555; color: #fff; }
        .del-btn { margin-top: 16px; width: 100%; padding: 8px; background: #c0392b; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 11px; }
        .status { position: absolute; bottom: 10px; left: 220px; background: rgba(22,33,62,0.9); padding: 6px 12px; border-radius: 4px; font-size: 11px; color: #888; }
        .conn { stroke: #4fc3f7; stroke-width: 2; fill: none; }
        .conn.conn-hover { stroke: #ff5722; stroke-width: 3; }
        .conn-temp { stroke: #4fc3f7; stroke-width: 2; fill: none; stroke-dasharray: 5,5; }
        .prop button.pick-btn { width: 100%; padding: 8px; background: #4fc3f7; color: #000; border: none; border-radius: 4px; cursor: pointer; font-size: 11px; font-weight: 600; }
        .prop button.pick-btn:hover { filter: brightness(1.1); }
        .modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000; justify-content: center; align-items: center; }
        .modal-overlay.active { display: flex; }
        .modal { background: #1a1a2e; border: 2px solid #4fc3f7; border-radius: 8px; padding: 16px; max-width: 90vw; max-height: 90vh; }
        .modal h3 { margin-bottom: 12px; color: #4fc3f7; font-size: 14px; }
        .modal-canvas-wrap { position: relative; cursor: crosshair; display: inline-block; }
        .modal-canvas-wrap img { display: block; max-width: 80vw; max-height: 70vh; }
        .modal-canvas-wrap canvas { position: absolute; top: 0; left: 0; pointer-events: auto; }
        .modal-btns { margin-top: 12px; display: flex; gap: 8px; justify-content: flex-end; }
        .modal-btns button { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }
        .modal-btns .apply { background: #4CAF50; color: #fff; }
        .modal-btns .cancel { background: #555; color: #fff; }
        .project-bar { display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: #16213e; border-bottom: 1px solid #333; height: 40px; }
        .project-bar label { font-size: 11px; color: #888; white-space: nowrap; }
        .project-bar input { flex: 1; background: #0f0f1a; border: 1px solid #333; color: #fff; padding: 6px 10px; border-radius: 4px; font-family: monospace; font-size: 12px; }
        .project-bar input:focus { border-color: #4fc3f7; outline: none; }
        .project-bar .project-status { font-size: 10px; padding: 3px 8px; border-radius: 3px; }
        .project-bar .project-status.valid { background: #2e7d32; color: #fff; }
        .project-bar .project-status.invalid { background: #c62828; color: #fff; }
        .project-bar .project-status.empty { background: #555; color: #aaa; }
    </style>
</head>
<body>
    <div class="project-bar">
        <label>Project Directory:</label>
        <input type="text" id="project-dir" placeholder="/path/to/project" onchange="validateProjectDir()">
        <span class="project-status empty" id="project-status">Not Set</span>
    </div>
    <div class="container">
        <div class="sidebar" id="sidebar"></div>
        <div class="canvas-area" id="canvas-area">
            <svg id="svg-layer"></svg>
            <div id="canvas"></div>
            <div class="toolbar">
                <button id="run-btn" onclick="execute()">▶ Run</button>
                <button id="abort-btn" class="abort" onclick="abort()" style="display:none;">⏹ Abort</button>
                <button class="sec" onclick="save()">Save</button>
                <button class="sec" onclick="load()">Load</button>
                <button class="sec" onclick="clear()">Clear</button>
                <span class="zoom-controls">
                    <button class="sec small" onclick="zoomOut()">−</button>
                    <span id="zoom-level">100%</span>
                    <button class="sec small" onclick="zoomIn()">+</button>
                </span>
            </div>
            <div class="status" id="status">Drag nodes to canvas</div>
        </div>
        <div class="props" id="props"><div class="props-empty">Select a node</div></div>
    </div>
    <div class="modal-overlay" id="roi-modal">
        <div class="modal">
            <h3>Draw ROI - Click and drag to select region</h3>
            <div class="modal-canvas-wrap" id="roi-canvas-wrap">
                <img id="roi-img">
                <canvas id="roi-canvas"></canvas>
            </div>
            <div class="modal-btns">
                <button class="cancel" onclick="closeROIModal()">Cancel</button>
                <button class="apply" onclick="applyROI()">Apply</button>
            </div>
        </div>
    </div>
<script>
const DEFS = ''' + json.dumps(get_nodes_json()) + ''';
let nodes = [], conns = [], sel = null, dragN = null, dragC = null, nid = 0;
let off = {x: 0, y: 0}, pan = {x: 0, y: 0}, panning = false, panStart = {};
let zoom = 1;
const ZOOM_MIN = 0.25, ZOOM_MAX = 2, ZOOM_STEP = 0.1;
const CANVAS_OFFSET = 5000;  // Canvas/SVG positioned at -5000,-5000
let projectDir = '';

async function validateProjectDir() {
    const input = document.getElementById('project-dir');
    const statusEl = document.getElementById('project-status');
    const path = input.value.trim();

    if (!path) {
        projectDir = '';
        statusEl.className = 'project-status empty';
        statusEl.textContent = 'Not Set';
        return;
    }

    try {
        const r = await fetch('/api/validate-path?path=' + encodeURIComponent(path));
        const data = await r.json();

        if (data.valid) {
            projectDir = data.path;
            input.value = data.path;  // Use normalized path
            statusEl.className = 'project-status valid';
            statusEl.textContent = 'Valid';
        } else {
            projectDir = '';
            statusEl.className = 'project-status invalid';
            statusEl.textContent = data.error || 'Invalid';
        }
    } catch (err) {
        projectDir = '';
        statusEl.className = 'project-status invalid';
        statusEl.textContent = 'Error';
    }
}

function init() {
    const sb = document.getElementById('sidebar');
    let html = '<h2>VDG Nodes</h2>';
    const cats = {};
    DEFS.forEach(d => { (cats[d.category] = cats[d.category] || []).push(d); });
    for (const [c, ds] of Object.entries(cats)) {
        html += '<h3>' + c + '</h3>';
        ds.forEach(d => { html += '<button class="node-btn" draggable="true" data-t="' + d.id + '">' + d.title + '</button>'; });
    }
    sb.innerHTML = html;
    sb.querySelectorAll('.node-btn').forEach(b => {
        b.ondragstart = e => e.dataTransfer.setData('t', b.dataset.t);
    });
    
    const area = document.getElementById('canvas-area');
    area.ondragover = e => e.preventDefault();
    area.ondrop = e => {
        e.preventDefault();
        const t = e.dataTransfer.getData('t');
        if (t) {
            const rect = area.getBoundingClientRect();
            const x = (e.clientX - rect.left - pan.x) / zoom;
            const y = (e.clientY - rect.top - pan.y) / zoom;
            addNode(t, x, y);
        }
    };
    area.onmousedown = e => {
        if (e.target === area || e.target.id === 'canvas') {
            panning = true; panStart = {x: e.clientX - pan.x, y: e.clientY - pan.y}; desel();
        }
    };
    area.onmousemove = e => {
        if (panning) { pan.x = e.clientX - panStart.x; pan.y = e.clientY - panStart.y; updTrans(); drawConns(); }
        if (dragC) updTemp(e);
    };
    area.onmouseup = () => { panning = false; if (dragC) cancelConn(); };

    // Zoom with mouse wheel
    area.onwheel = e => {
        e.preventDefault();
        const rect = area.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        // Mouse position in canvas coordinates before zoom
        const canvasX = (mx - pan.x) / zoom;
        const canvasY = (my - pan.y) / zoom;

        // Apply zoom
        const oldZoom = zoom;
        if (e.deltaY < 0) zoom = Math.min(ZOOM_MAX, zoom + ZOOM_STEP);
        else zoom = Math.max(ZOOM_MIN, zoom - ZOOM_STEP);

        // Adjust pan to keep mouse position fixed
        pan.x = mx - canvasX * zoom;
        pan.y = my - canvasY * zoom;

        updTrans();
        drawConns();
        status('Zoom: ' + Math.round(zoom * 100) + '%');
    };

    // Apply initial transform
    updTrans();
}

function updTrans() {
    // Canvas positioned at -5000,-5000, needs offset compensation
    const tx = pan.x + CANVAS_OFFSET, ty = pan.y + CANVAS_OFFSET;
    document.getElementById('canvas').style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + zoom + ')';
    // SVG is not transformed - paths are drawn in screen coordinates
    document.getElementById('zoom-level').textContent = Math.round(zoom * 100) + '%';
}

function zoomIn() {
    const area = document.getElementById('canvas-area');
    const rect = area.getBoundingClientRect();
    const cx = rect.width / 2, cy = rect.height / 2;
    const canvasX = (cx - pan.x) / zoom, canvasY = (cy - pan.y) / zoom;
    zoom = Math.min(ZOOM_MAX, zoom + ZOOM_STEP);
    pan.x = cx - canvasX * zoom; pan.y = cy - canvasY * zoom;
    updTrans(); drawConns();
}

function zoomOut() {
    const area = document.getElementById('canvas-area');
    const rect = area.getBoundingClientRect();
    const cx = rect.width / 2, cy = rect.height / 2;
    const canvasX = (cx - pan.x) / zoom, canvasY = (cy - pan.y) / zoom;
    zoom = Math.max(ZOOM_MIN, zoom - ZOOM_STEP);
    pan.x = cx - canvasX * zoom; pan.y = cy - canvasY * zoom;
    updTrans(); drawConns();
}

function addNode(t, x, y) {
    const d = DEFS.find(dd => dd.id === t);
    const n = {id: 'n' + (++nid), type: t, x, y, params: {}};
    d.params.forEach(p => n.params[p.name] = p.default);
    nodes.push(n);
    render(n);
    select(n);
    status('Added ' + d.title);
}

function render(n) {
    const d = DEFS.find(dd => dd.id === n.type);
    const el = document.createElement('div');
    el.className = 'node'; el.id = n.id;
    el.style.cssText = 'left:' + n.x + 'px;top:' + n.y + 'px';
    let h = '<div class="node-hdr" style="background:' + d.color + '">' + d.title + '</div><div class="node-body">';
    d.inputs.forEach(i => { h += '<div class="port in' + (i.optional ? ' opt' : '') + '"><span class="dot" data-n="' + n.id + '" data-p="' + i.name + '" data-d="in"></span>' + i.name + '</div>'; });
    d.outputs.forEach(o => { h += '<div class="port out"><span class="dot" data-n="' + n.id + '" data-p="' + o.name + '" data-d="out"></span>' + o.name + '</div>'; });
    h += '</div>';
    el.innerHTML = h;
    
    el.onmousedown = e => {
        if (e.target.classList.contains('dot')) { startConn(e.target); e.stopPropagation(); return; }
        select(n); dragN = n; off = {x: e.clientX - n.x * zoom - pan.x, y: e.clientY - n.y * zoom - pan.y}; e.stopPropagation();
    };
    document.addEventListener('mousemove', e => {
        if (dragN === n) { n.x = (e.clientX - off.x - pan.x) / zoom; n.y = (e.clientY - off.y - pan.y) / zoom; el.style.left = n.x + 'px'; el.style.top = n.y + 'px'; drawConns(); }
    });
    document.addEventListener('mouseup', () => { dragN = null; });
    el.querySelectorAll('.dot').forEach(dot => { dot.onmouseup = () => { if (dragC) endConn(dot); }; });
    
    document.getElementById('canvas').appendChild(el);
}

function select(n) { desel(); sel = n; document.getElementById(n.id).classList.add('selected'); showProps(n); }
function desel() { if (sel) { const e = document.getElementById(sel.id); if (e) e.classList.remove('selected'); } sel = null; document.getElementById('props').innerHTML = '<div class="props-empty">Select a node</div>'; }

function showProps(n) {
    const d = DEFS.find(dd => dd.id === n.type);
    let h = '<h3>' + d.title + '</h3>';
    d.params.forEach(p => {
        if (p.type === 'button') {
            h += '<div class="prop"><button class="pick-btn" onclick="pickROI(\\'' + n.id + '\\')">' + p.default + '</button></div>';
        } else {
            h += '<div class="prop"><label>' + p.name + '</label>';
            if (p.choices) {
                h += '<select data-p="' + p.name + '">' + p.choices.map(c => '<option' + (n.params[p.name] === c ? ' selected' : '') + '>' + c + '</option>').join('') + '</select>';
            } else if (p.type === 'bool') {
                h += '<input type="checkbox" data-p="' + p.name + '"' + (n.params[p.name] ? ' checked' : '') + '>';
            } else if (p.type === 'int' || p.type === 'float') {
                h += '<input type="number" data-p="' + p.name + '" value="' + (n.params[p.name] ?? '') + '" step="' + (p.type === 'float' ? '0.1' : '1') + '"' + (p.min != null ? ' min="' + p.min + '"' : '') + (p.max != null ? ' max="' + p.max + '"' : '') + '>';
            } else if (p.type === 'file') {
                // File input with suggestions dropdown
                const fileType = n.type.includes('video') ? 'video' : n.type.includes('image') ? 'image' : n.type.includes('track') ? 'track' : 'all';
                h += '<div class="file-input-wrap"><input type="text" data-p="' + p.name + '" data-filetype="' + fileType + '" value="' + (n.params[p.name] || '') + '" list="filelist-' + p.name + '" autocomplete="off">';
                h += '<button type="button" class="file-browse-btn" data-p="' + p.name + '" data-filetype="' + fileType + '">▼</button></div>';
                h += '<datalist id="filelist-' + p.name + '"></datalist>';
            } else {
                h += '<input type="text" data-p="' + p.name + '" value="' + (n.params[p.name] || '') + '">';
            }
            h += '</div>';
        }
    });
    h += '<button class="del-btn" onclick="delNode()">Delete Node</button>';
    document.getElementById('props').innerHTML = h;
    document.querySelectorAll('#props input, #props select').forEach(el => {
        el.onchange = e => {
            const v = e.target.type === 'checkbox' ? e.target.checked : (e.target.type === 'number' ? (e.target.step === '0.1' ? parseFloat(e.target.value) : parseInt(e.target.value)) : e.target.value);
            n.params[e.target.dataset.p] = v;
        };
    });
    // Set up file input suggestions
    document.querySelectorAll('#props .file-browse-btn').forEach(btn => {
        btn.onclick = () => loadFileSuggestions(btn.dataset.p, btn.dataset.filetype);
    });
    document.querySelectorAll('#props input[data-filetype]').forEach(inp => {
        inp.onfocus = () => loadFileSuggestions(inp.dataset.p, inp.dataset.filetype);
    });
}

async function loadFileSuggestions(paramName, fileType) {
    try {
        const r = await fetch('/api/files?type=' + fileType);
        const data = await r.json();
        const dl = document.getElementById('filelist-' + paramName);
        if (dl) {
            dl.innerHTML = data.files.map(f => '<option value="' + f + '">').join('');
        }
    } catch (err) {
        console.error('Failed to load file suggestions:', err);
    }
}

function delNode() {
    if (!sel) return;
    conns = conns.filter(c => c.sn !== sel.id && c.tn !== sel.id);
    document.getElementById(sel.id)?.remove();
    nodes = nodes.filter(n => n.id !== sel.id);
    drawConns(); desel(); status('Deleted');
}

function startConn(dot) { dragC = {sn: dot.dataset.n, sp: dot.dataset.p, sd: dot.dataset.d}; }
function endConn(dot) {
    if (!dragC || dragC.sd === dot.dataset.d) { cancelConn(); return; }
    const c = dragC.sd === 'out' ? {sn: dragC.sn, sp: dragC.sp, tn: dot.dataset.n, tp: dot.dataset.p} : {sn: dot.dataset.n, sp: dot.dataset.p, tn: dragC.sn, tp: dragC.sp};
    if (!conns.some(x => x.sn === c.sn && x.sp === c.sp && x.tn === c.tn && x.tp === c.tp)) { conns.push(c); drawConns(); status('Connected'); }
    dragC = null; remTemp();
}
function cancelConn() { dragC = null; remTemp(); }

function updTemp(e) {
    const svg = document.getElementById('svg-layer');
    let t = svg.querySelector('.conn-temp');
    if (!t) { t = document.createElementNS('http://www.w3.org/2000/svg', 'path'); t.classList.add('conn-temp'); svg.appendChild(t); }
    const dot = document.querySelector('[data-n="' + dragC.sn + '"][data-p="' + dragC.sp + '"]');
    if (!dot) return;
    const r = dot.getBoundingClientRect(), ar = document.getElementById('canvas-area').getBoundingClientRect();
    // Draw in screen coordinates relative to canvas-area
    const x1 = r.left + 4 - ar.left;
    const y1 = r.top + 4 - ar.top;
    const x2 = e.clientX - ar.left;
    const y2 = e.clientY - ar.top;
    const cx = 50 * zoom;
    t.setAttribute('d', 'M' + x1 + ' ' + y1 + ' C' + (x1+cx) + ' ' + y1 + ',' + (x2-cx) + ' ' + y2 + ',' + x2 + ' ' + y2);
}
function remTemp() { document.querySelector('.conn-temp')?.remove(); }

function drawConns() {
    const svg = document.getElementById('svg-layer'), ar = document.getElementById('canvas-area').getBoundingClientRect();
    svg.querySelectorAll('path').forEach(e => e.remove());
    conns.forEach((c, idx) => {
        const d1 = document.querySelector('[data-n="' + c.sn + '"][data-p="' + c.sp + '"]');
        const d2 = document.querySelector('[data-n="' + c.tn + '"][data-p="' + c.tp + '"]');
        if (!d1 || !d2) return;
        const r1 = d1.getBoundingClientRect(), r2 = d2.getBoundingClientRect();
        // Draw in screen coordinates relative to canvas-area
        const x1 = r1.left + 4 - ar.left;
        const y1 = r1.top + 4 - ar.top;
        const x2 = r2.left + 4 - ar.left;
        const y2 = r2.top + 4 - ar.top;
        const cx = 60 * zoom;  // Curve control offset scales with zoom
        const pathD = 'M' + x1 + ' ' + y1 + ' C' + (x1+cx) + ' ' + y1 + ',' + (x2-cx) + ' ' + y2 + ',' + x2 + ' ' + y2;

        // Visible path
        const visPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        visPath.classList.add('conn');
        visPath.setAttribute('d', pathD);
        visPath.style.pointerEvents = 'none';

        // Invisible wide path for hit detection
        const hitPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        hitPath.setAttribute('d', pathD);
        hitPath.setAttribute('stroke', 'rgba(0,0,0,0.001)');
        hitPath.setAttribute('stroke-width', '20');
        hitPath.setAttribute('fill', 'none');
        hitPath.style.pointerEvents = 'auto';
        hitPath.style.cursor = 'pointer';
        hitPath.dataset.idx = idx;

        hitPath.oncontextmenu = e => { e.preventDefault(); e.stopPropagation(); deleteConn(idx); };
        hitPath.onclick = e => { e.stopPropagation(); if (e.shiftKey || e.metaKey) deleteConn(idx); };
        hitPath.onmouseenter = () => visPath.classList.add('conn-hover');
        hitPath.onmouseleave = () => visPath.classList.remove('conn-hover');

        svg.appendChild(visPath);
        svg.appendChild(hitPath);
    });
}

function deleteConn(idx) {
    conns.splice(idx, 1);
    drawConns();
    status('Connection deleted');
}

function status(m) { document.getElementById('status').textContent = m; }

// Path resolution helper
function resolvePath(filepath) {
    if (!filepath) return filepath;
    // If absolute path (starts with /), return as-is
    if (filepath.startsWith('/')) return filepath;
    // If relative and we have a project dir, join them
    if (projectDir) {
        return projectDir + (projectDir.endsWith('/') ? '' : '/') + filepath;
    }
    return filepath;
}

// ROI Picker
let roiNode = null, roiRect = null, roiDrawing = false, roiStart = null;
let origW = 0, origH = 0;

function pickROI(nodeId) {
    roiNode = nodes.find(n => n.id === nodeId);
    if (!roiNode) { alert('Node not found'); return; }

    // Find connected video input
    const conn = conns.find(c => c.tn === nodeId && c.tp === 'video_in');
    if (!conn) { alert('Connect a video input to the ROI node first'); return; }

    const srcNode = nodes.find(n => n.id === conn.sn);
    if (!srcNode || srcNode.type !== 'video_input') { alert('Source must be a video_input node'); return; }

    const filepath = resolvePath(srcNode.params.filepath);
    if (!filepath) { alert('Video input has no filepath set'); return; }

    const firstFrame = srcNode.params.first_frame || 1;

    // Fetch frame
    const img = document.getElementById('roi-img');
    const canvas = document.getElementById('roi-canvas');

    status('Loading frame ' + firstFrame + '...');
    img.onload = () => {
        origW = parseInt(img.getAttribute('data-orig-w')) || img.naturalWidth;
        origH = parseInt(img.getAttribute('data-orig-h')) || img.naturalHeight;
        roiRect = null;

        // Show modal first, then set up canvas after layout
        document.getElementById('roi-modal').classList.add('active');

        requestAnimationFrame(() => {
            // Now the image is laid out, get its display size
            const displayW = img.offsetWidth || img.clientWidth || img.naturalWidth;
            const displayH = img.offsetHeight || img.clientHeight || img.naturalHeight;
            canvas.width = displayW;
            canvas.height = displayH;
            canvas.style.width = displayW + 'px';
            canvas.style.height = displayH + 'px';
            drawROIRect();
            status('Draw a rectangle on frame ' + firstFrame);
        });
    };
    img.onerror = () => { alert('Failed to load frame'); status('Error loading frame'); };

    fetch('/api/frame?filepath=' + encodeURIComponent(filepath) + '&frame=' + firstFrame)
        .then(r => {
            origW = parseInt(r.headers.get('X-Original-Width')) || 0;
            origH = parseInt(r.headers.get('X-Original-Height')) || 0;
            return r.blob();
        })
        .then(b => {
            img.setAttribute('data-orig-w', origW);
            img.setAttribute('data-orig-h', origH);
            img.src = URL.createObjectURL(b);
        })
        .catch(e => { alert('Failed to fetch frame: ' + e.message); });
}

function closeROIModal() {
    document.getElementById('roi-modal').classList.remove('active');
    roiNode = null;
    roiRect = null;
}

function applyROI() {
    if (!roiNode || !roiRect) { alert('Draw a rectangle first'); return; }

    const img = document.getElementById('roi-img');
    const scaleX = origW / img.offsetWidth;
    const scaleY = origH / img.offsetHeight;

    console.log('applyROI debug:', {
        roiRect: roiRect,
        origW: origW, origH: origH,
        imgOffsetW: img.offsetWidth, imgOffsetH: img.offsetHeight,
        scaleX: scaleX, scaleY: scaleY
    });

    const node = roiNode;  // Save reference before closing
    node.params.x = Math.round(roiRect.x * scaleX);
    node.params.y = Math.round(roiRect.y * scaleY);
    node.params.width = Math.round(roiRect.w * scaleX);
    node.params.height = Math.round(roiRect.h * scaleY);

    console.log('Final params:', node.params);

    closeROIModal();
    showProps(node);
    status('ROI set: ' + node.params.x + ',' + node.params.y + ' ' + node.params.width + 'x' + node.params.height);
}

function drawROIRect() {
    const canvas = document.getElementById('roi-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (roiRect) {
        ctx.strokeStyle = '#4fc3f7';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(roiRect.x, roiRect.y, roiRect.w, roiRect.h);
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(79, 195, 247, 0.1)';
        ctx.fillRect(roiRect.x, roiRect.y, roiRect.w, roiRect.h);
    }
}

// Canvas mouse events for ROI drawing
document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('roi-canvas');

    canvas.onmousedown = e => {
        const r = canvas.getBoundingClientRect();
        roiStart = { x: e.clientX - r.left, y: e.clientY - r.top };
        roiDrawing = true;
        roiRect = { x: roiStart.x, y: roiStart.y, w: 0, h: 0 };
    };

    canvas.onmousemove = e => {
        if (!roiDrawing) return;
        const r = canvas.getBoundingClientRect();
        const x = e.clientX - r.left, y = e.clientY - r.top;
        roiRect = {
            x: Math.min(roiStart.x, x),
            y: Math.min(roiStart.y, y),
            w: Math.abs(x - roiStart.x),
            h: Math.abs(y - roiStart.y)
        };
        drawROIRect();
    };

    canvas.onmouseup = () => { roiDrawing = false; };
    canvas.onmouseleave = () => { roiDrawing = false; };
});

let isRunning = false;

async function execute() {
    if (isRunning) return;
    isRunning = true;
    document.getElementById('run-btn').style.display = 'none';
    document.getElementById('abort-btn').style.display = '';
    status('Running...');
    const g = {
        projectDir: projectDir,
        nodes: nodes.map(n => ({id: n.id, type: n.type, data: {params: n.params}})),
        edges: conns.map(c => ({source: c.sn, sourceHandle: c.sp, target: c.tn, targetHandle: c.tp}))
    };
    try {
        const r = await fetch('/api/execute', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(g)});
        const j = await r.json();
        if (j.aborted) {
            status('Aborted');
            alert('Execution aborted by user.\\n\\n' + j.message);
        } else if (j.success) {
            status('Done!');
            alert('Execution complete!\\n\\n' + j.message);
        } else {
            status('Error');
            let errMsg = 'Errors:\\n';
            j.errors.forEach(e => { errMsg += '• ' + e.type + ' (' + e.node_id + '): ' + e.error + '\\n'; });
            errMsg += '\\nLog:\\n' + j.message;
            alert(errMsg);
        }
    } catch (err) {
        status('Error');
        alert('Request failed: ' + err.message);
    } finally {
        isRunning = false;
        document.getElementById('run-btn').style.display = '';
        document.getElementById('abort-btn').style.display = 'none';
    }
}

async function abort() {
    if (!isRunning) return;
    status('Aborting...');
    try {
        await fetch('/api/abort', {method: 'POST'});
    } catch (err) {
        console.error('Abort request failed:', err);
    }
}

async function save() {
    const data = {projectDir, nodes, conns};
    const content = JSON.stringify(data, null, 2);

    // Try to use File System Access API for native save dialog
    if (window.showSaveFilePicker) {
        try {
            const handle = await window.showSaveFilePicker({
                suggestedName: 'vdg-graph.json',
                types: [{
                    description: 'VDG Graph',
                    accept: {'application/json': ['.json']}
                }]
            });
            const writable = await handle.createWritable();
            await writable.write(content);
            await writable.close();
            status('Saved: ' + handle.name);
            return;
        } catch (err) {
            if (err.name === 'AbortError') {
                status('Save cancelled');
                return;
            }
            // Fall through to download method
        }
    }

    // Fallback: download method
    const b = new Blob([content], {type: 'application/json'});
    const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = 'vdg-graph.json'; a.click();
    status('Saved');
}

function load() {
    const i = document.createElement('input'); i.type = 'file'; i.accept = '.json';
    i.onchange = e => {
        const r = new FileReader();
        r.onload = async ev => {
            const d = JSON.parse(ev.target.result);
            clear();
            nodes = d.nodes || []; conns = d.conns || [];
            nid = Math.max(0, ...nodes.map(n => parseInt(n.id.slice(1)) || 0));
            nodes.forEach(render); drawConns();

            // Restore project directory
            if (d.projectDir) {
                document.getElementById('project-dir').value = d.projectDir;
                await validateProjectDir();
            }
            status('Loaded');
        };
        r.readAsText(e.target.files[0]);
    };
    i.click();
}

function clear() {
    document.getElementById('canvas').innerHTML = '';
    document.getElementById('svg-layer').innerHTML = '';
    nodes = []; conns = []; nid = 0; pan = {x: 0, y: 0}; zoom = 1;
    updTrans();
    // Clear project directory
    projectDir = '';
    document.getElementById('project-dir').value = '';
    document.getElementById('project-status').className = 'project-status empty';
    document.getElementById('project-status').textContent = 'Not Set';
    desel(); status('Cleared');
}

init();
</script>
</body>
</html>'''


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_TEMPLATE


@app.get("/api/nodes")
async def get_nodes():
    return get_nodes_json()


@app.get("/api/frame")
async def get_frame(filepath: str, frame: int = 1):
    """Get a single frame from a video as JPEG."""
    from fastapi.responses import Response
    import cv2
    from pathlib import Path

    if not Path(filepath).exists():
        return Response(content=b"File not found", status_code=404)

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return Response(content=b"Cannot open video", status_code=400)

    try:
        # Seek to frame (0-indexed internally, but we use 1-indexed API)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
        ret, img = cap.read()
        if not ret:
            return Response(content=b"Cannot read frame", status_code=400)

        # Get dimensions for response header
        height, width = img.shape[:2]

        # Resize if too large (max 1280px wide for preview)
        max_width = 1280
        if width > max_width:
            scale = max_width / width
            img = cv2.resize(img, (max_width, int(height * scale)))

        # Encode as JPEG
        _, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])

        return Response(
            content=jpeg.tobytes(),
            media_type="image/jpeg",
            headers={"X-Original-Width": str(width), "X-Original-Height": str(height)}
        )
    finally:
        cap.release()


@app.get("/api/validate-path")
async def validate_path(path: str):
    """Validate a directory path exists and is accessible."""
    from pathlib import Path

    if not path:
        return {"valid": False, "error": "Empty path"}

    target = Path(path).expanduser().resolve()

    if not target.exists():
        return {"valid": False, "error": "Does not exist"}

    if not target.is_dir():
        return {"valid": False, "error": "Not a directory"}

    # Check if readable
    try:
        list(target.iterdir())
    except PermissionError:
        return {"valid": False, "error": "Permission denied"}

    # Refresh file cache for this directory
    refresh_file_cache(str(target))

    return {"valid": True, "path": str(target)}


@app.get("/api/files")
async def list_files(type: str = "all", refresh: bool = False):
    """List files in the project directory, filtered by type.

    Args:
        type: Filter by file type ('video', 'image', 'track', 'all')
        refresh: Force refresh the cache
    """
    if refresh or not _file_cache.get('all'):
        refresh_file_cache()

    files = _file_cache.get(type, _file_cache.get('all', []))
    return {"files": files, "directory": _cache_directory}


@app.post("/api/abort")
async def abort_execution():
    """Abort the currently running graph execution."""
    global _abort_flag
    _abort_flag = True
    print("\n⚠ ABORT REQUESTED")
    return {"status": "abort requested"}


def _run_graph_sync(graph: dict) -> dict:
    """Run graph execution synchronously (called from thread pool)."""
    try:
        executor = GraphExecutor()
        result = executor.execute(graph)
        if result.get('aborted'):
            print("\n⚠ Execution aborted by user")
        elif result['success']:
            print("\n✓ Execution completed successfully")
        else:
            print("\n✗ Execution failed")
        return result
    except Exception as e:
        import traceback
        print(f"\n✗ Execution error: {e}")
        return {'success': False, 'message': str(e), 'errors': [{'node_id': 'graph', 'error': traceback.format_exc()}]}


@app.post("/api/execute")
async def execute_graph(graph: dict):
    """Execute the node graph."""
    import asyncio

    global _abort_flag
    _abort_flag = False  # Reset abort flag at start

    print("\n" + "=" * 50)
    print("EXECUTING GRAPH")
    print("=" * 50)

    # Run in thread pool so abort requests can be processed
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_graph_sync, graph)

    print("=" * 50 + "\n")
    return result


class GraphExecutor:
    """Executes a VDG node graph with streaming optimization."""

    # Nodes that can process frames in a streaming fashion
    STREAMABLE_NODES = {
        'video_input',      # Yields frames
        'feature_tracker',  # Processes frame-by-frame
        'feature_tracker_2p',  # Tracks two ROIs frame-by-frame
        'apply_transform',  # Applies pre-computed transform per frame
        'frame_average',    # Accumulates without storing frames
        'clahe',            # Single-frame operation
        'gamma',            # Single-frame gamma correction
        'rgb_multiply',     # Single-frame RGB channel multiply
        'video_output',     # Writes frames as they come
        'image_output',     # Final output (end of stream)
    }

    # Nodes that CANNOT stream - they need all data before producing output
    NON_STREAMABLE_NODES = {
        'stabilizer',       # Needs all track data to compute transforms
        'gaussian_filter',  # Needs neighboring frames for smoothing
    }

    # Nodes that are just data/config (no video processing)
    DATA_NODES = {
        'roi',
        'track_input',
        'track_output',
    }

    def __init__(self):
        self.outputs = {}  # node_id -> {port_name: value}
        self.log = []
        self.project_dir = None  # Project directory for relative paths
        self.aborted = False

    def _check_abort(self) -> bool:
        """Check if abort has been requested."""
        global _abort_flag
        if _abort_flag and not self.aborted:
            self.aborted = True
            self._log("⚠ ABORT: Stopping execution...")
        return self.aborted

    def _resolve_path(self, filepath: str) -> str:
        """Resolve a filepath relative to project directory.

        If filepath is absolute, returns it as-is.
        If filepath is relative and project_dir is set, joins them.
        Supports subdirectories (e.g., 'input/clip.mp4').
        """
        from pathlib import Path

        if not filepath:
            return filepath

        path = Path(filepath)

        # If absolute, return as-is
        if path.is_absolute():
            return str(path)

        # If relative and we have a project dir, resolve against it
        if self.project_dir:
            return str(Path(self.project_dir) / filepath)

        # No project dir - return as-is (will likely fail if truly relative)
        return filepath

    def _log(self, message: str):
        """Add message to log and print to terminal."""
        self.log.append(message)
        print(message)

    def _get_node_type(self, node: dict) -> str:
        return node.get('type', '')

    def _find_streaming_chains(self, nodes: dict, edges: list) -> list:
        """
        Find chains of nodes that can be executed in streaming mode.

        Creates SEPARATE chains for each distinct initial video path from video_input.
        Each chain can have internal branching (e.g., apply_transform → frame_average + video_output).

        Returns list of chain dicts, where each chain contains:
        - 'root': the video_input node ID
        - 'nodes': list of node IDs in topological order
        - 'video_edges': list of (src, tgt, src_port, tgt_port) for video connections
        """
        # Build adjacency info
        outgoing = {nid: [] for nid in nodes}  # node -> [(target_node, src_port, tgt_port)]
        incoming = {nid: [] for nid in nodes}  # node -> [(source_node, src_port, tgt_port)]

        for e in edges:
            src, tgt = e['source'], e['target']
            outgoing[src].append((tgt, e['sourceHandle'], e['targetHandle']))
            incoming[tgt].append((src, e['sourceHandle'], e['targetHandle']))

        # Video port names
        VIDEO_OUT_PORTS = ('video', 'video_out', 'mask', 'mask_out')
        VIDEO_IN_PORTS = ('video', 'video_in', 'mask', 'mask_in')

        chains = []

        for nid, node in nodes.items():
            ntype = self._get_node_type(node)

            # Start chains from video_input
            if ntype == 'video_input':
                # Find all DIRECT video outputs from this video_input
                # Each becomes the start of a separate chain
                direct_video_outs = [
                    (tgt, sp, tp) for tgt, sp, tp in outgoing[nid]
                    if sp in VIDEO_OUT_PORTS and tp in VIDEO_IN_PORTS
                ]

                # Group by target node to avoid duplicates (same node via different ports)
                seen_starts = set()

                for start_node, start_sp, start_tp in direct_video_outs:
                    if start_node in seen_starts:
                        continue

                    start_type = self._get_node_type(nodes[start_node])

                    # Skip non-streamable starting nodes
                    if start_type in self.NON_STREAMABLE_NODES:
                        continue
                    if start_type not in self.STREAMABLE_NODES:
                        continue

                    seen_starts.add(start_node)

                    # BFS from this starting node to find all downstream streamable nodes
                    chain_nodes = set([nid, start_node])
                    chain_edges = [(nid, start_node, start_sp, start_tp)]
                    queue = [start_node]
                    visited = set([nid, start_node])

                    while queue:
                        current = queue.pop(0)

                        # Find all video output connections
                        for tgt, sp, tp in outgoing[current]:
                            if sp not in VIDEO_OUT_PORTS or tp not in VIDEO_IN_PORTS:
                                continue

                            tgt_type = self._get_node_type(nodes[tgt])

                            # Skip non-streamable nodes
                            if tgt_type in self.NON_STREAMABLE_NODES:
                                continue
                            if tgt_type not in self.STREAMABLE_NODES:
                                continue

                            # Add edge to chain
                            chain_edges.append((current, tgt, sp, tp))
                            chain_nodes.add(tgt)

                            if tgt not in visited:
                                visited.add(tgt)
                                queue.append(tgt)

                    if len(chain_nodes) > 1:
                        # Topological sort of chain nodes
                        sorted_nodes = self._topological_sort_chain(nid, chain_nodes, chain_edges)
                        chains.append({
                            'root': nid,
                            'nodes': sorted_nodes,
                            'video_edges': chain_edges,
                        })

        return chains

    def _topological_sort_chain(self, root: str, nodes: set, edges: list) -> list:
        """Topological sort of nodes in a streaming chain."""
        # Build in-degree count for chain nodes only
        in_degree = {n: 0 for n in nodes}
        adj = {n: [] for n in nodes}

        for src, tgt, _, _ in edges:
            if src in nodes and tgt in nodes:
                adj[src].append(tgt)
                in_degree[tgt] += 1

        # Kahn's algorithm
        result = []
        queue = [n for n in nodes if in_degree[n] == 0]

        while queue:
            # Sort for deterministic order
            queue.sort()
            node = queue.pop(0)
            result.append(node)

            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def _execute_streaming_chain(self, chain: dict, nodes: dict, edges: list, errors: list) -> None:
        """Execute a chain of nodes in streaming mode (frame by frame).

        Chain is a dict with:
        - 'root': video_input node ID
        - 'nodes': list of node IDs in topological order
        - 'video_edges': list of (src, tgt, src_port, tgt_port) for video connections
        """
        from vdg.core.video import VideoReader, FFmpegReader
        import numpy as np
        import time

        if not chain or not chain.get('nodes'):
            return

        chain_nodes = chain['nodes']
        video_edges = chain.get('video_edges', [])
        root_id = chain['root']

        # Build a description of the chain for logging
        chain_desc = self._describe_chain(chain, nodes)
        chain_start = time.time()
        self._log(f"\n>>> STREAMING PIPELINE: {chain_desc}")

        # Get video source info
        source_node = nodes[root_id]
        source_params = source_node.get('data', {}).get('params', {}) or source_node.get('params', {})
        filepath = self._resolve_path(source_params.get('filepath', '').strip())
        first_frame = source_params.get('first_frame', 1)
        last_frame = source_params.get('last_frame', -1)
        last_frame = None if last_frame == -1 else last_frame
        use_hardware = source_params.get('use_hardware', False)
        hlg_convert = source_params.get('hlg_convert', False)

        if not filepath:
            errors.append({'node_id': root_id, 'type': 'video_input', 'error': 'No filepath specified'})
            return

        from pathlib import Path
        if not Path(filepath).exists():
            errors.append({'node_id': root_id, 'type': 'video_input', 'error': f'File not found: {filepath}'})
            return

        from vdg.core.video import get_video_properties
        props = get_video_properties(filepath)
        self._log(f"  Source: {filepath}")
        self._log(f"  Video: {props.width}x{props.height}, {props.fps:.2f}fps")

        # Store video reference for non-streaming nodes that might need it
        self.outputs[root_id] = {
            'video_out': {'filepath': filepath, 'first_frame': first_frame, 'last_frame': last_frame, 'props': props, 'use_hardware': use_hardware},
            'props': props
        }

        # Build mapping: node -> list of (upstream_node, src_port, tgt_port)
        node_video_inputs = {nid: [] for nid in chain_nodes}
        for src, tgt, sp, tp in video_edges:
            node_video_inputs[tgt].append((src, sp, tp))

        # Prepare streaming state for each node in chain
        chain_state = {}
        transform_frames = None  # Track frame range from transforms if present

        for nid in chain_nodes:
            if nid == root_id:
                continue  # Skip video_input

            node = nodes[nid]
            ntype = self._get_node_type(node)
            params = node.get('data', {}).get('params', {}) or node.get('params', {})

            # Gather non-video inputs (transforms, ROI, etc.)
            non_video_inputs = {}
            for e in edges:
                if e['target'] == nid and e['targetHandle'] not in ('video', 'video_in', 'mask', 'mask_in'):
                    src_out = self.outputs.get(e['source'], {})
                    non_video_inputs[e['targetHandle']] = src_out.get(e['sourceHandle'])

            # If this is apply_transform, extract frame range from transforms
            if ntype == 'apply_transform':
                transform_data = non_video_inputs.get('transforms', {})
                if transform_data and 'frames' in transform_data:
                    transform_frames = transform_data['frames']
                    self._log(f"  Transform data covers frames {transform_frames[0]}-{transform_frames[-1]} ({len(transform_frames)} frames)")

            chain_state[nid] = {
                'type': ntype,
                'params': params,
                'inputs': non_video_inputs,
                'video_inputs': node_video_inputs[nid],  # (upstream, src_port, tgt_port)
                'accumulator': None,
                'alpha_accumulator': None,
                'frame_count': 0,
                'tracker': None,
                'tracker2': None,  # For feature_tracker_2p
                'track_data': {},
                'track_data1': {},  # For feature_tracker_2p
                'track_data2': {},  # For feature_tracker_2p
            }

        # Determine frame range - use transform frames if available
        if transform_frames:
            # Override video range with transform range
            first_frame = transform_frames[0]
            last_frame = transform_frames[-1]
            self._log(f"  Limiting to transform frame range: {first_frame}-{last_frame}")

        # Stream frames through the chain
        frame_count = 0

        # Choose reader based on HLG convert setting
        if hlg_convert:
            # HLG/HDR to SDR tonemapping via FFmpeg
            hlg_filter = (
                "zscale=t=linear:npl=100,format=gbrpf32le,"
                "zscale=p=bt709,tonemap=hable:desat=0,"
                "zscale=t=bt709:m=bt709:r=tv,format=bgr24"
            )
            self._log(f"  HLG Convert: enabled (FFmpeg tonemapping)")
            reader_ctx = FFmpegReader(filepath, first_frame, last_frame, vf=hlg_filter)
        else:
            self._log(f"  Hardware decode: {use_hardware}")
            reader_ctx = VideoReader(filepath, first_frame, last_frame, use_hardware=use_hardware)

        with reader_ctx as reader:
            total_frames = reader.frame_count
            self._log(f"  Streaming {total_frames} frames...")

            for frame_num, frame in reader:
                # Check for abort
                if self._check_abort():
                    break

                # Track per-node outputs for this frame
                # Each node stores: {'video_out': frame, 'mask_out': mask, ...}
                frame_outputs = {
                    root_id: {
                        'video_out': frame,
                        'video': frame,  # Alias for old port name
                        '_data': {'frame_num': frame_num, 'props': props}
                    }
                }

                # Process through each node in topological order
                for nid in chain_nodes:
                    if nid == root_id:
                        continue

                    state = chain_state[nid]
                    ntype = state['type']
                    params = state['params']
                    inputs = state['inputs']

                    # Get video input(s) from upstream node(s)
                    current_frame = None
                    current_mask = None
                    current_data = {'frame_num': frame_num, 'props': props}

                    for upstream, src_port, tgt_port in state['video_inputs']:
                        upstream_out = frame_outputs.get(upstream, {})
                        if tgt_port in ('video_in', 'video'):
                            current_frame = upstream_out.get(src_port)
                            if '_data' in upstream_out:
                                current_data = upstream_out['_data']
                        elif tgt_port in ('mask_in', 'mask'):
                            current_mask = upstream_out.get(src_port)

                    if current_frame is None:
                        # Node has no video input ready - skip
                        continue

                    try:
                        if ntype == 'feature_tracker':
                            out_frame, out_data = self._stream_feature_tracker(
                                current_frame, current_data, state, params, inputs
                            )
                            frame_outputs[nid] = {'video_out': out_frame, '_data': out_data}

                        elif ntype == 'feature_tracker_2p':
                            out_frame, out_data = self._stream_feature_tracker_2p(
                                current_frame, current_data, state, params, inputs
                            )
                            frame_outputs[nid] = {'video_out': out_frame, '_data': out_data}

                        elif ntype == 'apply_transform':
                            out_frame, out_data = self._stream_apply_transform(
                                current_frame, current_data, state, params, inputs
                            )
                            # apply_transform produces both video_out and mask_out
                            frame_outputs[nid] = {
                                'video_out': out_frame,
                                'mask_out': out_data.get('mask'),
                                'mask': out_data.get('mask'),  # Alias for old port name
                                '_data': out_data
                            }

                        elif ntype == 'frame_average':
                            # Pass mask if available
                            if current_mask is not None:
                                inputs['_mask'] = current_mask
                            out_frame, out_data = self._stream_frame_average(
                                current_frame, current_data, state, params, inputs
                            )
                            frame_outputs[nid] = {'video_out': out_frame, '_data': out_data}

                        elif ntype == 'video_output':
                            self._stream_video_output(
                                current_frame, current_data, state, params, frame_count == 0
                            )
                            frame_outputs[nid] = {'_data': current_data}

                        elif ntype == 'clahe':
                            out_frame, out_data = self._stream_clahe(
                                current_frame, current_data, state, params
                            )
                            frame_outputs[nid] = {'video_out': out_frame, 'image_out': out_frame, '_data': out_data}

                        elif ntype == 'gamma':
                            out_frame, out_data = self._stream_gamma(
                                current_frame, current_data, state, params, inputs
                            )
                            # gamma can also process mask
                            out_mask = None
                            if current_mask is not None:
                                out_mask = current_mask  # Mask passes through unchanged
                            frame_outputs[nid] = {
                                'video_out': out_frame,
                                'image_out': out_frame,
                                'mask_out': out_mask,
                                '_data': out_data
                            }

                        elif ntype == 'rgb_multiply':
                            out_frame, out_data = self._stream_rgb_multiply(
                                current_frame, current_data, state, params
                            )
                            frame_outputs[nid] = {
                                'video_out': out_frame,
                                'image_out': out_frame,
                                '_data': out_data
                            }

                    except Exception as ex:
                        errors.append({'node_id': nid, 'type': ntype, 'error': str(ex)})
                        self._log(f"  ✗ Error in {ntype}: {ex}")
                        import traceback
                        traceback.print_exc()
                        return

                frame_count += 1
                if frame_count % 100 == 0:
                    self._log(f"  Processed {frame_count}/{total_frames} frames...")

        stream_elapsed = time.time() - chain_start
        fps = frame_count / stream_elapsed if stream_elapsed > 0 else 0
        self._log(f"  Streamed {frame_count} frames in {stream_elapsed:.2f}s ({fps:.1f} fps)")

        # Finalize each node and store outputs
        for nid in chain_nodes:
            if nid == root_id:
                continue

            state = chain_state[nid]
            ntype = state['type']
            params = state['params']

            try:
                if ntype == 'feature_tracker':
                    # Close preview video writer if it exists
                    if state.get('preview_writer'):
                        state['preview_writer'].release()
                        self._log(f"  Preview video saved")
                    self.outputs[nid] = {
                        'track_data': state['track_data'],
                        'points': state.get('last_points'),
                    }
                    self._log(f"  ✓ feature_tracker: {len(state['track_data'])} tracked frames")

                elif ntype == 'feature_tracker_2p':
                    # Close preview video writer if it exists
                    if state.get('preview_writer'):
                        state['preview_writer'].release()
                        self._log(f"  Preview video saved")
                    self.outputs[nid] = {
                        'track_data1': state['track_data1'],
                        'track_data2': state['track_data2'],
                    }
                    self._log(f"  ✓ feature_tracker_2p: track1={len(state['track_data1'])}, track2={len(state['track_data2'])} frames")

                elif ntype == 'frame_average':
                    result, alpha = self._finalize_frame_average(state, params)
                    self.outputs[nid] = {'image_out': result, 'alpha_out': alpha}
                    self._log(f"  ✓ frame_average: {result.shape[1]}x{result.shape[0]} output")

                elif ntype == 'video_output':
                    self._finalize_video_output(state)
                    self._log(f"  ✓ video_output: {state.get('filepath')}")

                elif ntype == 'apply_transform':
                    # No persistent output needed - frames already passed through
                    self.outputs[nid] = {'video_out': None, 'mask_out': None, 'mask': None}
                    self._log(f"  ✓ apply_transform complete")

                elif ntype == 'clahe':
                    self.outputs[nid] = {}
                    self._log(f"  ✓ clahe complete")

                elif ntype == 'gamma':
                    self.outputs[nid] = {'video_out': None, 'image_out': None, 'mask_out': None}
                    self._log(f"  ✓ gamma complete")

            except Exception as ex:
                errors.append({'node_id': nid, 'type': ntype, 'error': str(ex)})
                self._log(f"  ✗ Error finalizing {ntype}: {ex}")

    def _describe_chain(self, chain: dict, nodes: dict) -> str:
        """Create a human-readable description of a streaming chain."""
        chain_nodes = chain['nodes']
        video_edges = chain.get('video_edges', [])

        if len(chain_nodes) <= 4:
            return ' → '.join(self._get_node_type(nodes[nid]) for nid in chain_nodes)

        # For longer chains, show branching structure
        # Find nodes with multiple outputs
        out_count = {}
        for src, tgt, _, _ in video_edges:
            out_count[src] = out_count.get(src, 0) + 1

        desc_parts = []
        for nid in chain_nodes:
            ntype = self._get_node_type(nodes[nid])
            if out_count.get(nid, 0) > 1:
                ntype = f"{ntype}[→{out_count[nid]}]"
            desc_parts.append(ntype)

        return ' → '.join(desc_parts)

    def _stream_feature_tracker(self, frame, data, state, params, inputs):
        """Process one frame through feature tracker."""
        from vdg.tracking import FeatureTracker
        import cv2

        if state['tracker'] is None:
            roi = inputs.get('roi')
            state['tracker'] = FeatureTracker(
                num_features=params.get('num_features', 30),
                min_distance=params.get('min_distance', 30),
                initial_roi=roi,
                enforce_bbox=params.get('enforce_bbox', True),
                win_size=params.get('win_size', 21),
                pyramid_levels=params.get('pyramid_levels', 3),
            )
            state['tracker'].initialize(frame)
            state['initial_roi'] = roi

            # Initialize preview video writer if path is set
            preview_path = _ensure_video_extension(params.get('preview_path', '').strip())
            if preview_path:
                h, w = frame.shape[:2]
                fps = data.get('props').fps if data.get('props') else 30
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                state['preview_writer'] = cv2.VideoWriter(preview_path, fourcc, fps, (w, h))
                self._log(f"  Preview output: {preview_path}")
        else:
            state['tracker'].update(frame)

        tracker = state['tracker']
        if tracker.points is not None and len(tracker.points) > 0:
            pts = tracker.points.reshape(-1, 2)
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            state['track_data'][data['frame_num']] = {'x': float(cx), 'y': float(cy)}

        state['last_points'] = tracker.points
        state['frame_count'] += 1

        # Write preview frame if enabled
        if state.get('preview_writer'):
            preview = frame.copy()
            # Draw tracked points
            if tracker.points is not None:
                for pt in tracker.points.reshape(-1, 2):
                    cv2.circle(preview, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
                # Draw centroid
                pts = tracker.points.reshape(-1, 2)
                cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                cv2.circle(preview, (cx, cy), 8, (0, 255, 255), 2)
                cv2.drawMarker(preview, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
            # Draw current ROI/bbox
            if tracker.current_roi:
                rx, ry, rw, rh = [int(v) for v in tracker.current_roi]
                cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
            # Draw initial ROI
            if state.get('initial_roi'):
                rx, ry, rw, rh = [int(v) for v in state['initial_roi']]
                cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 1)
            state['preview_writer'].write(preview)

        return frame, data

    def _stream_feature_tracker_2p(self, frame, data, state, params, inputs):
        """Process one frame through two-point feature tracker."""
        from vdg.tracking import FeatureTracker
        import cv2

        if state['tracker'] is None:
            roi1 = inputs.get('roi1')
            roi2 = inputs.get('roi2')
            tracker_params = {
                'num_features': params.get('num_features', 30),
                'min_distance': params.get('min_distance', 30),
                'enforce_bbox': params.get('enforce_bbox', True),
                'win_size': params.get('win_size', 21),
                'pyramid_levels': params.get('pyramid_levels', 3),
            }
            state['tracker'] = FeatureTracker(initial_roi=roi1, **tracker_params)
            state['tracker2'] = FeatureTracker(initial_roi=roi2, **tracker_params)
            state['tracker'].initialize(frame)
            state['tracker2'].initialize(frame)
            state['initial_roi1'] = roi1
            state['initial_roi2'] = roi2

            # Initialize preview video writer if path is set
            preview_path = _ensure_video_extension(params.get('preview_path', '').strip())
            if preview_path:
                h, w = frame.shape[:2]
                fps = data.get('props').fps if data.get('props') else 30
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                state['preview_writer'] = cv2.VideoWriter(preview_path, fourcc, fps, (w, h))
                self._log(f"  Preview output: {preview_path}")
        else:
            state['tracker'].update(frame)
            state['tracker2'].update(frame)

        tracker1 = state['tracker']
        tracker2 = state['tracker2']

        # Record track 1
        if tracker1.points is not None and len(tracker1.points) > 0:
            pts = tracker1.points.reshape(-1, 2)
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            state['track_data1'][data['frame_num']] = {'x': float(cx), 'y': float(cy)}

        # Record track 2
        if tracker2.points is not None and len(tracker2.points) > 0:
            pts = tracker2.points.reshape(-1, 2)
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            state['track_data2'][data['frame_num']] = {'x': float(cx), 'y': float(cy)}

        state['frame_count'] += 1

        # Write preview frame if enabled
        if state.get('preview_writer'):
            preview = frame.copy()
            # Draw tracker 1 (green)
            if tracker1.points is not None:
                for pt in tracker1.points.reshape(-1, 2):
                    cv2.circle(preview, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
                pts = tracker1.points.reshape(-1, 2)
                cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                cv2.circle(preview, (cx, cy), 8, (0, 255, 0), 2)
                cv2.drawMarker(preview, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
            if tracker1.current_roi:
                rx, ry, rw, rh = [int(v) for v in tracker1.current_roi]
                cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            if state.get('initial_roi1'):
                rx, ry, rw, rh = [int(v) for v in state['initial_roi1']]
                cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (0, 200, 0), 1)

            # Draw tracker 2 (blue)
            if tracker2.points is not None:
                for pt in tracker2.points.reshape(-1, 2):
                    cv2.circle(preview, (int(pt[0]), int(pt[1])), 4, (255, 0, 0), -1)
                pts = tracker2.points.reshape(-1, 2)
                cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                cv2.circle(preview, (cx, cy), 8, (255, 0, 0), 2)
                cv2.drawMarker(preview, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 16, 2)
            if tracker2.current_roi:
                rx, ry, rw, rh = [int(v) for v in tracker2.current_roi]
                cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
            if state.get('initial_roi2'):
                rx, ry, rw, rh = [int(v) for v in state['initial_roi2']]
                cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (200, 0, 0), 1)

            state['preview_writer'].write(preview)

        return frame, data

    def _stream_apply_transform(self, frame, data, state, params, inputs):
        """Apply transform to one frame."""
        import cv2
        import numpy as np
        import math

        transform_input = inputs.get('transforms') or {}
        transforms = transform_input.get('transforms', {})
        frame_num = data['frame_num']
        t = transforms.get(frame_num)

        x_pad = params.get('x_pad', 0)
        y_pad = params.get('y_pad', 0)
        x_off = params.get('x_offset', 0)
        y_off = params.get('y_offset', 0)

        in_h, in_w = frame.shape[:2]
        out_w = in_w + x_pad
        out_h = in_h + y_pad

        if t is None:
            # No transform for this frame - just center it in padded canvas
            M = np.float32([[1, 0, x_pad / 2], [0, 1, y_pad / 2]])
        else:
            # Build transform matrix matching original stabilizer.py:
            # 1. Translate tracking point to origin
            # 2. Rotate and scale around origin
            # 3. Translate to reference position (+ padding offset)
            pnt_x = t.get('pnt_x', in_w / 2)
            pnt_y = t.get('pnt_y', in_h / 2)
            ref_x = t.get('ref_x', pnt_x)
            ref_y = t.get('ref_y', pnt_y)
            rotation = t.get('rotation', 0)
            scale = t.get('scale', 1)

            # M_offset: translate tracking point to origin
            M_offset = np.float32([
                [1, 0, -pnt_x],
                [0, 1, -pnt_y],
                [0, 0, 1]
            ])

            # M_rot: rotate and scale around origin
            rot_deg = -math.degrees(rotation)
            M_rot = cv2.getRotationMatrix2D((0, 0), rot_deg, scale)
            M_rot = np.vstack([M_rot, [0, 0, 1]])

            # M_place: translate to reference position + padding
            M_place = np.float32([
                [1, 0, ref_x + x_pad / 2 + x_off],
                [0, 1, ref_y + y_pad / 2 + y_off],
                [0, 0, 1]
            ])

            # Combine: M = M_place @ M_rot @ M_offset
            M = np.matmul(M_rot, M_offset)
            M = np.matmul(M_place, M)
            M = M[:2]  # Take 2x3 for warpAffine

        transformed = cv2.warpAffine(frame, M, (out_w, out_h))

        # Also create mask for alpha accumulation
        mask = np.ones((in_h, in_w), dtype=np.uint8) * 255
        mask_warped = cv2.warpAffine(mask, M, (out_w, out_h))
        data['mask'] = mask_warped

        state['frame_count'] += 1
        return transformed, data

    def _stream_frame_average(self, frame, data, state, params, inputs):
        """Accumulate one frame into average."""
        import numpy as np

        if state['accumulator'] is None:
            state['accumulator'] = np.zeros(frame.shape, dtype=np.float64)
            state['alpha_accumulator'] = np.zeros(frame.shape[:2], dtype=np.float64)

        state['accumulator'] += frame.astype(np.float64)

        # Use mask if provided (from data['mask'] or inputs['_mask']), otherwise compute from luminance
        mask = data.get('mask') if data.get('mask') is not None else inputs.get('_mask')
        if mask is not None:
            # Handle multi-channel mask (take first channel)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            state['alpha_accumulator'] += mask.astype(np.float64) / 255.0
        else:
            frame_f = frame.astype(np.float32) / 255.0
            alpha = np.power(np.clip(frame_f.mean(axis=2) * 255 / 16.0, 0, 1), 4)
            state['alpha_accumulator'] += alpha

        state['frame_count'] += 1
        return None, data  # No frame output during streaming

    def _finalize_frame_average(self, state, params):
        """Finalize frame average and return result."""
        import numpy as np

        if state['frame_count'] == 0:
            raise ValueError("No frames accumulated")

        result = state['accumulator'] / state['frame_count']
        alpha = state['alpha_accumulator'] / state['frame_count']

        comp_mode = params.get('comp_mode', 'on_black')
        brightness = params.get('brightness', 1.0)

        result *= brightness

        if comp_mode == 'on_white':
            alpha_3ch = np.dstack([alpha, alpha, alpha])
            white = np.ones_like(result) * 255
            result = result * alpha_3ch + white * (1 - alpha_3ch)
        elif comp_mode == 'unpremult':
            alpha_3ch = np.dstack([alpha, alpha, alpha])
            alpha_3ch = np.clip(alpha_3ch, 0.001, 1.0)
            result = result / alpha_3ch

        result = np.clip(result, 0, 255).astype(np.uint8)
        alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)

        return result, alpha

    def _stream_video_output(self, frame, data, state, params, is_first):
        """Write one frame to video output."""
        import cv2
        from pathlib import Path

        if is_first:
            filepath = self._resolve_path(params.get('filepath', 'output.mp4').strip())
            # Ensure parent directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            fps = data['props'].fps if data.get('props') else 30
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            state['writer'] = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
            state['filepath'] = filepath

        if state.get('writer'):
            state['writer'].write(frame)

        state['frame_count'] += 1

    def _finalize_video_output(self, state):
        """Close video writer."""
        if state.get('writer'):
            state['writer'].release()
            # Refresh file cache
            refresh_file_cache()

    def _stream_clahe(self, frame, data, state, params):
        """Apply CLAHE to one frame."""
        import cv2

        clip = params.get('clip_limit', 40.0)
        grid = params.get('grid_size', 8)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))

        channels = cv2.split(frame)
        enhanced = [clahe.apply(ch) for ch in channels]
        result = cv2.merge(enhanced)

        state['frame_count'] += 1
        return result, data

    def _stream_gamma(self, frame, data, state, params, inputs):
        """Apply gamma correction to one frame (not to mask)."""
        import numpy as np

        mode = params.get('mode', 'to_linear')
        gamma = params.get('gamma', 2.2)

        # Determine the gamma exponent based on mode
        if mode == 'to_linear':
            # sRGB to linear: raise to power of gamma
            exponent = gamma
        else:
            # linear to sRGB: raise to power of 1/gamma
            exponent = 1.0 / gamma

        # Apply gamma to video frame only (not mask)
        # Normalize to 0-1, apply gamma, scale back
        if frame.dtype == np.uint8:
            max_val = 255.0
        elif frame.dtype == np.uint16:
            max_val = 65535.0
        else:
            max_val = 1.0

        frame_float = frame.astype(np.float32) / max_val
        frame_corrected = np.power(frame_float, exponent)
        frame_corrected = np.clip(frame_corrected * max_val, 0, max_val).astype(frame.dtype)

        # Pass through mask unchanged (it's already linear)
        mask = data.get('mask')
        if mask is not None:
            data['mask'] = mask  # unchanged

        state['frame_count'] += 1
        return frame_corrected, data

    def _stream_rgb_multiply(self, frame, data, state, params):
        """Apply RGB channel multipliers to one frame."""
        import numpy as np

        if state.get('frame_count') is None:
            state['frame_count'] = 0

        red = params.get('red', 1.0)
        green = params.get('green', 1.0)
        blue = params.get('blue', 1.0)

        # Normalize to 0-1, apply multipliers, scale back
        if frame.dtype == np.uint8:
            max_val = 255.0
        elif frame.dtype == np.uint16:
            max_val = 65535.0
        else:
            max_val = 1.0

        frame_float = frame.astype(np.float32) / max_val

        # Apply multipliers per channel (BGR order from OpenCV)
        frame_float[:, :, 0] *= blue
        frame_float[:, :, 1] *= green
        frame_float[:, :, 2] *= red

        frame_corrected = np.clip(frame_float * max_val, 0, max_val).astype(frame.dtype)

        state['frame_count'] += 1
        return frame_corrected, data

    def _get_chain_dependencies(self, chain: dict, nodes: dict, edges: list) -> set:
        """Get all non-video dependencies for a streaming chain."""
        chain_nodes = chain.get('nodes', [])
        chain_set = set(chain_nodes)
        dependencies = set()

        for nid in chain_nodes:
            for e in edges:
                if e['target'] == nid:
                    src = e['source']
                    # Skip dependencies within the chain (video connections)
                    if src not in chain_set:
                        dependencies.add(src)

        return dependencies

    def execute(self, graph: dict) -> dict:
        import time
        exec_start = time.time()

        # Extract project directory
        self.project_dir = graph.get('projectDir', '').strip() or None
        if self.project_dir:
            self._log(f"Project directory: {self.project_dir}")

        nodes = {n['id']: n for n in graph.get('nodes', [])}
        edges = graph.get('edges', [])

        # Build dependency graph
        deps = {nid: set() for nid in nodes}
        for e in edges:
            deps[e['target']].add(e['source'])

        # Topological sort
        order = []
        visited = set()
        def visit(nid):
            if nid in visited:
                return
            visited.add(nid)
            for dep in deps[nid]:
                visit(dep)
            order.append(nid)
        for nid in nodes:
            visit(nid)

        # Find streaming chains
        streaming_chains = self._find_streaming_chains(nodes, edges)

        # Track which nodes belong to which chains (a node can be in multiple chains if it's the start)
        node_to_chains = {}  # node_id -> list of chain indices
        for i, chain in enumerate(streaming_chains):
            for nid in chain['nodes']:
                if nid not in node_to_chains:
                    node_to_chains[nid] = []
                node_to_chains[nid].append(i)

        # Calculate dependencies for each chain (by index)
        chain_deps = {}
        for i, chain in enumerate(streaming_chains):
            chain_deps[i] = self._get_chain_dependencies(chain, nodes, edges)

        self._log(f"Found {len(streaming_chains)} streaming chain(s)")
        for i, chain in enumerate(streaming_chains):
            chain_desc = self._describe_chain(chain, nodes)
            self._log(f"  Chain {i}: {chain_desc}")

        # Pre-populate video_input outputs with props (needed by stabilizer etc. before streaming runs)
        from vdg.core.video import get_video_properties
        from pathlib import Path
        video_inputs_loaded = set()
        for chain in streaming_chains:
            root_id = chain['root']
            source_node = nodes[root_id]
            source_type = self._get_node_type(source_node)
            if source_type == 'video_input' and root_id not in video_inputs_loaded:
                params = source_node.get('data', {}).get('params', {}) or source_node.get('params', {})
                filepath = self._resolve_path(params.get('filepath', '').strip())
                if filepath and Path(filepath).exists():
                    props = get_video_properties(filepath)
                    first_frame = params.get('first_frame', 1)
                    last_frame = params.get('last_frame', -1)
                    last_frame = None if last_frame == -1 else last_frame
                    self.outputs[root_id] = {
                        'video_out': {'filepath': filepath, 'first_frame': first_frame, 'last_frame': last_frame, 'props': props},
                        'props': props
                    }
                    video_inputs_loaded.add(root_id)
                    self._log(f"  Pre-loaded props for {root_id}: {props.width}x{props.height}")

        errors = []

        # Execute in order, using streaming for chains
        executed = set()
        executed_chains = set()  # Track which chains have been executed by index
        pending_chains = set(range(len(streaming_chains)))  # Chain indices waiting to execute

        for nid in order:
            # Check for abort
            if self._check_abort():
                break

            if nid in executed:
                continue

            node = nodes[nid]
            ntype = self._get_node_type(node)

            # Check if this node starts any pending streaming chains
            chains_starting_here = [i for i in node_to_chains.get(nid, [])
                                    if i in pending_chains and streaming_chains[i]['root'] == nid]

            if chains_starting_here:
                # Try to execute each chain starting at this node
                for chain_idx in chains_starting_here:
                    chain = streaming_chains[chain_idx]
                    deps = chain_deps.get(chain_idx, set())
                    deps_satisfied = all(dep in executed for dep in deps)

                    if deps_satisfied:
                        self._execute_streaming_chain(chain, nodes, edges, errors)
                        executed.update(chain['nodes'])
                        executed_chains.add(chain_idx)
                        pending_chains.discard(chain_idx)

                # If any chains were executed starting here, continue to next node
                if nid in executed:
                    continue

            # Check if this node is part of any pending chain (not start)
            in_pending_chain = any(i in pending_chains and streaming_chains[i]['root'] != nid
                                   for i in node_to_chains.get(nid, []))
            if in_pending_chain:
                # Part of a chain but not the start - skip, will be handled by chain
                continue

            # Execute node normally
            params = node.get('data', {}).get('params', {}) or node.get('params', {})

            # Gather inputs
            inputs = {}
            for e in edges:
                if e['target'] == nid:
                    src_out = self.outputs.get(e['source'], {})
                    inputs[e['targetHandle']] = src_out.get(e['sourceHandle'])

            # Execute node
            try:
                handler = NODE_HANDLERS.get(ntype)
                if handler:
                    self._log(f"Executing {ntype} ({nid})...")
                    node_start = time.time()
                    result = handler(inputs, params, self)
                    node_elapsed = time.time() - node_start
                    self.outputs[nid] = result or {}
                    self._log(f"  ✓ {ntype} complete ({node_elapsed:.2f}s)")
                else:
                    self._log(f"  ⚠ No handler for {ntype}")
                    self.outputs[nid] = {}
            except Exception as ex:
                import traceback
                errors.append({'node_id': nid, 'type': ntype, 'error': str(ex)})
                self._log(f"  ✗ Error in {ntype}: {ex}")

            executed.add(nid)

            # After each node, check if any pending chains can now execute
            for chain_idx in list(pending_chains):
                chain = streaming_chains[chain_idx]
                deps = chain_deps.get(chain_idx, set())
                deps_satisfied = all(dep in executed for dep in deps)
                if deps_satisfied:
                    self._execute_streaming_chain(chain, nodes, edges, errors)
                    executed.update(chain['nodes'])
                    executed_chains.add(chain_idx)
                    pending_chains.discard(chain_idx)

        # Execute any remaining pending chains (shouldn't happen if deps are correct)
        for chain_idx in list(pending_chains):
            chain = streaming_chains[chain_idx]
            if chain['root'] not in executed:
                self._log(f"  ⚠ Executing deferred chain starting at {chain['root']}")
                self._execute_streaming_chain(chain, nodes, edges, errors)
                executed.update(chain['nodes'])

        total_elapsed = time.time() - exec_start
        self._log(f"\n{'─' * 40}")
        self._log(f"Total execution time: {total_elapsed:.2f}s")

        return {
            'success': len(errors) == 0 and not self.aborted,
            'aborted': self.aborted,
            'message': '\n'.join(self.log),
            'errors': errors
        }


# =============================================================================
# NODE HANDLERS - Actual execution logic
# =============================================================================

def handle_video_input(inputs: dict, params: dict, executor) -> dict:
    """Load video and return video reference (not all frames in memory)."""
    from vdg.core.video import VideoReader, get_video_properties
    from pathlib import Path

    filepath = executor._resolve_path(params.get('filepath', '').strip())
    if not filepath:
        raise ValueError("No video file specified")

    # Check if file exists
    if not Path(filepath).exists():
        raise ValueError(f"Video file not found: {filepath}")

    first = params.get('first_frame', 1)
    last = params.get('last_frame', -1)
    last = None if last == -1 else last

    executor._log(f"  Video: {filepath}")
    executor._log(f"  Frame range: {first} - {last or 'end'}")

    props = get_video_properties(filepath)
    executor._log(f"  Properties: {props.width}x{props.height}, {props.fps:.2f}fps, {props.frame_count} frames")

    # Return video reference - frames will be loaded on-demand by nodes that need them
    return {
        'video_out': {
            'filepath': filepath,
            'first_frame': first,
            'last_frame': last,
            'props': props,
            'frames': None,  # Frames loaded on-demand, not upfront
        },
        'props': props
    }


def handle_image_input(inputs: dict, params: dict, executor) -> dict:
    """Load an image file."""
    import cv2
    from pathlib import Path

    filepath = executor._resolve_path(params.get('filepath', '').strip())
    if not filepath:
        raise ValueError("No image file specified")

    path = Path(filepath)
    if not path.exists():
        raise ValueError(f"Image file not found: {filepath}")

    # Load image with OpenCV (supports PNG, TIFF, EXR with 16-bit)
    # Use IMREAD_UNCHANGED to preserve bit depth and channels
    image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Failed to load image: {filepath}")

    # Convert grayscale to RGB if needed
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        # Has alpha - strip it (just load RGB)
        image = image[:, :, :3]

    executor._log(f"  Loaded: {filepath}")
    executor._log(f"  Size: {image.shape[1]}x{image.shape[0]}, dtype={image.dtype}")

    return {'image_out': image}


def handle_roi(inputs: dict, params: dict, executor) -> dict:
    """Return ROI tuple."""
    roi = (params.get('x', 0), params.get('y', 0),
           params.get('width', 100), params.get('height', 100))
    executor._log(f"  ROI: {roi}")
    return {'roi': roi}


def handle_feature_tracker(inputs: dict, params: dict, executor) -> dict:
    """Track features in video - streams from disk."""
    from vdg.tracking import FeatureTracker
    from vdg.core.video import VideoReader
    import numpy as np
    import cv2

    video_data = inputs.get('video_in')
    if video_data is None:
        video_data = inputs.get('video_out')
    roi = inputs.get('roi')

    if video_data is None:
        raise ValueError("No video input")

    tracker = FeatureTracker(
        num_features=params.get('num_features', 30),
        min_distance=params.get('min_distance', 30),
        initial_roi=roi,
        enforce_bbox=params.get('enforce_bbox', True),
        win_size=params.get('win_size', 21),
        pyramid_levels=params.get('pyramid_levels', 3),
    )

    track_data = {}
    preview_writer = None
    preview_path = _ensure_video_extension(params.get('preview_path', '').strip())

    # Check if we have a video reference (streaming) or pre-loaded frames
    if isinstance(video_data, dict) and video_data.get('filepath'):
        # STREAMING MODE
        filepath = video_data['filepath']
        first_frame = video_data.get('first_frame', 1)
        last_frame = video_data.get('last_frame')
        use_hardware = video_data.get('use_hardware', False)

        executor._log(f"  Streaming from {filepath}")

        frame_count = 0
        with VideoReader(filepath, first_frame, last_frame, use_hardware=use_hardware) as reader:
            total_frames = reader.frame_count
            for frame_num, frame in reader:
                if frame_count == 0:
                    tracker.initialize(frame)
                    # Initialize preview writer
                    if preview_path:
                        h, w = frame.shape[:2]
                        fps = reader.properties.fps
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        preview_writer = cv2.VideoWriter(preview_path, fourcc, fps, (w, h))
                        executor._log(f"  Preview output: {preview_path}")
                else:
                    tracker.update(frame)

                if tracker.points is not None and len(tracker.points) > 0:
                    pts = tracker.points.reshape(-1, 2)
                    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                    track_data[frame_num] = {'x': float(cx), 'y': float(cy)}

                # Write preview frame
                if preview_writer:
                    preview = frame.copy()
                    if tracker.points is not None:
                        for pt in tracker.points.reshape(-1, 2):
                            cv2.circle(preview, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
                        pts = tracker.points.reshape(-1, 2)
                        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                        cv2.circle(preview, (cx, cy), 8, (0, 255, 255), 2)
                        cv2.drawMarker(preview, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
                    if tracker.current_roi:
                        rx, ry, rw, rh = [int(v) for v in tracker.current_roi]
                        cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
                    if roi:
                        rx, ry, rw, rh = [int(v) for v in roi]
                        cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 1)
                    preview_writer.write(preview)

                frame_count += 1
                if frame_count % 100 == 0:
                    executor._log(f"  Tracked {frame_count}/{total_frames} frames...")

        if preview_writer:
            preview_writer.release()
            executor._log(f"  Preview video saved")

        executor._log(f"  Tracked {frame_count} frames, {len(track_data)} valid points")

    elif isinstance(video_data, dict) and video_data.get('frames'):
        # PRE-LOADED MODE (legacy)
        frames = video_data['frames']
        frame_nums = video_data.get('frame_nums', list(range(1, len(frames) + 1)))

        executor._log(f"  Processing {len(frames)} pre-loaded frames")

        for i, (frame_num, frame) in enumerate(zip(frame_nums, frames)):
            if i == 0:
                tracker.initialize(frame)
            else:
                tracker.update(frame)

            if tracker.points is not None and len(tracker.points) > 0:
                pts = tracker.points.reshape(-1, 2)
                cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                track_data[frame_num] = {'x': float(cx), 'y': float(cy)}

            if (i + 1) % 100 == 0:
                executor._log(f"  Tracked {i + 1} frames...")

        executor._log(f"  Tracked {len(frames)} frames, {len(track_data)} valid points")
    else:
        raise ValueError("No video frames available")

    return {'points': tracker.points, 'track_data': track_data}


def handle_feature_tracker_2p(inputs: dict, params: dict, executor) -> dict:
    """Track features in two ROIs simultaneously - single pass through video."""
    from vdg.tracking import FeatureTracker
    from vdg.core.video import VideoReader
    import numpy as np
    import cv2

    video_data = inputs.get('video_in')
    if video_data is None:
        video_data = inputs.get('video_out')
    roi1 = inputs.get('roi1')
    roi2 = inputs.get('roi2')

    if video_data is None:
        raise ValueError("No video input")

    # Create two trackers with the same parameters
    tracker_params = {
        'num_features': params.get('num_features', 30),
        'min_distance': params.get('min_distance', 30),
        'enforce_bbox': params.get('enforce_bbox', True),
        'win_size': params.get('win_size', 21),
        'pyramid_levels': params.get('pyramid_levels', 3),
    }

    tracker1 = FeatureTracker(initial_roi=roi1, **tracker_params)
    tracker2 = FeatureTracker(initial_roi=roi2, **tracker_params)

    track_data1 = {}
    track_data2 = {}
    preview_writer = None
    preview_path = _ensure_video_extension(params.get('preview_path', '').strip())

    # Check if we have a video reference (streaming) or pre-loaded frames
    if isinstance(video_data, dict) and video_data.get('filepath'):
        # STREAMING MODE
        filepath = video_data['filepath']
        first_frame = video_data.get('first_frame', 1)
        last_frame = video_data.get('last_frame')
        use_hardware = video_data.get('use_hardware', False)

        executor._log(f"  Streaming from {filepath}")
        executor._log(f"  ROI 1: {roi1}")
        executor._log(f"  ROI 2: {roi2}")

        frame_count = 0
        with VideoReader(filepath, first_frame, last_frame, use_hardware=use_hardware) as reader:
            total_frames = reader.frame_count
            for frame_num, frame in reader:
                if frame_count == 0:
                    tracker1.initialize(frame)
                    tracker2.initialize(frame)
                    # Initialize preview writer
                    if preview_path:
                        h, w = frame.shape[:2]
                        fps = reader.properties.fps
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        preview_writer = cv2.VideoWriter(preview_path, fourcc, fps, (w, h))
                        executor._log(f"  Preview output: {preview_path}")
                else:
                    tracker1.update(frame)
                    tracker2.update(frame)

                # Record track 1
                if tracker1.points is not None and len(tracker1.points) > 0:
                    pts = tracker1.points.reshape(-1, 2)
                    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                    track_data1[frame_num] = {'x': float(cx), 'y': float(cy)}

                # Record track 2
                if tracker2.points is not None and len(tracker2.points) > 0:
                    pts = tracker2.points.reshape(-1, 2)
                    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                    track_data2[frame_num] = {'x': float(cx), 'y': float(cy)}

                # Write preview frame
                if preview_writer:
                    preview = frame.copy()
                    # Draw tracker 1 (green)
                    if tracker1.points is not None:
                        for pt in tracker1.points.reshape(-1, 2):
                            cv2.circle(preview, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
                        pts = tracker1.points.reshape(-1, 2)
                        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                        cv2.circle(preview, (cx, cy), 8, (0, 255, 0), 2)
                        cv2.drawMarker(preview, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
                    if tracker1.current_roi:
                        rx, ry, rw, rh = [int(v) for v in tracker1.current_roi]
                        cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
                    if roi1:
                        rx, ry, rw, rh = [int(v) for v in roi1]
                        cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (0, 200, 0), 1)

                    # Draw tracker 2 (blue)
                    if tracker2.points is not None:
                        for pt in tracker2.points.reshape(-1, 2):
                            cv2.circle(preview, (int(pt[0]), int(pt[1])), 4, (255, 0, 0), -1)
                        pts = tracker2.points.reshape(-1, 2)
                        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                        cv2.circle(preview, (cx, cy), 8, (255, 0, 0), 2)
                        cv2.drawMarker(preview, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 16, 2)
                    if tracker2.current_roi:
                        rx, ry, rw, rh = [int(v) for v in tracker2.current_roi]
                        cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
                    if roi2:
                        rx, ry, rw, rh = [int(v) for v in roi2]
                        cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (200, 0, 0), 1)

                    preview_writer.write(preview)

                frame_count += 1
                if frame_count % 100 == 0:
                    executor._log(f"  Tracked {frame_count}/{total_frames} frames...")

        if preview_writer:
            preview_writer.release()
            executor._log(f"  Preview video saved")

        executor._log(f"  Tracked {frame_count} frames")
        executor._log(f"  Track 1: {len(track_data1)} valid points")
        executor._log(f"  Track 2: {len(track_data2)} valid points")

    elif isinstance(video_data, dict) and video_data.get('frames'):
        # PRE-LOADED MODE (legacy)
        frames = video_data['frames']
        frame_nums = video_data.get('frame_nums', list(range(1, len(frames) + 1)))

        executor._log(f"  Processing {len(frames)} pre-loaded frames")

        for i, (frame_num, frame) in enumerate(zip(frame_nums, frames)):
            if i == 0:
                tracker1.initialize(frame)
                tracker2.initialize(frame)
            else:
                tracker1.update(frame)
                tracker2.update(frame)

            if tracker1.points is not None and len(tracker1.points) > 0:
                pts = tracker1.points.reshape(-1, 2)
                cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                track_data1[frame_num] = {'x': float(cx), 'y': float(cy)}

            if tracker2.points is not None and len(tracker2.points) > 0:
                pts = tracker2.points.reshape(-1, 2)
                cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                track_data2[frame_num] = {'x': float(cx), 'y': float(cy)}

            if (i + 1) % 100 == 0:
                executor._log(f"  Tracked {i + 1} frames...")

        executor._log(f"  Tracked {len(frames)} frames")
        executor._log(f"  Track 1: {len(track_data1)} valid points")
        executor._log(f"  Track 2: {len(track_data2)} valid points")
    else:
        raise ValueError("No video frames available")

    return {'track_data1': track_data1, 'track_data2': track_data2}


def handle_stabilizer(inputs: dict, params: dict, executor) -> dict:
    """Compute stabilization transforms from track data."""
    track1 = inputs.get('track1', {})
    track2 = inputs.get('track2', {})
    props_input = inputs.get('props')

    if not track1:
        raise ValueError("No track data for track1")

    mode = params.get('mode', 'two_point')
    ref_frame = params.get('ref_frame', -1)
    swap_xy = params.get('swap_xy', False)
    x_flip = params.get('x_flip', False)
    y_flip = params.get('y_flip', False)

    # props can be either a VideoProperties object or a video_out dict
    # If it's a dict (video_out), extract props and frame range from it
    video_first_frame = None
    video_last_frame = None
    props = props_input

    if isinstance(props_input, dict):
        # It's a video_out dict - extract frame range and props
        video_first_frame = props_input.get('first_frame')
        video_last_frame = props_input.get('last_frame')
        props = props_input.get('props')

    # Get video dimensions for denormalization
    if props and hasattr(props, 'width'):
        width, height = props.width, props.height
    else:
        width, height = 1920, 1080
        executor._log(f"  Warning: No video props, assuming {width}x{height}")

    portrait = height > width

    # Helper to get pixel coordinates from track point
    # Matches orient_coordinates() in vdg/tracking/stabilizer.py
    def get_pixel_coords(pt):
        raw_x, raw_y = pt['x'], pt['y']

        if pt.get('normalized', False):
            # Apply orientation transform (Blender coord system → OpenCV)
            if portrait:
                # Video rotated 90deg CCW in Blender
                x = raw_y
                y = raw_x
            else:
                # Landscape: default y-flip (Blender uses bottom-left origin)
                x = raw_x
                y = 1.0 - raw_y

            # Swap XY if requested
            if swap_xy:
                x, y = y, x

            # Flip coordinates (in normalized space)
            if x_flip:
                x = 1.0 - x
            if y_flip:
                y = 1.0 - y

            # Scale to pixel coordinates
            x = x * width
            y = y * height
        else:
            # Already pixel coordinates, just apply flips
            x, y = raw_x, raw_y
            if swap_xy:
                x, y = y, x
            if x_flip:
                x = width - x
            if y_flip:
                y = height - y

        return x, y

    # Find common frames
    if track2:
        frames = sorted(set(track1.keys()) & set(track2.keys()))
    else:
        frames = sorted(track1.keys())

    if not frames:
        raise ValueError("No valid frames in track data")

    # Filter frames to video's frame range if specified
    original_frame_count = len(frames)
    if video_first_frame is not None:
        frames = [f for f in frames if f >= video_first_frame]
    if video_last_frame is not None:
        frames = [f for f in frames if f <= video_last_frame]

    if not frames:
        raise ValueError(f"No track frames within video range ({video_first_frame}-{video_last_frame})")

    if len(frames) != original_frame_count:
        executor._log(f"  Filtered tracks to video range: {frames[0]}-{frames[-1]} ({len(frames)} of {original_frame_count} frames)")

    if ref_frame == -1:
        ref_frame = frames[0]
    elif ref_frame not in frames:
        # Clamp ref_frame to valid range if it's outside the filtered frames
        old_ref = ref_frame
        ref_frame = min(frames, key=lambda f: abs(f - ref_frame))
        executor._log(f"  Warning: ref_frame {old_ref} not in range, using {ref_frame}")

    executor._log(f"  Mode: {mode}, ref_frame: {ref_frame}, {len(frames)} frames")

    # Get reference positions (in pixels)
    ref1_pt = track1.get(ref_frame, track1[frames[0]])
    ref1_x, ref1_y = get_pixel_coords(ref1_pt)
    ref2_x, ref2_y = None, None
    if track2:
        ref2_pt = track2.get(ref_frame, track2[frames[0]])
        ref2_x, ref2_y = get_pixel_coords(ref2_pt)

    transforms = {}
    for f in frames:
        p1_x, p1_y = get_pixel_coords(track1[f])
        p2_x, p2_y = get_pixel_coords(track2[f]) if track2 else (None, None)

        # Compute transform based on mode
        # Store tracking point and reference positions for proper rotation center
        if mode == 'single' or p2_x is None:
            # Translation only
            transforms[f] = {
                'pnt_x': p1_x, 'pnt_y': p1_y,
                'ref_x': ref1_x, 'ref_y': ref1_y,
                'rotation': 0, 'scale': 1
            }
        else:
            # Two-point: translation + rotation + scale
            import math

            # Current vector
            vx = p2_x - p1_x
            vy = p2_y - p1_y
            # Reference vector
            rvx = ref2_x - ref1_x
            rvy = ref2_y - ref1_y

            # Scale
            curr_len = math.sqrt(vx*vx + vy*vy)
            ref_len = math.sqrt(rvx*rvx + rvy*rvy)
            scale = ref_len / curr_len if curr_len > 0 else 1

            # Rotation
            curr_angle = math.atan2(vy, vx)
            ref_angle = math.atan2(rvy, rvx)
            rotation = ref_angle - curr_angle

            transforms[f] = {
                'pnt_x': p1_x, 'pnt_y': p1_y,
                'ref_x': ref1_x, 'ref_y': ref1_y,
                'rotation': rotation, 'scale': scale
            }

    # Bundle frames with transforms so apply_transform can access the frame range
    return {'transforms': {'transforms': transforms, 'frames': frames}, 'props': props}


def handle_apply_transform(inputs: dict, params: dict, executor) -> dict:
    """Apply transforms to video frames - streams input, stores output."""
    from vdg.core.video import VideoReader
    import cv2
    import numpy as np
    import math

    video_data = inputs.get('video_in')
    transform_data = inputs.get('transforms', {})
    transforms = transform_data.get('transforms', {})

    if video_data is None:
        raise ValueError("No video input - connect to video_in port")

    x_pad = params.get('x_pad', 0)
    y_pad = params.get('y_pad', 0)
    x_off = params.get('x_offset', 0)
    y_off = params.get('y_offset', 0)

    def build_transform_matrix(t, in_w, in_h, x_pad, y_pad, x_off, y_off):
        """Build transform matrix matching original stabilizer.py."""
        out_w = in_w + x_pad
        out_h = in_h + y_pad

        if t is None:
            return np.float32([[1, 0, x_pad / 2], [0, 1, y_pad / 2]]), out_w, out_h

        pnt_x = t.get('pnt_x', in_w / 2)
        pnt_y = t.get('pnt_y', in_h / 2)
        ref_x = t.get('ref_x', pnt_x)
        ref_y = t.get('ref_y', pnt_y)
        rotation = t.get('rotation', 0)
        scale = t.get('scale', 1)

        # M_offset: translate tracking point to origin
        M_offset = np.float32([
            [1, 0, -pnt_x],
            [0, 1, -pnt_y],
            [0, 0, 1]
        ])

        # M_rot: rotate and scale around origin
        rot_deg = -math.degrees(rotation)
        M_rot = cv2.getRotationMatrix2D((0, 0), rot_deg, scale)
        M_rot = np.vstack([M_rot, [0, 0, 1]])

        # M_place: translate to reference position + padding
        M_place = np.float32([
            [1, 0, ref_x + x_pad / 2 + x_off],
            [0, 1, ref_y + y_pad / 2 + y_off],
            [0, 0, 1]
        ])

        # Combine: M = M_place @ M_rot @ M_offset
        M = np.matmul(M_rot, M_offset)
        M = np.matmul(M_place, M)
        return M[:2], out_w, out_h

    stabilized_frames = []
    mask_frames = []

    # Check if we have a video reference (streaming) or pre-loaded frames
    if isinstance(video_data, dict) and video_data.get('filepath'):
        # STREAMING MODE
        filepath = video_data['filepath']
        first_frame = video_data.get('first_frame', 1)
        last_frame = video_data.get('last_frame')
        props = video_data.get('props')
        use_hardware = video_data.get('use_hardware', False)

        executor._log(f"  Streaming from {filepath}")

        in_w, in_h = props.width, props.height
        out_w = in_w + x_pad
        out_h = in_h + y_pad
        executor._log(f"  Input: {in_w}x{in_h}, Output: {out_w}x{out_h}")

        frame_count = 0
        with VideoReader(filepath, first_frame, last_frame, use_hardware=use_hardware) as reader:
            total_frames = reader.frame_count
            for frame_num, frame in reader:
                t = transforms.get(frame_num)
                M, out_w, out_h = build_transform_matrix(t, in_w, in_h, x_pad, y_pad, x_off, y_off)

                # Apply transform
                stabilized = cv2.warpAffine(frame, M, (out_w, out_h))
                stabilized_frames.append(stabilized)

                # Create mask
                mask = np.ones((in_h, in_w), dtype=np.uint8) * 255
                mask_warped = cv2.warpAffine(mask, M, (out_w, out_h))
                mask_frames.append(mask_warped)

                frame_count += 1
                if frame_count % 100 == 0:
                    executor._log(f"  Transformed {frame_count}/{total_frames} frames...")

        executor._log(f"  Stabilized {frame_count} frames")

    elif isinstance(video_data, dict) and video_data.get('frames'):
        # PRE-LOADED MODE (legacy dict format)
        frames = video_data['frames']
        frame_nums = video_data.get('frame_nums', list(range(1, len(frames) + 1)))
        props = video_data.get('props')

        in_h, in_w = frames[0].shape[:2]
        out_w = in_w + x_pad
        out_h = in_h + y_pad
        executor._log(f"  Input: {in_w}x{in_h}, Output: {out_w}x{out_h}")

        for i, (frame_num, frame) in enumerate(zip(frame_nums, frames)):
            t = transforms.get(frame_num)
            M, out_w, out_h = build_transform_matrix(t, in_w, in_h, x_pad, y_pad, x_off, y_off)

            stabilized = cv2.warpAffine(frame, M, (out_w, out_h))
            stabilized_frames.append(stabilized)

            mask = np.ones((in_h, in_w), dtype=np.uint8) * 255
            mask_warped = cv2.warpAffine(mask, M, (out_w, out_h))
            mask_frames.append(mask_warped)

            if (i + 1) % 100 == 0:
                executor._log(f"  Transformed {i + 1} frames...")

        executor._log(f"  Stabilized {len(stabilized_frames)} frames")
    elif isinstance(video_data, list) and len(video_data) > 0:
        # Plain list of frames (from gamma node, etc.)
        frames = video_data
        # Use transform frame numbers if available, otherwise 1-indexed
        transform_frame_data = inputs.get('transforms', {})
        transform_frames = transform_frame_data.get('frames', [])
        if transform_frames:
            frame_nums = transform_frames
        else:
            frame_nums = list(range(1, len(frames) + 1))

        in_h, in_w = frames[0].shape[:2]
        out_w = in_w + x_pad
        out_h = in_h + y_pad
        executor._log(f"  Input: {in_w}x{in_h}, Output: {out_w}x{out_h}")
        executor._log(f"  Processing {len(frames)} frames from list")

        for i, frame in enumerate(frames):
            frame_num = frame_nums[i] if i < len(frame_nums) else i + 1
            t = transforms.get(frame_num)
            M, out_w, out_h = build_transform_matrix(t, in_w, in_h, x_pad, y_pad, x_off, y_off)

            stabilized = cv2.warpAffine(frame, M, (out_w, out_h))
            stabilized_frames.append(stabilized)

            mask = np.ones((in_h, in_w), dtype=np.uint8) * 255
            mask_warped = cv2.warpAffine(mask, M, (out_w, out_h))
            mask_frames.append(mask_warped)

            if (i + 1) % 100 == 0:
                executor._log(f"  Transformed {i + 1}/{len(frames)} frames...")

        executor._log(f"  Stabilized {len(stabilized_frames)} frames")
    else:
        raise ValueError("No video frames available")

    return {'video_out': stabilized_frames, 'mask': mask_frames, 'width': out_w, 'height': out_h}


def handle_frame_average(inputs: dict, params: dict, executor) -> dict:
    """Average frames together - streams from disk to minimize memory usage."""
    from vdg.core.video import VideoReader
    import numpy as np

    video_data = inputs.get('video_in')
    if video_data is None:
        video_data = inputs.get('video_out')
    if video_data is None:
        video_data = {}
    masks = inputs.get('mask_in')
    if masks is None:
        masks = inputs.get('mask_out')
    if masks is None:
        masks = []

    comp_mode = params.get('comp_mode', 'on_black')
    brightness = params.get('brightness', 1.0)

    # Check if we have a video reference (streaming mode) or pre-loaded frames
    if isinstance(video_data, dict) and video_data.get('filepath'):
        # STREAMING MODE - read frames one at a time, don't store in memory
        filepath = video_data['filepath']
        first_frame = video_data.get('first_frame', 1)
        last_frame = video_data.get('last_frame')
        props = video_data.get('props')
        use_hardware = video_data.get('use_hardware', False)

        executor._log(f"  Streaming from {filepath}")
        executor._log(f"  Mode: {comp_mode}, brightness: {brightness}")

        # Initialize accumulator on first frame
        acc = None
        alpha_acc = None
        frame_count = 0

        with VideoReader(filepath, first_frame, last_frame, use_hardware=use_hardware) as reader:
            total_frames = reader.frame_count
            for frame_num, frame in reader:
                # Initialize accumulators on first frame
                if acc is None:
                    acc = np.zeros(frame.shape, dtype=np.float64)
                    alpha_acc = np.zeros(frame.shape[:2], dtype=np.float64)

                # Accumulate
                acc += frame.astype(np.float64)

                # Compute alpha from luminance
                frame_f = frame.astype(np.float32) / 255.0
                alpha = np.power(np.clip(frame_f.mean(axis=2) * 255 / 16.0, 0, 1), 4)
                alpha_acc += alpha

                frame_count += 1
                if frame_count % 100 == 0:
                    executor._log(f"  Processed {frame_count}/{total_frames} frames...")

        executor._log(f"  Processed {frame_count} frames total")

    elif isinstance(video_data, dict) and video_data.get('frames'):
        # PRE-LOADED MODE - frames already in memory (legacy support)
        frames = video_data['frames']
        executor._log(f"  Processing {len(frames)} pre-loaded frames")

        acc = np.zeros(frames[0].shape, dtype=np.float64)
        alpha_acc = np.zeros(frames[0].shape[:2], dtype=np.float64)

        for i, frame in enumerate(frames):
            acc += frame.astype(np.float64)
            if masks and i < len(masks):
                alpha_acc += masks[i].astype(np.float64) / 255.0

        frame_count = len(frames)

    elif isinstance(video_data, list):
        # LIST MODE - list of frames passed directly
        frames = video_data
        executor._log(f"  Processing {len(frames)} frames from list")

        acc = np.zeros(frames[0].shape, dtype=np.float64)
        alpha_acc = np.zeros(frames[0].shape[:2], dtype=np.float64)

        for i, frame in enumerate(frames):
            acc += frame.astype(np.float64)
            if masks and i < len(masks):
                alpha_acc += masks[i].astype(np.float64) / 255.0

        frame_count = len(frames)
    else:
        raise ValueError("No video input - connect a Video Input node")

    if frame_count == 0:
        raise ValueError("No frames to average")

    # Average
    result = acc / frame_count
    alpha = alpha_acc / frame_count

    # Apply brightness
    result *= brightness

    # Composite mode
    if comp_mode == 'on_white':
        alpha_3ch = np.dstack([alpha, alpha, alpha])
        white = np.ones_like(result) * 255
        result = result * alpha_3ch + white * (1 - alpha_3ch)
    elif comp_mode == 'unpremult':
        alpha_3ch = np.dstack([alpha, alpha, alpha])
        alpha_3ch = np.clip(alpha_3ch, 0.001, 1.0)  # Avoid division by zero
        result = result / alpha_3ch

    # Output as float32 in 0-1 range to preserve precision for downstream processing
    result = np.clip(result / 255.0, 0, 1).astype(np.float32)
    alpha = np.clip(alpha, 0, 1).astype(np.float32)

    executor._log(f"  Output: {result.shape[1]}x{result.shape[0]}, float32 (0-1 range)")

    return {'image_out': result, 'alpha_out': alpha}


def handle_image_output(inputs: dict, params: dict, executor) -> dict:
    """Save image to file."""
    import cv2
    import numpy as np
    from pathlib import Path

    image = inputs.get('image_in')
    if image is None:
        image = inputs.get('image_out')
    if image is None:
        raise ValueError("No image to save")

    filepath = executor._resolve_path(params.get('filepath', '').strip() or 'output.png')

    # Ensure parent directory exists
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure valid extension
    if not path.suffix or path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.exr']:
        filepath = str(path.with_suffix('.png'))
        executor._log(f"  Warning: Added .png extension: {filepath}")

    bit_depth = params.get('bit_depth', '16')

    # Convert from float32 (0-1 range) to integer format for saving
    if image.dtype == np.float32 or image.dtype == np.float64:
        if bit_depth == '16':
            image = np.clip(image * 65535, 0, 65535).astype(np.uint16)
        else:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
    elif bit_depth == '16' and image.dtype == np.uint8:
        # Convert 8-bit to 16-bit
        image = (image.astype(np.uint16) * 257)  # Scale 0-255 to 0-65535

    success = cv2.imwrite(filepath, image)
    if not success:
        raise ValueError(f"Failed to write image to {filepath}")

    executor._log(f"  Saved {filepath} ({bit_depth}-bit, {image.shape[1]}x{image.shape[0]})")

    # Refresh file cache
    refresh_file_cache()

    return {'filepath': filepath}


def handle_video_output(inputs: dict, params: dict, executor) -> dict:
    """Save video to file."""
    import cv2

    frames = inputs.get('video_in')
    if frames is None:
        frames = inputs.get('video_out')
    if frames is None:
        frames = []
    props = inputs.get('props')

    # Handle different input formats
    if isinstance(frames, dict) and 'frames' in frames:
        actual_frames = frames['frames']
        props = frames.get('props', props)
    elif isinstance(frames, list):
        actual_frames = frames
    else:
        raise ValueError("No frames to save")

    if not actual_frames:
        raise ValueError("No frames to save")

    filepath = executor._resolve_path(params.get('filepath', 'output.mp4').strip())

    # Ensure parent directory exists
    from pathlib import Path
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Get dimensions from first frame
    h, w = actual_frames[0].shape[:2]
    fps = props.fps if props and hasattr(props, 'fps') else 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))

    for frame in actual_frames:
        writer.write(frame)

    writer.release()
    executor._log(f"  Saved {filepath} ({len(actual_frames)} frames, {w}x{h})")

    # Refresh file cache
    refresh_file_cache()

    return {'filepath': filepath}


def handle_track_input(inputs: dict, params: dict, executor) -> dict:
    """Load track data from a .crv file."""
    from vdg.tracking.track_io import read_crv_file
    from pathlib import Path

    filepath = executor._resolve_path(params.get('filepath', '').strip())
    if not filepath:
        raise ValueError("No track file specified")

    if not Path(filepath).exists():
        raise ValueError(f"Track file not found: {filepath}")

    executor._log(f"  Loading track file: {filepath}")

    # Read the CRV file - returns dict of frame -> (x, y) with normalized coords
    raw_data = read_crv_file(filepath)

    # Convert to the format used by other nodes: dict of frame -> {'x': x, 'y': y}
    # Note: CRV files typically store normalized coordinates (0-1)
    # We need to convert them to pixel coordinates if we know the video dimensions
    # For now, assume they need to be scaled by video dimensions later
    track_data = {}
    for frame, (x, y) in raw_data.items():
        track_data[frame] = {'x': x, 'y': y, 'normalized': True}

    executor._log(f"  Loaded {len(track_data)} frames of track data")

    return {'track_data': track_data}


def handle_track_output(inputs: dict, params: dict, executor) -> dict:
    """Save track data to .crv file."""
    track_data = inputs.get('track_data', {})
    props = inputs.get('props')

    if not track_data:
        raise ValueError("No track data to save")

    filepath = executor._resolve_path(params.get('filepath', 'track.crv').strip())

    # Ensure parent directory exists
    from pathlib import Path
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Get dimensions from props
    if props and hasattr(props, 'width'):
        w, h = props.width, props.height
    else:
        w, h = 1920, 1080

    with open(filepath, 'w') as f:
        for frame_num in sorted(track_data.keys()):
            pt = track_data[frame_num]
            x_norm = pt['x'] / w
            y_norm = pt['y'] / h
            f.write(f"{frame_num} [[ {x_norm}, {y_norm}]]\n")

    executor._log(f"  Saved {filepath} ({len(track_data)} frames)")

    # Refresh file cache
    refresh_file_cache()

    return {'filepath': filepath}


def handle_clahe(inputs: dict, params: dict, executor) -> dict:
    """Apply CLAHE contrast enhancement."""
    import cv2

    image = inputs.get('image_in')
    if image is None:
        image = inputs.get('image_out')
    if image is None:
        raise ValueError("No image input")

    clip = params.get('clip_limit', 40.0)
    grid = params.get('grid_size', 8)

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))

    # Apply to each channel
    channels = cv2.split(image)
    enhanced = [clahe.apply(ch) for ch in channels]
    result = cv2.merge(enhanced)

    executor._log(f"  Applied CLAHE (clip={clip}, grid={grid})")

    return {'image_out': result}


def handle_gamma(inputs: dict, params: dict, executor) -> dict:
    """Apply gamma correction for linear/sRGB conversion."""
    import numpy as np
    from vdg.core.video import VideoReader

    video_in = inputs.get('video_in')
    image_in = inputs.get('image_in')
    mask_in = inputs.get('mask_in')

    # Accept either video or image input
    input_data = video_in if video_in is not None else image_in

    if input_data is None:
        raise ValueError("No video or image input")

    mode = params.get('mode', 'to_linear')
    gamma = params.get('gamma', 2.2)

    # Determine the gamma exponent based on mode
    if mode == 'to_linear':
        exponent = gamma
    else:
        exponent = 1.0 / gamma

    executor._log(f"  Gamma correction: {mode} (gamma={gamma})")

    def apply_gamma(frame, exponent):
        if frame.dtype == np.uint8:
            max_val = 255.0
        elif frame.dtype == np.uint16:
            max_val = 65535.0
        else:
            max_val = 1.0

        frame_float = frame.astype(np.float32) / max_val
        frame_corrected = np.power(frame_float, exponent)
        return np.clip(frame_corrected * max_val, 0, max_val).astype(frame.dtype)

    # Handle video reference (filepath with frames=None) - load frames on demand
    if isinstance(input_data, dict) and 'filepath' in input_data and input_data.get('frames') is None:
        filepath = input_data['filepath']
        first_frame = input_data.get('first_frame', 1)
        last_frame = input_data.get('last_frame')
        executor._log(f"  Loading video from {filepath}")

        corrected_frames = []
        with VideoReader(filepath, first_frame=first_frame, last_frame=last_frame) as reader:
            for frame_num, frame in reader:
                corrected_frames.append(apply_gamma(frame, exponent))

        executor._log(f"  Processed {len(corrected_frames)} frames")
        return {'video_out': corrected_frames, 'image_out': None, 'mask_out': mask_in}
    # Handle dict with frames list
    elif isinstance(input_data, dict) and 'frames' in input_data and input_data['frames'] is not None:
        frames = input_data['frames']
        corrected_frames = [apply_gamma(f, exponent) for f in frames]
        executor._log(f"  Processed {len(corrected_frames)} frames")
        return {'video_out': corrected_frames, 'image_out': None, 'mask_out': mask_in}
    elif isinstance(input_data, list):
        corrected_frames = [apply_gamma(f, exponent) for f in input_data]
        executor._log(f"  Processed {len(corrected_frames)} frames")
        return {'video_out': corrected_frames, 'image_out': None, 'mask_out': mask_in}
    elif isinstance(input_data, np.ndarray):
        # Single image
        corrected = apply_gamma(input_data, exponent)
        executor._log(f"  Processed single image")
        return {'video_out': None, 'image_out': corrected, 'mask_out': mask_in}
    else:
        raise ValueError("Unsupported input format")


def handle_rgb_multiply(inputs: dict, params: dict, executor) -> dict:
    """Multiply RGB channels individually for color correction."""
    import numpy as np
    from vdg.core.video import VideoReader

    video_in = inputs.get('video_in')
    image_in = inputs.get('image_in')

    # Accept either video or image input
    input_data = video_in if video_in is not None else image_in

    if input_data is None:
        raise ValueError("No video or image input")

    red = params.get('red', 1.0)
    green = params.get('green', 1.0)
    blue = params.get('blue', 1.0)

    executor._log(f"  RGB multiply: R={red}, G={green}, B={blue}")

    def apply_rgb_multiply(frame, r, g, b):
        """Apply RGB multipliers to a frame."""
        if frame.dtype == np.uint8:
            max_val = 255.0
        elif frame.dtype == np.uint16:
            max_val = 65535.0
        else:
            max_val = 1.0

        frame_float = frame.astype(np.float32) / max_val

        # Apply multipliers per channel (assuming BGR order from OpenCV)
        frame_float[:, :, 0] *= b  # Blue channel
        frame_float[:, :, 1] *= g  # Green channel
        frame_float[:, :, 2] *= r  # Red channel

        return np.clip(frame_float * max_val, 0, max_val).astype(frame.dtype)

    # Handle video reference (filepath with frames=None) - load frames on demand
    if isinstance(input_data, dict) and 'filepath' in input_data and input_data.get('frames') is None:
        filepath = input_data['filepath']
        first_frame = input_data.get('first_frame', 1)
        last_frame = input_data.get('last_frame')
        executor._log(f"  Loading video from {filepath}")

        corrected_frames = []
        with VideoReader(filepath, first_frame=first_frame, last_frame=last_frame) as reader:
            for frame_num, frame in reader:
                corrected_frames.append(apply_rgb_multiply(frame, red, green, blue))

        executor._log(f"  Processed {len(corrected_frames)} frames")
        return {'video_out': corrected_frames, 'image_out': None}
    # Handle dict with frames list
    elif isinstance(input_data, dict) and 'frames' in input_data and input_data['frames'] is not None:
        frames = input_data['frames']
        corrected_frames = [apply_rgb_multiply(f, red, green, blue) for f in frames]
        executor._log(f"  Processed {len(corrected_frames)} frames")
        return {'video_out': corrected_frames, 'image_out': None}
    elif isinstance(input_data, list):
        corrected_frames = [apply_rgb_multiply(f, red, green, blue) for f in input_data]
        executor._log(f"  Processed {len(corrected_frames)} frames")
        return {'video_out': corrected_frames, 'image_out': None}
    elif isinstance(input_data, np.ndarray):
        # Single image
        corrected = apply_rgb_multiply(input_data, red, green, blue)
        executor._log(f"  Processed single image")
        return {'video_out': None, 'image_out': corrected}
    else:
        raise ValueError("Unsupported input format")


def handle_post_process(inputs: dict, params: dict, executor) -> dict:
    """Apply post-processing operation to image + alpha."""
    from vdg.postprocess.operations import apply_operation, get_operations
    import numpy as np

    image = inputs.get('image_in')
    if image is None:
        image = inputs.get('image_out')
    alpha = inputs.get('alpha_in')
    if alpha is None:
        alpha = inputs.get('alpha_out')

    if image is None:
        raise ValueError("No image input provided")
    if alpha is None:
        raise ValueError("No alpha input provided")

    operation = params.get('operation', 'comp_on_white')
    available = get_operations()

    if operation not in available:
        raise ValueError(f"Unknown operation: {operation}. Available: {available}")

    # Collect additional params for the operation
    op_params = {
        'gamma': params.get('gamma', 2.2),
        'contrast': params.get('contrast', 60.0),
        'threshold': params.get('threshold', 0.0015),
        'blur_size': params.get('blur_size', 5.0),
        'power': params.get('power', 8.0),
    }

    executor._log(f"  Applying {operation}")
    result = apply_operation(operation, image, alpha, **op_params)
    executor._log(f"  Output: {result.shape[1]}x{result.shape[0]}, dtype={result.dtype}")

    # Apply trim if enabled
    if params.get('trim', False):
        # Normalize alpha to find content
        if alpha.dtype == np.uint16:
            alpha_norm = alpha.astype(np.float32) / 65535.0
        else:
            alpha_norm = alpha.astype(np.float32) / 255.0

        if alpha_norm.ndim == 3:
            alpha_1ch = alpha_norm[:, :, 0]
        else:
            alpha_1ch = alpha_norm

        # Find bounding box of non-black alpha
        content_mask = alpha_1ch > 0.001
        rows_with_content = np.any(content_mask, axis=1)
        cols_with_content = np.any(content_mask, axis=0)

        row_indices = np.where(rows_with_content)[0]
        col_indices = np.where(cols_with_content)[0]

        if len(row_indices) > 0 and len(col_indices) > 0:
            min_y, max_y = row_indices[0], row_indices[-1] + 1
            min_x, max_x = col_indices[0], col_indices[-1] + 1
            result = result[min_y:max_y, min_x:max_x].copy()
            executor._log(f"  Trimmed to: {result.shape[1]}x{result.shape[0]}")

    return {'image_out': result}


def handle_gaussian_filter(inputs: dict, params: dict, executor) -> dict:
    """Apply Gaussian smoothing to track data."""
    track_data = inputs.get('track_data', {})

    if not track_data:
        raise ValueError("No track data")

    sigma = params.get('sigma', 5.0)

    try:
        from scipy.ndimage import gaussian_filter1d
        import numpy as np

        frames = sorted(track_data.keys())
        x_vals = np.array([track_data[f]['x'] for f in frames])
        y_vals = np.array([track_data[f]['y'] for f in frames])

        x_smooth = gaussian_filter1d(x_vals, sigma=sigma)
        y_smooth = gaussian_filter1d(y_vals, sigma=sigma)

        result = {f: {'x': x_smooth[i], 'y': y_smooth[i]} for i, f in enumerate(frames)}
        executor._log(f"  Smoothed {len(frames)} points (sigma={sigma})")

        return {'track_data': result}
    except ImportError:
        executor._log(f"  Warning: scipy not installed, returning unfiltered data")
        return {'track_data': track_data}


# Register all handlers
NODE_HANDLERS = {
    'video_input': handle_video_input,
    'image_input': handle_image_input,
    'roi': handle_roi,
    'feature_tracker': handle_feature_tracker,
    'feature_tracker_2p': handle_feature_tracker_2p,
    'stabilizer': handle_stabilizer,
    'apply_transform': handle_apply_transform,
    'frame_average': handle_frame_average,
    'image_output': handle_image_output,
    'video_output': handle_video_output,
    'track_output': handle_track_output,
    'track_input': handle_track_input,
    'clahe': handle_clahe,
    'gamma': handle_gamma,
    'post_process': handle_post_process,
    'gaussian_filter': handle_gaussian_filter,
    'rgb_multiply': handle_rgb_multiply,
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VDG Node Editor - Web-based node editor for video processing")
    parser.add_argument('-d', '--directory', type=str, help='Project directory for file suggestions')
    parser.add_argument('-p', '--port', type=int, default=8000, help='Port to run server on (default: 8000)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    args = parser.parse_args()

    # Set initial project directory if provided
    if args.directory:
        from pathlib import Path
        dir_path = Path(args.directory).expanduser().resolve()
        if dir_path.is_dir():
            refresh_file_cache(str(dir_path))
            print(f"Project directory: {dir_path}")
        else:
            print(f"Warning: Directory not found: {args.directory}")
            refresh_file_cache()
    else:
        refresh_file_cache()

    print("=" * 50)
    print("VDG Node Editor")
    print(f"Open http://{args.host}:{args.port}")
    print("=" * 50)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
