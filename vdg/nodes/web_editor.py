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
        outputs=[NodePort("video", "video"), NodePort("props", "props")],
        params=[
            NodeParam("filepath", "file", ""),
            NodeParam("first_frame", "int", 1, min=1),
            NodeParam("last_frame", "int", -1),
            NodeParam("use_hardware", "bool", False),
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
        inputs=[NodePort("video", "video", optional=True)],
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
        inputs=[NodePort("video", "video"), NodePort("roi", "roi", optional=True)],
        outputs=[NodePort("points", "points"), NodePort("track_data", "track_data")],
        params=[
            NodeParam("num_features", "int", 30, min=1, max=200),
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
        inputs=[NodePort("video", "video"), NodePort("mask", "video", optional=True)],
        outputs=[NodePort("image", "image"), NodePort("alpha", "image")],
        params=[
            NodeParam("comp_mode", "choice", "on_black", choices=["on_black", "on_white", "unpremult"]),
            NodeParam("brightness", "float", 1.0, min=0.0, max=4.0),
        ],
        color="#FF9800",
    ),
    NodeDefinition(
        id="clahe", title="CLAHE", category="Processing",
        inputs=[NodePort("image", "image")],
        outputs=[NodePort("image", "image")],
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
        id="video_output", title="Video Output", category="Output",
        inputs=[NodePort("video", "video"), NodePort("props", "props")],
        outputs=[],
        params=[
            NodeParam("filepath", "string", "output.mp4"),
            NodeParam("use_hardware", "bool", True),
            NodeParam("bitrate", "string", "20M"),
        ],
        color="#9C27B0",
    ),
    NodeDefinition(
        id="image_output", title="Image Output", category="Output",
        inputs=[NodePort("image", "image")],
        outputs=[],
        params=[
            NodeParam("filepath", "string", "output.png"),
            NodeParam("bit_depth", "choice", "16", choices=["8", "16"]),
        ],
        color="#9C27B0",
    ),
    NodeDefinition(
        id="track_output", title="Track Output", category="Output",
        inputs=[NodePort("track_data", "track_data"), NodePort("props", "props")],
        outputs=[],
        params=[
            NodeParam("filepath", "string", "track.crv"),
            NodeParam("track_type", "choice", "2", choices=["1", "2"]),
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
        .container { display: flex; height: 100vh; }
        .sidebar { width: 200px; background: #16213e; padding: 12px; overflow-y: auto; border-right: 1px solid #333; }
        .sidebar h2 { font-size: 14px; margin-bottom: 12px; color: #4fc3f7; }
        .sidebar h3 { font-size: 10px; margin: 12px 0 6px; text-transform: uppercase; color: #666; }
        .node-btn { display: block; width: 100%; padding: 6px 10px; margin: 3px 0; background: #252545; color: #fff; border: 1px solid #333; border-radius: 4px; cursor: grab; text-align: left; font-size: 12px; }
        .node-btn:hover { background: #2d2d5a; border-color: #4fc3f7; }
        .canvas-area { flex: 1; position: relative; overflow: hidden; }
        #canvas { position: absolute; width: 5000px; height: 5000px; background-image: radial-gradient(#333 1px, transparent 1px); background-size: 20px 20px; z-index: 1; }
        #svg-layer { position: absolute; width: 5000px; height: 5000px; z-index: 2; pointer-events: none; }
        #svg-layer path { pointer-events: stroke; }
        .toolbar { position: absolute; top: 10px; right: 10px; display: flex; gap: 6px; z-index: 100; }
        .toolbar button { padding: 8px 14px; background: #4CAF50; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }
        .toolbar button:hover { filter: brightness(1.1); }
        .toolbar button.sec { background: #555; }
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
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar" id="sidebar"></div>
        <div class="canvas-area" id="canvas-area">
            <svg id="svg-layer"></svg>
            <div id="canvas"></div>
            <div class="toolbar">
                <button onclick="execute()">▶ Run</button>
                <button class="sec" onclick="save()">Save</button>
                <button class="sec" onclick="load()">Load</button>
                <button class="sec" onclick="clear()">Clear</button>
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
        if (t) addNode(t, e.clientX - area.getBoundingClientRect().left - pan.x, e.clientY - area.getBoundingClientRect().top - pan.y);
    };
    area.onmousedown = e => {
        if (e.target === area || e.target.id === 'canvas') {
            panning = true; panStart = {x: e.clientX - pan.x, y: e.clientY - pan.y}; desel();
        }
    };
    area.onmousemove = e => {
        if (panning) { pan.x = e.clientX - panStart.x; pan.y = e.clientY - panStart.y; updTrans(); }
        if (dragC) updTemp(e);
    };
    area.onmouseup = () => { panning = false; if (dragC) cancelConn(); };
}

function updTrans() {
    document.getElementById('canvas').style.transform = 'translate(' + pan.x + 'px,' + pan.y + 'px)';
    document.getElementById('svg-layer').style.transform = 'translate(' + pan.x + 'px,' + pan.y + 'px)';
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
        select(n); dragN = n; off = {x: e.clientX - n.x - pan.x, y: e.clientY - n.y - pan.y}; e.stopPropagation();
    };
    document.addEventListener('mousemove', e => {
        if (dragN === n) { n.x = e.clientX - off.x - pan.x; n.y = e.clientY - off.y - pan.y; el.style.left = n.x + 'px'; el.style.top = n.y + 'px'; drawConns(); }
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
    const x1 = r.left + 4 - ar.left - pan.x, y1 = r.top + 4 - ar.top - pan.y;
    const x2 = e.clientX - ar.left - pan.x, y2 = e.clientY - ar.top - pan.y;
    t.setAttribute('d', 'M' + x1 + ' ' + y1 + ' C' + (x1+50) + ' ' + y1 + ',' + (x2-50) + ' ' + y2 + ',' + x2 + ' ' + y2);
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
        const x1 = r1.left + 4 - ar.left - pan.x, y1 = r1.top + 4 - ar.top - pan.y;
        const x2 = r2.left + 4 - ar.left - pan.x, y2 = r2.top + 4 - ar.top - pan.y;
        const pathD = 'M' + x1 + ' ' + y1 + ' C' + (x1+60) + ' ' + y1 + ',' + (x2-60) + ' ' + y2 + ',' + x2 + ' ' + y2;

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

// ROI Picker
let roiNode = null, roiRect = null, roiDrawing = false, roiStart = null;
let origW = 0, origH = 0;

function pickROI(nodeId) {
    roiNode = nodes.find(n => n.id === nodeId);
    if (!roiNode) { alert('Node not found'); return; }

    // Find connected video input
    const conn = conns.find(c => c.tn === nodeId && c.tp === 'video');
    if (!conn) { alert('Connect a video input to the ROI node first'); return; }

    const srcNode = nodes.find(n => n.id === conn.sn);
    if (!srcNode || srcNode.type !== 'video_input') { alert('Source must be a video_input node'); return; }

    const filepath = srcNode.params.filepath;
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

async function execute() {
    status('Running...');
    const g = { nodes: nodes.map(n => ({id: n.id, type: n.type, data: {params: n.params}})), edges: conns.map(c => ({source: c.sn, sourceHandle: c.sp, target: c.tn, targetHandle: c.tp})) };
    try {
        const r = await fetch('/api/execute', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(g)});
        const j = await r.json();
        if (j.success) {
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
    }
}

function save() {
    const b = new Blob([JSON.stringify({nodes, conns}, null, 2)], {type: 'application/json'});
    const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = 'vdg-graph.json'; a.click();
    status('Saved');
}

function load() {
    const i = document.createElement('input'); i.type = 'file'; i.accept = '.json';
    i.onchange = e => {
        const r = new FileReader();
        r.onload = ev => {
            const d = JSON.parse(ev.target.result);
            clear();
            nodes = d.nodes || []; conns = d.conns || [];
            nid = Math.max(0, ...nodes.map(n => parseInt(n.id.slice(1)) || 0));
            nodes.forEach(render); drawConns();
            status('Loaded');
        };
        r.readAsText(e.target.files[0]);
    };
    i.click();
}

function clear() {
    nodes.forEach(n => document.getElementById(n.id)?.remove());
    nodes = []; conns = []; drawConns(); desel(); status('Cleared');
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


@app.post("/api/execute")
async def execute_graph(graph: dict):
    """Execute the node graph."""
    print("\n" + "=" * 50)
    print("EXECUTING GRAPH")
    print("=" * 50)
    try:
        executor = GraphExecutor()
        result = executor.execute(graph)
        if result['success']:
            print("\n✓ Execution completed successfully")
        else:
            print("\n✗ Execution failed")
        print("=" * 50 + "\n")
        return result
    except Exception as e:
        import traceback
        print(f"\n✗ Execution error: {e}")
        print("=" * 50 + "\n")
        return {'success': False, 'message': str(e), 'errors': [{'node_id': 'graph', 'error': traceback.format_exc()}]}


class GraphExecutor:
    """Executes a VDG node graph with streaming optimization."""

    # Nodes that can process frames in a streaming fashion
    STREAMABLE_NODES = {
        'video_input',      # Yields frames
        'feature_tracker',  # Processes frame-by-frame
        'apply_transform',  # Applies pre-computed transform per frame
        'frame_average',    # Accumulates without storing frames
        'clahe',            # Single-frame operation
        'gamma',            # Single-frame gamma correction
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

    def _log(self, message: str):
        """Add message to log and print to terminal."""
        self.log.append(message)
        print(message)

    def _get_node_type(self, node: dict) -> str:
        return node.get('type', '')

    def _find_streaming_chains(self, nodes: dict, edges: list) -> list:
        """
        Find chains of nodes that can be executed in streaming mode.

        Returns list of chains, where each chain is a list of node IDs
        that can stream frames through together.
        """
        # Build adjacency info
        outgoing = {nid: [] for nid in nodes}  # node -> [(target_node, src_port, tgt_port)]
        incoming = {nid: [] for nid in nodes}  # node -> [(source_node, src_port, tgt_port)]

        for e in edges:
            src, tgt = e['source'], e['target']
            outgoing[src].append((tgt, e['sourceHandle'], e['targetHandle']))
            incoming[tgt].append((src, e['sourceHandle'], e['targetHandle']))

        # Find video streaming chains
        # A chain starts at video_input and continues through streamable nodes
        # until hitting a non-streamable node or branching
        chains = []

        for nid, node in nodes.items():
            ntype = self._get_node_type(node)

            # Start chains from video_input
            if ntype == 'video_input':
                # Find all video outputs from this video_input
                video_outs = [
                    (tgt, sp, tp) for tgt, sp, tp in outgoing[nid]
                    if sp in ('video', 'video_out') and tp in ('video', 'video_in')
                ]

                # Trace each video output as a potential chain
                for start_node, _, _ in video_outs:
                    start_type = self._get_node_type(nodes[start_node])

                    # Skip non-streamable starting nodes
                    if start_type in self.NON_STREAMABLE_NODES:
                        continue
                    if start_type not in self.STREAMABLE_NODES:
                        continue

                    # Start a chain from video_input through this streamable node
                    chain = [nid, start_node]
                    current = start_node

                    while True:
                        # Find video output connections from current node
                        next_video_outs = [
                            (tgt, sp, tp) for tgt, sp, tp in outgoing[current]
                            if sp in ('video', 'video_out') and tp in ('video', 'video_in')
                        ]

                        if len(next_video_outs) != 1:
                            # Branching or end of chain
                            break

                        next_node, _, _ = next_video_outs[0]
                        next_type = self._get_node_type(nodes[next_node])

                        if next_type in self.NON_STREAMABLE_NODES:
                            # Hit a barrier - stop chain here
                            break

                        if next_type in self.STREAMABLE_NODES:
                            chain.append(next_node)
                            current = next_node
                        else:
                            break

                    if len(chain) > 1:
                        chains.append(chain)

        return chains

    def _execute_streaming_chain(self, chain: list, nodes: dict, edges: list, errors: list) -> None:
        """Execute a chain of nodes in streaming mode (frame by frame)."""
        from vdg.core.video import VideoReader
        import numpy as np

        if not chain:
            return

        self._log(f"\n>>> STREAMING PIPELINE: {' → '.join(self._get_node_type(nodes[nid]) for nid in chain)}")

        # Get video source info
        source_node = nodes[chain[0]]
        source_params = source_node.get('data', {}).get('params', {}) or source_node.get('params', {})
        filepath = source_params.get('filepath', '').strip()
        first_frame = source_params.get('first_frame', 1)
        last_frame = source_params.get('last_frame', -1)
        last_frame = None if last_frame == -1 else last_frame
        use_hardware = source_params.get('use_hardware', False)

        if not filepath:
            errors.append({'node_id': chain[0], 'type': 'video_input', 'error': 'No filepath specified'})
            return

        from pathlib import Path
        if not Path(filepath).exists():
            errors.append({'node_id': chain[0], 'type': 'video_input', 'error': f'File not found: {filepath}'})
            return

        from vdg.core.video import get_video_properties
        props = get_video_properties(filepath)
        self._log(f"  Source: {filepath}")
        self._log(f"  Video: {props.width}x{props.height}, {props.fps:.2f}fps")

        # Store video reference for non-streaming nodes that might need it
        self.outputs[chain[0]] = {
            'video': {'filepath': filepath, 'first_frame': first_frame, 'last_frame': last_frame, 'props': props, 'use_hardware': use_hardware},
            'props': props
        }

        # Prepare streaming state for each node in chain
        chain_state = {}
        transform_frames = None  # Track frame range from transforms if present

        for nid in chain[1:]:  # Skip video_input
            node = nodes[nid]
            ntype = self._get_node_type(node)
            params = node.get('data', {}).get('params', {}) or node.get('params', {})

            # Gather non-video inputs (transforms, ROI, etc.)
            non_video_inputs = {}
            for e in edges:
                if e['target'] == nid and e['targetHandle'] not in ('video', 'video_in'):
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
                'accumulator': None,
                'alpha_accumulator': None,
                'frame_count': 0,
                'tracker': None,
                'track_data': {},
            }

        # Determine frame range - use transform frames if available
        if transform_frames:
            # Override video range with transform range
            first_frame = transform_frames[0]
            last_frame = transform_frames[-1]
            self._log(f"  Limiting to transform frame range: {first_frame}-{last_frame}")

        # Stream frames through the chain
        frame_count = 0
        self._log(f"  Hardware decode: {use_hardware}")
        with VideoReader(filepath, first_frame, last_frame, use_hardware=use_hardware) as reader:
            total_frames = reader.frame_count
            self._log(f"  Streaming {total_frames} frames...")

            for frame_num, frame in reader:
                current_frame = frame
                current_data = {'frame_num': frame_num, 'props': props}

                # Process through each node in chain
                for nid in chain[1:]:
                    state = chain_state[nid]
                    ntype = state['type']
                    params = state['params']
                    inputs = state['inputs']

                    try:
                        if ntype == 'feature_tracker':
                            current_frame, current_data = self._stream_feature_tracker(
                                current_frame, current_data, state, params, inputs
                            )
                        elif ntype == 'apply_transform':
                            current_frame, current_data = self._stream_apply_transform(
                                current_frame, current_data, state, params, inputs
                            )
                        elif ntype == 'frame_average':
                            current_frame, current_data = self._stream_frame_average(
                                current_frame, current_data, state, params, inputs
                            )
                        elif ntype == 'video_output':
                            self._stream_video_output(
                                current_frame, current_data, state, params, frame_count == 0
                            )
                        elif ntype == 'clahe':
                            current_frame, current_data = self._stream_clahe(
                                current_frame, current_data, state, params
                            )
                        elif ntype == 'gamma':
                            current_frame, current_data = self._stream_gamma(
                                current_frame, current_data, state, params, inputs
                            )
                    except Exception as ex:
                        errors.append({'node_id': nid, 'type': ntype, 'error': str(ex)})
                        self._log(f"  ✗ Error in {ntype}: {ex}")
                        return

                frame_count += 1
                if frame_count % 100 == 0:
                    self._log(f"  Processed {frame_count}/{total_frames} frames...")

        self._log(f"  Streamed {frame_count} frames")

        # Finalize each node and store outputs
        for nid in chain[1:]:
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

                elif ntype == 'frame_average':
                    result, alpha = self._finalize_frame_average(state, params)
                    self.outputs[nid] = {'image': result, 'alpha': alpha}
                    self._log(f"  ✓ frame_average: {result.shape[1]}x{result.shape[0]} output")

                elif ntype == 'video_output':
                    self._finalize_video_output(state)
                    self._log(f"  ✓ video_output: {state.get('filepath')}")

                elif ntype == 'apply_transform':
                    # No persistent output needed - frames already passed through
                    self.outputs[nid] = {'video_out': None, 'mask': None}
                    self._log(f"  ✓ apply_transform complete")

                elif ntype == 'clahe':
                    self.outputs[nid] = {}
                    self._log(f"  ✓ clahe complete")

                elif ntype == 'gamma':
                    self.outputs[nid] = {'video_out': None, 'mask_out': None}
                    self._log(f"  ✓ gamma complete")

            except Exception as ex:
                errors.append({'node_id': nid, 'type': ntype, 'error': str(ex)})
                self._log(f"  ✗ Error finalizing {ntype}: {ex}")

    def _stream_feature_tracker(self, frame, data, state, params, inputs):
        """Process one frame through feature tracker."""
        from vdg.tracking import FeatureTracker
        import cv2

        if state['tracker'] is None:
            roi = inputs.get('roi')
            state['tracker'] = FeatureTracker(
                num_features=params.get('num_features', 30),
                initial_roi=roi,
                enforce_bbox=params.get('enforce_bbox', True),
                win_size=params.get('win_size', 21),
                pyramid_levels=params.get('pyramid_levels', 3),
            )
            state['tracker'].initialize(frame)
            state['initial_roi'] = roi

            # Initialize preview video writer if path is set
            preview_path = params.get('preview_path', '').strip()
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

    def _stream_apply_transform(self, frame, data, state, params, inputs):
        """Apply transform to one frame."""
        import cv2
        import numpy as np
        import math

        transforms = inputs.get('transforms', {}).get('transforms', {})
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

        # Use mask if provided, otherwise compute from luminance
        if 'mask' in data and data['mask'] is not None:
            state['alpha_accumulator'] += data['mask'].astype(np.float64) / 255.0
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

        if is_first:
            filepath = params.get('filepath', 'output.mp4').strip()
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

    def _get_chain_dependencies(self, chain: list, nodes: dict, edges: list) -> set:
        """Get all non-video dependencies for a streaming chain."""
        chain_set = set(chain)
        dependencies = set()

        for nid in chain:
            for e in edges:
                if e['target'] == nid:
                    src = e['source']
                    # Skip dependencies within the chain (video connections)
                    if src not in chain_set:
                        dependencies.add(src)

        return dependencies

    def execute(self, graph: dict) -> dict:
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
            for nid in chain:
                if nid not in node_to_chains:
                    node_to_chains[nid] = []
                node_to_chains[nid].append(i)

        # Calculate dependencies for each chain (by index)
        chain_deps = {}
        for i, chain in enumerate(streaming_chains):
            chain_deps[i] = self._get_chain_dependencies(chain, nodes, edges)

        self._log(f"Found {len(streaming_chains)} streaming chain(s)")
        for i, chain in enumerate(streaming_chains):
            chain_types = [self._get_node_type(nodes[nid]) for nid in chain]
            self._log(f"  Chain {i}: {' → '.join(chain_types)}")

        # Pre-populate video_input outputs with props (needed by stabilizer etc. before streaming runs)
        from vdg.core.video import get_video_properties
        from pathlib import Path
        video_inputs_loaded = set()
        for chain in streaming_chains:
            source_node = nodes[chain[0]]
            source_type = self._get_node_type(source_node)
            if source_type == 'video_input' and chain[0] not in video_inputs_loaded:
                params = source_node.get('data', {}).get('params', {}) or source_node.get('params', {})
                filepath = params.get('filepath', '').strip()
                if filepath and Path(filepath).exists():
                    props = get_video_properties(filepath)
                    first_frame = params.get('first_frame', 1)
                    last_frame = params.get('last_frame', -1)
                    last_frame = None if last_frame == -1 else last_frame
                    self.outputs[chain[0]] = {
                        'video': {'filepath': filepath, 'first_frame': first_frame, 'last_frame': last_frame, 'props': props},
                        'props': props
                    }
                    video_inputs_loaded.add(chain[0])
                    self._log(f"  Pre-loaded props for {chain[0]}: {props.width}x{props.height}")

        errors = []

        # Execute in order, using streaming for chains
        executed = set()
        executed_chains = set()  # Track which chains have been executed by index
        pending_chains = set(range(len(streaming_chains)))  # Chain indices waiting to execute

        for nid in order:
            if nid in executed:
                continue

            node = nodes[nid]
            ntype = self._get_node_type(node)

            # Check if this node starts any pending streaming chains
            chains_starting_here = [i for i in node_to_chains.get(nid, [])
                                    if i in pending_chains and streaming_chains[i][0] == nid]

            if chains_starting_here:
                # Try to execute each chain starting at this node
                for chain_idx in chains_starting_here:
                    chain = streaming_chains[chain_idx]
                    deps = chain_deps.get(chain_idx, set())
                    deps_satisfied = all(dep in executed for dep in deps)

                    if deps_satisfied:
                        self._execute_streaming_chain(chain, nodes, edges, errors)
                        executed.update(chain)
                        executed_chains.add(chain_idx)
                        pending_chains.discard(chain_idx)

                # If any chains were executed starting here, continue to next node
                if nid in executed:
                    continue

            # Check if this node is part of any pending chain (not start)
            in_pending_chain = any(i in pending_chains and streaming_chains[i][0] != nid
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
                    result = handler(inputs, params, self)
                    self.outputs[nid] = result or {}
                    self._log(f"  ✓ {ntype} complete")
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
                    executed.update(chain)
                    executed_chains.add(chain_idx)
                    pending_chains.discard(chain_idx)

        # Execute any remaining pending chains (shouldn't happen if deps are correct)
        for chain_idx in list(pending_chains):
            chain = streaming_chains[chain_idx]
            if chain[0] not in executed:
                self._log(f"  ⚠ Executing deferred chain starting at {chain[0]}")
                self._execute_streaming_chain(chain, nodes, edges, errors)
                executed.update(chain)

        return {
            'success': len(errors) == 0,
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

    filepath = params.get('filepath', '').strip()
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
        'video': {
            'filepath': filepath,
            'first_frame': first,
            'last_frame': last,
            'props': props,
            'frames': None,  # Frames loaded on-demand, not upfront
        },
        'props': props
    }


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

    video_data = inputs.get('video')
    roi = inputs.get('roi')

    if video_data is None:
        raise ValueError("No video input")

    tracker = FeatureTracker(
        num_features=params.get('num_features', 30),
        initial_roi=roi,
        enforce_bbox=params.get('enforce_bbox', True),
        win_size=params.get('win_size', 21),
        pyramid_levels=params.get('pyramid_levels', 3),
    )

    track_data = {}
    preview_writer = None
    preview_path = params.get('preview_path', '').strip()

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


def handle_stabilizer(inputs: dict, params: dict, executor) -> dict:
    """Compute stabilization transforms from track data."""
    track1 = inputs.get('track1', {})
    track2 = inputs.get('track2', {})
    props = inputs.get('props')

    if not track1:
        raise ValueError("No track data for track1")

    mode = params.get('mode', 'two_point')
    ref_frame = params.get('ref_frame', -1)
    swap_xy = params.get('swap_xy', False)
    x_flip = params.get('x_flip', False)
    y_flip = params.get('y_flip', False)

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

    if ref_frame == -1:
        ref_frame = frames[0]

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

    video_data = inputs.get('video', {})
    masks = inputs.get('mask', [])

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

    result = np.clip(result, 0, 255).astype(np.uint8)
    alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)

    executor._log(f"  Output: {result.shape[1]}x{result.shape[0]}")

    return {'image': result, 'alpha': alpha}


def handle_image_output(inputs: dict, params: dict, executor) -> dict:
    """Save image to file."""
    import cv2
    import numpy as np
    from pathlib import Path

    image = inputs.get('image')
    if image is None:
        raise ValueError("No image to save")

    filepath = params.get('filepath', '').strip()
    if not filepath:
        filepath = 'output.png'

    # Ensure valid extension
    path = Path(filepath)
    if not path.suffix or path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.exr']:
        filepath = str(path.with_suffix('.png'))
        executor._log(f"  Warning: Added .png extension: {filepath}")

    bit_depth = params.get('bit_depth', '16')

    if bit_depth == '16':
        # Convert to 16-bit
        if image.dtype == np.uint8:
            image = (image.astype(np.uint16) * 257)  # Scale 0-255 to 0-65535

    success = cv2.imwrite(filepath, image)
    if not success:
        raise ValueError(f"Failed to write image to {filepath}")

    executor._log(f"  Saved {filepath} ({bit_depth}-bit, {image.shape[1]}x{image.shape[0]})")

    return {'filepath': filepath}


def handle_video_output(inputs: dict, params: dict, executor) -> dict:
    """Save video to file."""
    import cv2

    frames = inputs.get('video', [])
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

    filepath = params.get('filepath', 'output.mp4').strip()

    # Get dimensions from first frame
    h, w = actual_frames[0].shape[:2]
    fps = props.fps if props and hasattr(props, 'fps') else 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))

    for frame in actual_frames:
        writer.write(frame)

    writer.release()
    executor._log(f"  Saved {filepath} ({len(actual_frames)} frames, {w}x{h})")

    return {'filepath': filepath}


def handle_track_input(inputs: dict, params: dict, executor) -> dict:
    """Load track data from a .crv file."""
    from vdg.tracking.track_io import read_crv_file
    from pathlib import Path

    filepath = params.get('filepath', '').strip()
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

    filepath = params.get('filepath', 'track.crv').strip()

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

    return {'filepath': filepath}


def handle_clahe(inputs: dict, params: dict, executor) -> dict:
    """Apply CLAHE contrast enhancement."""
    import cv2

    image = inputs.get('image')
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

    return {'image': result}


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
    'roi': handle_roi,
    'feature_tracker': handle_feature_tracker,
    'stabilizer': handle_stabilizer,
    'apply_transform': handle_apply_transform,
    'frame_average': handle_frame_average,
    'image_output': handle_image_output,
    'video_output': handle_video_output,
    'track_output': handle_track_output,
    'track_input': handle_track_input,
    'clahe': handle_clahe,
    'gamma': handle_gamma,
    'gaussian_filter': handle_gaussian_filter,
    'color_correction': lambda i, p, e: i,  # TODO
}


def main():
    print("=" * 50)
    print("VDG Node Editor")
    print("Open http://localhost:8000")
    print("=" * 50)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
