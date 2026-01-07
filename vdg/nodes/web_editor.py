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
        inputs=[],
        outputs=[NodePort("roi", "roi")],
        params=[
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
            NodeParam("y_flip", "bool", False),
        ],
        color="#2196F3",
    ),
    NodeDefinition(
        id="apply_transform", title="Apply Transform", category="Processing",
        inputs=[NodePort("video", "video"), NodePort("transforms", "transforms")],
        outputs=[NodePort("video", "video"), NodePort("mask", "video")],
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
            NodeParam("gamma", "float", 1.0, min=0.1, max=4.0),
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
        #canvas { position: absolute; width: 5000px; height: 5000px; background-image: radial-gradient(#333 1px, transparent 1px); background-size: 20px 20px; }
        #svg-layer { position: absolute; width: 5000px; height: 5000px; pointer-events: none; }
        .toolbar { position: absolute; top: 10px; right: 10px; display: flex; gap: 6px; z-index: 100; }
        .toolbar button { padding: 8px 14px; background: #4CAF50; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }
        .toolbar button:hover { filter: brightness(1.1); }
        .toolbar button.sec { background: #555; }
        .node { position: absolute; min-width: 140px; background: #252545; border: 2px solid #444; border-radius: 6px; cursor: move; user-select: none; font-size: 11px; }
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
        .conn-temp { stroke: #4fc3f7; stroke-width: 2; fill: none; stroke-dasharray: 5,5; }
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
        if (e.target === area || e.target.id === 'canvas' || e.target.id === 'svg-layer') {
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
    svg.querySelectorAll('.conn').forEach(e => e.remove());
    conns.forEach(c => {
        const d1 = document.querySelector('[data-n="' + c.sn + '"][data-p="' + c.sp + '"]');
        const d2 = document.querySelector('[data-n="' + c.tn + '"][data-p="' + c.tp + '"]');
        if (!d1 || !d2) return;
        const r1 = d1.getBoundingClientRect(), r2 = d2.getBoundingClientRect();
        const x1 = r1.left + 4 - ar.left - pan.x, y1 = r1.top + 4 - ar.top - pan.y;
        const x2 = r2.left + 4 - ar.left - pan.x, y2 = r2.top + 4 - ar.top - pan.y;
        const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        p.classList.add('conn');
        p.setAttribute('d', 'M' + x1 + ' ' + y1 + ' C' + (x1+60) + ' ' + y1 + ',' + (x2-60) + ' ' + y2 + ',' + x2 + ' ' + y2);
        svg.appendChild(p);
    });
}

function status(m) { document.getElementById('status').textContent = m; }

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


@app.post("/api/execute")
async def execute_graph(graph: dict):
    """Execute the node graph."""
    try:
        executor = GraphExecutor()
        result = executor.execute(graph)
        return result
    except Exception as e:
        import traceback
        return {'success': False, 'message': str(e), 'errors': [{'node_id': 'graph', 'error': traceback.format_exc()}]}


class GraphExecutor:
    """Executes a VDG node graph."""
    
    def __init__(self):
        self.outputs = {}  # node_id -> {port_name: value}
        self.log = []
    
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
        
        # Execute in order
        errors = []
        for nid in order:
            node = nodes[nid]
            ntype = node['type']
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
                    self.log.append(f"Executing {ntype} ({nid})...")
                    result = handler(inputs, params, self.log)
                    self.outputs[nid] = result or {}
                    self.log.append(f"  ✓ {ntype} complete")
                else:
                    self.log.append(f"  ⚠ No handler for {ntype}")
                    self.outputs[nid] = {}
            except Exception as ex:
                import traceback
                errors.append({'node_id': nid, 'type': ntype, 'error': str(ex)})
                self.log.append(f"  ✗ Error in {ntype}: {ex}")
        
        return {
            'success': len(errors) == 0,
            'message': '\n'.join(self.log),
            'errors': errors
        }


# =============================================================================
# NODE HANDLERS - Actual execution logic
# =============================================================================

def handle_video_input(inputs: dict, params: dict, log: list) -> dict:
    """Load video and return frames + properties."""
    from vdg.core.video import VideoReader
    import numpy as np
    
    filepath = params.get('filepath', '')
    if not filepath:
        raise ValueError("No video file specified")
    
    first = params.get('first_frame', 1)
    last = params.get('last_frame', -1)
    last = None if last == -1 else last
    
    log.append(f"  Opening {filepath} (frames {first}-{last or 'end'})")
    reader = VideoReader(filepath, first, last)
    reader.open()
    
    props = reader.properties
    log.append(f"  Video: {props.width}x{props.height}, {props.fps}fps")
    
    # Load all frames into memory so multiple nodes can use them
    frames = []
    frame_nums = []
    for frame_num, frame in reader:
        frames.append(frame.copy())
        frame_nums.append(frame_num)
        if len(frames) % 100 == 0:
            log.append(f"  Loaded {len(frames)} frames...")
    
    reader.close()
    log.append(f"  Loaded {len(frames)} frames total")
    
    return {
        'video': {'frames': frames, 'frame_nums': frame_nums, 'props': props, 'filepath': filepath},
        'props': props
    }


def handle_roi(inputs: dict, params: dict, log: list) -> dict:
    """Return ROI tuple."""
    roi = (params.get('x', 0), params.get('y', 0), 
           params.get('width', 100), params.get('height', 100))
    log.append(f"  ROI: {roi}")
    return {'roi': roi}


def handle_feature_tracker(inputs: dict, params: dict, log: list) -> dict:
    """Track features in video."""
    from vdg.tracking import FeatureTracker
    import numpy as np
    
    video_data = inputs.get('video')
    roi = inputs.get('roi')
    
    if video_data is None:
        raise ValueError("No video input")
    
    frames = video_data.get('frames', [])
    frame_nums = video_data.get('frame_nums', [])
    
    if not frames:
        raise ValueError("No frames in video data")
    
    tracker = FeatureTracker(
        num_features=params.get('num_features', 30),
        initial_roi=roi,
        enforce_bbox=params.get('enforce_bbox', True),
    )
    
    track_data = {}
    
    for i, (frame_num, frame) in enumerate(zip(frame_nums, frames)):
        if i == 0:
            tracker.initialize(frame)
        else:
            tracker.update(frame)
        
        if tracker.points is not None and len(tracker.points) > 0:
            # Calculate centroid
            pts = tracker.points.reshape(-1, 2)
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            track_data[frame_num] = {'x': float(cx), 'y': float(cy)}
        
        if (i + 1) % 100 == 0:
            log.append(f"  Tracked {i + 1} frames...")
    
    log.append(f"  Tracked {len(frames)} frames, {len(track_data)} valid points")
    
    return {'points': tracker.points, 'track_data': track_data}


def handle_stabilizer(inputs: dict, params: dict, log: list) -> dict:
    """Compute stabilization transforms from track data."""
    track1 = inputs.get('track1', {})
    track2 = inputs.get('track2', {})
    props = inputs.get('props')
    
    if not track1:
        raise ValueError("No track data for track1")
    
    mode = params.get('mode', 'two_point')
    ref_frame = params.get('ref_frame', -1)
    
    # Find common frames
    if track2:
        frames = sorted(set(track1.keys()) & set(track2.keys()))
    else:
        frames = sorted(track1.keys())
    
    if not frames:
        raise ValueError("No valid frames in track data")
    
    if ref_frame == -1:
        ref_frame = frames[0]
    
    log.append(f"  Mode: {mode}, ref_frame: {ref_frame}, {len(frames)} frames")
    
    # Get reference positions
    ref1 = track1.get(ref_frame, track1[frames[0]])
    ref2 = track2.get(ref_frame, track2[frames[0]]) if track2 else None
    
    transforms = {}
    for f in frames:
        p1 = track1[f]
        p2 = track2[f] if track2 else None
        
        # Compute transform based on mode
        if mode == 'single' or not p2:
            # Translation only
            dx = ref1['x'] - p1['x']
            dy = ref1['y'] - p1['y']
            transforms[f] = {'dx': dx, 'dy': dy, 'rotation': 0, 'scale': 1}
        else:
            # Two-point: translation + rotation + scale
            import math
            
            # Current vector
            vx = p2['x'] - p1['x']
            vy = p2['y'] - p1['y']
            # Reference vector
            rvx = ref2['x'] - ref1['x']
            rvy = ref2['y'] - ref1['y']
            
            # Scale
            curr_len = math.sqrt(vx*vx + vy*vy)
            ref_len = math.sqrt(rvx*rvx + rvy*rvy)
            scale = ref_len / curr_len if curr_len > 0 else 1
            
            # Rotation
            curr_angle = math.atan2(vy, vx)
            ref_angle = math.atan2(rvy, rvx)
            rotation = ref_angle - curr_angle
            
            # Translation (after applying rotation and scale to p1)
            dx = ref1['x'] - p1['x']
            dy = ref1['y'] - p1['y']
            
            transforms[f] = {'dx': dx, 'dy': dy, 'rotation': rotation, 'scale': scale}
    
    return {'transforms': transforms, 'frames': frames, 'props': props}


def handle_apply_transform(inputs: dict, params: dict, log: list) -> dict:
    """Apply transforms to video frames."""
    import cv2
    import numpy as np
    
    video_data = inputs.get('video')
    transform_data = inputs.get('transforms', {})
    transforms = transform_data.get('transforms', {})
    
    if video_data is None:
        raise ValueError("No video input")
    
    frames = video_data.get('frames', [])
    frame_nums = video_data.get('frame_nums', [])
    props = video_data.get('props')
    
    if not frames:
        raise ValueError("No frames in video data")
    
    x_pad = params.get('x_pad', 0)
    y_pad = params.get('y_pad', 0)
    x_off = params.get('x_offset', 0)
    y_off = params.get('y_offset', 0)
    
    in_h, in_w = frames[0].shape[:2]
    out_w = in_w + x_pad
    out_h = in_h + y_pad
    
    log.append(f"  Input: {in_w}x{in_h}, Output: {out_w}x{out_h}")
    
    stabilized_frames = []
    mask_frames = []
    
    for i, (frame_num, frame) in enumerate(zip(frame_nums, frames)):
        t = transforms.get(frame_num, {'dx': 0, 'dy': 0, 'rotation': 0, 'scale': 1})
        
        # Build transform matrix
        cx, cy = frame.shape[1] / 2, frame.shape[0] / 2
        
        # Rotation + scale around center
        rot_deg = np.degrees(t.get('rotation', 0))
        scale = t.get('scale', 1)
        M = cv2.getRotationMatrix2D((cx, cy), rot_deg, scale)
        
        # Add translation
        M[0, 2] += t['dx'] + x_pad / 2 + x_off
        M[1, 2] += t['dy'] + y_pad / 2 + y_off
        
        # Apply transform
        stabilized = cv2.warpAffine(frame, M, (out_w, out_h))
        stabilized_frames.append(stabilized)
        
        # Create mask
        mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255
        mask_warped = cv2.warpAffine(mask, M, (out_w, out_h))
        mask_frames.append(mask_warped)
        
        if (i + 1) % 100 == 0:
            log.append(f"  Transformed {i + 1} frames...")
    
    log.append(f"  Stabilized {len(stabilized_frames)} frames")
    
    return {'video': stabilized_frames, 'mask': mask_frames, 'width': out_w, 'height': out_h}


def handle_frame_average(inputs: dict, params: dict, log: list) -> dict:
    """Average frames together."""
    import numpy as np
    
    frames = inputs.get('video', [])
    masks = inputs.get('mask', [])
    
    if not frames:
        raise ValueError("No frames to average")
    
    if isinstance(frames[0], np.ndarray):
        # Already have frame list
        pass
    else:
        raise ValueError("Expected list of frames")
    
    comp_mode = params.get('comp_mode', 'on_black')
    gamma = params.get('gamma', 1.0)
    brightness = params.get('brightness', 1.0)
    
    log.append(f"  Averaging {len(frames)} frames, mode: {comp_mode}")
    
    # Accumulate
    acc = np.zeros(frames[0].shape, dtype=np.float64)
    alpha_acc = np.zeros(frames[0].shape[:2], dtype=np.float64)
    
    for i, frame in enumerate(frames):
        acc += frame.astype(np.float64)
        if masks and i < len(masks):
            alpha_acc += masks[i].astype(np.float64) / 255.0
    
    # Average
    n = len(frames)
    result = acc / n
    alpha = alpha_acc / n
    
    # Apply gamma and brightness
    if gamma != 1.0:
        result = np.power(result / 255.0, 1.0 / gamma) * 255.0
    result *= brightness
    
    # Composite mode
    if comp_mode == 'on_white':
        alpha_3ch = np.dstack([alpha, alpha, alpha])
        white = np.ones_like(result) * 255
        result = result * alpha_3ch + white * (1 - alpha_3ch)
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    
    log.append(f"  Averaged to {result.shape[1]}x{result.shape[0]}")
    
    return {'image': result, 'alpha': alpha}


def handle_image_output(inputs: dict, params: dict, log: list) -> dict:
    """Save image to file."""
    import cv2
    import numpy as np
    
    image = inputs.get('image')
    if image is None:
        raise ValueError("No image to save")
    
    filepath = params.get('filepath', 'output.png')
    bit_depth = params.get('bit_depth', '16')
    
    if bit_depth == '16':
        # Convert to 16-bit
        if image.dtype == np.uint8:
            image = (image.astype(np.uint16) * 257)  # Scale 0-255 to 0-65535
    
    cv2.imwrite(filepath, image)
    log.append(f"  Saved {filepath} ({bit_depth}-bit)")
    
    return {'filepath': filepath}


def handle_video_output(inputs: dict, params: dict, log: list) -> dict:
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
    
    filepath = params.get('filepath', 'output.mp4')
    
    # Get dimensions from first frame
    h, w = actual_frames[0].shape[:2]
    fps = props.fps if props and hasattr(props, 'fps') else 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
    
    for frame in actual_frames:
        writer.write(frame)
    
    writer.release()
    log.append(f"  Saved {filepath} ({len(actual_frames)} frames, {w}x{h})")
    
    return {'filepath': filepath}


def handle_track_output(inputs: dict, params: dict, log: list) -> dict:
    """Save track data to .crv file."""
    track_data = inputs.get('track_data', {})
    props = inputs.get('props')
    
    if not track_data:
        raise ValueError("No track data to save")
    
    filepath = params.get('filepath', 'track.crv')
    
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
    
    log.append(f"  Saved {filepath} ({len(track_data)} frames)")
    
    return {'filepath': filepath}


def handle_clahe(inputs: dict, params: dict, log: list) -> dict:
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
    
    log.append(f"  Applied CLAHE (clip={clip}, grid={grid})")
    
    return {'image': result}


def handle_gaussian_filter(inputs: dict, params: dict, log: list) -> dict:
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
        log.append(f"  Smoothed {len(frames)} points (sigma={sigma})")
        
        return {'track_data': result}
    except ImportError:
        log.append(f"  Warning: scipy not installed, returning unfiltered data")
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
    'track_input': lambda i, p, l: {'track_data': {}},  # TODO
    'clahe': handle_clahe,
    'gaussian_filter': handle_gaussian_filter,
    'color_correction': lambda i, p, l: i,  # TODO
}


def main():
    print("=" * 50)
    print("VDG Node Editor")
    print("Open http://localhost:8000")
    print("=" * 50)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
