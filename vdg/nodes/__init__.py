"""
VDG Node Editors
=================

This module provides node-based visual programming interfaces for VDG.

Options:

1. Web Editor (React Flow + FastAPI):
   pip install fastapi uvicorn
   python -m vdg.nodes.web_editor
   
2. Ryven (Pure Python/Qt):
   pip install ryven
   Then import vdg.nodes.ryven_nodes in Ryven

3. Export to other formats (ComfyUI, Blender, etc.):
   See node_export.py for conversion utilities
"""

__all__ = ['web_editor', 'ryven_nodes']
