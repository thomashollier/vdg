#!/usr/bin/env python3
"""
Command-line runner for VDG workflow JSON files.

Usage:
    python -m vdg.nodes.runner workflow.json [--set node_id.param=value ...]

Examples:
    # Run a workflow
    python -m vdg.nodes.runner workflows/post_process.json

    # Override parameters
    python -m vdg.nodes.runner workflow.json --set n1.filepath=input.mov --set n4.filepath=output.png

    # Set working directory
    python -m vdg.nodes.runner workflow.json -d /path/to/project
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_workflow(filepath: str) -> dict:
    """Load a workflow JSON file and convert to execution format."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Convert nodes list to dict
    nodes = {n['id']: n for n in data.get('nodes', [])}

    # Convert conns format (sn/sp/tn/tp) to edges format (source/sourceHandle/target/targetHandle)
    conns = data.get('conns', [])
    edges = [
        {
            'source': c['sn'],
            'sourceHandle': c['sp'],
            'target': c['tn'],
            'targetHandle': c['tp'],
        }
        for c in conns
    ]

    return {'nodes': nodes, 'edges': edges}


def apply_overrides(graph: dict, overrides: list[str]) -> None:
    """Apply parameter overrides from command line.

    Format: node_id.param=value
    """
    for override in overrides:
        if '=' not in override:
            print(f"Warning: Invalid override format '{override}', expected 'node_id.param=value'")
            continue

        key, value = override.split('=', 1)
        if '.' not in key:
            print(f"Warning: Invalid override key '{key}', expected 'node_id.param'")
            continue

        node_id, param = key.rsplit('.', 1)

        if node_id not in graph['nodes']:
            print(f"Warning: Node '{node_id}' not found in workflow")
            continue

        node = graph['nodes'][node_id]
        if 'params' not in node:
            node['params'] = {}

        # Try to preserve type
        old_value = node['params'].get(param)
        if isinstance(old_value, bool):
            value = value.lower() in ('true', '1', 'yes')
        elif isinstance(old_value, int):
            try:
                value = int(value)
            except ValueError:
                pass
        elif isinstance(old_value, float):
            try:
                value = float(value)
            except ValueError:
                pass

        node['params'][param] = value
        print(f"  {node_id}.{param} = {value}")


def run_workflow(workflow_path: str, overrides: list[str] = None, directory: str = None) -> bool:
    """Load and execute a workflow.

    Returns True on success, False on failure.
    """
    # Import here to avoid circular imports and speed up --help
    from vdg.nodes.web_editor import GraphExecutor, refresh_file_cache

    workflow_path = Path(workflow_path).expanduser().resolve()
    if not workflow_path.exists():
        print(f"Error: Workflow file not found: {workflow_path}")
        return False

    # Set working directory
    if directory:
        dir_path = Path(directory).expanduser().resolve()
        if dir_path.is_dir():
            os.chdir(dir_path)
            refresh_file_cache(str(dir_path))
            print(f"Working directory: {dir_path}")
        else:
            print(f"Warning: Directory not found: {directory}")
    else:
        # Use workflow file's directory as working directory
        os.chdir(workflow_path.parent)
        refresh_file_cache(str(workflow_path.parent))

    print(f"Loading workflow: {workflow_path.name}")
    try:
        graph = load_workflow(str(workflow_path))
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in workflow file: {e}")
        return False
    except Exception as e:
        print(f"Error loading workflow: {e}")
        return False

    # Show nodes
    print(f"  {len(graph['nodes'])} nodes, {len(graph['edges'])} connections")

    # Apply overrides
    if overrides:
        print("Applying overrides:")
        apply_overrides(graph, overrides)

    # Execute
    print("\nExecuting...")
    print("=" * 50)

    executor = GraphExecutor()
    result = executor.execute(graph)

    print("=" * 50)

    if result.get('aborted'):
        print("\n⚠ Execution aborted")
        return False
    elif result.get('success'):
        print("\n✓ Execution completed successfully")
        return True
    else:
        print("\n✗ Execution failed")
        if result.get('errors'):
            for err in result['errors']:
                print(f"  - {err.get('node_id', 'unknown')}: {err.get('error', 'unknown error')}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run VDG workflow JSON files from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s workflow.json
  %(prog)s workflow.json --set n1.filepath=video.mov
  %(prog)s workflow.json -d /path/to/project --set n4.filepath=output.png
        """,
    )
    parser.add_argument('workflow', help='Path to workflow JSON file')
    parser.add_argument('-d', '--directory', help='Working directory for relative paths')
    parser.add_argument(
        '--set', '-s',
        action='append',
        dest='overrides',
        metavar='NODE.PARAM=VALUE',
        help='Override node parameter (can be used multiple times)',
    )
    parser.add_argument('--list-nodes', '-l', action='store_true', help='List nodes and their parameters, then exit')

    args = parser.parse_args()

    # List mode
    if args.list_nodes:
        workflow_path = Path(args.workflow).expanduser().resolve()
        if not workflow_path.exists():
            print(f"Error: Workflow file not found: {workflow_path}")
            sys.exit(1)

        with open(workflow_path, 'r') as f:
            data = json.load(f)

        print(f"Workflow: {workflow_path.name}\n")
        for node in data.get('nodes', []):
            print(f"{node['id']} ({node['type']})")
            for param, value in node.get('params', {}).items():
                print(f"  {param}: {value}")
            print()
        sys.exit(0)

    # Run workflow
    success = run_workflow(args.workflow, args.overrides, args.directory)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
