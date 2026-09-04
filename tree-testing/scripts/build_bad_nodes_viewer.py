#!/usr/bin/env python3
"""Build bad-nodes HTML viewer and canvas JSON."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.analysis.bad_nodes_viewer import export_bad_nodes_viewer_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build bad-nodes graph viewer")
    parser.add_argument(
        "--canvas-trees",
        type=Path,
        default=ROOT / "output" / "canvas_trees.json",
        help="Source τ-tree structures",
    )
    parser.add_argument(
        "--bad-nodes-dir",
        "--results-dir",
        type=Path,
        default=ROOT.parent / "tree-of-reasoning" / "results" / "bad_nodes",
        help="Bad-nodes JSONL directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output",
        help="Directory for bad_nodes_canvas.json and HTML",
    )
    parser.add_argument(
        "--mech-interp-dir",
        type=Path,
        default=None,
        help="Optional mech_interp JSONL directory (adds depth, τ-leaves, breadth)",
    )
    parser.add_argument(
        "--dataset-id",
        default="tau001",
        help="Dataset id for mech_interp tree keys",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=ROOT / "templates" / "bad_nodes_graph.template.html",
        help="HTML template path",
    )
    args = parser.parse_args()

    payload = export_bad_nodes_viewer_data(
        canvas_trees_path=args.canvas_trees,
        bad_nodes_dir=args.bad_nodes_dir,
        output_path=args.output_dir / "bad_nodes_canvas.json",
        mech_interp_dir=args.mech_interp_dir,
        dataset_id=args.dataset_id,
    )

    html_out = args.output_dir / "bad_nodes_graph.html"
    shutil.copyfile(args.template, html_out)

    print(f"Wrote {args.output_dir / 'bad_nodes_canvas.json'} ({len(payload['runs'])} runs)")
    print(f"Wrote {html_out}")
    print("Open via HTTP server, e.g.:")
    print(f"  cd {ROOT} && python3 -m http.server 8767")
    print(f"  http://localhost:8767/output/bad_nodes_graph.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
