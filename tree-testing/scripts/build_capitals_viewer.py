#!/usr/bin/env python3
"""Build capitals experiment dashboard (summary + tree viewer)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.analysis.capitals_viewer import export_capitals_dashboard  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build capitals bad-nodes dashboard")
    parser.add_argument(
        "--mech-dir",
        type=Path,
        default=ROOT.parent / "tree-of-reasoning" / "results" / "capitals_mech_interp",
        help="Stage-1 τ-tree JSONL directory",
    )
    parser.add_argument(
        "--bad-nodes-dir",
        type=Path,
        default=ROOT.parent / "tree-of-reasoning" / "results" / "capitals_bad_nodes",
        help="Stage-2 bad-nodes JSONL directory",
    )
    parser.add_argument(
        "--model",
        default="deepseek-r1-7b",
        help="Model id JSONL filename stem",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "capitals",
        help="Output directory for JSON + HTML",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=ROOT / "templates" / "capitals_dashboard.template.html",
        help="HTML template path",
    )
    args = parser.parse_args()

    mech_path = args.mech_dir / f"{args.model}.jsonl"
    bad_path = args.bad_nodes_dir / f"{args.model}.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = export_capitals_dashboard(
        mech_interp_path=mech_path,
        bad_nodes_path=bad_path,
        output_path=args.output_dir / "capitals_dashboard.json",
    )

    html_out = args.output_dir / "capitals_dashboard.html"
    shutil.copyfile(args.template, html_out)

    print(f"Mech interp trees: {payload['mech_interp_trees']}")
    print(f"Bad-nodes trees: {payload['bad_nodes_trees']}")
    print(f"Viewer runs (with trees): {payload['viewer_runs']}")
    print(f"Wrote {args.output_dir / 'capitals_dashboard.json'}")
    print(f"Wrote {html_out}")
    print("Serve via HTTP:")
    print(f"  cd {ROOT} && python3 -m http.server 8768")
    print(f"  http://localhost:8768/output/capitals/capitals_dashboard.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
