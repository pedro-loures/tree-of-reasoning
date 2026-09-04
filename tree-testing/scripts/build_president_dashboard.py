#!/usr/bin/env python3
"""Build president experiment dashboard (capitals-style, mention categories)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.analysis.president_viewer import export_president_dashboard  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build president experiment dashboard")
    parser.add_argument(
        "--mech-dir",
        type=Path,
        default=ROOT.parent / "tree-of-reasoning" / "results" / "president_mech_interp",
        help="Stage-1 τ-tree JSONL directory",
    )
    parser.add_argument(
        "--bad-nodes-dir",
        type=Path,
        default=ROOT.parent / "tree-of-reasoning" / "results" / "president_bad_nodes",
        help="Stage-2 all-leaf completion JSONL directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "president",
        help="Output directory for JSON + HTML",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=ROOT / "templates" / "president_dashboard.template.html",
        help="HTML template path",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_out = args.output_dir / "president_dashboard.json"
    payload = export_president_dashboard(
        mech_interp_dir=args.mech_dir,
        bad_nodes_dir=args.bad_nodes_dir,
        output_path=json_out,
    )

    html_out = args.output_dir / "president_dashboard.html"
    shutil.copyfile(args.template, html_out)

    print(f"Mech interp trees: {payload['mech_interp_trees']}")
    print(f"Bad-nodes trees: {payload['bad_nodes_trees']}")
    print(f"Viewer runs: {payload['viewer_runs']}")
    print(f"Wrote {json_out}")
    print(f"Wrote {html_out}")
    print("Serve via HTTP:")
    print(f"  cd {ROOT} && python3 -m http.server 8769")
    print(f"  http://localhost:8769/output/president/president_dashboard.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
