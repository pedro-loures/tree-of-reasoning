#!/usr/bin/env python3
"""Build the unified GitHub Pages static site from local experiment results."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.pipelines.analysis.pages_export import (  # noqa: E402
    copy_site_assets,
    export_capitals_sharded,
    export_elections_sharded,
    export_interactive_sharded,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build unified GitHub Pages site")
    parser.add_argument(
        "--capitals-mech",
        type=Path,
        default=REPO_ROOT / "tree-of-reasoning" / "results" / "capitals_mech_interp" / "deepseek-r1-7b.jsonl",
    )
    parser.add_argument(
        "--capitals-bad",
        type=Path,
        default=REPO_ROOT / "tree-of-reasoning" / "results" / "capitals_bad_nodes" / "deepseek-r1-7b.jsonl",
    )
    parser.add_argument(
        "--president-mech",
        type=Path,
        default=REPO_ROOT / "tree-of-reasoning" / "results" / "president_mech_interp",
    )
    parser.add_argument(
        "--president-bad",
        type=Path,
        default=REPO_ROOT / "tree-of-reasoning" / "results" / "president_bad_nodes",
    )
    parser.add_argument(
        "--interactive-dir",
        type=Path,
        default=REPO_ROOT / "tree-of-reasoning" / "results" / "dashboard_sessions",
    )
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=ROOT / "site",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "docs",
    )
    parser.add_argument("--skip-capitals", action="store_true")
    parser.add_argument("--skip-elections", action="store_true")
    parser.add_argument("--skip-interactive", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    copy_site_assets(args.site_dir, args.output_dir)
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True)

    if not args.skip_capitals and args.capitals_mech.exists() and args.capitals_bad.exists():
        cap_manifest = export_capitals_sharded(
            args.capitals_mech,
            args.capitals_bad,
            data_dir / "capitals",
        )
        print(f"Capitals: {len(cap_manifest.get('shards', []))} trees")
    else:
        print("Capitals: skipped (inputs missing)")

    if not args.skip_elections and args.president_mech.exists() and args.president_bad.exists():
        elec_manifest = export_elections_sharded(
            args.president_mech,
            args.president_bad,
            data_dir / "elections",
        )
        print(f"Elections: {len(elec_manifest.get('shards', []))} trees")
    else:
        print("Elections: skipped (inputs missing)")

    if not args.skip_interactive and args.interactive_dir.exists():
        int_manifest = export_interactive_sharded(
            args.interactive_dir,
            data_dir / "interactive",
            repo_root=REPO_ROOT,
        )
        print(f"Interactive: {len(int_manifest.get('shards', []))} trees")
    else:
        print("Interactive: skipped (inputs missing)")

    print(f"Built site at {args.output_dir}")
    print("Preview: cd docs && python3 -m http.server 8080")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
