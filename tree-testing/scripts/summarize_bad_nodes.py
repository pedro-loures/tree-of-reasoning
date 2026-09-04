#!/usr/bin/env python3
"""Aggregate exclusively-bad node experiment outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.analysis.bad_nodes import export_bad_nodes_summary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize bad-nodes experiment results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT.parent / "tree-of-reasoning" / "results" / "bad_nodes",
        help="Directory containing bad_nodes JSONL outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output",
        help="Directory for summary JSON",
    )
    args = parser.parse_args()

    result = export_bad_nodes_summary(args.results_dir, args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    print("rows:")
    for row in result["rows"]:
        country = row.get("country_id")
        country_label = f" country={country}" if country else ""
        print(
            f"  {row['model_id']}{country_label} {row['instruction_variant']} "
            f"pl={row['prefix_length']}: "
            f"candidates={row['total_candidates']} "
            f"exclusively_bad={row['exclusively_bad_count']} "
            f"ditched={row['ditched_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
