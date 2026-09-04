#!/usr/bin/env python3
"""Find repeated token paths across branches in capitals tau trees."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.analysis.token_path_motifs import (  # noqa: E402
    load_capitals_records,
    run_motif_analysis,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze repeated token paths per tree")
    parser.add_argument(
        "--mech-dir",
        type=Path,
        default=ROOT.parent / "tree-of-reasoning" / "results" / "capitals_mech_interp",
        help="Directory with mech_interp JSONL files",
    )
    parser.add_argument(
        "--model",
        default="deepseek-r1-7b",
        help="Model id JSONL filename stem",
    )
    parser.add_argument(
        "--instruction-variant",
        action="append",
        dest="instruction_variants",
        choices=["legacy", "plain"],
        help="Restrict to instruction variant (repeatable)",
    )
    parser.add_argument(
        "--country",
        action="append",
        dest="countries",
        help="Restrict to country id (repeatable)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Plain pilot countries only (brazil, fiji, india, morocco, new_zealand)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=6,
        help="Minimum motif length in tokens (default 6 = ignore <=5)",
    )
    parser.add_argument(
        "--min-branches",
        type=int,
        default=2,
        help="Minimum distinct root-to-leaf paths sharing the motif",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "token_path_motifs",
        help="Output directory for CSV tables and index.html",
    )
    args = parser.parse_args()

    records = load_capitals_records(args.mech_dir)
    records = [
        record
        for record in records
        if record.model_id == args.model
    ]
    if args.instruction_variants:
        records = [record for record in records if record.instruction_variant in args.instruction_variants]
    if args.countries:
        records = [record for record in records if record.country_id in args.countries]
    if args.pilot:
        from src.pipelines.analysis.token_path_motifs import PILOT_COUNTRIES

        records = [record for record in records if record.country_id in PILOT_COUNTRIES]

    if not records:
        print("No matching trees found.", file=sys.stderr)
        return 1

    subdir = "pilot" if args.pilot else "plain"
    if args.countries and len(args.countries) == 1:
        subdir = f"{subdir}_{args.countries[0]}"

    result = run_motif_analysis(
        records,
        args.output_dir.resolve(),
        min_length=args.min_length,
        min_branches=args.min_branches,
        subdir=subdir,
    )

    print(f"Analyzed {result['trees']} trees")
    print(f"Wrote index: {result['index_html']}")
    print(f"Wrote manifest: {result['manifest']}")
    for summary in result["summaries"]:
        print(
            f"  {summary.country_id}: leaves={summary.leaf_count} "
            f"motifs={summary.motif_count}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
