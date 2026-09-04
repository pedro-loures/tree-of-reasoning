#!/usr/bin/env python3
"""Backfill reasoning_token_count and related trace metrics in saved JSONL results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trace_retag import retag_token_metrics  # noqa: E402


def backfill_file(path: Path) -> int:
    rows: list[dict] = []
    updated = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            token_metrics = row.get("trace_token_metrics") or []
            prompt_prefix = row.get("root_prefix") or ""
            if not token_metrics or not prompt_prefix:
                rows.append(row)
                continue

            retagged, summary = retag_token_metrics(token_metrics, prompt_prefix)
            row["trace_token_metrics"] = retagged
            row["trace_metrics"].update(summary)
            updated += 1
            rows.append(row)

    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill trace metrics in JSONL results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / ".." / "tree-of-reasoning" / "results",
        help="Directory containing model JSONL files",
    )
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()

    total = 0
    for path in sorted(results_dir.glob("*.jsonl")):
        count = backfill_file(path)
        total += count
        print(f"Updated {count} rows in {path}")
    print(f"Done. Updated {total} rows total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
