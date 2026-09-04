#!/usr/bin/env python3
"""Categorize president experiment results into three mention lists."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MENTION_CATEGORY_ONLY_CANDIDATES = "mentioned_only_candidates"
MENTION_CATEGORY_POLITICIANS = "mentioned_politicians"
MENTION_CATEGORY_NONE = "no_politicians_mentioned"

CATEGORIES = (
    MENTION_CATEGORY_ONLY_CANDIDATES,
    MENTION_CATEGORY_POLITICIANS,
    MENTION_CATEGORY_NONE,
)


def load_results_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _text_snippet(row: dict[str, Any], limit: int = 400) -> str:
    text = row.get("trace_metrics", {}).get("generated_text", "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def build_entry(row: dict[str, Any]) -> dict[str, Any]:
    mentions = row.get("politician_mentions", {}).get("greedy", {})
    return {
        "model_id": row.get("model_id"),
        "seed": row.get("seed"),
        "instruction": row.get("instruction"),
        "prefix_length": row.get("prefix_length"),
        "category": mentions.get("category", MENTION_CATEGORY_NONE),
        "mentions": mentions.get("mentions", []),
        "text_snippet": _text_snippet(row),
    }


def write_category_files(
    entries: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_category: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORIES}
    for entry in entries:
        category = entry["category"]
        if category in by_category:
            by_category[category].append(entry)

    for category, items in by_category.items():
        path = output_dir / f"{category}.jsonl"
        with path.open("w") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts = Counter(entry["category"] for entry in entries)
    model_counts: dict[str, Counter[str]] = defaultdict(Counter)
    seed_counts: dict[int, Counter[str]] = defaultdict(Counter)
    party_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()

    for entry in entries:
        category = entry["category"]
        model_counts[entry["model_id"]][category] += 1
        seed_counts[int(entry["seed"])][category] += 1
        for mention in entry["mentions"]:
            party_counts[mention.get("party", "")] += 1
            if mention.get("is_presidential_candidate_2026"):
                candidate_counts[mention.get("full_name", "")] += 1

    return {
        "total_responses": len(entries),
        "category_counts": dict(category_counts),
        "by_model": {model: dict(counts) for model, counts in model_counts.items()},
        "by_seed": {str(seed): dict(counts) for seed, counts in seed_counts.items()},
        "party_mentions": dict(party_counts.most_common()),
        "presidential_candidate_mentions": dict(candidate_counts.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze president mention experiment results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT.parent / "tree-of-reasoning" / "results" / "president_mech_interp",
        help="Directory containing per-model JSONL result files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "president",
        help="Directory for categorized lists and summary.json",
    )
    args = parser.parse_args()

    entries: list[dict[str, Any]] = []
    for path in sorted(args.results_dir.glob("*.jsonl")):
        for row in load_results_jsonl(path):
            if "politician_mentions" not in row:
                continue
            entries.append(build_entry(row))

    if not entries:
        print(f"No politician mention results found in {args.results_dir}", flush=True)
        return 1

    write_category_files(entries, args.output_dir)
    summary = build_summary(entries)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"Wrote {len(entries)} categorized responses to {args.output_dir}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    for category in CATEGORIES:
        count = summary["category_counts"].get(category, 0)
        print(f"  {category}: {count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
