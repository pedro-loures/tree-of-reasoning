"""Aggregate exclusively-bad node experiment results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_bad_node_results(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.jsonl")):
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def build_summary_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for row in sorted(
        rows,
        key=lambda item: (
            item.get("country_id", ""),
            item.get("model_id", ""),
            item.get("instruction_variant", ""),
            int(item.get("prefix_length", 0)),
        ),
    ):
        summary = row.get("summary", {})
        table.append(
            {
                "tree_key": row.get("tree_key"),
                "country_id": row.get("country_id"),
                "country_name": row.get("country_name"),
                "model_id": row.get("model_id"),
                "instruction_variant": row.get("instruction_variant"),
                "prefix_length": row.get("prefix_length"),
                "total_leaves": summary.get("total_leaves"),
                "total_candidates": summary.get("total_candidates"),
                "exclusively_bad_count": summary.get("exclusively_bad_count"),
                "ditched_count": summary.get("ditched_count"),
                "leaf_subtype_counts": summary.get("leaf_subtype_counts", {}),
            }
        )
    return table


def export_bad_nodes_summary(results_dir: Path, output_dir: Path) -> dict[str, Any]:
    rows = load_bad_node_results(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = build_summary_table(rows)
    summary_path = output_dir / "bad_nodes_summary.json"
    payload = {
        "source": str(results_dir),
        "tree_count": len(rows),
        "rows": table,
    }
    summary_path.write_text(json.dumps(payload, indent=2))
    return {
        "tree_count": len(rows),
        "rows": table,
        "summary_path": str(summary_path),
    }
