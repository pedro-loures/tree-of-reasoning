"""Metric extraction and aggregation from experiment JSONL rows."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

METRIC_KEYS = [
    "total_nodes",
    "max_depth",
    "leaf_count",
    "mass_above_tau",
    "breadth_warning_count",
    "reasoning_token_count",
    "mean_entropy_reasoning",
    "mean_logprob_selected",
    "answer_correct",
    "top_1_correct",
    "top_k_any_correct",
    "mentions_lorem",
]

METRIC_LABELS = {
    "total_nodes": ("Total tree nodes", "count"),
    "max_depth": ("Max tree depth", "tokens"),
    "leaf_count": ("Leaf count", "count"),
    "mass_above_tau": ("Mass above tau", "probability"),
    "breadth_warning_count": ("Breadth warnings", "count"),
    "reasoning_token_count": ("Reasoning tokens", "tokens"),
    "mean_entropy_reasoning": ("Mean reasoning entropy", "nats"),
    "mean_logprob_selected": ("Mean selected logprob", "log prob"),
    "answer_correct": ("Greedy answer correct", "rate"),
    "top_1_correct": ("Top-1 leaf correct", "rate"),
    "top_k_any_correct": ("Top-k any correct", "rate"),
    "mentions_lorem": ("Mentions lorem", "rate"),
}


def metric_value(row: dict[str, Any], key: str) -> object | None:
    if key in row.get("tree_metrics", {}):
        return row["tree_metrics"][key]
    if key in row.get("trace_metrics", {}):
        value = row["trace_metrics"][key]
        if key.endswith("_correct") or key == "mentions_lorem":
            return int(bool(value))
        return value
    if key in row.get("top_k_metrics", {}):
        value = row["top_k_metrics"][key]
        if key.endswith("_correct"):
            return int(bool(value))
        return value
    return None


def top_k_completions_summary(row: dict[str, Any]) -> list[dict[str, Any]]:
    completions = row.get("top_k_metrics", {}).get("top_k_completions", [])
    return [
        {
            "rank": index + 1,
            "leaf_id": item["leaf_id"],
            "path_prob": round(item["path_prob"], 6),
            "answer_text": item.get("answer_text", ""),
            "answer_correct": bool(item.get("answer_correct", False)),
            "reasoning_complete": bool(item.get("reasoning_complete", False)),
        }
        for index, item in enumerate(completions)
    ]


def run_metrics(
    row: dict[str, Any],
    dataset_id: str,
    model_id: str,
    prefix_length: int,
    seed: int,
    tree_key: str,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "prefix_length": prefix_length,
        "seed": seed,
        "tree_key": tree_key,
        **{key: metric_value(row, key) for key in METRIC_KEYS},
        "top_k_completions": top_k_completions_summary(row),
    }


def aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["model_id"], run["prefix_length"])].append(run)

    aggregates: list[dict[str, Any]] = []
    for (model_id, prefix_length), items in sorted(groups.items()):
        row: dict[str, Any] = {
            "model_id": model_id,
            "prefix_length": prefix_length,
            "runs": len(items),
        }
        for key in METRIC_KEYS:
            values = [item[key] for item in items if item.get(key) is not None]
            row[key] = sum(values) / len(values) if values else None
        aggregates.append(row)
    return aggregates
