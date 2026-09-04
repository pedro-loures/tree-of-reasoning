"""Top-k word and position edge analysis from tau trees."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, TypeVar

from src.pipelines.analysis.io import RunRecord
from src.pipelines.analysis.tree_parser import build_incoming_token_map

T = TypeVar("T")

WORD_EDGE_DEFINITION = "sum_breadth_by_incoming_token_internal_nodes_only"
POSITION_EDGE_DEFINITION = "sum_breadth_by_tree_depth_internal_nodes_only"


def _is_internal_node(node: dict) -> bool:
    return bool(node.get("child_ids"))


def _internal_nodes(nodes: list[dict]) -> list[dict]:
    return [
        node
        for node in nodes
        if node["id"] != "root" and _is_internal_node(node)
    ]


def count_edges_per_word(nodes: list[dict]) -> dict[str, int]:
    incoming = build_incoming_token_map(nodes)
    counts: Counter[str] = Counter()
    for node in _internal_nodes(nodes):
        token = incoming.get(node["id"], "?")
        counts[token] += int(node.get("breadth", 0))
    return dict(counts)


def count_edges_per_position(nodes: list[dict]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for node in _internal_nodes(nodes):
        counts[int(node["depth"])] += int(node.get("breadth", 0))
    return dict(counts)


def top_k_ranked(
    counts: dict[T, int],
    k: int,
    key_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked_most = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ranked_least = sorted(counts.items(), key=lambda item: (item[1], item[0]))
    most = [{key_name: key, "edge_count": count} for key, count in ranked_most[:k]]
    least = [{key_name: key, "edge_count": count} for key, count in ranked_least[:k]]
    return most, least


def top_k_words(counts: dict[str, int], k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return top_k_ranked(counts, k, "word")


def top_k_positions(counts: dict[int, int], k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return top_k_ranked(counts, k, "position")


def all_edges_by_position(position_counts: dict[int, int]) -> list[dict[str, Any]]:
    return [
        {"position": position, "edge_count": count}
        for position, count in sorted(position_counts.items())
    ]


@dataclass
class WordEdgeResult:
    per_execution: list[dict[str, Any]]
    aggregated: list[dict[str, Any]]


def analyze_runs(runs: list[RunRecord], top_k: int) -> WordEdgeResult:
    per_execution: list[dict[str, Any]] = []
    group_word_counts: dict[tuple[str, int], Counter[str]] = {}
    group_position_counts: dict[tuple[str, int], Counter[int]] = {}

    for record in runs:
        word_counts = count_edges_per_word(record.tree_nodes)
        position_counts = count_edges_per_position(record.tree_nodes)
        most_words, least_words = top_k_words(word_counts, top_k)
        most_positions, least_positions = top_k_positions(position_counts, top_k)
        per_execution.append(
            {
                "model_id": record.model_id,
                "prefix_length": record.prefix_length,
                "seed": record.seed,
                "most_edges": most_words,
                "least_edges": least_words,
                "most_edges_by_position": most_positions,
                "least_edges_by_position": least_positions,
                "all_edges_by_position": all_edges_by_position(position_counts),
            }
        )
        key = (record.model_id, record.prefix_length)
        if key not in group_word_counts:
            group_word_counts[key] = Counter()
            group_position_counts[key] = Counter()
        group_word_counts[key].update(word_counts)
        group_position_counts[key].update(position_counts)

    aggregated: list[dict[str, Any]] = []
    for (model_id, prefix_length), word_counts in sorted(group_word_counts.items()):
        position_counts = group_position_counts[(model_id, prefix_length)]
        most_words, least_words = top_k_words(dict(word_counts), top_k)
        most_positions, least_positions = top_k_positions(dict(position_counts), top_k)
        seed_count = sum(
            1
            for item in per_execution
            if item["model_id"] == model_id and item["prefix_length"] == prefix_length
        )
        aggregated.append(
            {
                "model_id": model_id,
                "prefix_length": prefix_length,
                "runs": seed_count,
                "most_edges": most_words,
                "least_edges": least_words,
                "most_edges_by_position": most_positions,
                "least_edges_by_position": least_positions,
                "all_edges_by_position": all_edges_by_position(dict(position_counts)),
            }
        )

    return WordEdgeResult(per_execution=per_execution, aggregated=aggregated)


def build_payload(
    runs: list[RunRecord],
    top_k: int,
    source: str,
) -> dict[str, Any]:
    result = analyze_runs(runs, top_k)
    return {
        "top_k": top_k,
        "word_edge_definition": WORD_EDGE_DEFINITION,
        "position_edge_definition": POSITION_EDGE_DEFINITION,
        "source": source,
        "per_execution": result.per_execution,
        "aggregated": result.aggregated,
    }
