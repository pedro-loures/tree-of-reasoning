"""Identify internal tau-tree nodes whose subtrees yield exclusively bad answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.completion.leaf_completer import LeafCompletionResult
from src.data.countries import instruction_variant
from src.tree.tau_builder import TauTreeNode, TauTreeResult

from src.utils.politician_mentions import (
    MENTION_CATEGORY_ONLY_CANDIDATES,
    MENTION_CATEGORY_POLITICIANS,
    mentions_presidential_candidate,
)

FailureSubtype = Literal[
    "wrong_city",
    "incomplete",
    "empty",
    "lorem_drift",
    "mentioned_politicians",
    "mentioned_candidates",
    "no_politicians_mentioned",
]
BadSubtype = Literal[
    "correct",
    "wrong_city",
    "incomplete",
    "empty",
    "lorem_drift",
    "mentioned_politicians",
    "mentioned_candidates",
    "no_politicians_mentioned",
]


@dataclass
class CandidateNodeResult:
    node_id: str
    depth: int
    n_leaves: int
    status: Literal["exclusively_bad", "ditched"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "n_leaves": self.n_leaves,
            "status": self.status,
        }


def make_tree_key(
    dataset_id: str,
    model_id: str,
    variant: str,
    prefix_length: int,
    seed: int,
    country_id: str | None = None,
) -> str:
    if country_id:
        return f"{dataset_id}:{model_id}:{country_id}:{variant}:{prefix_length}:{seed}"
    return f"{dataset_id}:{model_id}:{variant}:{prefix_length}:{seed}"


def tree_from_dict(tree_dict: dict[str, Any]) -> TauTreeResult:
    nodes = [
        TauTreeNode(
            id=node["id"],
            depth=int(node["depth"]),
            prefix_text=node["prefix_text"],
            log_path_prob=float(node["log_path_prob"]),
            path_prob=float(node["path_prob"]),
            parent_id=node.get("parent_id"),
            child_ids=list(node.get("child_ids", [])),
            child_token_ids=list(node.get("child_token_ids", [])),
            child_tokens=list(node.get("child_tokens", [])),
            breadth=int(node.get("breadth", 0)),
        )
        for node in tree_dict["nodes"]
    ]
    from src.tree.tau_builder import BreadthWarning

    warnings = [
        BreadthWarning(
            node_id=warning["node_id"],
            depth=int(warning["depth"]),
            breadth=int(warning["breadth"]),
        )
        for warning in tree_dict.get("breadth_warnings", [])
    ]
    return TauTreeResult(
        nodes=nodes,
        breadth_warnings=warnings,
        breadth_warning_count=int(tree_dict.get("breadth_warning_count", len(warnings))),
        tau=float(tree_dict["tau"]),
        max_tree_depth=int(tree_dict["max_tree_depth"]),
        root_prefix=tree_dict["root_prefix"],
    )


def build_node_index(tree: TauTreeResult) -> dict[str, TauTreeNode]:
    return {node.id: node for node in tree.nodes}


def leaf_descendants(node_id: str, nodes_by_id: dict[str, TauTreeNode]) -> list[str]:
    node = nodes_by_id[node_id]
    if not node.child_ids:
        return [node_id]
    leaves: list[str] = []
    for child_id in node.child_ids:
        leaves.extend(leaf_descendants(child_id, nodes_by_id))
    return leaves


def candidate_nodes(nodes_by_id: dict[str, TauTreeNode]) -> list[tuple[int, int, str]]:
    """Return internal nodes with at least two leaf descendants."""
    candidates: list[tuple[int, int, str]] = []
    for node_id, node in nodes_by_id.items():
        if not node.child_ids:
            continue
        leaves = leaf_descendants(node_id, nodes_by_id)
        if len(leaves) >= 2:
            candidates.append((node.depth, len(leaves), node_id))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return candidates


def classify_bad_subtypes(result: LeafCompletionResult) -> list[FailureSubtype]:
    """Return all failure tags that apply to a leaf (not mutually exclusive)."""
    if result.answer_correct:
        return []

    if result.mention_category is not None:
        subtypes: list[FailureSubtype] = []
        if not result.reasoning_complete:
            subtypes.append("incomplete")
        if not result.answer_text.strip():
            subtypes.append("empty")
        if result.mentions_lorem:
            subtypes.append("lorem_drift")
        if result.reasoning_complete:
            if mentions_presidential_candidate(result.mentions or []) or (
                result.mention_category == MENTION_CATEGORY_ONLY_CANDIDATES
            ):
                subtypes.append("mentioned_candidates")
            elif result.mention_category == MENTION_CATEGORY_POLITICIANS:
                subtypes.append("mentioned_politicians")
        return subtypes or ["mentioned_candidates"]

    subtypes: list[FailureSubtype] = []
    if not result.reasoning_complete:
        subtypes.append("incomplete")
    if not result.answer_text.strip():
        subtypes.append("empty")
    if result.mentions_lorem:
        subtypes.append("lorem_drift")
    if result.reasoning_complete and result.answer_text.strip():
        subtypes.append("wrong_city")
    return subtypes


def classify_bad_subtype(result: LeafCompletionResult) -> BadSubtype:
    """Single-label view for backward compatibility (first matching failure tag)."""
    if result.answer_correct:
        return "correct"
    subtypes = classify_bad_subtypes(result)
    return subtypes[0] if subtypes else "wrong_city"


def evaluate_candidates(
    nodes_by_id: dict[str, TauTreeNode],
    leaf_cache: dict[str, LeafCompletionResult],
) -> list[CandidateNodeResult]:
    results: list[CandidateNodeResult] = []
    for depth, n_leaves, node_id in candidate_nodes(nodes_by_id):
        leaf_ids = leaf_descendants(node_id, nodes_by_id)
        any_correct = any(leaf_cache[leaf_id].answer_correct for leaf_id in leaf_ids)
        results.append(
            CandidateNodeResult(
                node_id=node_id,
                depth=depth,
                n_leaves=n_leaves,
                status="ditched" if any_correct else "exclusively_bad",
            )
        )
    return results


def summarize_bad_nodes(
    leaf_cache: dict[str, LeafCompletionResult],
    candidate_results: list[CandidateNodeResult],
) -> dict[str, Any]:
    exclusively_bad = sum(1 for item in candidate_results if item.status == "exclusively_bad")
    ditched = sum(1 for item in candidate_results if item.status == "ditched")
    subtypes: dict[str, int] = {}
    for result in leaf_cache.values():
        failure_subtypes = classify_bad_subtypes(result)
        if not failure_subtypes:
            subtypes["correct"] = subtypes.get("correct", 0) + 1
            continue
        for subtype in failure_subtypes:
            subtypes[subtype] = subtypes.get(subtype, 0) + 1
    return {
        "total_leaves": len(leaf_cache),
        "total_candidates": len(candidate_results),
        "exclusively_bad_count": exclusively_bad,
        "ditched_count": ditched,
        "leaf_subtype_counts": subtypes,
    }
