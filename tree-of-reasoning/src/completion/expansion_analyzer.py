"""Collect expandable exclusively-bad nodes and record expansion metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.completion.bad_node_analyzer import CandidateNodeResult
from src.tree.tau_builder import TauTreeNode


@dataclass
class ExpansionRecord:
    node_id: str
    tau_original: float
    tau_star: float
    leaves_before: int
    leaves_after: int
    new_leaf_ids: list[str]
    binary_search_probes: int
    hit_tau_floor: bool
    outcome_after_rescore: Literal["exclusively_bad", "ditched", "skipped"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "tau_original": self.tau_original,
            "tau_star": self.tau_star,
            "leaves_before": self.leaves_before,
            "leaves_after": self.leaves_after,
            "new_leaf_ids": self.new_leaf_ids,
            "binary_search_probes": self.binary_search_probes,
            "hit_tau_floor": self.hit_tau_floor,
            "outcome_after_rescore": self.outcome_after_rescore,
        }


def collect_expandable_nodes(
    candidate_results: list[CandidateNodeResult],
    nodes_by_id: dict[str, TauTreeNode],
    *,
    target_leaves: int = 10,
) -> list[CandidateNodeResult]:
    """Return exclusively bad nodes with fewer than target leaves, shallow first."""
    expandable = [
        item
        for item in candidate_results
        if item.status == "exclusively_bad" and item.n_leaves < target_leaves
    ]
    expandable.sort(key=lambda item: (item.depth, item.n_leaves, item.node_id))
    return expandable


def outcome_for_node(
    node_id: str,
    nodes_by_id: dict[str, TauTreeNode],
    candidate_by_id: dict[str, CandidateNodeResult],
) -> Literal["exclusively_bad", "ditched", "skipped"]:
    candidate = candidate_by_id.get(node_id)
    if candidate is None:
        return "skipped"
    return candidate.status  # type: ignore[return-value]


def summarize_expansion(
    expansion_records: list[ExpansionRecord],
    new_leaves_completed: int,
) -> dict[str, Any]:
    expanded = list(expansion_records)
    ditched = sum(1 for record in expanded if record.outcome_after_rescore == "ditched")
    still_bad = sum(1 for record in expanded if record.outcome_after_rescore == "exclusively_bad")
    tau_values = [record.tau_star for record in expanded if record.tau_star]
    avg_tau = sum(tau_values) / len(tau_values) if tau_values else None
    return {
        "nodes_expanded": len(expanded),
        "nodes_ditched_after": ditched,
        "nodes_still_exclusively_bad": still_bad,
        "new_leaves_completed": new_leaves_completed,
        "avg_tau_star": round(avg_tau, 6) if avg_tau is not None else None,
        "hit_tau_floor_count": sum(1 for record in expanded if record.hit_tau_floor),
    }


def attach_expansion_to_candidates(
    candidate_results: list[CandidateNodeResult],
    expansion_by_node: dict[str, ExpansionRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidate_results:
        row = candidate.to_dict()
        record = expansion_by_node.get(candidate.node_id)
        if record is not None:
            row["expansion"] = record.to_dict()
        rows.append(row)
    return rows
