"""Rescore existing dashboard trees with new expected answers."""

from __future__ import annotations

from typing import Any

from src.completion.bad_node_analyzer import CandidateNodeResult, summarize_bad_nodes
from src.completion.leaf_completer import leaf_completion_from_dict
from src.dashboard.payload import build_run_entry, leaf_descendants_compact
from src.dashboard.service import parse_expected_answers
from src.utils.answer_scoring import evaluate_expected_answers, normalize_answer_mode


def _child_ids(node: dict[str, Any]) -> list[str]:
    return list(node.get("c") or node.get("child_ids") or [])


def candidate_nodes_compact(tree_nodes: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    by_id = {node["id"]: node for node in tree_nodes}
    candidates: list[tuple[int, int, str]] = []
    for node_id, node in by_id.items():
        if not _child_ids(node):
            continue
        leaves = leaf_descendants_compact(node_id, by_id)
        if len(leaves) >= 2:
            candidates.append((int(node["d"]), len(leaves), node_id))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return candidates


def evaluate_candidates_compact(
    tree_nodes: list[dict[str, Any]],
    leaf_cache: dict[str, Any],
) -> list[CandidateNodeResult]:
    by_id = {node["id"]: node for node in tree_nodes}
    results: list[CandidateNodeResult] = []
    for depth, n_leaves, node_id in candidate_nodes_compact(tree_nodes):
        leaf_ids = leaf_descendants_compact(node_id, by_id)
        any_correct = any(
            leaf_cache[leaf_id].answer_correct
            for leaf_id in leaf_ids
            if leaf_id in leaf_cache
        )
        results.append(
            CandidateNodeResult(
                node_id=node_id,
                depth=depth,
                n_leaves=n_leaves,
                status="ditched" if any_correct else "exclusively_bad",
            )
        )
    return results


def tree_metrics_from_summary(tree_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_depth": tree_summary.get("max_depth"),
        "total_nodes": tree_summary.get("total_nodes"),
        "leaf_count": tree_summary.get("leaf_count"),
        "mass_above_tau": tree_summary.get("mass_above_tau"),
        "breadth_warning_count": tree_summary.get("breadth_warning_count"),
        "breadth_by_depth": {},
        "max_child_breadth_by_depth": {},
    }


def rescore_run(
    run: dict[str, Any],
    expected_answers: str | None,
    answer_mode: str | None = "or",
) -> dict[str, Any]:
    cache = {
        leaf_id: leaf_completion_from_dict(leaf_data)
        for leaf_id, leaf_data in run["leaf_completions"].items()
    }
    accepted = parse_expected_answers(expected_answers)
    resolved_mode = normalize_answer_mode(answer_mode)

    if accepted:
        for result in cache.values():
            if result.reasoning_complete:
                correct, hits, mode = evaluate_expected_answers(
                    result.answer_text,
                    expected_answers,
                    resolved_mode,
                )
                result.answer_correct = correct
                result.answer_mode = mode
                result.answer_matches = hits
            else:
                result.answer_correct = False
                result.answer_mode = resolved_mode
                result.answer_matches = {}
    else:
        for result in cache.values():
            result.answer_correct = result.reasoning_complete
            result.answer_mode = None
            result.answer_matches = None
        resolved_mode = None

    candidate_results = evaluate_candidates_compact(run["tree_nodes"], cache)
    summary = summarize_bad_nodes(cache, candidate_results)
    leaf_completions = [item.to_dict() for item in cache.values()]
    tree_summary = run.get("tree_summary") or {}

    return build_run_entry(
        tree_key=run["tree_key"],
        prompt=run["prompt"],
        model_id=run["model_id"],
        tau=run["tau"],
        tree_nodes=run["tree_nodes"],
        tree_metrics=tree_metrics_from_summary(tree_summary),
        leaf_completions=leaf_completions,
        candidate_nodes=[item.to_dict() for item in candidate_results],
        summary=summary,
        expected_answers=expected_answers.strip() if expected_answers else None,
        answer_mode=resolved_mode if accepted else None,
    )
