"""Build viewer JSON payloads for the interactive dashboard."""

from __future__ import annotations

from datetime import date
from typing import Any


def _max_breadth(tree_metrics: dict[str, Any]) -> int | None:
    values = [int(value) for value in (tree_metrics.get("max_child_breadth_by_depth") or {}).values()]
    return max(values) if values else None


def _mean_internal_breadth(tree_metrics: dict[str, Any]) -> float | None:
    breadth_by_depth = tree_metrics.get("breadth_by_depth") or {}
    if not breadth_by_depth:
        return None
    values = [int(value) for value in breadth_by_depth.values()]
    return round(sum(values) / len(values), 2)


def compute_leaf_probability_masses(leaf_completions: list[dict[str, Any]]) -> dict[str, float | None]:
    total_mass = 0.0
    prob_good = 0.0
    prob_bad = 0.0
    for leaf in leaf_completions:
        path_prob = float(leaf.get("path_prob", 0.0))
        total_mass += path_prob
        if leaf.get("answer_correct", False):
            prob_good += path_prob
        else:
            prob_bad += path_prob
    if total_mass <= 0:
        return {
            "prob_mass_total": 0.0,
            "prob_good": 0.0,
            "prob_bad": 0.0,
            "prob_good_pct": None,
            "prob_bad_pct": None,
        }
    return {
        "prob_mass_total": round(total_mass, 6),
        "prob_good": round(prob_good, 6),
        "prob_bad": round(prob_bad, 6),
        "prob_good_pct": round(100.0 * prob_good / total_mass, 1),
        "prob_bad_pct": round(100.0 * prob_bad / total_mass, 1),
    }


def classify_color_class(bad_pct: float) -> str:
    if bad_pct >= 1.0:
        return "exclusively_bad"
    if bad_pct <= 0.0:
        return "exclusively_good"
    if bad_pct > 0.75:
        return "mostly_bad"
    return "mixed"


def leaf_descendants_compact(node_id: str, by_id: dict[str, dict[str, Any]]) -> list[str]:
    node = by_id[node_id]
    child_ids = node.get("c") or node.get("child_ids") or []
    if not child_ids:
        return [node_id]
    leaves: list[str] = []
    for child_id in child_ids:
        leaves.extend(leaf_descendants_compact(child_id, by_id))
    return leaves


def compute_node_error_stats(
    tree_nodes: list[dict[str, Any]],
    leaf_completions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_id = {node["id"]: node for node in tree_nodes}
    stats: dict[str, dict[str, Any]] = {}

    for node_id in by_id:
        leaf_ids = leaf_descendants_compact(node_id, by_id)
        if not leaf_ids:
            continue

        n_bad = sum(
            1
            for leaf_id in leaf_ids
            if not leaf_completions.get(leaf_id, {}).get("answer_correct", False)
        )
        n_leaves = len(leaf_ids)
        bad_pct = n_bad / n_leaves
        stats[node_id] = {
            "n_leaves": n_leaves,
            "n_bad": n_bad,
            "n_good": n_leaves - n_bad,
            "bad_pct": round(bad_pct, 6),
            "bad_pct_display": round(bad_pct * 100, 1),
            "color_class": classify_color_class(bad_pct),
        }
    return stats


def _count_exclusively_good_internal(node_stats: dict[str, dict[str, Any]]) -> int:
    return sum(
        1
        for stats in node_stats.values()
        if stats.get("color_class") == "exclusively_good" and int(stats.get("n_leaves", 0)) >= 2
    )


def build_tree_summary(
    *,
    tree_key: str,
    prompt: str,
    model_id: str,
    tau: float,
    tree_metrics: dict[str, Any],
    summary: dict[str, Any],
    leaf_completions: list[dict[str, Any]],
    node_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    subtypes = summary.get("leaf_subtype_counts", {})
    leaf_total = int(summary.get("total_leaves", 0))
    leaf_correct = int(subtypes.get("correct", 0))
    candidates = int(summary.get("total_candidates", 0))
    exclusively_bad = int(summary.get("exclusively_bad_count", 0))
    prob_masses = compute_leaf_probability_masses(leaf_completions)
    prompt_preview = prompt.replace("\n", " ").strip()
    if len(prompt_preview) > 72:
        prompt_preview = prompt_preview[:69] + "..."

    return {
        "tree_key": tree_key,
        "prompt": prompt,
        "prompt_preview": prompt_preview,
        "model_id": model_id,
        "tau": tau,
        "max_depth": tree_metrics.get("max_depth"),
        "total_nodes": tree_metrics.get("total_nodes"),
        "leaf_count": tree_metrics.get("leaf_count"),
        "mass_above_tau": tree_metrics.get("mass_above_tau"),
        "breadth_warning_count": tree_metrics.get("breadth_warning_count"),
        "max_breadth": _max_breadth(tree_metrics),
        "mean_breadth_by_depth": _mean_internal_breadth(tree_metrics),
        "total_leaves": leaf_total,
        "total_candidates": candidates,
        "exclusively_bad_count": exclusively_bad,
        "exclusively_bad_pct": round(100.0 * exclusively_bad / candidates, 1) if candidates else None,
        "ditched_count": int(summary.get("ditched_count", 0)),
        "exclusively_good_count": _count_exclusively_good_internal(node_stats),
        "leaf_correct": leaf_correct,
        "leaf_wrong": max(0, leaf_total - leaf_correct),
        "leaf_correct_pct": round(100.0 * leaf_correct / leaf_total, 1) if leaf_total else None,
        **prob_masses,
    }


def build_run_entry(
    *,
    tree_key: str,
    prompt: str,
    model_id: str,
    tau: float,
    tree_nodes: list[dict[str, Any]],
    tree_metrics: dict[str, Any],
    leaf_completions: list[dict[str, Any]],
    candidate_nodes: list[dict[str, Any]],
    summary: dict[str, Any],
    expected_answers: str | None = None,
    answer_mode: str | None = None,
    embeddings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    leaf_map = {item["leaf_id"]: item for item in leaf_completions}
    node_status = {item["node_id"]: item["status"] for item in candidate_nodes}
    node_stats = compute_node_error_stats(tree_nodes, leaf_map)
    tree_summary = build_tree_summary(
        tree_key=tree_key,
        prompt=prompt,
        model_id=model_id,
        tau=tau,
        tree_metrics=tree_metrics,
        summary=summary,
        leaf_completions=leaf_completions,
        node_stats=node_stats,
    )
    return {
        "tree_key": tree_key,
        "prompt": prompt,
        "model_id": model_id,
        "tau": tau,
        "expected_answers": expected_answers,
        "answer_mode": answer_mode,
        "summary": summary,
        "candidate_nodes": candidate_nodes,
        "tree_summary": tree_summary,
        "tree_nodes": tree_nodes,
        "node_status": node_status,
        "leaf_completions": leaf_map,
        "node_stats": node_stats,
        "embeddings": embeddings,
    }


def merge_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    tree_summaries: list[dict[str, Any]] = []
    payload_runs: list[dict[str, Any]] = []
    trees: dict[str, list[dict[str, Any]]] = {}
    node_status: dict[str, dict[str, str]] = {}
    leaf_completions: dict[str, dict[str, dict[str, Any]]] = {}
    node_stats: dict[str, dict[str, dict[str, Any]]] = {}

    for run in runs:
        tree_key = run["tree_key"]
        tree_summaries.append(run["tree_summary"])
        payload_runs.append(
            {
                "tree_key": tree_key,
                "prompt": run["prompt"],
                "model_id": run["model_id"],
                "tau": run["tau"],
                "expected_answers": run.get("expected_answers"),
                "answer_mode": run.get("answer_mode"),
                "summary": run["summary"],
                "candidate_nodes": run["candidate_nodes"],
                "embeddings": run.get("embeddings"),
                "exclusively_bad_nodes": [
                    item["node_id"]
                    for item in run["candidate_nodes"]
                    if item.get("status") == "exclusively_bad"
                ],
            }
        )
        trees[tree_key] = run["tree_nodes"]
        node_status[tree_key] = run["node_status"]
        leaf_completions[tree_key] = run["leaf_completions"]
        node_stats[tree_key] = run["node_stats"]

    models = sorted({run["model_id"] for run in runs})
    taus = sorted({float(run["tau"]) for run in runs})

    return {
        "source": "interactive dashboard",
        "generated_at": date.today().isoformat(),
        "models": models,
        "taus": taus,
        "tree_summaries": tree_summaries,
        "runs": payload_runs,
        "trees": trees,
        "node_status": node_status,
        "leaf_completions": leaf_completions,
        "node_stats": node_stats,
    }
