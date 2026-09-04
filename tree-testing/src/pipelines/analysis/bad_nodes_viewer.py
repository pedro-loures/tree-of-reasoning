"""Export combined tree + bad-node data for the HTML graph viewer."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from src.pipelines.analysis.io import load_all


def _max_breadth(tree_metrics: dict[str, Any]) -> int | None:
    values = [int(value) for value in (tree_metrics.get("max_child_breadth_by_depth") or {}).values()]
    return max(values) if values else None


def _mean_internal_breadth(tree_metrics: dict[str, Any]) -> float | None:
    breadth_by_depth = tree_metrics.get("breadth_by_depth") or {}
    if not breadth_by_depth:
        return None
    values = [int(value) for value in breadth_by_depth.values()]
    return round(sum(values) / len(values), 2)


def _metrics_from_tree_nodes(nodes: list[dict[str, Any]]) -> dict[str, int | None]:
    if not nodes:
        return {}
    return {
        "max_depth": max(int(node.get("d", 0)) for node in nodes),
        "leaf_count": sum(1 for node in nodes if not (node.get("c") or [])),
        "total_nodes": len(nodes),
    }


def compute_leaf_probability_masses(leaf_completions: list[dict[str, Any]]) -> dict[str, float | None]:
    total_mass = 0.0
    prob_good = 0.0
    prob_bad = 0.0
    prob_other = 0.0
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
            "prob_other": 0.0,
            "prob_good_pct": None,
            "prob_bad_pct": None,
            "prob_other_pct": None,
        }
    return {
        "prob_mass_total": round(total_mass, 6),
        "prob_good": round(prob_good, 6),
        "prob_bad": round(prob_bad, 6),
        "prob_other": round(prob_other, 6),
        "prob_good_pct": round(100.0 * prob_good / total_mass, 1),
        "prob_bad_pct": round(100.0 * prob_bad / total_mass, 1),
        "prob_other_pct": round(100.0 * prob_other / total_mass, 1),
    }


def classify_color_class(bad_pct: float) -> str:
    if bad_pct >= 1.0:
        return "exclusively_bad"
    if bad_pct <= 0.0:
        return "exclusively_good"
    if bad_pct > 0.75:
        return "mostly_bad"
    return "mixed"


def compute_node_error_stats(
    tree_nodes: list[dict[str, Any]],
    leaf_completions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute bad-answer rate for every node with at least one leaf descendant."""
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


def leaf_descendants_compact(node_id: str, by_id: dict[str, dict[str, Any]]) -> list[str]:
    node = by_id[node_id]
    child_ids = node.get("c") or node.get("child_ids") or []
    if not child_ids:
        return [node_id]
    leaves: list[str] = []
    for child_id in child_ids:
        leaves.extend(leaf_descendants_compact(child_id, by_id))
    return leaves


def load_bad_node_rows(bad_nodes_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(bad_nodes_dir.glob("*.jsonl")):
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _canvas_run_index(canvas: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {run["tree_key"]: run for run in canvas.get("runs", [])}


def _count_exclusively_good_internal(node_stats: dict[str, dict[str, Any]]) -> int:
    return sum(
        1
        for stats in node_stats.values()
        if stats.get("color_class") == "exclusively_good" and int(stats.get("n_leaves", 0)) >= 2
    )


def build_tree_summary(
    bad_row: dict[str, Any],
    canvas_run: dict[str, Any] | None,
    node_stats: dict[str, dict[str, Any]],
    mech_row: dict[str, Any] | None = None,
    tree_nodes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = bad_row.get("summary", {})
    subtypes = summary.get("leaf_subtype_counts", {})
    leaf_total = int(summary.get("total_leaves", 0))
    leaf_correct = int(subtypes.get("correct", 0))
    candidates = int(summary.get("total_candidates", 0))
    exclusively_bad = int(summary.get("exclusively_bad_count", 0))
    prob_masses = compute_leaf_probability_masses(bad_row.get("leaf_completions", []))

    tree_metrics = (mech_row or {}).get("tree_metrics", {})
    if not tree_metrics and tree_nodes:
        tree_metrics = _metrics_from_tree_nodes(tree_nodes)

    return {
        "tree_key": bad_row["tree_key"],
        "model_id": bad_row["model_id"],
        "instruction_variant": bad_row.get("instruction_variant", "legacy"),
        "prefix_length": int(bad_row["prefix_length"]),
        "seed": int(bad_row.get("seed", 0)),
        "country_id": bad_row.get("country_id"),
        "country_name": bad_row.get("country_name"),
        "max_depth": tree_metrics.get("max_depth"),
        "total_nodes": tree_metrics.get("total_nodes") or (canvas_run or {}).get("total_nodes"),
        "leaf_count": tree_metrics.get("leaf_count"),
        "mass_above_tau": tree_metrics.get("mass_above_tau") or (canvas_run or {}).get("mass_above_tau"),
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


def build_aggregates(tree_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per model × prefix × variant (seed 0 only for tau001)."""
    return sorted(
        tree_summaries,
        key=lambda row: (
            row["model_id"],
            row["instruction_variant"],
            int(row["prefix_length"]),
        ),
    )


def export_bad_nodes_viewer_data(
    canvas_trees_path: Path,
    bad_nodes_dir: Path,
    output_path: Path,
    mech_interp_dir: Path | None = None,
    dataset_id: str = "tau001",
) -> dict[str, Any]:
    canvas = json.loads(canvas_trees_path.read_text())
    bad_rows = load_bad_node_rows(bad_nodes_dir)
    canvas_runs = _canvas_run_index(canvas)
    mech_by_key: dict[str, dict[str, Any]] = {}
    if mech_interp_dir and mech_interp_dir.is_dir():
        mech_by_key = {
            record.tree_key: record.raw
            for record in load_all(mech_interp_dir, dataset_id=dataset_id)
        }

    runs: list[dict[str, Any]] = []
    tree_summaries: list[dict[str, Any]] = []
    node_status: dict[str, dict[str, str]] = {}
    leaf_completions: dict[str, dict[str, dict[str, Any]]] = {}
    node_stats: dict[str, dict[str, dict[str, Any]]] = {}

    for row in bad_rows:
        tree_key = row["tree_key"]
        if tree_key not in canvas["trees"]:
            continue

        status_map = {item["node_id"]: item["status"] for item in row.get("candidate_nodes", [])}
        node_status[tree_key] = status_map
        leaf_map = {item["leaf_id"]: item for item in row.get("leaf_completions", [])}
        leaf_completions[tree_key] = leaf_map
        tree_nodes = canvas["trees"][tree_key]
        stats = compute_node_error_stats(tree_nodes, leaf_map)
        node_stats[tree_key] = stats
        tree_summaries.append(
            build_tree_summary(
                row,
                canvas_runs.get(tree_key),
                stats,
                mech_row=mech_by_key.get(tree_key),
                tree_nodes=tree_nodes,
            )
        )

        runs.append(
            {
                "tree_key": tree_key,
                "country_id": row.get("country_id"),
                "country_name": row.get("country_name"),
                "model_id": row["model_id"],
                "instruction_variant": row.get("instruction_variant", "legacy"),
                "prefix_length": row["prefix_length"],
                "seed": row["seed"],
                "summary": row.get("summary", {}),
                "candidate_nodes": row.get("candidate_nodes", []),
                "exclusively_bad_nodes": [
                    item["node_id"]
                    for item in row.get("candidate_nodes", [])
                    if item.get("status") == "exclusively_bad"
                ],
            }
        )

    runs.sort(
        key=lambda item: (
            item.get("country_id") or "",
            item["model_id"],
            item["instruction_variant"],
            int(item["prefix_length"]),
            int(item["seed"]),
        )
    )

    tree_keys = {run["tree_key"] for run in runs}
    trees = {key: canvas["trees"][key] for key in tree_keys if key in canvas["trees"]}
    models = sorted({row["model_id"] for row in tree_summaries})
    prefix_lengths = sorted({int(row["prefix_length"]) for row in tree_summaries})

    payload = {
        "source": f"{canvas_trees_path} + {bad_nodes_dir}",
        "generated_at": date.today().isoformat(),
        "models": models,
        "prefix_lengths": prefix_lengths,
        "tree_summaries": tree_summaries,
        "aggregates": build_aggregates(tree_summaries),
        "runs": runs,
        "trees": trees,
        "node_status": node_status,
        "leaf_completions": leaf_completions,
        "node_stats": node_stats,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")))
    return payload
