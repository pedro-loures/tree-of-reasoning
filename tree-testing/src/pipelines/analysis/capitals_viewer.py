"""Export capitals experiment dashboard JSON (trees + aggregated summaries)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Callable

from src.pipelines.analysis.bad_nodes_viewer import compute_node_error_stats
from src.pipelines.analysis.capitals_regions import REGION_LABELS, region_for_country
from src.pipelines.analysis.tree_parser import compact_tree


def _instruction_variant(instruction: str) -> str:
    if instruction.strip().lower().startswith("ignore the previous text"):
        return "legacy"
    return "plain"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _make_tree_key(row: dict[str, Any]) -> str:
    if "tree_key" in row:
        return row["tree_key"]
    country_id = row.get("country_id", "")
    variant = _instruction_variant(row["instruction"])
    return (
        f"capitals:{row['model_id']}:{country_id}:{variant}:"
        f"{int(row['prefix_length'])}:{int(row.get('seed', 0))}"
    )


def _max_breadth(tree_metrics: dict[str, Any]) -> int | None:
    values = [int(value) for value in (tree_metrics.get("max_child_breadth_by_depth") or {}).values()]
    return max(values) if values else None


def _mean_internal_breadth(tree_metrics: dict[str, Any]) -> float | None:
    breadth_by_depth = tree_metrics.get("breadth_by_depth") or {}
    if not breadth_by_depth:
        return None
    values = [int(value) for value in breadth_by_depth.values()]
    return round(sum(values) / len(values), 2)


def compute_leaf_probability_masses(
    leaf_completions: list[dict[str, Any]],
    *,
    is_good: Callable[[dict[str, Any]], bool] | None = None,
    is_bad: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, float | None]:
    """Sum path probabilities by outcome class (good / bad / other).

    When ``is_good`` / ``is_bad`` are omitted, correctness uses ``answer_correct``
    and every non-correct leaf counts as bad.
    """
    total_mass = 0.0
    prob_good = 0.0
    prob_bad = 0.0
    prob_other = 0.0

    for leaf in leaf_completions:
        path_prob = float(leaf.get("path_prob", 0.0))
        total_mass += path_prob
        good = is_good(leaf) if is_good is not None else bool(leaf.get("answer_correct", False))
        bad = is_bad(leaf) if is_bad is not None else not good
        if good:
            prob_good += path_prob
        elif bad:
            prob_bad += path_prob
        else:
            prob_other += path_prob

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


def _node_expansions_from_row(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expansions: dict[str, dict[str, Any]] = {}
    for item in row.get("candidate_nodes", []):
        expansion = item.get("expansion")
        if expansion:
            expansions[item["node_id"]] = expansion
    return expansions


def build_tree_summary(
    mech: dict[str, Any],
    bad: dict[str, Any] | None,
    *,
    expanded: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tree_metrics = mech.get("tree_metrics", {})
    trace_metrics = mech.get("trace_metrics", {})
    top_k_metrics = mech.get("top_k_metrics", {})
    country_id = mech.get("country_id", "unknown")
    region_id = region_for_country(country_id)
    variant = _instruction_variant(mech["instruction"])

    summary: dict[str, Any] = {
        "tree_key": _make_tree_key(mech),
        "country_id": country_id,
        "country_name": mech.get("country_name", country_id),
        "region_id": region_id,
        "region_label": REGION_LABELS.get(region_id, region_id),
        "instruction_variant": variant,
        "prefix_length": int(mech["prefix_length"]),
        "seed": int(mech.get("seed", 0)),
        "model_id": mech["model_id"],
        "max_depth": tree_metrics.get("max_depth"),
        "total_nodes": tree_metrics.get("total_nodes"),
        "leaf_count": tree_metrics.get("leaf_count"),
        "mass_above_tau": tree_metrics.get("mass_above_tau"),
        "breadth_warning_count": tree_metrics.get("breadth_warning_count"),
        "max_breadth": _max_breadth(tree_metrics),
        "mean_breadth_by_depth": _mean_internal_breadth(tree_metrics),
        "reasoning_token_count": trace_metrics.get("reasoning_token_count"),
        "mean_entropy_reasoning": trace_metrics.get("mean_entropy_reasoning"),
        "mean_logprob_selected": trace_metrics.get("mean_logprob_selected"),
        "greedy_correct": int(bool(trace_metrics.get("answer_correct", False))),
        "top_1_correct": int(bool(top_k_metrics.get("top_1_correct", False))),
        "top_k_any_correct": int(bool(top_k_metrics.get("top_k_any_correct", False))),
        "has_bad_nodes": bad is not None,
        "has_expansion": expanded is not None,
    }

    if expanded:
        exp_summary = expanded.get("expansion_summary") or {}
        summary.update(
            {
                "nodes_expanded": int(exp_summary.get("nodes_expanded", 0)),
                "nodes_ditched_after": int(exp_summary.get("nodes_ditched_after", 0)),
                "avg_tau_star": exp_summary.get("avg_tau_star"),
            }
        )
    else:
        summary.update(
            {
                "nodes_expanded": None,
                "nodes_ditched_after": None,
                "avg_tau_star": None,
            }
        )

    if bad:
        bad_summary = bad.get("summary", {})
        subtypes = bad_summary.get("leaf_subtype_counts", {})
        leaf_correct = int(subtypes.get("correct", 0))
        leaf_total = int(bad_summary.get("total_leaves", 0))
        prob_masses = compute_leaf_probability_masses(bad.get("leaf_completions", []))
        summary.update(
            {
                "exclusively_bad_count": int(bad_summary.get("exclusively_bad_count", 0)),
                "ditched_count": int(bad_summary.get("ditched_count", 0)),
                "total_candidates": int(bad_summary.get("total_candidates", 0)),
                "total_leaves": leaf_total,
                "leaf_correct": leaf_correct,
                "leaf_wrong": max(0, leaf_total - leaf_correct),
                **prob_masses,
            }
        )
    else:
        summary.update(
            {
                "exclusively_bad_count": None,
                "ditched_count": None,
                "total_candidates": None,
                "total_leaves": None,
                "leaf_correct": None,
                "leaf_wrong": None,
                "prob_mass_total": None,
                "prob_good": None,
                "prob_bad": None,
                "prob_other": None,
                "prob_good_pct": None,
                "prob_bad_pct": None,
                "prob_other_pct": None,
            }
        )

    return summary


def export_capitals_dashboard(
    mech_interp_path: Path,
    bad_nodes_path: Path,
    output_path: Path,
    *,
    expanded_bad_nodes_path: Path | None = None,
) -> dict[str, Any]:
    mech_rows = _load_jsonl(mech_interp_path)
    bad_rows = _load_jsonl(bad_nodes_path)
    expanded_rows = _load_jsonl(expanded_bad_nodes_path) if expanded_bad_nodes_path else []
    bad_by_key = {_make_tree_key(row): row for row in bad_rows}
    expanded_by_key = {_make_tree_key(row): row for row in expanded_rows}

    tree_summaries: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    trees: dict[str, list[dict]] = {}
    node_status: dict[str, dict[str, str]] = {}
    leaf_completions: dict[str, dict[str, dict[str, Any]]] = {}
    node_stats: dict[str, dict[str, dict[str, Any]]] = {}
    node_expansions: dict[str, dict[str, dict[str, Any]]] = {}

    for mech in mech_rows:
        tree_key = _make_tree_key(mech)
        expanded = expanded_by_key.get(tree_key)
        bad = expanded if expanded is not None else bad_by_key.get(tree_key)
        tree_summaries.append(build_tree_summary(mech, bad, expanded=expanded))

        root_prefix = str(mech.get("root_prefix", ""))
        if expanded and expanded.get("tree"):
            tree_nodes = expanded["tree"]["nodes"]
            root_prefix = str(expanded.get("root_prefix", root_prefix))
        else:
            tree_nodes = mech["tree"]["nodes"]
        compact = compact_tree(tree_nodes, root_prefix=root_prefix)
        trees[tree_key] = compact

        if bad:
            status_map = {item["node_id"]: item["status"] for item in bad.get("candidate_nodes", [])}
            node_status[tree_key] = status_map
            leaf_map = {item["leaf_id"]: item for item in bad.get("leaf_completions", [])}
            leaf_completions[tree_key] = leaf_map
            node_stats[tree_key] = compute_node_error_stats(tree_nodes, leaf_map)
            expansions = _node_expansions_from_row(bad)
            if expansions:
                node_expansions[tree_key] = expansions
            runs.append(
                {
                    "tree_key": tree_key,
                    "country_id": mech.get("country_id"),
                    "country_name": mech.get("country_name"),
                    "region_id": region_for_country(mech.get("country_id")),
                    "region_label": REGION_LABELS.get(region_for_country(mech.get("country_id")), ""),
                    "model_id": mech["model_id"],
                    "instruction_variant": _instruction_variant(mech["instruction"]),
                    "prefix_length": int(mech["prefix_length"]),
                    "seed": int(mech.get("seed", 0)),
                    "summary": bad.get("summary", {}),
                    "baseline_summary": bad.get("baseline_summary"),
                    "expansion_summary": bad.get("expansion_summary"),
                    "has_expansion": expanded is not None,
                    "greedy_correct": mech["trace_metrics"]["answer_correct"],
                    "top_k_any_correct": mech["top_k_metrics"]["top_k_any_correct"],
                }
            )

    tree_summaries.sort(
        key=lambda item: (
            item.get("country_id") or "",
            item["instruction_variant"],
            int(item["prefix_length"]),
        )
    )
    runs.sort(
        key=lambda item: (
            item.get("country_id") or "",
            item["instruction_variant"],
            int(item["prefix_length"]),
        )
    )

    prefix_lengths = sorted({int(row["prefix_length"]) for row in tree_summaries})

    source_parts = [str(mech_interp_path), str(bad_nodes_path)]
    if expanded_bad_nodes_path is not None:
        source_parts.append(str(expanded_bad_nodes_path))
    payload = {
        "source": " + ".join(source_parts),
        "generated_at": date.today().isoformat(),
        "mech_interp_trees": len(mech_rows),
        "bad_nodes_trees": len(bad_rows),
        "expanded_bad_nodes_trees": len(expanded_rows),
        "viewer_runs": len(runs),
        "tree_summaries": tree_summaries,
        "prefix_lengths": prefix_lengths,
        "instruction_variants": ["legacy", "plain"],
        "runs": runs,
        "trees": trees,
        "node_status": node_status,
        "node_expansions": node_expansions,
        "leaf_completions": leaf_completions,
        "node_stats": node_stats,
        "regions": [{"id": k, "label": v} for k, v in REGION_LABELS.items()],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")))
    return payload
