"""Tests for capitals dashboard probability mass helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.analysis.capitals_viewer import (  # noqa: E402
    compute_leaf_probability_masses,
    export_capitals_dashboard,
)

def test_export_includes_node_expansions(tmp_path):
    mech_path = tmp_path / "mech.jsonl"
    bad_path = tmp_path / "bad.jsonl"
    expanded_path = tmp_path / "expanded.jsonl"
    out_path = tmp_path / "dashboard.json"

    mech_row = {
        "country_id": "brazil",
        "country_name": "Brazil",
        "model_id": "deepseek-r1-7b",
        "instruction": "What is the capital?",
        "prefix_length": 0,
        "seed": 0,
        "root_prefix": "Q",
        "tree": {"nodes": [{"id": "root", "depth": 0, "prefix_text": "Q", "path_prob": 1.0, "child_ids": []}]},
        "tree_metrics": {"max_depth": 0, "total_nodes": 1, "leaf_count": 1, "mass_above_tau": 1.0},
        "trace_metrics": {"reasoning_token_count": 1, "mean_entropy_reasoning": 0.1, "mean_logprob_selected": -0.1, "answer_correct": True},
        "top_k_metrics": {"top_1_correct": True, "top_k_any_correct": True},
    }
    bad_row = {
        "tree_key": "capitals:deepseek-r1-7b:brazil:plain:0:0",
        "summary": {"exclusively_bad_count": 1, "ditched_count": 0, "total_candidates": 1, "total_leaves": 2, "leaf_subtype_counts": {"correct": 0}},
        "candidate_nodes": [{"node_id": "d1_1", "status": "exclusively_bad", "n_leaves": 2}],
        "leaf_completions": [{"leaf_id": "d2_1", "path_prob": 0.5, "answer_correct": False}],
    }
    expanded_row = {
        **bad_row,
        "tree": mech_row["tree"],
        "expansion_summary": {"nodes_expanded": 1, "nodes_ditched_after": 0, "avg_tau_star": 0.006},
        "candidate_nodes": [
            {
                "node_id": "d1_1",
                "status": "exclusively_bad",
                "n_leaves": 12,
                "expansion": {
                    "node_id": "d1_1",
                    "tau_original": 0.01,
                    "tau_star": 0.006,
                    "leaves_before": 2,
                    "leaves_after": 12,
                    "new_leaf_ids": ["d3_1"],
                    "binary_search_probes": 4,
                    "hit_tau_floor": False,
                    "outcome_after_rescore": "exclusively_bad",
                },
            }
        ],
    }

    mech_path.write_text(json.dumps(mech_row) + "\n")
    bad_path.write_text(json.dumps(bad_row) + "\n")
    expanded_path.write_text(json.dumps(expanded_row) + "\n")

    payload = export_capitals_dashboard(
        mech_interp_path=mech_path,
        bad_nodes_path=bad_path,
        output_path=out_path,
        expanded_bad_nodes_path=expanded_path,
    )
    assert payload["expanded_bad_nodes_trees"] == 1
    tree_key = "capitals:deepseek-r1-7b:brazil:plain:0:0"
    assert payload["node_expansions"][tree_key]["d1_1"]["tau_star"] == 0.006
    assert payload["tree_summaries"][0]["nodes_expanded"] == 1



def test_probability_masses_weight_by_path_prob():
    leaves = [
        {"leaf_id": "a", "path_prob": 0.4, "answer_correct": True},
        {"leaf_id": "b", "path_prob": 0.35, "answer_correct": False},
        {"leaf_id": "c", "path_prob": 0.25, "answer_correct": False},
    ]
    masses = compute_leaf_probability_masses(leaves)
    assert masses["prob_mass_total"] == 1.0
    assert masses["prob_good_pct"] == 40.0
    assert masses["prob_bad_pct"] == 60.0
    assert masses["prob_other_pct"] == 0.0


def test_probability_masses_support_custom_bad_filter():
    leaves = [
        {"leaf_id": "a", "path_prob": 0.5, "answer_correct": True},
        {"leaf_id": "b", "path_prob": 0.3, "answer_correct": False},
        {"leaf_id": "c", "path_prob": 0.2, "answer_correct": False},
    ]
    masses = compute_leaf_probability_masses(
        leaves,
        is_good=lambda leaf: bool(leaf["answer_correct"]),
        is_bad=lambda leaf: leaf["leaf_id"] == "b",
    )
    assert masses["prob_good_pct"] == 50.0
    assert masses["prob_bad_pct"] == 30.0
    assert masses["prob_other_pct"] == 20.0
