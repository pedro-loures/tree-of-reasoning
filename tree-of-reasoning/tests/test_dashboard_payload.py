"""Tests for interactive dashboard payload helpers."""

from __future__ import annotations

from src.dashboard.payload import build_run_entry, compute_node_error_stats, merge_runs


def test_build_run_entry_and_merge() -> None:
    tree_nodes = [
        {"id": "root", "d": 0, "p": 1.0, "c": ["n1", "n2"]},
        {"id": "n1", "d": 1, "p": 0.6, "c": [], "tok": "A"},
        {"id": "n2", "d": 1, "p": 0.4, "c": [], "tok": "B"},
    ]
    leaf_completions = [
        {"leaf_id": "n1", "path_prob": 0.6, "answer_correct": True, "answer_text": "Brasília"},
        {"leaf_id": "n2", "path_prob": 0.4, "answer_correct": False, "answer_text": "Rio"},
    ]
    candidate_nodes = [
        {"node_id": "root", "depth": 0, "n_leaves": 2, "status": "ditched"},
    ]
    summary = {
        "total_leaves": 2,
        "total_candidates": 1,
        "exclusively_bad_count": 0,
        "ditched_count": 1,
        "leaf_subtype_counts": {"correct": 1, "wrong_city": 1},
    }
    tree_metrics = {
        "max_depth": 1,
        "total_nodes": 3,
        "leaf_count": 2,
        "mass_above_tau": 1.0,
        "breadth_warning_count": 0,
        "breadth_by_depth": {"0": 2},
        "max_child_breadth_by_depth": {"0": 2},
    }

    run = build_run_entry(
        tree_key="interactive:test:0.01:abc",
        prompt="What is the capital of Brazil?",
        model_id="deepseek-r1-7b",
        tau=0.01,
        tree_nodes=tree_nodes,
        tree_metrics=tree_metrics,
        leaf_completions=leaf_completions,
        candidate_nodes=candidate_nodes,
        summary=summary,
    )

    assert run["tree_summary"]["prompt_preview"].startswith("What is the capital")
    assert run["node_stats"]["root"]["n_bad"] == 1
    assert run["node_stats"]["root"]["color_class"] == "mixed"

    payload = merge_runs([run])
    assert payload["runs"][0]["tree_key"] == "interactive:test:0.01:abc"
    assert payload["trees"]["interactive:test:0.01:abc"][0]["id"] == "root"


def test_compute_node_error_stats_all_bad() -> None:
    tree_nodes = [
        {"id": "root", "d": 0, "p": 1.0, "c": ["n1", "n2"]},
        {"id": "n1", "d": 1, "p": 0.5, "c": []},
        {"id": "n2", "d": 1, "p": 0.5, "c": []},
    ]
    leaf_map = {
        "n1": {"answer_correct": False},
        "n2": {"answer_correct": False},
    }
    stats = compute_node_error_stats(tree_nodes, leaf_map)
    assert stats["root"]["color_class"] == "exclusively_bad"
