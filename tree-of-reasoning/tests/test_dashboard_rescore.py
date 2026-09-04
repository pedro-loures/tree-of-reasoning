"""Tests for rescoring existing dashboard trees."""

from __future__ import annotations

from src.dashboard.payload import build_run_entry
from src.dashboard.rescore import rescore_run


def _sample_run() -> dict:
    tree_nodes = [
        {"id": "root", "d": 0, "p": 1.0, "c": ["n1", "n2"]},
        {"id": "n1", "d": 1, "p": 0.6, "c": [], "tok": "A"},
        {"id": "n2", "d": 1, "p": 0.4, "c": [], "tok": "B"},
    ]
    leaf_completions = [
        {"leaf_id": "n1", "path_prob": 0.6, "answer_correct": True, "answer_text": "Brasília", "reasoning_complete": True},
        {"leaf_id": "n2", "path_prob": 0.4, "answer_correct": False, "answer_text": "Rio", "reasoning_complete": True},
    ]
    candidate_nodes = [{"node_id": "root", "depth": 0, "n_leaves": 2, "status": "ditched"}]
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
    return build_run_entry(
        tree_key="interactive:test:0.01:abc",
        prompt="What is the capital of Brazil?",
        model_id="deepseek-r1-7b",
        tau=0.01,
        tree_nodes=tree_nodes,
        tree_metrics=tree_metrics,
        leaf_completions=leaf_completions,
        candidate_nodes=candidate_nodes,
        summary=summary,
        expected_answers="Brasília",
        answer_mode="or",
    )


def test_rescore_changes_leaf_correctness() -> None:
    run = _sample_run()
    updated = rescore_run(run, expected_answers="Rio", answer_mode="or")
    assert updated["leaf_completions"]["n1"]["answer_correct"] is False
    assert updated["leaf_completions"]["n2"]["answer_correct"] is True
    assert updated["expected_answers"] == "Rio"


def test_rescore_and_mode_marks_root_exclusively_bad() -> None:
    run = _sample_run()
    updated = rescore_run(run, expected_answers="Brasília, Brazil", answer_mode="and")
    assert updated["leaf_completions"]["n1"]["answer_correct"] is False
    assert updated["candidate_nodes"][0]["status"] == "exclusively_bad"
    assert updated["summary"]["exclusively_bad_count"] == 1
