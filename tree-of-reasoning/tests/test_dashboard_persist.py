"""Tests for dashboard tree persistence."""

from __future__ import annotations

import json

from src.dashboard.payload import build_run_entry, merge_runs
from src.dashboard.persist import (
    list_saved_runs,
    load_run_file,
    runs_from_session_payload,
    save_run,
)


def _sample_run() -> dict:
    tree_nodes = [
        {"id": "root", "d": 0, "p": 1.0, "c": ["n1"]},
        {"id": "n1", "d": 1, "p": 1.0, "c": [], "tok": "A"},
    ]
    leaf_completions = [
        {"leaf_id": "n1", "path_prob": 1.0, "answer_correct": True, "answer_text": "Brasília"},
    ]
    candidate_nodes = [{"node_id": "root", "depth": 0, "n_leaves": 1, "status": "ditched"}]
    summary = {
        "total_leaves": 1,
        "total_candidates": 1,
        "exclusively_bad_count": 0,
        "ditched_count": 1,
        "leaf_subtype_counts": {"correct": 1},
    }
    tree_metrics = {
        "max_depth": 1,
        "total_nodes": 2,
        "leaf_count": 1,
        "mass_above_tau": 1.0,
        "breadth_warning_count": 0,
        "breadth_by_depth": {"0": 1},
        "max_child_breadth_by_depth": {"0": 1},
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
    )


def test_save_and_load_run_roundtrip(tmp_path) -> None:
    run = _sample_run()
    path = save_run(run, tmp_path)
    assert path.exists()

    loaded = load_run_file(path)
    assert loaded["tree_key"] == run["tree_key"]
    assert loaded["tree_nodes"][1]["id"] == "n1"
    assert loaded["leaf_completions"]["n1"]["answer_text"] == "Brasília"


def test_list_saved_runs(tmp_path) -> None:
    run = _sample_run()
    save_run(run, tmp_path)
    entries = list_saved_runs(tmp_path)
    assert len(entries) == 1
    assert entries[0]["tree_key"] == run["tree_key"]
    assert entries[0]["filename"].endswith(".json")


def test_runs_from_session_payload() -> None:
    run = _sample_run()
    payload = merge_runs([run])
    rebuilt = runs_from_session_payload(payload)
    assert len(rebuilt) == 1
    assert rebuilt[0]["tree_key"] == run["tree_key"]
    assert rebuilt[0]["tree_nodes"] == run["tree_nodes"]
