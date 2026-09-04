"""Unit tests for expansion analyzer helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.completion.bad_node_analyzer import CandidateNodeResult  # noqa: E402
from src.completion.expansion_analyzer import (  # noqa: E402
    ExpansionRecord,
    attach_expansion_to_candidates,
    collect_expandable_nodes,
    summarize_expansion,
)
from src.tree.tau_builder import TauTreeNode  # noqa: E402


def test_collect_expandable_nodes_shallow_first():
    candidates = [
        CandidateNodeResult("deep", depth=5, n_leaves=2, status="exclusively_bad"),
        CandidateNodeResult("shallow", depth=2, n_leaves=3, status="exclusively_bad"),
        CandidateNodeResult("ok", depth=2, n_leaves=10, status="exclusively_bad"),
        CandidateNodeResult("ditched", depth=2, n_leaves=2, status="ditched"),
    ]
    nodes = {
        "deep": TauTreeNode("deep", 5, "p", 0.0, 1.0),
        "shallow": TauTreeNode("shallow", 2, "p", 0.0, 1.0),
    }
    expandable = collect_expandable_nodes(candidates, nodes, target_leaves=10)
    assert [item.node_id for item in expandable] == ["shallow", "deep"]


def test_attach_expansion_to_candidates():
    candidates = [
        CandidateNodeResult("n1", depth=2, n_leaves=12, status="exclusively_bad"),
    ]
    record = ExpansionRecord(
        node_id="n1",
        tau_original=0.01,
        tau_star=0.006,
        leaves_before=2,
        leaves_after=12,
        new_leaf_ids=["d3_9"],
        binary_search_probes=5,
        hit_tau_floor=False,
        outcome_after_rescore="exclusively_bad",
    )
    rows = attach_expansion_to_candidates(candidates, {"n1": record})
    assert rows[0]["expansion"]["tau_star"] == 0.006


def test_summarize_expansion():
    records = [
        ExpansionRecord(
            node_id="a",
            tau_original=0.01,
            tau_star=0.006,
            leaves_before=2,
            leaves_after=11,
            new_leaf_ids=["l1"],
            binary_search_probes=4,
            hit_tau_floor=False,
            outcome_after_rescore="exclusively_bad",
        ),
        ExpansionRecord(
            node_id="b",
            tau_original=0.01,
            tau_star=0.004,
            leaves_before=3,
            leaves_after=10,
            new_leaf_ids=["l2"],
            binary_search_probes=6,
            hit_tau_floor=False,
            outcome_after_rescore="ditched",
        ),
    ]
    summary = summarize_expansion(records, new_leaves_completed=2)
    assert summary["nodes_expanded"] == 2
    assert summary["nodes_ditched_after"] == 1
    assert summary["nodes_still_exclusively_bad"] == 1
