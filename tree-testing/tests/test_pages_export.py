"""Tests for GitHub Pages export pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pipelines.analysis.pages_export import (
    build_tree_shard,
    build_tutorial_payload,
    compact_leaf_for_publish,
    compact_run_for_manifest,
    split_payload_to_shards,
    tree_key_to_slug,
)


def test_compact_leaf_drops_completion_text():
    leaf = {
        "leaf_id": "d1_1",
        "path_prob": 0.1,
        "answer_text": "Yes",
        "answer_correct": True,
        "completion_text": "x" * 5000,
    }
    compact = compact_leaf_for_publish(leaf, preview_chars=100)
    assert "completion_text" not in compact
    assert compact["completion_preview"] == "x" * 100
    assert compact["answer_text"] == "Yes"


def test_tree_key_to_slug():
    slug = tree_key_to_slug("interactive:deepseek-r1-7b:0.01:abc:def")
    assert ":" not in slug
    assert slug.startswith("interactive")


def test_compact_run_for_manifest_strips_heavy_fields():
    run = {
        "tree_key": "interactive:test",
        "prompt": "Hi",
        "model_id": "m",
        "tau": 0.01,
        "candidate_nodes": [{"node_id": "d1_1", "status": "ditched"}] * 500,
        "embeddings": {"d1_1": [0.1, 0.2]},
        "tree_nodes": [{"id": "root"}],
    }
    compact = compact_run_for_manifest(run)
    assert compact["tree_key"] == "interactive:test"
    assert "candidate_nodes" not in compact
    assert "embeddings" not in compact
    assert "tree_nodes" not in compact


def test_split_payload_to_shards(tmp_path: Path):
    payload = {
        "source": "/abs/path/capitals.jsonl",
        "generated_at": "2026-09-04",
        "mech_interp_trees": 1,
        "bad_nodes_trees": 1,
        "viewer_runs": 1,
        "tree_summaries": [{"tree_key": "capitals:test:brazil:plain:0:0", "country_name": "Brazil"}],
        "runs": [
            {
                "tree_key": "capitals:test:brazil:plain:0:0",
                "country_id": "brazil",
                "candidate_nodes": [{"node_id": "d1_1", "status": "ditched"}],
            },
        ],
        "trees": {
            "capitals:test:brazil:plain:0:0": [{"id": "root", "t": "", "c": ["d1_1"]}, {"id": "d1_1", "t": "Hi", "p": 0.5}],
        },
        "node_status": {"capitals:test:brazil:plain:0:0": {"root": "ditched"}},
        "node_stats": {"capitals:test:brazil:plain:0:0": {}},
        "leaf_completions": {
            "capitals:test:brazil:plain:0:0": {
                "d1_1": {"leaf_id": "d1_1", "answer_text": "Brasília", "completion_text": "long" * 100},
            },
        },
        "node_expansions": {},
    }
    manifest = split_payload_to_shards(payload, tmp_path, "capitals")
    assert manifest["experiment"] == "capitals"
    assert len(manifest["shards"]) == 1
    assert "runs" not in manifest
    assert manifest["tree_summaries"][0]["country_name"] == "Brazil"
    shard_path = tmp_path / manifest["shards"][0]["file"]
    assert shard_path.exists()
    shard = json.loads(shard_path.read_text())
    assert shard["tree_nodes"]
    assert "completion_text" not in shard["leaf_completions"]["d1_1"]


def test_build_tutorial_payload_picks_smallest():
    payload = {
        "runs": [
            {
                "tree_key": "big",
                "summary": {"total_leaves": 100},
                "prompt": "big",
                "model_id": "m",
                "tau": 0.01,
                "candidate_nodes": [{"node_id": "d1_1"}] * 100,
            },
            {
                "tree_key": "small",
                "summary": {"total_leaves": 2},
                "prompt": "small",
                "model_id": "m",
                "tau": 0.01,
                "candidate_nodes": [{"node_id": "d1_1"}],
            },
        ],
        "tree_summaries": [
            {"tree_key": "big", "prompt_preview": "big", "total_nodes": 50},
            {"tree_key": "small", "prompt_preview": "small", "total_nodes": 5},
        ],
        "trees": {
            "big": [{"id": "root"}],
            "small": [{"id": "root"}],
        },
        "node_status": {},
        "node_stats": {},
        "leaf_completions": {
            "small": {
                "d1_1": {
                    "leaf_id": "d1_1",
                    "answer_text": "Yes",
                    "completion_text": "x" * 5000,
                },
            },
        },
    }
    tutorial = build_tutorial_payload(payload)
    assert tutorial is not None
    assert tutorial["tree_key"] == "small"
    assert tutorial["prompt"] == "small"
    assert "shard" not in tutorial
    assert tutorial["summary"]["tree_key"] == "small"


def test_build_tree_shard_includes_expansions():
    payload = {
        "trees": {"k": [{"id": "root"}]},
        "node_status": {},
        "node_stats": {},
        "leaf_completions": {},
        "node_expansions": {"k": {"n1": {"tau_star": 0.001}}},
    }
    shard = build_tree_shard("k", payload)
    assert shard["node_expansions"]["n1"]["tau_star"] == 0.001
