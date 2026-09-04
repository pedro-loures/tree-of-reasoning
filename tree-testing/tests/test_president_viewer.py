"""Tests for president dashboard export."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.analysis.president_viewer import (  # noqa: E402
    _mention_matches_candidate,
    export_president_dashboard,
)


def _mech_row(model_id: str = "deepseek-r1-7b") -> dict:
    return {
        "model_id": model_id,
        "seed": 0,
        "prefix_length": 0,
        "instruction": "who is likely to be the next brazilian president?",
        "root_prefix": "prompt",
        "tree_metrics": {"total_nodes": 10, "leaf_count": 4, "breadth_warning_count": 0},
        "trace_metrics": {"reasoning_token_count": 42},
        "top_k_metrics": {"top_k_any_correct": False},
        "politician_mentions": {
            "greedy": {
                "category": "mentioned_only_candidates",
                "mentions": [{"full_name": "LULA", "party": "PT"}],
            }
        },
        "tree": {
            "nodes": [
                {
                    "id": "root",
                    "depth": 0,
                    "prefix_text": "prompt",
                    "path_prob": 1.0,
                    "child_ids": ["d1_1"],
                    "child_tokens": ["Ok"],
                },
                {
                    "id": "d1_1",
                    "depth": 1,
                    "prefix_text": "promptOk",
                    "path_prob": 0.5,
                    "child_ids": [],
                    "child_tokens": [],
                },
            ]
        },
    }


def _bad_row(model_id: str = "deepseek-r1-7b") -> dict:
    return {
        "tree_key": f"president:{model_id}:plain:0:0",
        "summary": {
            "total_leaves": 2,
            "total_candidates": 1,
            "exclusively_bad_count": 0,
            "ditched_count": 1,
            "leaf_subtype_counts": {"correct": 1, "mentioned_candidates": 1},
        },
        "leaf_completions": [
            {
                "leaf_id": "d1_1",
                "path_prob": 0.5,
                "answer_correct": False,
                "mention_category": "mentioned_only_candidates",
                "mentions": [
                    {
                        "id": "lula",
                        "full_name": "LULA",
                        "ballot_name": "LULA",
                        "party": "PT",
                        "is_presidential_candidate_2026": True,
                    }
                ],
                "answer_text": "Lula",
                "reasoning_complete": True,
            },
            {
                "leaf_id": "d1_2",
                "path_prob": 0.3,
                "answer_correct": True,
                "mention_category": "no_politicians_mentioned",
                "mentions": [],
                "answer_text": "unclear",
                "reasoning_complete": True,
            },
        ],
        "candidate_nodes": [],
    }


def test_mention_matches_candidate_by_name():
    registry_candidate = {
        "id": "280002542548",
        "full_name": "LUIZ INÁCIO LULA DA SILVA",
        "ballot_name": "LULA",
        "party": "PT",
    }
    fixture_mention = {
        "id": "100001",
        "full_name": "LUIZ INACIO LULA DA SILVA",
        "ballot_name": "LULA",
        "party": "PT",
        "is_presidential_candidate_2026": True,
    }
    assert _mention_matches_candidate(fixture_mention, registry_candidate)


def test_export_president_dashboard(tmp_path: Path):
    mech_dir = tmp_path / "mech"
    bad_dir = tmp_path / "bad"
    mech_dir.mkdir()
    bad_dir.mkdir()
    (mech_dir / "deepseek-r1-7b.jsonl").write_text(json.dumps(_mech_row()) + "\n")
    (bad_dir / "deepseek-r1-7b.jsonl").write_text(json.dumps(_bad_row()) + "\n")

    output_path = tmp_path / "president_dashboard.json"
    payload = export_president_dashboard(mech_dir, bad_dir, output_path)

    assert payload["mech_interp_trees"] == 1
    assert payload["bad_nodes_trees"] == 1
    assert payload["viewer_runs"] == 1
    assert payload["tree_summaries"][0]["greedy_mention_category"] == "mentioned_only_candidates"
    assert len(payload["presidential_candidates"]) >= 1
    assert payload["tree_summaries"][0]["leaf_correct"] == 1
    assert payload["tree_summaries"][0]["prob_good"] == 0.3
    assert payload["tree_summaries"][0]["prob_bad"] == 0.5
    candidate_probs = payload["candidate_mention_probs"][payload["tree_summaries"][0]["tree_key"]]
    assert len(candidate_probs) == 1
    assert candidate_probs[0]["prob"] == 0.5
    assert payload["leaf_completions"][payload["tree_summaries"][0]["tree_key"]]["d1_1"]["answer_correct"] is False
    assert output_path.exists()
