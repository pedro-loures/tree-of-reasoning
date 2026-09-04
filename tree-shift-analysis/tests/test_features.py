"""Tests for feature computation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import build_node_rows, cosine_distance, topk_entropy  # noqa: E402
from src.io import NodeEmbedding, RunRecord  # noqa: E402


def test_cosine_distance_identical_vectors():
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float16)
    assert cosine_distance(vector, vector) == 0.0


def test_cosine_distance_orthogonal_vectors():
    left = np.array([1.0, 0.0], dtype=np.float16)
    right = np.array([0.0, 1.0], dtype=np.float16)
    assert abs(cosine_distance(left, right) - 1.0) < 1e-6


def test_topk_entropy_uniform():
    log_probs = np.log(np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32))
    entropy = topk_entropy(log_probs)
    assert abs(entropy - np.log(4.0)) < 1e-6


def test_build_node_rows_filters_leaves():
    run = RunRecord(
        run_key="demo:legacy:0:0",
        model_id="demo",
        instruction_variant="legacy",
        prefix_length=0,
        seed=0,
        tree_nodes=[
            {"id": "root", "depth": 0, "breadth": 2, "parent_id": None, "prefix_text": "root"},
            {
                "id": "d1_1",
                "depth": 1,
                "breadth": 0,
                "parent_id": "root",
                "prefix_text": "rootA",
            },
            {
                "id": "d1_2",
                "depth": 1,
                "breadth": 3,
                "parent_id": "root",
                "prefix_text": "rootB",
            },
        ],
        layers=[4],
        embeddings_by_node={
            "root": NodeEmbedding(
                node_id="root",
                parent_id="",
                hidden_by_layer={4: np.array([1.0, 0.0], dtype=np.float16)},
                top_k_logprobs=np.array([-0.1, -2.0], dtype=np.float32),
            ),
            "d1_2": NodeEmbedding(
                node_id="d1_2",
                parent_id="root",
                hidden_by_layer={4: np.array([0.0, 1.0], dtype=np.float16)},
                top_k_logprobs=np.array([-0.2, -1.5], dtype=np.float32),
            ),
        },
    )

    rows = build_node_rows(run, min_breadth=1)
    assert len(rows) == 1
    assert rows[0]["node_id"] == "d1_2"
    assert rows[0]["breadth"] == 3
    assert abs(rows[0]["cos_dist_l4"] - 1.0) < 1e-6
