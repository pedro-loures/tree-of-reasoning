"""Tests for JSONL + npz loading."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io import (  # noqa: E402
    instruction_variant,
    load_all_runs,
    load_embeddings_npz,
    _run_key,
)


def _write_fixture(results_dir: Path, row: dict | None = None) -> None:
    embeddings_dir = results_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    node_ids = ["root", "d1_1"]
    parent_ids = ["", "root"]
    layers = np.array([4, 8], dtype=np.int32)
    hidden_states = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ],
        dtype=np.float16,
    )
    top_k_logprobs = np.array([[-0.1, -1.0], [-0.2, -1.2]], dtype=np.float32)
    np.savez_compressed(
        embeddings_dir / "demo.npz",
        node_ids=np.array(node_ids, dtype=object),
        parent_ids=np.array(parent_ids, dtype=object),
        layers=layers,
        hidden_states=hidden_states,
        top_k_token_ids=np.zeros((2, 2), dtype=np.int32),
        top_k_logprobs=top_k_logprobs,
    )

    if row is None:
        row = {
            "model_id": "demo-model",
            "instruction": "ignore the previous text, what is the capital of brazil",
            "prefix_length": 0,
            "seed": 0,
            "tree": {
                "nodes": [
                    {"id": "root", "depth": 0, "breadth": 1, "parent_id": None},
                    {"id": "d1_1", "depth": 1, "breadth": 2, "parent_id": "root"},
                ]
            },
            "embeddings": {"path": "demo.npz"},
        }
    (results_dir / "demo-model.jsonl").write_text(json.dumps(row) + "\n")


def test_instruction_variant_brazil_plain():
    assert instruction_variant({"instruction": "what is the capital of brazil"}) == "plain"
    assert (
        instruction_variant({"instruction": "ignore the previous text, what is the capital of brazil"})
        == "legacy"
    )


def test_instruction_variant_capitals_plain():
    assert instruction_variant({"instruction": "what is the capital of albania"}) == "plain"
    assert (
        instruction_variant({"instruction": "ignore the previous text, what is the capital of albania"})
        == "legacy"
    )


def test_run_key_includes_country():
    assert _run_key("deepseek-r1-7b", "legacy", 0, 0) == "deepseek-r1-7b:legacy:0:0"
    assert (
        _run_key("deepseek-r1-7b", "legacy", 0, 0, country_id="albania")
        == "deepseek-r1-7b:albania:legacy:0:0"
    )


def test_load_embeddings_npz(tmp_path: Path):
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    node_ids = ["root", "d1_1"]
    parent_ids = ["", "root"]
    layers = np.array([4], dtype=np.int32)
    hidden_states = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float16)
    top_k_logprobs = np.array([[-0.1], [-0.2]], dtype=np.float32)
    np.savez_compressed(
        embeddings_dir / "demo.npz",
        node_ids=np.array(node_ids, dtype=object),
        parent_ids=np.array(parent_ids, dtype=object),
        layers=layers,
        hidden_states=hidden_states,
        top_k_token_ids=np.zeros((2, 1), dtype=np.int32),
        top_k_logprobs=top_k_logprobs,
    )

    active_layers, embeddings = load_embeddings_npz(embeddings_dir / "demo.npz")
    assert active_layers == [4]
    assert embeddings["d1_1"].parent_id == "root"
    assert embeddings["d1_1"].hidden_by_layer[4][0] == 0.0


def test_load_all_runs(tmp_path: Path):
    _write_fixture(tmp_path)
    runs, skipped = load_all_runs(tmp_path)
    assert len(runs) == 1
    assert skipped == []
    assert runs[0].model_id == "demo-model"
    assert runs[0].run_key == "demo-model:legacy:0:0"
    assert "d1_1" in runs[0].embeddings_by_node


def test_load_all_runs_capitals_country(tmp_path: Path):
    _write_fixture(
        tmp_path,
        row={
            "model_id": "deepseek-r1-7b",
            "country_id": "albania",
            "instruction": "what is the capital of albania",
            "prefix_length": 0,
            "seed": 0,
            "tree": {
                "nodes": [
                    {"id": "root", "depth": 0, "breadth": 1, "parent_id": None},
                    {"id": "d1_1", "depth": 1, "breadth": 2, "parent_id": "root"},
                ]
            },
            "embeddings": {"path": "demo.npz"},
        },
    )
    runs, skipped = load_all_runs(tmp_path)
    assert len(runs) == 1
    assert skipped == []
    assert runs[0].instruction_variant == "plain"
    assert runs[0].country_id == "albania"
    assert runs[0].region_id == "europe"
    assert runs[0].run_key == "deepseek-r1-7b:albania:plain:0:0"
