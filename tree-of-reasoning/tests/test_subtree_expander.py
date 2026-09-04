"""Unit tests for subtree tau-star expansion (no GPU)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tree.subtree_expander import (  # noqa: E402
    binary_search_tau_star,
    count_leaves_at_tau,
    expand_subtree_at_tau,
    leaf_descendants,
)
from src.tree.tau_builder import TauTreeNode, TauTreeResult  # noqa: E402


class _MockHfRunner:
    def __init__(
        self,
        logprobs_by_prefix: dict[str, torch.Tensor],
        *,
        default_logprobs: torch.Tensor | None = None,
    ) -> None:
        self._logprobs_by_prefix = logprobs_by_prefix
        if default_logprobs is not None:
            self._default = default_logprobs
        else:
            self._default = next(iter(logprobs_by_prefix.values()))

    def decode_token_id(self, token_id: int) -> str:
        return f"t{token_id}"

    def forward_batch(self, prefix_texts, capture_layers=None, top_k=20):
        del capture_layers, top_k

        class _Features:
            def __init__(self, logprobs: torch.Tensor) -> None:
                self.logprobs = logprobs

        return [
            _Features(self._logprobs_by_prefix.get(prefix, self._default))
            for prefix in prefix_texts
        ]


def _linear_tree() -> tuple[TauTreeResult, dict[str, TauTreeNode]]:
    root = TauTreeNode(
        id="root",
        depth=0,
        prefix_text="root",
        log_path_prob=0.0,
        path_prob=1.0,
    )
    child = TauTreeNode(
        id="d1_1",
        depth=1,
        prefix_text="roota",
        log_path_prob=math.log(0.5),
        path_prob=0.5,
        parent_id="root",
    )
    leaf = TauTreeNode(
        id="d2_1",
        depth=2,
        prefix_text="rootab",
        log_path_prob=math.log(0.2),
        path_prob=0.2,
        parent_id="d1_1",
    )
    root.child_ids = [child.id]
    root.child_token_ids = [0]
    root.child_tokens = ["a"]
    root.breadth = 1
    child.child_ids = [leaf.id]
    child.child_token_ids = [1]
    child.child_tokens = ["b"]
    child.breadth = 1
    tree = TauTreeResult(
        nodes=[root, child, leaf],
        breadth_warnings=[],
        breadth_warning_count=0,
        tau=0.01,
        max_tree_depth=8,
        root_prefix="root",
    )
    return tree, {node.id: node for node in tree.nodes}


def test_count_leaves_at_tau_branching():
    anchor = TauTreeNode(
        id="anchor",
        depth=1,
        prefix_text="p",
        log_path_prob=math.log(0.5),
        path_prob=0.5,
    )
    logprobs = torch.log(torch.tensor([0.4, 0.3, 0.05, 0.01]))
    terminal_logprobs = torch.log(torch.tensor([0.001, 0.001, 0.001, 0.001]))
    runner = _MockHfRunner({"p": logprobs}, default_logprobs=terminal_logprobs)

    assert count_leaves_at_tau(runner, anchor, tau=0.1, max_depth=4) == 2
    assert count_leaves_at_tau(runner, anchor, tau=0.02, max_depth=4) == 3


def test_binary_search_tau_star_finds_highest_tau():
    anchor = TauTreeNode(
        id="anchor",
        depth=0,
        prefix_text="p",
        log_path_prob=0.0,
        path_prob=1.0,
    )
    logprobs = torch.log(torch.tensor([0.5, 0.4, 0.08, 0.02, 0.005]))
    terminal_logprobs = torch.log(torch.tensor([0.001, 0.001, 0.001, 0.001, 0.001]))
    runner = _MockHfRunner(
        {"p": logprobs, "pt0": terminal_logprobs, "pt1": terminal_logprobs},
        default_logprobs=terminal_logprobs,
    )

    result = binary_search_tau_star(
        runner,
        anchor,
        tau_original=0.01,
        tau_floor=0.001,
        target_leaves=3,
        max_depth=2,
        tau_search_epsilon=0.001,
    )
    assert result.leaf_count >= 3
    assert result.tau_star >= 0.001


def test_expand_subtree_adds_nodes_without_duplicates():
    tree, nodes_by_id = _linear_tree()
    logprobs_leaf = torch.log(torch.tensor([0.5, 0.4, 0.01]))
    runner = _MockHfRunner(
        {
            "roota": logprobs_leaf,
            "rootab": logprobs_leaf,
        }
    )
    before_ids = {node.id for node in tree.nodes}
    expansion = expand_subtree_at_tau(
        runner,
        tree,
        anchor_id="d1_1",
        tau=0.05,
        max_depth=4,
    )
    assert expansion.new_nodes
    assert all(node.id not in before_ids for node in expansion.new_nodes)
    assert expansion.leaves_after >= expansion.leaves_before


def test_leaf_descendants():
    _, nodes_by_id = _linear_tree()
    assert leaf_descendants("root", nodes_by_id) == ["d2_1"]
    assert leaf_descendants("d1_1", nodes_by_id) == ["d2_1"]
