"""Aggregate metrics from tau-tree results."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.tree.tau_builder import TauTreeResult


def compute_tree_metrics(tree: TauTreeResult) -> dict[str, Any]:
    nodes_by_depth: Counter[int] = Counter()
    max_child_breadth_by_depth: dict[int, int] = {}
    for node in tree.nodes:
        nodes_by_depth[node.depth] += 1
        current = max_child_breadth_by_depth.get(node.depth, 0)
        max_child_breadth_by_depth[node.depth] = max(current, node.breadth)

    depths = [node.depth for node in tree.nodes]
    max_depth = max(depths) if depths else 0
    leaf_nodes = tree.leaves
    leaf_mass = sum(leaf.path_prob for leaf in leaf_nodes)

    return {
        "breadth_by_depth": {str(depth): count for depth, count in sorted(nodes_by_depth.items())},
        "max_child_breadth_by_depth": {
            str(depth): count for depth, count in sorted(max_child_breadth_by_depth.items())
        },
        "max_depth": max_depth,
        "total_nodes": len(tree.nodes),
        "leaf_count": len(leaf_nodes),
        "mass_above_tau": leaf_mass,
        "breadth_warning_count": tree.breadth_warning_count,
        "breadth_warnings": [warning.to_dict() for warning in tree.breadth_warnings],
    }
