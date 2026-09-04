"""Tests for tree PNG plotting."""

from __future__ import annotations

from pathlib import Path

from src.pipelines.analysis.tree_plots import plot_tree


def test_plot_tree_writes_png(tmp_path: Path) -> None:
    nodes = [
        {"id": "root", "d": 0, "p": 1.0, "c": ["a"]},
        {"id": "a", "d": 1, "p": 0.5, "c": [], "tok": "Okay"},
    ]
    out_path = tmp_path / "tree.png"
    plot_tree(nodes, out_path, dpi=60)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
