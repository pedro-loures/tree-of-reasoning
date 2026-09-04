#!/usr/bin/env python3
"""Render probe JSON as a token tree with cumulative path probabilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _fmt(p: float) -> str:
    if p >= 0.01:
        return f"{p:.4f}"
    if p >= 0.0001:
        return f"{p:.6f}"
    return f"{p:.2e}"


def _children_of(parent: dict, nodes: list[dict]) -> list[tuple[str, str, dict]]:
    """Return [(child_id, branch_token, child_node), ...] for a parent node."""
    kids: list[tuple[str, str, dict]] = []
    for token in parent["branch_tokens"]:
        expected = parent["prefix_text"] + token
        for node in nodes:
            if node["prefix_text"] == expected and node["depth"] == parent["depth"] + 1:
                kids.append((node["id"], token, node))
                break
    return kids


def _child_path_prob(parent: dict, token: str, child: dict, distribution: list[dict]) -> float:
    if "path_prob" in child:
        return child["path_prob"]
    parent_p = parent.get("path_prob", 1.0)
    for entry in distribution:
        if entry["token"] == token:
            return parent_p * entry["prob"]
    return parent_p


def render_tree(result: dict) -> str:
    nodes = result["nodes"]
    by_id = {n["id"]: n for n in nodes}
    root = by_id["root"]

    threshold = result.get("path_prob_threshold")
    mode = result.get("branch_mode", "top_k")

    lines = [
        f"Model: {result['model']}",
        "",
        "Cumulative probability = product of marginal token probs along the path.",
        "Format: token (cumulative_prob)",
    ]
    if mode == "tau" and threshold is not None:
        lines.append(f"Branching: prune paths with cumulative prob < {threshold:.0%}")
    lines.append("")

    def walk(node_id: str, indent: str) -> None:
        node = by_id[node_id]
        kids = _children_of(node, nodes)
        for i, (child_id, token, child) in enumerate(kids):
            last = i == len(kids) - 1
            branch = "└── " if last else "├── "
            cum = _child_path_prob(node, token, child, node["distribution"])
            lines.append(f"{indent}{branch}{token!r} ({_fmt(cum)})")
            child_indent = indent + ("    " if last else "│   ")
            if _children_of(child, nodes):
                walk(child_id, child_indent)

    walk("root", "")
    return "\n".join(lines)


def main() -> int:
    results_dir = Path(__file__).resolve().parents[1] / "results"
    paths = sorted(results_dir.glob("*.json"))
    # Prefer latest result per model_id when multiple runs exist
    latest: dict[str, Path] = {}
    for path in paths:
        with path.open() as f:
            model_id = json.load(f).get("model_id", path.stem)
        latest[model_id] = path
    paths = sorted(latest.values())
    if not paths:
        print("No JSON results found.", file=sys.stderr)
        return 1

    for path in paths:
        with path.open() as f:
            result = json.load(f)
        text = render_tree(result)
        out_path = path.with_suffix(".tree.txt")
        out_path.write_text(text + "\n")
        print(text)
        print(f"\nSaved: {out_path}\n")
        print("=" * 60)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
