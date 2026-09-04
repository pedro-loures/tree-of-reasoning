#!/usr/bin/env python3
"""Export a labeled τ-tree PNG for paper figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.analysis.tree_parser import compact_tree, prune_tree_by_path_prob  # noqa: E402
from src.pipelines.analysis.tree_plots import (  # noqa: E402
    MODEL_LABELS,
    compute_fit_spacing,
    plot_labeled_tree,
)

PAPER_WIDTH = 6.875
PAPER_HEIGHT = 8.0
MIN_X_SPACING = 0.55
MIN_Y_SPACING = 0.36
PATH_PROB_THRESHOLDS = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10]


def select_nodes_for_page(
    all_nodes: list[dict],
    *,
    fig_width: float = PAPER_WIDTH,
    fig_height: float = PAPER_HEIGHT,
    min_path_prob: float | None = None,
) -> tuple[list[dict], float]:
    if min_path_prob is not None and min_path_prob > 0:
        return prune_tree_by_path_prob(all_nodes, min_path_prob), min_path_prob

    chosen_nodes = prune_tree_by_path_prob(all_nodes, PATH_PROB_THRESHOLDS[-1])
    chosen_threshold = PATH_PROB_THRESHOLDS[-1]
    for threshold in PATH_PROB_THRESHOLDS:
        nodes = prune_tree_by_path_prob(all_nodes, threshold)
        leaf_count = max(sum(1 for node in nodes if not node["c"]), 1)
        max_depth = max(node["d"] for node in nodes)
        x_spacing, y_spacing = compute_fit_spacing(
            leaf_count,
            max_depth,
            fig_width=fig_width,
            fig_height=fig_height,
        )
        if x_spacing >= MIN_X_SPACING and y_spacing >= MIN_Y_SPACING:
            return nodes, threshold
    return chosen_nodes, chosen_threshold


def load_record(
    jsonl_path: Path,
    *,
    instruction: str,
    prefix_length: int,
    seed: int,
) -> dict:
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if (
            record.get("instruction") == instruction
            and record.get("prefix_length") == prefix_length
            and record.get("seed") == seed
        ):
            return record
    raise SystemExit(
        f"No matching record in {jsonl_path} "
        f"(instruction={instruction!r}, prefix_length={prefix_length}, seed={seed})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export labeled τ-tree PNG")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=ROOT.parent / "tree-of-reasoning" / "results" / "mech_interp" / "deepseek-r1-7b.jsonl",
    )
    parser.add_argument(
        "--instruction",
        default="what is the capital of brazil",
        help="Plain Brazil prompt (no Lorem prefix)",
    )
    parser.add_argument("--prefix-length", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "tree-of-reasoning" / "results" / "figures" / "brazil_plain_tree.png",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--min-path-prob",
        type=float,
        default=None,
        help="Hide subtrees below this cumulative path probability (auto-tuned for page fit by default)",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=PAPER_WIDTH,
        help="Figure width in inches (LaTeX \\textwidth is ~6.875in)",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=PAPER_HEIGHT,
        help="Figure height in inches for single-page layout",
    )
    parser.add_argument(
        "--x-spacing",
        type=float,
        default=None,
        help="Horizontal spacing between leaf columns in inches",
    )
    parser.add_argument(
        "--y-spacing",
        type=float,
        default=None,
        help="Vertical spacing between depth levels in inches",
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help="Large appendix layout instead of single-page fit",
    )
    parser.add_argument(
        "--full-tree",
        action="store_true",
        help="Show every node in the τ-tree (may be unreadable in PDF)",
    )
    args = parser.parse_args()

    record = load_record(
        args.jsonl,
        instruction=args.instruction,
        prefix_length=args.prefix_length,
        seed=args.seed,
    )
    all_nodes = compact_tree(record["tree"]["nodes"], record.get("root_prefix"))
    if args.full_tree or (args.min_path_prob is not None and args.min_path_prob <= 0):
        nodes = all_nodes
        min_path_prob = 0.0
        subtitle = None
    elif args.large:
        min_path_prob = args.min_path_prob if args.min_path_prob is not None else 0.03
        nodes = prune_tree_by_path_prob(all_nodes, min_path_prob)
        subtitle = (
            f"Showing {len(nodes)} of {len(all_nodes)} nodes "
            f"(path probability ≥ {min_path_prob:g}; unexpanded τ-tree)"
        )
    else:
        nodes, min_path_prob = select_nodes_for_page(
            all_nodes,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            min_path_prob=args.min_path_prob,
        )
        subtitle = (
            f"Showing {len(nodes)} of {len(all_nodes)} nodes "
            f"(path probability ≥ {min_path_prob:g}; unexpanded τ-tree)"
        )
    model_id = record.get("model_id", "deepseek-r1-7b")
    model_label = MODEL_LABELS.get(model_id, model_id)
    tau = record["tree"]["tau"]
    title = f"{model_label} · plain prompt · τ={tau:g} · \"{args.instruction}\""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_labeled_tree(
        nodes,
        args.output,
        title=title,
        subtitle=subtitle,
        dpi=args.dpi,
        paper=args.large and not args.full_tree,
        paper_fit=not args.large and not args.full_tree,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        x_spacing=args.x_spacing,
        y_spacing=args.y_spacing,
    )

    print(f"Nodes shown: {len(nodes)} / {len(all_nodes)}")
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
