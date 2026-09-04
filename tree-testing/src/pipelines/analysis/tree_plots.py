"""Matplotlib tree PNG generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec

MODEL_LABELS = {
    "deepseek-r1-7b": "DeepSeek-R1-7B",
    "qwq-32b-awq": "QwQ-32B-AWQ",
}


def tree_positions(nodes: list[dict], root_id: str = "root") -> dict[str, tuple[float, float]]:
    by_id = {node["id"]: node for node in nodes}

    def leaf_ids(node_id: str) -> list[str]:
        node = by_id[node_id]
        if not node["c"]:
            return [node_id]
        leaves: list[str] = []
        for child_id in node["c"]:
            leaves.extend(leaf_ids(child_id))
        return leaves

    leaf_order = leaf_ids(root_id)
    leaf_x = {leaf_id: float(index) for index, leaf_id in enumerate(leaf_order)}
    positions: dict[str, tuple[float, float]] = {}

    def assign(node_id: str) -> float:
        node = by_id[node_id]
        if not node["c"]:
            x = leaf_x[node_id]
        else:
            x = sum(assign(child_id) for child_id in node["c"]) / len(node["c"])
        y = -float(node["d"])
        positions[node_id] = (x, y)
        return x

    assign(root_id)
    return positions


def collect_edges(nodes: list[dict], positions: dict[str, tuple[float, float]]) -> list[tuple]:
    by_id = {node["id"]: node for node in nodes}
    edges: list[tuple] = []
    for node in nodes:
        x0, y0 = positions[node["id"]]
        for child_id in node["c"]:
            child = by_id[child_id]
            x1, y1 = positions[child_id]
            edges.append(((x0, y0), (x1, y1), float(child["p"])))
    return edges


def display_token(token: str | None, node_id: str = "") -> str:
    if node_id == "root" or not token:
        return "prompt"
    if token == "\n":
        return "\\n"
    if token == "\t":
        return "\\t"
    if token == " ":
        return "·"
    if len(token) > 16:
        return token[:14] + "…"
    return token


def fmt_prob(prob: float) -> str:
    if prob >= 0.0001:
        return f"{prob:.4f}"
    return f"{prob:.2e}"


LABEL_BBOX = {
    "boxstyle": "round,pad=0.22",
    "facecolor": "white",
    "edgecolor": "#d0d0d0",
    "linewidth": 0.4,
    "alpha": 0.95,
}

LABEL_BBOX_COMPACT = {
    "boxstyle": "round,pad=0.15",
    "facecolor": "white",
    "edgecolor": "#d0d0d0",
    "linewidth": 0.35,
    "alpha": 0.95,
}


def node_label(node: dict, *, compact: bool = False) -> str:
    token = display_token(node.get("tok"), node["id"])
    prob = fmt_prob(float(node["p"]))
    if compact:
        return f"{token} ({prob})"
    return f"{prob}\n{token}"


def compute_fit_spacing(
    leaf_count: int,
    max_depth: int,
    *,
    fig_width: float,
    fig_height: float,
    margin_x: float = 0.55,
    margin_top: float = 1.15,
    margin_bottom: float = 0.35,
) -> tuple[float, float]:
    usable_w = max(fig_width - 2 * margin_x, 1.0)
    usable_h = max(fig_height - margin_top - margin_bottom, 1.0)
    x_spacing = usable_w / max(leaf_count - 1, 1)
    y_spacing = usable_h / max(max_depth, 1)
    return x_spacing, y_spacing


def scale_positions(
    positions: dict[str, tuple[float, float]],
    *,
    x_spacing: float,
    y_spacing: float,
) -> dict[str, tuple[float, float]]:
    return {
        node_id: (x * x_spacing, y * y_spacing)
        for node_id, (x, y) in positions.items()
    }


def edge_color(prob: float, max_prob: float) -> str:
    ratio = prob / max_prob if max_prob > 0 else 0.0
    if ratio >= 0.35:
        return "#111111"
    if ratio >= 0.12:
        return "#555555"
    return "#aaaaaa"


def draw_labeled_tree(
    ax,
    nodes: list[dict],
    *,
    max_width: float = 3.0,
    min_width: float = 0.15,
    token_fontsize: float = 6.5,
    prob_fontsize: float = 5.5,
    label_pad_points: float = 10.0,
    x_spacing: float = 1.0,
    y_spacing: float = 1.0,
    label_bbox: dict | None = None,
    compact_labels: bool = False,
    stagger_labels: bool = True,
    pad_x_frac: float = 0.12,
    pad_y_frac: float = 0.16,
    pad_x_min: float = 1.4,
    pad_y_min: float = 1.6,
) -> None:
    positions = scale_positions(
        tree_positions(nodes),
        x_spacing=x_spacing,
        y_spacing=y_spacing,
    )
    edges = collect_edges(nodes, positions)

    max_prob = max((prob for _, _, prob in edges), default=1.0)
    for (start, end), (_, _, prob) in zip(
        [(s, e) for s, e, _ in edges],
        edges,
    ):
        width = min_width + (max_width - min_width) * (prob / max_prob if max_prob > 0 else 0)
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=edge_color(prob, max_prob),
            linewidth=width,
            solid_capstyle="round",
            zorder=1,
        )

    for node in nodes:
        x, y = positions[node["id"]]
        is_internal = bool(node["c"])
        marker_size = 4.0 if is_internal else 5.0
        marker_color = "#333333" if is_internal else "#555555"
        ax.plot(
            x,
            y,
            "o",
            markersize=marker_size,
            markerfacecolor=marker_color,
            markeredgecolor="#111111",
            markeredgewidth=0.5,
            zorder=3,
        )

        label = node_label(node, compact=compact_labels)
        if is_internal:
            label_xy = (0, label_pad_points)
            label_va = "bottom"
        else:
            label_xy = (0, -label_pad_points)
            label_va = "top"

        if stagger_labels:
            depth_offset = ((node["d"] % 2) * 2 - 1) * (4.0 if compact_labels else 6.0)
            label_xy = (label_xy[0] + depth_offset, label_xy[1])

        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=label_xy,
            ha="center",
            va=label_va,
            fontsize=token_fontsize,
            linespacing=0.95,
            color="#111111",
            zorder=4,
            bbox=label_bbox,
        )

    xs = [xy[0] for xy in positions.values()]
    ys = [xy[1] for xy in positions.values()]
    pad_x = max((max(xs) - min(xs)) * pad_x_frac, x_spacing * pad_x_min)
    pad_y = max((max(ys) - min(ys)) * pad_y_frac, y_spacing * pad_y_min)
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.set_aspect("auto")
    ax.axis("off")


def plot_labeled_tree(
    nodes: list[dict],
    out_path: Path,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    dpi: int = 300,
    token_fontsize: float = 6.5,
    prob_fontsize: float = 5.5,
    paper: bool = False,
    paper_fit: bool = False,
    fig_width: float | None = None,
    fig_height: float | None = None,
    x_spacing: float | None = None,
    y_spacing: float | None = None,
) -> None:
    leaf_count = max(sum(1 for node in nodes if not node["c"]), 1)
    max_depth = max(node["d"] for node in nodes)
    compact_labels = False
    stagger_labels = True

    if paper_fit:
        fig_width = 6.875 if fig_width is None else fig_width
        fig_height = 8.0 if fig_height is None else fig_height
        fit_x, fit_y = compute_fit_spacing(
            leaf_count,
            max_depth,
            fig_width=fig_width,
            fig_height=fig_height,
        )
        x_spacing = fit_x if x_spacing is None else x_spacing
        y_spacing = fit_y if y_spacing is None else y_spacing
        token_fontsize = min(8.5, max(7.0, min(x_spacing * 10.5, y_spacing * 18.0)))
        prob_fontsize = token_fontsize
        label_pad_points = 9.0
        label_bbox = LABEL_BBOX_COMPACT.copy()
        compact_labels = True
        fig_h = fig_height
        pad_x_frac = 0.05
        pad_y_frac = 0.07
        pad_x_min = 0.35
        pad_y_min = 0.35
    elif paper:
        pad_x_frac = 0.12
        pad_y_frac = 0.16
        pad_x_min = 1.4
        pad_y_min = 1.6
        token_fontsize = 10.0
        prob_fontsize = 9.0
        x_spacing = 1.55 if x_spacing is None else x_spacing
        y_spacing = 1.05 if y_spacing is None else y_spacing
        label_pad_points = 20.0
        label_bbox = LABEL_BBOX.copy()
        auto_width = (leaf_count - 1) * x_spacing + 3.0
        auto_height = max_depth * y_spacing + 3.8
        fig_width = auto_width if fig_width is None else max(fig_width, auto_width)
        fig_h = auto_height
    else:
        pad_x_frac = 0.12
        pad_y_frac = 0.16
        pad_x_min = 1.4
        pad_y_min = 1.6
        x_spacing = 0.42 if x_spacing is None else x_spacing
        y_spacing = 0.50 if y_spacing is None else y_spacing
        fig_h = max(12.0, max_depth * y_spacing)
        fig_width = max(16.0, leaf_count * x_spacing) if fig_width is None else fig_width
        label_pad_points = 10.0
        label_bbox = None

    fig, ax = plt.subplots(figsize=(fig_width, fig_h), dpi=dpi)
    draw_labeled_tree(
        ax,
        nodes,
        token_fontsize=token_fontsize,
        prob_fontsize=prob_fontsize,
        label_pad_points=label_pad_points,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
        label_bbox=label_bbox,
        compact_labels=compact_labels,
        stagger_labels=stagger_labels,
        pad_x_frac=pad_x_frac,
        pad_y_frac=pad_y_frac,
        pad_x_min=pad_x_min,
        pad_y_min=pad_y_min,
    )
    if title:
        title_size = 9 if paper_fit else (13 if paper else 12)
        fig.suptitle(title, fontsize=title_size, y=0.985, fontweight="normal")
    if subtitle:
        subtitle_size = 7 if paper_fit else 9
        fig.text(
            0.5,
            0.965 if title else 0.99,
            subtitle,
            ha="center",
            va="top",
            fontsize=subtitle_size,
            color="#444444",
        )
    top_margin = 0.90 if paper_fit and subtitle else (0.93 if subtitle else 0.96)
    fig.subplots_adjust(left=0.02, right=0.98, top=top_margin, bottom=0.02)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def draw_tree(ax, nodes: list[dict], max_width: float = 3.0, min_width: float = 0.15) -> None:
    positions = tree_positions(nodes)
    edges = collect_edges(nodes, positions)

    segments = [(start, end) for start, end, _ in edges]
    probs = [prob for _, _, prob in edges]
    max_prob = max(probs) if probs else 1.0
    widths = [
        min_width + (max_width - min_width) * (prob / max_prob if max_prob > 0 else 0)
        for prob in probs
    ]

    if segments:
        lc = LineCollection(segments, colors="black", linewidths=widths, capstyle="round")
        ax.add_collection(lc)

    xs = [xy[0] for xy in positions.values()]
    ys = [xy[1] for xy in positions.values()]
    pad_x = max((max(xs) - min(xs)) * 0.05, 0.5)
    pad_y = max((max(ys) - min(ys)) * 0.05, 0.5)
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def plot_tree(nodes: list[dict], out_path: Path, dpi: int = 120) -> None:
    fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)
    draw_tree(ax, nodes)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def plot_model_grid(
    model_id: str,
    instruction_variant: str,
    dataset_id: str,
    trees: dict[str, list[dict]],
    prefix_lengths: list[int],
    seeds: list[int],
    out_path: Path,
    grid_dpi: int = 100,
) -> None:
    n_rows = len(prefix_lengths)
    n_cols = len(seeds)
    fig_w = 3.2 * n_cols + 1.0
    fig_h = 2.4 * n_rows + 0.8
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=grid_dpi)
    gs = GridSpec(
        n_rows,
        n_cols + 1,
        figure=fig,
        width_ratios=[0.22] + [1.0] * n_cols,
        wspace=0.06,
        hspace=0.10,
        left=0.06,
        right=0.98,
        top=0.90,
        bottom=0.08,
    )

    for row_idx, prefix_length in enumerate(prefix_lengths):
        label_ax = fig.add_subplot(gs[row_idx, 0])
        label_ax.axis("off")
        label_ax.text(
            0.5,
            0.5,
            f"{prefix_length} words",
            rotation=90,
            ha="center",
            va="center",
            fontsize=9,
        )

        for col_idx, seed in enumerate(seeds):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])
            if row_idx == 0:
                ax.set_title(f"seed {seed}", fontsize=9, pad=4)
            tree_key = f"{dataset_id}:{model_id}:{instruction_variant}:{prefix_length}:{seed}"
            nodes = trees.get(tree_key)
            if nodes:
                draw_tree(ax, nodes)
            else:
                ax.axis("off")

    model_label = MODEL_LABELS.get(model_id, model_id)
    fig.suptitle(f"{model_label} · {instruction_variant} prompt", fontsize=11, y=0.97)
    fig.text(0.02, 0.5, "Lorem prefix length (words)", ha="center", va="center", rotation=90, fontsize=9)
    fig.text(0.55, 0.02, "random seed (Lorem prefix sampling)", ha="center", fontsize=9)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)


def plot_pair_grid(
    models: list[str],
    instruction_variant: str,
    dataset_id: str,
    trees: dict[str, list[dict]],
    prefix_lengths: list[int],
    seeds: list[int],
    out_dir: Path,
    grid_dpi: int = 100,
) -> None:
    pair_dir = out_dir / "pairs" / instruction_variant
    pair_dir.mkdir(parents=True, exist_ok=True)
    for prefix_length in prefix_lengths:
        for seed in seeds:
            fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5.2), dpi=grid_dpi)
            if len(models) == 1:
                axes = [axes]
            for ax, model_id in zip(axes, models):
                tree_key = f"{dataset_id}:{model_id}:{instruction_variant}:{prefix_length}:{seed}"
                nodes = trees.get(tree_key)
                if nodes:
                    draw_tree(ax, nodes)
                else:
                    ax.axis("off")
                ax.set_title(MODEL_LABELS.get(model_id, model_id), fontsize=10, pad=6)
            fig.suptitle(
                f"prefix {prefix_length} words · seed {seed}",
                fontsize=11,
                y=0.98,
            )
            out_path = pair_dir / f"prefix_{prefix_length}_seed_{seed}.png"
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06, facecolor="white")
            plt.close(fig)


def plot_all(
    trees: dict[str, list[dict]],
    models: list[str],
    instruction_variants: list[str],
    prefix_lengths: list[int],
    seeds: list[int],
    datasets: list[dict[str, str]],
    out_dir: Path,
    dpi: int = 120,
    grid_dpi: int = 100,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_model_dir = out_dir / "trees"
    per_model_dir.mkdir(parents=True, exist_ok=True)

    for tree_key, nodes in trees.items():
        safe_name = tree_key.replace(":", "_")
        plot_tree(nodes, per_model_dir / f"{safe_name}.png", dpi=dpi)

    for dataset in datasets:
        dataset_id = dataset["id"]
        dataset_label = dataset.get("label", dataset_id)
        dataset_dir = out_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for model_id in models:
            for instruction_variant in instruction_variants:
                plot_model_grid(
                    model_id,
                    instruction_variant,
                    dataset_id,
                    trees,
                    prefix_lengths,
                    seeds,
                    dataset_dir / f"{model_id}_{instruction_variant}_grid.png",
                    grid_dpi=grid_dpi,
                )

        for instruction_variant in instruction_variants:
            plot_pair_grid(
                models,
                instruction_variant,
                dataset_id,
                trees,
                prefix_lengths,
                seeds,
                dataset_dir,
                grid_dpi=grid_dpi,
            )
