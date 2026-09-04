"""Scatter and summary plots for cosine shift vs breadth."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import layer_columns


def _scatter(ax: plt.Axes, x: pd.Series, y: pd.Series, xlabel: str, ylabel: str) -> None:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if valid.empty:
        ax.set_title("No data")
        return
    ax.scatter(valid["x"], valid["y"], alpha=0.35, s=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_cos_dist_vs_breadth_by_layer(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    dpi: int = 120,
) -> None:
    layer_cols = layer_columns(df)
    if not layer_cols:
        return

    cols = 2
    rows = int(np.ceil(len(layer_cols) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows), dpi=dpi)
    axes_list = np.array(axes).reshape(-1)

    for index, layer_col in enumerate(layer_cols):
        ax = axes_list[index]
        _scatter(ax, df[layer_col], df["breadth"], layer_col, "breadth")
        ax.set_title(f"Layer {layer_col.replace('cos_dist_l', '')}")

    for index in range(len(layer_cols), len(axes_list)):
        axes_list[index].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_cos_dist_vs_breadth_depth_bins(
    df: pd.DataFrame,
    output_path: Path,
    depth_bins: list[list[int]],
    layer_col: str | None = None,
    dpi: int = 120,
) -> None:
    if df.empty:
        return

    layer_cols = layer_columns(df)
    if not layer_cols:
        return
    layer_col = layer_col or layer_cols[-1]

    def depth_label(depth: int) -> str:
        for low, high in depth_bins:
            if low <= depth <= high:
                return f"{low}-{high}"
        return "other"

    frame = df.copy()
    frame["depth_bin"] = frame["depth"].map(lambda depth: depth_label(int(depth)))
    bins = sorted(frame["depth_bin"].unique())
    cols = 2
    rows = int(np.ceil(len(bins) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows), dpi=dpi)
    axes_list = np.array(axes).reshape(-1)

    for index, depth_bin in enumerate(bins):
        group = frame[frame["depth_bin"] == depth_bin]
        ax = axes_list[index]
        _scatter(ax, group[layer_col], group["breadth"], layer_col, "breadth")
        ax.set_title(f"Depth {depth_bin}")

    for index in range(len(bins), len(axes_list)):
        axes_list[index].axis("off")

    fig.suptitle(f"Cosine shift vs breadth by depth ({layer_col})")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_by_layer(
    correlations: dict[str, Any],
    output_path: Path,
    dpi: int = 120,
) -> None:
    layer_rho = correlations.get("layer_breadth_rho", {})
    if not layer_rho:
        return

    layers = sorted(layer_rho.keys(), key=lambda value: int(value.replace("l", "")) if value.startswith("l") else int(value))
    values = [layer_rho[layer] for layer in layers]
    labels = [str(layer) for layer in layers]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
    ax.bar(labels, values)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman rho (cos_dist vs breadth)")
    ax.set_title("Correlation by layer")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_all_plots(
    df: pd.DataFrame,
    correlations: dict[str, Any],
    output_dir: Path,
    depth_bins: list[list[int]],
    dpi: int = 120,
) -> list[Path]:
    plots_dir = output_dir / "plots"
    outputs: list[Path] = []

    by_layer = plots_dir / "cos_dist_vs_breadth_by_layer.png"
    plot_cos_dist_vs_breadth_by_layer(df, by_layer, "Cosine shift vs breadth by layer", dpi=dpi)
    outputs.append(by_layer)

    depth_plot = plots_dir / "cos_dist_vs_breadth_depth_bins.png"
    plot_cos_dist_vs_breadth_depth_bins(df, depth_plot, depth_bins=depth_bins, dpi=dpi)
    outputs.append(depth_plot)

    corr_plot = plots_dir / "correlation_by_layer.png"
    plot_correlation_by_layer(correlations, corr_plot, dpi=dpi)
    outputs.append(corr_plot)

    for model_id in sorted(df["model_id"].unique()):
        model_plot = plots_dir / f"cos_dist_vs_breadth_{model_id}.png"
        model_df = df[df["model_id"] == model_id]
        plot_cos_dist_vs_breadth_by_layer(
            model_df,
            model_plot,
            f"Cosine shift vs breadth ({model_id})",
            dpi=dpi,
        )
        outputs.append(model_plot)

    return outputs
