"""Pipeline orchestrator for cosine shift vs breadth analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.features import build_features_dataframe, layer_columns
from src.io import load_all_runs
from src.plots import build_all_plots
from src.stats import compute_correlations


@dataclass
class ExperimentConfig:
    results_dir: Path
    output_dir: Path
    min_breadth: int = 1
    layers: list[int] | None = None
    build_plots: bool = True
    plot_dpi: int = 120
    depth_bins: list[list[int]] | None = None


def load_config(config_path: Path, repo_root: Path) -> ExperimentConfig:
    raw = yaml.safe_load(config_path.read_text())
    results_dir = Path(raw["results_dir"])
    if not results_dir.is_absolute():
        results_dir = (repo_root / results_dir).resolve()
    output_dir = Path(raw["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    return ExperimentConfig(
        results_dir=results_dir,
        output_dir=output_dir,
        min_breadth=int(raw.get("min_breadth", 1)),
        layers=raw.get("layers"),
        build_plots=bool(raw.get("build_plots", True)),
        plot_dpi=int(raw.get("plot_dpi", 120)),
        depth_bins=raw.get("depth_bins"),
    )


class AnalysisPipeline:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self, model_ids: list[str] | None = None, build_plots: bool | None = None) -> dict[str, Any]:
        cfg = self.config
        runs, skipped = load_all_runs(cfg.results_dir, layers=cfg.layers)
        if model_ids:
            runs = [run for run in runs if run.model_id in model_ids]

        if not runs:
            raise FileNotFoundError(f"No runs with embeddings found in {cfg.results_dir}")

        df = build_features_dataframe(runs, min_breadth=cfg.min_breadth)
        if df.empty:
            raise ValueError("No internal nodes with valid parent embeddings found")

        correlations = compute_correlations(df, depth_bins=cfg.depth_bins)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        features_path = cfg.output_dir / "node_features.parquet"
        df.to_parquet(features_path, index=False)

        correlations_path = cfg.output_dir / "correlations.json"
        correlations_path.write_text(json.dumps(correlations, indent=2))

        summary = {
            "runs_loaded": len(runs),
            "runs_skipped": len(skipped),
            "internal_nodes": len(df),
            "models": sorted(df["model_id"].unique().tolist()),
            "layers": [column.replace("cos_dist_l", "") for column in layer_columns(df)],
            "skipped": skipped,
            "features_path": str(features_path),
            "correlations_path": str(correlations_path),
        }

        plot_paths: list[str] = []
        should_plot = build_plots if build_plots is not None else cfg.build_plots
        if should_plot:
            outputs = build_all_plots(
                df,
                correlations,
                cfg.output_dir,
                depth_bins=cfg.depth_bins or [[1, 5], [6, 10], [11, 15], [16, 20], [21, 999]],
                dpi=cfg.plot_dpi,
            )
            plot_paths = [str(path) for path in outputs]
        summary["plots"] = plot_paths

        summary_path = cfg.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        summary["summary_path"] = str(summary_path)
        return summary
