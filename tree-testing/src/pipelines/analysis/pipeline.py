"""Analysis pipeline orchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.pipelines.analysis.export import export_runs
from src.pipelines.analysis.io import load_from_sources
from src.pipelines.analysis.tree_plots import plot_all
from src.pipelines.analysis.viewer import build_graph_viewer as write_graph_viewer_html
from src.pipelines.analysis.viewer import build_viewer as write_viewer_html
from src.pipelines.analysis.word_edges import build_payload


@dataclass
class AnalysisConfig:
    results_dir: Path | None
    results_sources: list[tuple[str, str, Path]]
    output_dir: Path
    top_k_words: int = 10
    word_edges_output: str = "word_edges.json"
    export_canvas: bool = True
    build_viewer: bool = True
    build_graph_viewer: bool = True
    build_plots: bool = True
    plot_dpi: int = 120
    plot_grid_dpi: int = 100
    template_path: Path | None = None


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def load_config(config_path: Path, repo_root: Path) -> AnalysisConfig:
    raw = yaml.safe_load(config_path.read_text())
    results_sources: list[tuple[str, str, Path]] = []
    if raw.get("results_sources"):
        for entry in raw["results_sources"]:
            results_sources.append(
                (
                    str(entry["id"]),
                    str(entry.get("label", entry["id"])),
                    _resolve_path(repo_root, str(entry["path"])),
                )
            )
    results_dir: Path | None = None
    if raw.get("results_dir"):
        results_dir = _resolve_path(repo_root, str(raw["results_dir"]))
        if not results_sources:
            results_sources.append(("main", "main", results_dir))
    if not results_sources:
        raise ValueError("Config must define results_sources or results_dir")
    output_dir = Path(raw["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    template_path = raw.get("template_path")
    resolved_template = (
        (repo_root / template_path).resolve() if template_path else repo_root / "templates" / "results_viewer.template.html"
    )
    return AnalysisConfig(
        results_dir=results_dir,
        results_sources=results_sources,
        output_dir=output_dir,
        top_k_words=int(raw.get("top_k_words", 10)),
        word_edges_output=str(raw.get("word_edges_output", "word_edges.json")),
        export_canvas=bool(raw.get("export_canvas", True)),
        build_viewer=bool(raw.get("build_viewer", True)),
        build_graph_viewer=bool(raw.get("build_graph_viewer", True)),
        build_plots=bool(raw.get("build_plots", True)),
        plot_dpi=int(raw.get("plot_dpi", 120)),
        plot_grid_dpi=int(raw.get("plot_grid_dpi", 100)),
        template_path=resolved_template,
    )


class AnalysisPipeline:
    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config

    def run(
        self,
        export_canvas: bool | None = None,
        build_viewer: bool | None = None,
        build_graph_viewer: bool | None = None,
        build_plots: bool | None = None,
        top_k_words: int | None = None,
    ) -> dict[str, Any]:
        cfg = self.config
        do_export = export_canvas if export_canvas is not None else cfg.export_canvas
        do_viewer = build_viewer if build_viewer is not None else cfg.build_viewer
        do_graph_viewer = build_graph_viewer if build_graph_viewer is not None else cfg.build_graph_viewer
        do_plots = build_plots if build_plots is not None else cfg.build_plots
        top_k = top_k_words if top_k_words is not None else cfg.top_k_words

        records = load_from_sources([(dataset_id, path) for dataset_id, _, path in cfg.results_sources])
        if not records:
            source_dirs = ", ".join(str(path) for _, _, path in cfg.results_sources)
            raise FileNotFoundError(f"No JSONL results found in {source_dirs}")

        source = ", ".join(f"{dataset_id}:{path}" for dataset_id, _, path in cfg.results_sources)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        word_edges_path = cfg.output_dir / cfg.word_edges_output
        word_edges_payload = build_payload(records, top_k, source)
        word_edges_path.write_text(json.dumps(word_edges_payload, indent=2))

        export_result = None
        if do_export:
            dataset_labels = {dataset_id: label for dataset_id, label, _ in cfg.results_sources}
            export_result = export_runs(records, cfg.output_dir, source, dataset_labels=dataset_labels)

        if do_plots and export_result is not None:
            plot_all(
                export_result.trees,
                export_result.models,
                export_result.instruction_variants,
                export_result.prefix_lengths,
                export_result.seeds,
                export_result.datasets,
                cfg.output_dir / "plots",
                dpi=cfg.plot_dpi,
                grid_dpi=cfg.plot_grid_dpi,
            )

        if do_viewer:
            metrics_payload = export_result.base_payload if export_result else None
            if cfg.template_path and cfg.template_path.exists():
                write_viewer_html(cfg.output_dir, cfg.template_path, metrics_payload)

        if do_graph_viewer:
            graph_template = cfg.template_path.parent / "tree_graph.template.html" if cfg.template_path else None
            if graph_template and graph_template.exists():
                write_graph_viewer_html(cfg.output_dir, graph_template)

        return {
            "records": len(records),
            "word_edges": str(word_edges_path),
            "output_dir": str(cfg.output_dir),
        }
