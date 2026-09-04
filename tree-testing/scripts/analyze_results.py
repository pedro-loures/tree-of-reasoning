#!/usr/bin/env python3
"""CLI entrypoint for the analysis pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.analysis.pipeline import AnalysisPipeline, load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze tree-of-reasoning experiment results")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "analysis.yaml",
        help="Path to analysis config YAML",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Override results directory (JSONL from tree-of-reasoning)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory for derived artifacts",
    )
    parser.add_argument("--top-k", type=int, help="Top-k words for edge analysis")
    parser.add_argument("--no-export", action="store_true", help="Skip canvas JSON export")
    parser.add_argument("--no-viewer", action="store_true", help="Skip HTML viewer build")
    parser.add_argument("--no-plots", action="store_true", help="Skip tree PNG plots")
    args = parser.parse_args()

    config = load_config(args.config, ROOT)
    if args.results_dir:
        resolved = args.results_dir.resolve()
        config.results_dir = resolved
        config.results_sources = [("main", "main", resolved)]
    if args.output_dir:
        config.output_dir = args.output_dir.resolve()

    pipeline = AnalysisPipeline(config)
    summary = pipeline.run(
        export_canvas=not args.no_export,
        build_viewer=not args.no_viewer,
        build_plots=not args.no_plots,
        top_k_words=args.top_k,
    )

    print(f"Analyzed {summary['records']} runs")
    print(f"Word edges: {summary['word_edges']}")
    print(f"Output dir: {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
