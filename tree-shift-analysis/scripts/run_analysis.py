#!/usr/bin/env python3
"""CLI entrypoint for cosine shift vs breadth analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import AnalysisPipeline, load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze cosine shift vs breadth")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiment.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument("--results-dir", type=Path, help="Override results directory")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--model", action="append", dest="models", help="Restrict to model id(s)")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    config = load_config(args.config, ROOT)
    if args.results_dir:
        config.results_dir = args.results_dir.resolve()
    if args.output_dir:
        config.output_dir = args.output_dir.resolve()

    summary = AnalysisPipeline(config).run(
        model_ids=args.models,
        build_plots=not args.no_plots,
    )

    print(f"Runs loaded: {summary['runs_loaded']}")
    print(f"Internal nodes: {summary['internal_nodes']}")
    print(f"Features: {summary['features_path']}")
    print(f"Correlations: {summary['correlations_path']}")
    if summary.get("plots"):
        print("Plots:")
        for plot_path in summary["plots"]:
            print(f"  {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
