#!/usr/bin/env python3
"""CLI entrypoint for the prefix-length CoT experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.execution.pipeline import ExecutionPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run prefix-length CoT experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiment.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory for JSONL outputs",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Restrict to one or more model ids",
    )
    parser.add_argument(
        "--prefix-length",
        action="append",
        type=int,
        dest="prefix_lengths",
        help="Restrict to one or more prefix lengths",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        help="Restrict to one or more seeds",
    )
    parser.add_argument(
        "--country",
        action="append",
        dest="countries",
        help="Restrict to one or more country ids (capitals experiment)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot subset: prefix lengths 0 and 1000, seed 0 only",
    )
    args = parser.parse_args()

    prefix_lengths = args.prefix_lengths
    seeds = args.seeds
    country_ids = args.countries
    if args.pilot:
        prefix_lengths = [0, 1000]
        seeds = [0]
        if args.config.name == "capitals_experiment.yaml":
            country_ids = country_ids or ["brazil"]

    from src.models.common import load_experiment_config  # noqa: E402

    experiment, _, _ = load_experiment_config(args.config, repo_root=ROOT)
    results_dir = args.results_dir
    if results_dir == ROOT / "results" and experiment.results_dir != "results":
        results_dir = ROOT / experiment.results_dir

    pipeline = ExecutionPipeline(
        config_path=args.config,
        results_dir=results_dir,
        repo_root=ROOT,
    )
    output_paths = pipeline.run(
        model_ids=args.models,
        prefix_lengths=prefix_lengths,
        seeds=seeds,
        country_ids=country_ids,
    )
    print("Wrote results:")
    for path in output_paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
