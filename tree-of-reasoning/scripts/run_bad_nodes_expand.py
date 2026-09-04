#!/usr/bin/env python3
"""CLI entrypoint for per-node tau-star bad-node expansion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.bad_nodes_expand.pipeline import BadNodesExpandPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand exclusively-bad nodes to tau-star")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "capitals_bad_nodes_expand.yaml",
        help="Path to bad-nodes expansion config YAML",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Restrict to one or more model ids",
    )
    parser.add_argument(
        "--instruction-variant",
        action="append",
        dest="instruction_variants",
        choices=["legacy", "plain"],
        help="Restrict to legacy or plain instruction variant",
    )
    parser.add_argument(
        "--country",
        action="append",
        dest="countries",
        help="Restrict to one or more country ids",
    )
    parser.add_argument(
        "--prefix-length",
        action="append",
        type=int,
        dest="prefix_lengths",
        help="Restrict to one or more prefix lengths",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Plain-only pilot for deepseek-r1-7b",
    )
    args = parser.parse_args()

    model_ids = args.models
    instruction_variants = args.instruction_variants
    country_ids = args.countries
    prefix_lengths = args.prefix_lengths
    if args.pilot:
        model_ids = ["deepseek-r1-7b"]
        instruction_variants = ["plain"]

    pipeline = BadNodesExpandPipeline(config_path=args.config, repo_root=ROOT)
    output_paths = pipeline.run(
        model_ids=model_ids,
        instruction_variants=instruction_variants,
        country_ids=country_ids,
        prefix_lengths=prefix_lengths,
    )
    print("Wrote results:")
    for path in output_paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
