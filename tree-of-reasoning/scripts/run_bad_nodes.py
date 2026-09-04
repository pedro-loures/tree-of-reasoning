#!/usr/bin/env python3
"""CLI entrypoint for the exclusively-bad node experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.bad_nodes.pipeline import BadNodesPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run exclusively-bad node experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "bad_nodes.yaml",
        help="Path to bad-nodes config YAML",
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
        "--country",
        action="append",
        dest="countries",
        help="Restrict to one or more country ids (capitals experiment)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot subset: deepseek-r1-7b, prefix_length=1000, seed 0 only",
    )
    args = parser.parse_args()

    model_ids = args.models
    prefix_lengths = args.prefix_lengths
    country_ids = args.countries
    instructions = None
    if args.pilot:
        model_ids = ["deepseek-r1-7b"]
        if args.config.name == "capitals_bad_nodes.yaml":
            country_ids = country_ids or ["brazil"]
            prefix_lengths = [0, 1000]
        else:
            prefix_lengths = [1000]
            instructions = ["ignore the previous text, what is the capital of brazil"]

    pipeline = BadNodesPipeline(config_path=args.config, repo_root=ROOT)
    output_paths = pipeline.run(
        model_ids=model_ids,
        prefix_lengths=prefix_lengths,
        instructions=instructions,
        country_ids=country_ids,
    )
    print("Wrote results:")
    for path in output_paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
