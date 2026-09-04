#!/usr/bin/env python3
"""CLI entrypoint for running the reasoning tree probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.probe.tree_probe import (  # noqa: E402
    format_probe_report,
    load_config,
    run_probe_for_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run vLLM reasoning tree probe")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "probe.yaml",
        help="Path to probe config YAML",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run only this model id from config (e.g. deepseek-r1-7b)",
    )
    args = parser.parse_args()

    probe, models, vllm_cfg = load_config(args.config)
    results_dir = ROOT / probe.results_dir

    if args.model:
        models = [m for m in models if m.id == args.model]
        if not models:
            print(f"Unknown model id: {args.model}", file=sys.stderr)
            return 1

    for model_spec in models:
        print(f"\n{'=' * 60}")
        print(f"Running probe for {model_spec.hf_id}")
        print(f"{'=' * 60}")
        result = run_probe_for_model(model_spec, vllm_cfg, probe, results_dir)
        print(format_probe_report(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
