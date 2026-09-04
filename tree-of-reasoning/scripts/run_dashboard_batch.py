#!/usr/bin/env python3
"""Generate one or more τ-trees for the interactive dashboard and save to disk."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dashboard.persist import default_save_dir, save_run  # noqa: E402
from src.dashboard.service import DashboardConfig, DashboardService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-generate dashboard τ-trees")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiment.yaml",
        help="Experiment config with model definitions",
    )
    parser.add_argument("--model", default="deepseek-r1-7b", help="Model id")
    parser.add_argument("--tau", type=float, default=0.01, help="τ threshold")
    parser.add_argument("--prompt", action="append", dest="prompts", help="Prompt (repeatable)")
    parser.add_argument(
        "--prompts-file",
        type=Path,
        help="JSON file with a list of prompt strings",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory for saved dashboard trees",
    )
    parser.add_argument(
        "--expected-answers",
        default=None,
        help="Comma-separated expected answers (applied to every prompt)",
    )
    parser.add_argument("--answer-mode", default="or")
    args = parser.parse_args()

    prompts: list[str] = list(args.prompts or [])
    if args.prompts_file:
        file_prompts = json.loads(args.prompts_file.read_text(encoding="utf-8"))
        if not isinstance(file_prompts, list):
            raise SystemExit("--prompts-file must contain a JSON list of strings")
        prompts.extend(str(item) for item in file_prompts)
    if not prompts:
        raise SystemExit("Provide at least one --prompt or --prompts-file")

    save_dir = args.save_dir or default_save_dir()
    service = DashboardService(
        DashboardConfig(config_path=args.config, repo_root=ROOT),
    )

    for index, prompt in enumerate(prompts, start=1):
        print(f"[{index}/{len(prompts)}] Generating tree (τ={args.tau:g})")
        print(f"  prompt: {prompt}")
        run = service.generate_tree(
            prompt=prompt,
            model_id=args.model,
            tau=args.tau,
            expected_answers=args.expected_answers,
            answer_mode=args.answer_mode,
        )
        path = save_run(run, save_dir)
        summary = run.get("tree_summary") or {}
        print(
            f"  saved: {path.name} "
            f"({summary.get('total_nodes', '?')} nodes, {summary.get('leaf_count', '?')} leaves)"
        )

    print(f"Done. Saved {len(prompts)} tree(s) to {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
