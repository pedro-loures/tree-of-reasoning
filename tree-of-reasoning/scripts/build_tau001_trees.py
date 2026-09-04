#!/usr/bin/env python3
"""Build tau=0.001 trees for all bad-nodes conditions (seed 0 only)."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.lorem_sampler import build_prompt, lorem_prefix  # noqa: E402
from src.experiment.runner import append_result, load_completed_conditions  # noqa: E402
from src.models.common import Condition, load_experiment_config  # noqa: E402
from src.models.hf_runner import HfRunner  # noqa: E402
from src.tree.metrics import compute_tree_metrics  # noqa: E402
from src.tree.tau_builder import build_tau_tree  # noqa: E402


def build_record(
    hf: HfRunner,
    condition: Condition,
    model_id: str,
    model_hf_id: str,
    tau: float,
    max_depth: int,
    breadth_warning_threshold: int,
    numerical_floor: float,
    batch_size: int,
    probe_max_tokens: int,
    top_k_logprobs: int,
) -> dict:
    prompt = build_prompt(lorem_prefix(condition.prefix_length), condition.instruction)
    root_prefix, _ = hf.find_reasoning_root_prefix(prompt, probe_max_tokens=probe_max_tokens)
    build_result = build_tau_tree(
        hf,
        root_prefix=root_prefix,
        tau=tau,
        max_depth=max_depth,
        breadth_warning_threshold=breadth_warning_threshold,
        numerical_floor=numerical_floor,
        batch_size=batch_size,
        capture_hidden_states=False,
        top_k_logprobs=top_k_logprobs,
    )
    tree_metrics = compute_tree_metrics(build_result.tree)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "model_hf_id": model_hf_id,
        "instruction": condition.instruction,
        "prefix_length": condition.prefix_length,
        "seed": condition.seed,
        "prompt": prompt,
        "lorem_prefix": lorem_prefix(condition.prefix_length),
        "root_prefix": root_prefix,
        "tree": build_result.tree.to_dict(),
        "tree_metrics": tree_metrics,
        "top_k_metrics": {"top_k_completions": []},
        "trace_metrics": {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build tau=0.001 source trees for bad-nodes")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "tau001_experiment.yaml")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--prefix-length", action="append", type=int, dest="prefix_lengths")
    args = parser.parse_args()

    experiment, models, _ = load_experiment_config(args.config, repo_root=ROOT)
    if args.models:
        models = [model for model in models if model.id in args.models]

    results_dir = ROOT / experiment.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    for model_spec in models:
        output_path = results_dir / f"{model_spec.id}.jsonl"
        conditions = list(experiment.iter_conditions())
        if args.prefix_lengths is not None:
            conditions = [cond for cond in conditions if cond.prefix_length in args.prefix_lengths]

        completed = load_completed_conditions(output_path)
        pending = [cond for cond in conditions if (cond.instruction, cond.prefix_length, cond.seed) not in completed]
        if completed:
            print(f"Skipping {len(completed)} completed conditions for {model_spec.id}", flush=True)
        if not pending:
            print(f"Nothing to build for {model_spec.id}", flush=True)
            continue

        print(f"Building {len(pending)} trees for {model_spec.id}", flush=True)
        hf = HfRunner(model_spec)
        hf.load()
        try:
            for condition in pending:
                t0 = time.time()
                print(
                    f"  instruction={condition.instruction!r} "
                    f"prefix_length={condition.prefix_length} seed={condition.seed}",
                    flush=True,
                )
                record = build_record(
                    hf,
                    condition,
                    model_spec.id,
                    model_spec.hf_id,
                    tau=experiment.tau,
                    max_depth=experiment.max_tree_depth,
                    breadth_warning_threshold=experiment.breadth_warning_threshold,
                    numerical_floor=experiment.numerical_floor,
                    batch_size=model_spec.hf_batch_size or experiment.hf_batch_size,
                    probe_max_tokens=experiment.reasoning_probe_max_tokens,
                    top_k_logprobs=experiment.top_k_logprobs,
                )
                append_result(record, output_path)
                metrics = record["tree_metrics"]
                elapsed = time.time() - t0
                print(
                    f"    nodes={metrics['total_nodes']} leaves={metrics['leaf_count']} "
                    f"warnings={metrics['breadth_warning_count']} elapsed={elapsed:.0f}s",
                    flush=True,
                )
        finally:
            hf.unload()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Wrote trees under {results_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
