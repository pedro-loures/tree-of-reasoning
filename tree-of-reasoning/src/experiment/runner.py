"""Experiment runner for prefix-length sweep across three evaluation tracks."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.completion.leaf_completer import complete_top_k_leaves, summarize_top_k_completions
from src.data.lorem_sampler import build_prompt, lorem_prefix
from src.data.tse_loader import load_registry
from src.embeddings.storage import run_embedding_key, save_node_embeddings
from src.models.common import Condition, ExperimentConfig, ModelSpec, VllmConfig, load_experiment_config
from src.models.hf_runner import HfRunner
from src.models.vllm_runner import VllmRunner
from src.trace.path_probe import run_path_probe
from src.tree.metrics import compute_tree_metrics
from src.tree.tau_builder import TauTreeBuildResult, TauTreeResult, build_tau_tree, stack_node_embeddings
from src.utils.politician_mentions import PoliticianRegistry, analyze_completion_tracks


@dataclass
class RunContext:
    model_spec: ModelSpec
    experiment: ExperimentConfig
    vllm_cfg: VllmConfig
    results_dir: Path
    embeddings_dir: Path
    politician_registry: PoliticianRegistry | None = None


def append_result(result: dict[str, Any], results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a") as f:
        f.write(json.dumps(result) + "\n")


def condition_key(condition: Condition) -> tuple[Any, ...]:
    if condition.country_id:
        return (
            condition.country_id,
            condition.instruction,
            condition.prefix_length,
            condition.seed,
        )
    return (condition.instruction, condition.prefix_length, condition.seed)


def load_completed_conditions(path: Path) -> set[tuple[Any, ...]]:
    completed: set[tuple[Any, ...]] = set()
    for row in load_results_jsonl(path):
        country_id = row.get("country_id")
        if country_id:
            completed.add(
                (
                    country_id,
                    row["instruction"],
                    int(row["prefix_length"]),
                    int(row["seed"]),
                )
            )
        else:
            completed.add(
                (
                    row["instruction"],
                    int(row["prefix_length"]),
                    int(row["seed"]),
                )
            )
    return completed


def load_results_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_tree_for_prompt(
    hf_runner: HfRunner,
    ctx: RunContext,
    prompt: str,
) -> tuple[str, TauTreeResult, dict[str, Any], TauTreeBuildResult]:
    root_prefix, _ = hf_runner.find_reasoning_root_prefix(
        prompt,
        probe_max_tokens=ctx.experiment.reasoning_probe_max_tokens,
    )
    build_result = build_tau_tree(
        hf_runner,
        root_prefix=root_prefix,
        tau=ctx.experiment.tau,
        max_depth=ctx.experiment.max_tree_depth,
        breadth_warning_threshold=ctx.experiment.breadth_warning_threshold,
        numerical_floor=ctx.experiment.numerical_floor,
        batch_size=ctx.model_spec.hf_batch_size or ctx.experiment.hf_batch_size,
        capture_hidden_states=ctx.experiment.capture_hidden_states,
        top_k_logprobs=ctx.experiment.top_k_logprobs,
    )
    return root_prefix, build_result.tree, compute_tree_metrics(build_result.tree), build_result


def _save_embeddings(
    ctx: RunContext,
    condition: Condition,
    build_result: TauTreeBuildResult,
) -> dict[str, Any] | None:
    if not ctx.experiment.capture_hidden_states or not build_result.capture_layers:
        return None

    node_ids, parent_ids, hidden_states, top_k_token_ids, top_k_logprobs = stack_node_embeddings(
        build_result.tree,
        build_result.node_features,
        build_result.capture_layers,
    )
    if not node_ids:
        return None

    run_key = run_embedding_key(
        ctx.model_spec.id,
        condition.instruction,
        condition.prefix_length,
        condition.seed,
    )
    manifest = save_node_embeddings(
        ctx.embeddings_dir,
        run_key=run_key,
        node_ids=node_ids,
        parent_ids=parent_ids,
        layers=build_result.capture_layers,
        hidden_states=hidden_states,
        top_k_token_ids=top_k_token_ids,
        top_k_logprobs=top_k_logprobs,
    )
    return manifest.to_dict()


def _run_vllm_tracks(
    vllm_runner: VllmRunner,
    ctx: RunContext,
    prompt: str,
    tree: TauTreeResult,
    accepted_capitals: list[str] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    trace_metrics = run_path_probe(
        vllm_runner,
        prompt,
        max_tokens=ctx.experiment.max_completion_tokens,
        temperature=ctx.experiment.temperature,
        logprobs_limit=ctx.experiment.logprobs_limit,
        accepted_capitals=accepted_capitals,
    )
    top_k_completions = complete_top_k_leaves(
        vllm_runner,
        tree,
        k=ctx.experiment.top_k_leaves,
        max_tokens=ctx.experiment.max_completion_tokens,
        temperature=ctx.experiment.temperature,
        accepted_capitals=accepted_capitals,
    )
    top_k_metrics = summarize_top_k_completions(top_k_completions)
    trace_summary = {
        key: value for key, value in trace_metrics.items() if key != "token_metrics"
    }
    return trace_summary, trace_metrics.get("token_metrics", []), top_k_metrics


def run_single_condition(
    ctx: RunContext,
    condition: Condition,
) -> dict[str, Any]:
    lorem_text = lorem_prefix(condition.prefix_length)
    prompt = build_prompt(lorem_text, condition.instruction)

    hf_runner = HfRunner(ctx.model_spec)
    hf_runner.load()
    try:
        root_prefix, tree, tree_metrics, build_result = _build_tree_for_prompt(
            hf_runner, ctx, prompt
        )
        embeddings_manifest = _save_embeddings(ctx, condition, build_result)
    finally:
        hf_runner.unload()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    vllm_runner = VllmRunner(ctx.model_spec, ctx.vllm_cfg)
    vllm_runner.load()
    try:
        trace_summary, trace_tokens, top_k_metrics = _run_vllm_tracks(
            vllm_runner,
            ctx,
            prompt,
            tree,
            condition.accepted_capitals,
        )
    finally:
        vllm_runner.unload()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": ctx.model_spec.id,
        "model_hf_id": ctx.model_spec.hf_id,
        "instruction": condition.instruction,
        "prefix_length": condition.prefix_length,
        "seed": condition.seed,
        "prompt": prompt,
        "lorem_prefix": lorem_text,
        "root_prefix": root_prefix,
        "tree": tree.to_dict(),
        "tree_metrics": tree_metrics,
        "trace_metrics": trace_summary,
        "trace_token_metrics": trace_tokens,
        "top_k_metrics": top_k_metrics,
    }
    if condition.country_id:
        result["country_id"] = condition.country_id
        result["country_name"] = condition.country_name
        result["accepted_capitals"] = condition.accepted_capitals
    if embeddings_manifest is not None:
        result["embeddings"] = embeddings_manifest
    if ctx.politician_registry is not None:
        leaf_prefixes = {leaf.id: leaf.prefix_text for leaf in tree.leaves}
        result["politician_mentions"] = analyze_completion_tracks(
            ctx.politician_registry,
            greedy_generated_text=trace_summary.get("generated_text", ""),
            top_k_completions=top_k_metrics.get("top_k_completions", []),
            leaf_prefixes=leaf_prefixes,
        )
    return result


def run_experiment(
    config_path: Path,
    results_dir: Path,
    model_ids: list[str] | None = None,
    prefix_lengths: list[int] | None = None,
    seeds: list[int] | None = None,
    country_ids: list[str] | None = None,
    repo_root: Path | None = None,
) -> list[Path]:
    experiment, models, vllm_cfg = load_experiment_config(config_path, repo_root=repo_root)
    if model_ids:
        models = [model for model in models if model.id in model_ids]
    embeddings_dir = results_dir / Path(experiment.embeddings_dir).name

    politician_registry: PoliticianRegistry | None = None
    if experiment.politician_registry_path:
        registry_path = Path(experiment.politician_registry_path)
        if not registry_path.is_absolute():
            base = repo_root or config_path.parent.parent
            registry_path = (base / registry_path).resolve()
        politician_registry = PoliticianRegistry.from_records(load_registry(registry_path))

    output_paths: list[Path] = []
    for model_spec in models:
        ctx = RunContext(
            model_spec=model_spec,
            experiment=experiment,
            vllm_cfg=vllm_cfg,
            results_dir=results_dir,
            embeddings_dir=embeddings_dir,
            politician_registry=politician_registry,
        )
        output_path = results_dir / f"{model_spec.id}.jsonl"
        output_paths.append(output_path)

        conditions = experiment.iter_conditions()
        if country_ids is not None:
            conditions = [cond for cond in conditions if cond.country_id in country_ids]
        if prefix_lengths is not None:
            conditions = [cond for cond in conditions if cond.prefix_length in prefix_lengths]
        if seeds is not None:
            conditions = [cond for cond in conditions if cond.seed in seeds]

        completed = load_completed_conditions(output_path)
        pending = [cond for cond in conditions if condition_key(cond) not in completed]
        if completed:
            print(f"Skipping {len(completed)} completed conditions for {model_spec.id}", flush=True)
        if not pending:
            continue

        print(
            f"Running {len(pending)} conditions for {model_spec.id} "
            f"(total configured={len(conditions)})",
            flush=True,
        )
        for condition in pending:
            country_label = f" country={condition.country_id}" if condition.country_id else ""
            print(
                f"Running model={model_spec.id}{country_label} "
                f"instruction={condition.instruction!r} "
                f"prefix_length={condition.prefix_length} seed={condition.seed}",
                flush=True,
            )
            result = run_single_condition(ctx, condition)
            append_result(result, output_path)
            mention_label = ""
            if "politician_mentions" in result:
                mention_label = (
                    f" mention_category={result['politician_mentions']['greedy']['category']}"
                )
            print(
                f"  tree_nodes={result['tree_metrics']['total_nodes']} "
                f"warnings={result['tree_metrics']['breadth_warning_count']} "
                f"greedy_correct={result['trace_metrics']['answer_correct']} "
                f"top_k_any_correct={result['top_k_metrics']['top_k_any_correct']} "
                f"embeddings={'yes' if 'embeddings' in result else 'no'}"
                f"{mention_label}",
                flush=True,
            )

    return output_paths
