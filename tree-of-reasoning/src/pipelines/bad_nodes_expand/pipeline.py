"""Orchestration for per-node tau-star local expansion of exclusively bad nodes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.completion.bad_node_analyzer import (
    build_node_index,
    evaluate_candidates,
    make_tree_key,
    summarize_bad_nodes,
    tree_from_dict,
)
from src.completion.expansion_analyzer import (
    ExpansionRecord,
    attach_expansion_to_candidates,
    collect_expandable_nodes,
    summarize_expansion,
)
from src.completion.leaf_completer import (
    LeafCompletionResult,
    complete_leaves,
    leaf_completion_from_dict,
)
from src.data.countries import instruction_variant
from src.experiment.runner import load_results_jsonl
from src.models.common import ModelSpec, VllmConfig, load_experiment_config
from src.models.hf_runner import HfRunner
from src.models.vllm_runner import VllmRunner
from src.pipelines.bad_nodes.pipeline import (
    append_result,
    iter_canonical_records,
    load_completed_tree_keys,
)
from src.tree.subtree_expander import expand_anchor_to_target, leaf_descendants


@dataclass
class BadNodesExpandConfig:
    mech_interp_dir: Path
    bad_nodes_dir: Path
    output_dir: Path
    dataset_id: str
    canonical_seed: int
    tau_original: float
    tau_floor: float
    tau_search_epsilon: float
    target_leaves: int
    max_tree_depth: int
    breadth_warning_threshold: int
    numerical_floor: float
    hf_batch_size: int
    temperature: float
    max_completion_tokens: int


def load_bad_nodes_expand_config(
    config_path: Path,
    repo_root: Path,
) -> tuple[BadNodesExpandConfig, list[ModelSpec], VllmConfig]:
    import yaml

    with config_path.open() as handle:
        raw = yaml.safe_load(handle)

    expand_raw = raw["bad_nodes_expand"]
    mech_dir = Path(expand_raw["mech_interp_dir"])
    bad_dir = Path(expand_raw["bad_nodes_dir"])
    output_dir = Path(expand_raw["output_dir"])
    if not mech_dir.is_absolute():
        mech_dir = (repo_root / mech_dir).resolve()
    if not bad_dir.is_absolute():
        bad_dir = (repo_root / bad_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()

    experiment, models, vllm_cfg = load_experiment_config(config_path, repo_root=repo_root)
    return (
        BadNodesExpandConfig(
            mech_interp_dir=mech_dir,
            bad_nodes_dir=bad_dir,
            output_dir=output_dir,
            dataset_id=str(expand_raw.get("dataset_id", "capitals")),
            canonical_seed=int(expand_raw.get("canonical_seed", 0)),
            tau_original=float(expand_raw.get("tau_original", experiment.tau)),
            tau_floor=float(expand_raw.get("tau_floor", 0.001)),
            tau_search_epsilon=float(expand_raw.get("tau_search_epsilon", 1e-4)),
            target_leaves=int(expand_raw.get("target_leaves", 10)),
            max_tree_depth=int(expand_raw.get("max_tree_depth", experiment.max_tree_depth)),
            breadth_warning_threshold=int(
                expand_raw.get("breadth_warning_threshold", experiment.breadth_warning_threshold)
            ),
            numerical_floor=float(
                expand_raw.get("numerical_floor", experiment.numerical_floor)
            ),
            hf_batch_size=int(expand_raw.get("hf_batch_size", experiment.hf_batch_size)),
            temperature=float(expand_raw.get("temperature", experiment.temperature)),
            max_completion_tokens=int(
                expand_raw.get("max_completion_tokens", experiment.max_completion_tokens)
            ),
        ),
        models,
        vllm_cfg,
    )


def index_rows_by_tree_key(rows: list[dict[str, Any]], cfg: BadNodesExpandConfig) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        variant = instruction_variant(row["instruction"])
        tree_key = make_tree_key(
            cfg.dataset_id,
            row["model_id"],
            variant,
            int(row["prefix_length"]),
            int(row.get("seed", cfg.canonical_seed)),
            country_id=row.get("country_id"),
        )
        indexed[tree_key] = row
    return indexed


def seed_leaf_cache_from_bad_row(bad_row: dict[str, Any]) -> dict[str, LeafCompletionResult]:
    return {
        item["leaf_id"]: leaf_completion_from_dict(item)
        for item in bad_row.get("leaf_completions", [])
    }


def run_single_tree_hf_expansion(
    mech_row: dict[str, Any],
    bad_row: dict[str, Any],
    hf_runner: HfRunner,
    cfg: BadNodesExpandConfig,
) -> dict[str, Any]:
    """Binary-search tau-star and splice subtrees (HF only)."""
    tree = tree_from_dict(mech_row["tree"])
    nodes_by_id = build_node_index(tree)
    leaf_cache = seed_leaf_cache_from_bad_row(bad_row)
    baseline_summary = bad_row.get("summary", summarize_bad_nodes(leaf_cache, evaluate_candidates(nodes_by_id, leaf_cache)))

    if int(baseline_summary.get("exclusively_bad_count", 0)) == 0:
        raise ValueError("Tree has no exclusively bad nodes")

    candidate_results = evaluate_candidates(nodes_by_id, leaf_cache)
    expandable = collect_expandable_nodes(
        candidate_results,
        nodes_by_id,
        target_leaves=cfg.target_leaves,
    )

    expansion_records: dict[str, ExpansionRecord] = {}
    all_new_leaf_ids: list[str] = []

    for candidate in expandable:
        nodes_by_id = build_node_index(tree)
        current_leaves = len(leaf_descendants(candidate.node_id, nodes_by_id))
        if current_leaves >= cfg.target_leaves:
            continue

        expansion, _search = expand_anchor_to_target(
            hf_runner,
            tree,
            candidate.node_id,
            tau_original=cfg.tau_original,
            tau_floor=cfg.tau_floor,
            tau_search_epsilon=cfg.tau_search_epsilon,
            target_leaves=cfg.target_leaves,
            max_depth=cfg.max_tree_depth,
            numerical_floor=cfg.numerical_floor,
            batch_size=cfg.hf_batch_size,
        )
        all_new_leaf_ids.extend(expansion.new_leaf_ids)

        expansion_records[candidate.node_id] = ExpansionRecord(
            node_id=candidate.node_id,
            tau_original=cfg.tau_original,
            tau_star=expansion.tau_star,
            leaves_before=expansion.leaves_before,
            leaves_after=expansion.leaves_after,
            new_leaf_ids=expansion.new_leaf_ids,
            binary_search_probes=expansion.binary_search_probes,
            hit_tau_floor=expansion.hit_tau_floor,
            outcome_after_rescore="skipped",
        )

    return {
        "tree": tree,
        "leaf_cache": leaf_cache,
        "baseline_summary": baseline_summary,
        "expansion_records": expansion_records,
        "new_leaf_ids": sorted(set(all_new_leaf_ids)),
        "mech_row": mech_row,
        "bad_row": bad_row,
        "accepted_capitals": mech_row.get("accepted_capitals") or bad_row.get("accepted_capitals"),
    }


def run_single_tree_expansion(
    mech_row: dict[str, Any],
    bad_row: dict[str, Any],
    hf_runner: HfRunner,
    vllm_runner: VllmRunner,
    cfg: BadNodesExpandConfig,
) -> dict[str, Any]:
    hf_state = run_single_tree_hf_expansion(mech_row, bad_row, hf_runner, cfg)
    return finalize_single_tree_expansion(hf_state, vllm_runner, cfg)


def finalize_single_tree_expansion(
    hf_state: dict[str, Any],
    vllm_runner: VllmRunner,
    cfg: BadNodesExpandConfig,
) -> dict[str, Any]:
    tree = hf_state["tree"]
    leaf_cache: dict[str, LeafCompletionResult] = hf_state["leaf_cache"]
    expansion_records: dict[str, ExpansionRecord] = hf_state["expansion_records"]
    mech_row = hf_state["mech_row"]
    bad_row = hf_state["bad_row"]
    baseline_summary = hf_state["baseline_summary"]
    accepted_capitals = hf_state["accepted_capitals"]
    new_leaf_ids = hf_state["new_leaf_ids"]

    pending = [leaf_id for leaf_id in new_leaf_ids if leaf_id not in leaf_cache]
    if pending:
        completions = complete_leaves(
            vllm_runner,
            tree,
            leaf_ids=pending,
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            accepted_capitals=accepted_capitals,
        )
        for completion in completions:
            leaf_cache[completion.leaf_id] = completion

    nodes_by_id = build_node_index(tree)
    candidate_results = evaluate_candidates(nodes_by_id, leaf_cache)
    candidate_by_id = {item.node_id: item for item in candidate_results}

    for node_id, record in expansion_records.items():
        if node_id in candidate_by_id:
            record.outcome_after_rescore = candidate_by_id[node_id].status  # type: ignore[assignment]
        else:
            record.outcome_after_rescore = "skipped"

    summary = summarize_bad_nodes(leaf_cache, candidate_results)
    expansion_summary = summarize_expansion(
        list(expansion_records.values()),
        new_leaves_completed=len(pending),
    )

    variant = instruction_variant(mech_row["instruction"])
    country_id = mech_row.get("country_id")
    tree_key = make_tree_key(
        cfg.dataset_id,
        mech_row["model_id"],
        variant,
        int(mech_row["prefix_length"]),
        int(mech_row.get("seed", cfg.canonical_seed)),
        country_id=country_id,
    )

    result: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tree_key": tree_key,
        "dataset_id": cfg.dataset_id,
        "model_id": mech_row["model_id"],
        "instruction": mech_row["instruction"],
        "instruction_variant": variant,
        "prefix_length": int(mech_row["prefix_length"]),
        "seed": int(mech_row.get("seed", cfg.canonical_seed)),
        "tree": tree.to_dict(),
        "leaf_completions": [item.to_dict() for item in leaf_cache.values()],
        "candidate_nodes": attach_expansion_to_candidates(candidate_results, expansion_records),
        "summary": summary,
        "baseline_summary": baseline_summary,
        "expansion_summary": expansion_summary,
        "expanded_bad_nodes": sorted(expansion_records.keys()),
    }
    if country_id:
        result["country_id"] = country_id
        result["country_name"] = mech_row.get("country_name")
        result["accepted_capitals"] = accepted_capitals
    return result


class BadNodesExpandPipeline:
    def __init__(self, config_path: Path, repo_root: Path) -> None:
        self.config_path = config_path
        self.repo_root = repo_root
        self.cfg, self.models, self.vllm_cfg = load_bad_nodes_expand_config(config_path, repo_root)

    def run(
        self,
        model_ids: list[str] | None = None,
        instruction_variants: list[str] | None = None,
        country_ids: list[str] | None = None,
        prefix_lengths: list[int] | None = None,
    ) -> list[Path]:
        models = self.models
        if model_ids:
            models = [model for model in models if model.id in model_ids]

        output_paths: list[Path] = []
        for model_spec in models:
            mech_path = self.cfg.mech_interp_dir / f"{model_spec.id}.jsonl"
            bad_path = self.cfg.bad_nodes_dir / f"{model_spec.id}.jsonl"
            output_path = self.cfg.output_dir / f"{model_spec.id}.jsonl"
            output_paths.append(output_path)

            mech_rows = iter_canonical_records(
                mech_path,
                self.cfg.dataset_id,
                self.cfg.canonical_seed,
            )
            bad_by_key = index_rows_by_tree_key(
                iter_canonical_records(bad_path, self.cfg.dataset_id, self.cfg.canonical_seed),
                self.cfg,
            )
            mech_by_key = index_rows_by_tree_key(mech_rows, self.cfg)

            pending: list[str] = []
            for tree_key, bad_row in bad_by_key.items():
                if int(bad_row.get("summary", {}).get("exclusively_bad_count", 0)) == 0:
                    continue
                if instruction_variants is not None:
                    if bad_row.get("instruction_variant", instruction_variant(bad_row["instruction"])) not in instruction_variants:
                        continue
                if country_ids is not None and bad_row.get("country_id") not in country_ids:
                    continue
                if prefix_lengths is not None and int(bad_row["prefix_length"]) not in prefix_lengths:
                    continue
                if tree_key not in mech_by_key:
                    continue
                pending.append(tree_key)

            completed = load_completed_tree_keys(output_path)
            pending = [tree_key for tree_key in pending if tree_key not in completed]

            if completed:
                print(f"Skipping {len(completed)} completed expanded trees for {model_spec.id}", flush=True)
            if not pending:
                continue

            print(f"Running {len(pending)} expanded trees for {model_spec.id}", flush=True)
            for tree_key in pending:
                bad_row = bad_by_key[tree_key]
                mech_row = mech_by_key[tree_key]
                country_label = f" country={bad_row.get('country_id')}" if bad_row.get("country_id") else ""
                print(
                    f"  model={bad_row['model_id']}{country_label} "
                    f"variant={bad_row.get('instruction_variant')} "
                    f"prefix_length={bad_row['prefix_length']}",
                    flush=True,
                )
                hf_runner = HfRunner(model_spec)
                hf_runner.load()
                try:
                    hf_state = run_single_tree_hf_expansion(
                        mech_row,
                        bad_row,
                        hf_runner,
                        self.cfg,
                    )
                finally:
                    hf_runner.unload()

                vllm_runner = VllmRunner(model_spec, self.vllm_cfg)
                vllm_runner.load()
                try:
                    result = finalize_single_tree_expansion(
                        hf_state,
                        vllm_runner,
                        self.cfg,
                    )
                finally:
                    vllm_runner.unload()
                append_result(result, output_path)
                exp = result["expansion_summary"]
                summary = result["summary"]
                print(
                    f"    expanded={exp['nodes_expanded']} "
                    f"new_leaves={exp['new_leaves_completed']} "
                    f"still_bad={exp['nodes_still_exclusively_bad']} "
                    f"ditched={exp['nodes_ditched_after']} "
                    f"exclusively_bad={summary['exclusively_bad_count']}",
                    flush=True,
                )

        return output_paths
