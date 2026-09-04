"""Orchestration for the exclusively-bad node experiment."""

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
from src.completion.leaf_completer import (
    LeafCompletionResult,
    complete_leaves,
    rescore_leaf_cache,
    seed_cache_from_top_k,
)
from src.data.countries import instruction_variant, tree_variant_label
from src.data.tse_loader import load_registry
from src.experiment.runner import load_results_jsonl
from src.models.common import ModelSpec, VllmConfig, load_experiment_config
from src.models.vllm_runner import VllmRunner
from src.utils.politician_mentions import PoliticianRegistry


@dataclass
class BadNodesConfig:
    source_dir: Path
    output_dir: Path
    dataset_id: str
    canonical_seed: int
    temperature: float
    max_completion_tokens: int
    politician_registry_path: Path | None = None


def load_bad_nodes_config(config_path: Path, repo_root: Path) -> tuple[BadNodesConfig, list[ModelSpec], VllmConfig]:
    import yaml

    with config_path.open() as handle:
        raw = yaml.safe_load(handle)

    bad_nodes_raw = raw["bad_nodes"]
    source_dir = Path(bad_nodes_raw["source_dir"])
    output_dir = Path(bad_nodes_raw["output_dir"])
    if not source_dir.is_absolute():
        source_dir = (repo_root / source_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()

    politician_registry_path: Path | None = None
    if raw.get("politician_registry"):
        politician_registry_path = Path(raw["politician_registry"])
        if not politician_registry_path.is_absolute():
            politician_registry_path = (repo_root / politician_registry_path).resolve()

    experiment, models, vllm_cfg = load_experiment_config(config_path, repo_root=repo_root)
    return (
        BadNodesConfig(
            source_dir=source_dir,
            output_dir=output_dir,
            dataset_id=str(bad_nodes_raw.get("dataset_id", "mech_interp")),
            canonical_seed=int(bad_nodes_raw.get("canonical_seed", 0)),
            temperature=float(bad_nodes_raw.get("temperature", experiment.temperature)),
            max_completion_tokens=int(
                bad_nodes_raw.get("max_completion_tokens", experiment.max_completion_tokens)
            ),
            politician_registry_path=politician_registry_path,
        ),
        models,
        vllm_cfg,
    )


def iter_canonical_records(
    source_path: Path,
    dataset_id: str,
    canonical_seed: int,
) -> list[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    records: list[dict[str, Any]] = []
    for row in load_results_jsonl(source_path):
        if int(row.get("seed", 0)) != canonical_seed:
            continue
        country_id = row.get("country_id")
        if country_id:
            key = (country_id, row["model_id"], row["instruction"], int(row["prefix_length"]))
        else:
            key = (row["model_id"], row["instruction"], int(row["prefix_length"]))
        if key in seen:
            continue
        seen.add(key)
        records.append(row)
    records.sort(
        key=lambda row: (
            row.get("country_id", ""),
            row["model_id"],
            row["instruction"],
            int(row["prefix_length"]),
        )
    )
    return records


def load_completed_tree_keys(output_path: Path) -> set[str]:
    completed: set[str] = set()
    for row in load_results_jsonl(output_path):
        if "tree_key" in row:
            completed.add(row["tree_key"])
    return completed


def append_result(result: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as handle:
        handle.write(json.dumps(result) + "\n")


def pending_leaf_ids(tree_leaves: list[str], cache: dict[str, LeafCompletionResult]) -> list[str]:
    return [leaf_id for leaf_id in tree_leaves if leaf_id not in cache]


def run_single_tree(
    vllm_runner: VllmRunner,
    row: dict[str, Any],
    cfg: BadNodesConfig,
    politician_registry: PoliticianRegistry | None = None,
) -> dict[str, Any]:
    tree = tree_from_dict(row["tree"])
    nodes_by_id = build_node_index(tree)
    all_leaf_ids = [node_id for node_id, node in nodes_by_id.items() if not node.child_ids]
    accepted_capitals = row.get("accepted_capitals")
    leaf_prefixes = {node_id: node.prefix_text for node_id, node in nodes_by_id.items() if not node.child_ids}

    leaf_cache = seed_cache_from_top_k(row.get("top_k_metrics", {}))
    rescore_leaf_cache(
        leaf_cache,
        accepted_capitals,
        politician_registry=politician_registry,
        leaf_prefixes=leaf_prefixes,
    )
    to_complete = pending_leaf_ids(all_leaf_ids, leaf_cache)
    if to_complete:
        completions = complete_leaves(
            vllm_runner,
            tree,
            leaf_ids=to_complete,
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            accepted_capitals=accepted_capitals,
            politician_registry=politician_registry,
        )
        for completion in completions:
            leaf_cache[completion.leaf_id] = completion

    candidate_results = evaluate_candidates(nodes_by_id, leaf_cache)
    summary = summarize_bad_nodes(leaf_cache, candidate_results)
    variant = tree_variant_label(cfg.dataset_id, row["instruction"])
    country_id = row.get("country_id")
    tree_key = make_tree_key(
        cfg.dataset_id,
        row["model_id"],
        variant,
        int(row["prefix_length"]),
        int(row.get("seed", cfg.canonical_seed)),
        country_id=country_id,
    )
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tree_key": tree_key,
        "dataset_id": cfg.dataset_id,
        "model_id": row["model_id"],
        "instruction": row["instruction"],
        "instruction_variant": variant,
        "prefix_length": int(row["prefix_length"]),
        "seed": int(row.get("seed", cfg.canonical_seed)),
        "leaf_completions": [item.to_dict() for item in leaf_cache.values()],
        "candidate_nodes": [item.to_dict() for item in candidate_results],
        "summary": summary,
    }
    if country_id:
        result["country_id"] = country_id
        result["country_name"] = row.get("country_name")
        result["accepted_capitals"] = accepted_capitals
    return result


class BadNodesPipeline:
    def __init__(
        self,
        config_path: Path,
        repo_root: Path,
    ) -> None:
        self.config_path = config_path
        self.repo_root = repo_root
        self.cfg, self.models, self.vllm_cfg = load_bad_nodes_config(config_path, repo_root)

    def run(
        self,
        model_ids: list[str] | None = None,
        prefix_lengths: list[int] | None = None,
        instructions: list[str] | None = None,
        country_ids: list[str] | None = None,
    ) -> list[Path]:
        models = self.models
        if model_ids:
            models = [model for model in models if model.id in model_ids]

        output_paths: list[Path] = []
        for model_spec in models:
            source_path = self.cfg.source_dir / f"{model_spec.id}.jsonl"
            output_path = self.cfg.output_dir / f"{model_spec.id}.jsonl"
            output_paths.append(output_path)

            records = iter_canonical_records(
                source_path,
                self.cfg.dataset_id,
                self.cfg.canonical_seed,
            )
            if country_ids is not None:
                records = [row for row in records if row.get("country_id") in country_ids]
            if prefix_lengths is not None:
                records = [row for row in records if int(row["prefix_length"]) in prefix_lengths]
            if instructions is not None:
                records = [row for row in records if row["instruction"] in instructions]

            completed = load_completed_tree_keys(output_path)
            pending = []
            for row in records:
                variant = tree_variant_label(cfg.dataset_id, row["instruction"])
                tree_key = make_tree_key(
                    self.cfg.dataset_id,
                    row["model_id"],
                    variant,
                    int(row["prefix_length"]),
                    int(row.get("seed", self.cfg.canonical_seed)),
                    country_id=row.get("country_id"),
                )
                if tree_key not in completed:
                    pending.append(row)

            if completed:
                print(f"Skipping {len(completed)} completed trees for {model_spec.id}", flush=True)
            if not pending:
                continue

            print(f"Running {len(pending)} trees for {model_spec.id}", flush=True)
            politician_registry: PoliticianRegistry | None = None
            if self.cfg.politician_registry_path is not None:
                politician_registry = PoliticianRegistry.from_records(
                    load_registry(self.cfg.politician_registry_path)
                )
            vllm_runner = VllmRunner(model_spec, self.vllm_cfg)
            vllm_runner.load()
            try:
                for row in pending:
                    country_label = f" country={row['country_id']}" if row.get("country_id") else ""
                    print(
                        f"  model={row['model_id']}{country_label} "
                        f"instruction={row['instruction']!r} "
                        f"prefix_length={row['prefix_length']}",
                        flush=True,
                    )
                    result = run_single_tree(vllm_runner, row, self.cfg, politician_registry)
                    append_result(result, output_path)
                    summary = result["summary"]
                    print(
                        f"    leaves={summary['total_leaves']} "
                        f"candidates={summary['total_candidates']} "
                        f"exclusively_bad={summary['exclusively_bad_count']} "
                        f"ditched={summary['ditched_count']}",
                        flush=True,
                    )
            finally:
                vllm_runner.unload()

        return output_paths
