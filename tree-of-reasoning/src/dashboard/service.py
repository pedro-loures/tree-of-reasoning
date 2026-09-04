"""Run prompt-driven tau trees for the interactive dashboard."""

from __future__ import annotations

import gc
import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.completion.bad_node_analyzer import (
    build_node_index,
    evaluate_candidates,
    summarize_bad_nodes,
)
from src.completion.leaf_completer import complete_leaves
from src.dashboard.payload import build_run_entry
from src.dashboard.persist import default_save_dir
from src.dashboard.progress import ProgressTracker
from src.embeddings.storage import dashboard_embedding_key, save_node_embeddings
from src.models.common import ModelSpec, VllmConfig, load_experiment_config
from src.models.hf_runner import HfRunner
from src.models.vllm_runner import VllmRunner
from src.tree.metrics import compute_tree_metrics
from src.tree.tau_builder import TauTreeBuildResult, build_tau_tree, stack_node_embeddings
from src.utils.answer_scoring import normalize_answer_mode

LEAF_COMPLETION_CHUNK_SIZE = 8


def build_incoming_token_map(nodes: list[dict[str, Any]]) -> dict[str, str]:
    incoming: dict[str, str] = {}
    for node in nodes:
        for child_id, token in zip(node.get("child_ids", []), node.get("child_tokens", [])):
            incoming[child_id] = token
    return incoming


def compact_tree(nodes: list[dict[str, Any]], root_prefix: str | None = None) -> list[dict[str, Any]]:
    parent_token = build_incoming_token_map(nodes)
    if root_prefix is None:
        root_node = next((node for node in nodes if node.get("id") == "root"), None)
        root_prefix = root_node.get("prefix_text", "") if root_node else ""

    compact: list[dict[str, Any]] = []
    for node in nodes:
        entry: dict[str, Any] = {
            "id": node["id"],
            "d": node["depth"],
            "p": round(node["path_prob"], 6),
            "c": node["child_ids"],
        }
        if node["id"] != "root":
            entry["tok"] = parent_token.get(node["id"], "?")
        if node.get("child_tokens"):
            entry["ct"] = node["child_tokens"]
        prefix_text = node.get("prefix_text")
        if prefix_text and node["id"] != "root":
            suffix = prefix_text[len(root_prefix):] if prefix_text.startswith(root_prefix) else prefix_text
            if suffix:
                entry["suffix"] = suffix
        compact.append(entry)
    return compact


def parse_expected_answers(raw: str | list[str] | None) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        values = [item.strip() for item in raw if item and item.strip()]
        return values or None
    values = [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]
    return values or None


def make_tree_key(prompt: str, model_id: str, tau: float) -> str:
    digest = hashlib.sha1(f"{prompt}|{model_id}|{tau}".encode()).hexdigest()[:10]
    return f"interactive:{model_id}:{tau:g}:{digest}:{uuid.uuid4().hex[:6]}"


@dataclass
class DashboardConfig:
    config_path: Path
    repo_root: Path
    max_tree_depth: int = 512
    breadth_warning_threshold: int = 20
    numerical_floor: float = 1e-12
    reasoning_probe_max_tokens: int = 8
    max_completion_tokens: int = 4096
    temperature: float = 0.0
    top_k_logprobs: int = 20
    capture_hidden_states: bool = True
    embeddings_dir: Path | None = None

    def resolved_embeddings_dir(self) -> Path:
        return self.embeddings_dir or (default_save_dir() / "embeddings")


class DashboardService:
    def __init__(self, dashboard_config: DashboardConfig) -> None:
        self.dashboard_config = dashboard_config
        self._models: list[ModelSpec] = []
        self._vllm_cfg: VllmConfig | None = None
        self._load_config()

    def _load_config(self) -> None:
        _, models, vllm_cfg = load_experiment_config(
            self.dashboard_config.config_path,
            repo_root=self.dashboard_config.repo_root,
        )
        self._models = models
        self._vllm_cfg = vllm_cfg

    def list_models(self) -> list[dict[str, str]]:
        return [{"id": model.id, "hf_id": model.hf_id} for model in self._models]

    def _model_spec(self, model_id: str) -> ModelSpec:
        for model in self._models:
            if model.id == model_id:
                return model
        raise ValueError(f"Unknown model_id: {model_id}")

    def _save_embeddings(
        self,
        tree_key: str,
        build_result: TauTreeBuildResult,
    ) -> dict[str, Any] | None:
        if not self.dashboard_config.capture_hidden_states or not build_result.capture_layers:
            return None

        node_ids, parent_ids, hidden_states, top_k_token_ids, top_k_logprobs = stack_node_embeddings(
            build_result.tree,
            build_result.node_features,
            build_result.capture_layers,
        )
        if not node_ids:
            return None

        manifest = save_node_embeddings(
            self.dashboard_config.resolved_embeddings_dir(),
            run_key=dashboard_embedding_key(tree_key),
            node_ids=node_ids,
            parent_ids=parent_ids,
            layers=build_result.capture_layers,
            hidden_states=hidden_states,
            top_k_token_ids=top_k_token_ids,
            top_k_logprobs=top_k_logprobs,
        )
        return manifest.to_dict()

    def _save_embeddings_for_prompt(
        self,
        *,
        prompt: str,
        model_id: str,
        tau: float,
        tree_key: str,
    ) -> dict[str, Any] | None:
        """Rebuild the τ-tree with hidden-state capture and persist embeddings only."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        model_spec = self._model_spec(model_id)
        cfg = self.dashboard_config
        hf_runner = HfRunner(model_spec)
        hf_runner.load()
        try:
            root_prefix, _ = hf_runner.find_reasoning_root_prefix(
                prompt,
                probe_max_tokens=cfg.reasoning_probe_max_tokens,
            )
            build_result = build_tau_tree(
                hf_runner,
                root_prefix=root_prefix,
                tau=tau,
                max_depth=cfg.max_tree_depth,
                breadth_warning_threshold=cfg.breadth_warning_threshold,
                numerical_floor=cfg.numerical_floor,
                batch_size=model_spec.hf_batch_size or 4,
                capture_hidden_states=cfg.capture_hidden_states,
                top_k_logprobs=cfg.top_k_logprobs,
            )
            return self._save_embeddings(tree_key, build_result)
        finally:
            hf_runner.unload()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_tree(
        self,
        *,
        prompt: str,
        model_id: str,
        tau: float,
        expected_answers: str | list[str] | None = None,
        answer_mode: str | None = "or",
        progress: ProgressTracker | None = None,
    ) -> dict[str, Any]:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        model_spec = self._model_spec(model_id)
        accepted_answers = parse_expected_answers(expected_answers)
        resolved_answer_mode = normalize_answer_mode(answer_mode)
        cfg = self.dashboard_config
        vllm_cfg = self._vllm_cfg
        if vllm_cfg is None:
            raise RuntimeError("Dashboard config is not loaded")
        tree_key = make_tree_key(prompt, model_id, tau)
        embeddings_manifest: dict[str, Any] | None = None

        def report(stage: str, message: str, *, total: int | None = None) -> None:
            if progress is not None:
                progress.start_stage(stage, message, total=total)

        report("load_hf", f"Loading {model_spec.id} (HuggingFace)")
        hf_runner = HfRunner(model_spec)
        hf_runner.load()
        try:
            report("find_root", "Finding reasoning root prefix")
            root_prefix, _ = hf_runner.find_reasoning_root_prefix(
                prompt,
                probe_max_tokens=cfg.reasoning_probe_max_tokens,
            )

            tree_batches = 0

            def on_tree_progress(nodes: int, frontier: int) -> None:
                nonlocal tree_batches
                tree_batches += 1
                if progress is not None:
                    if tree_batches == 1:
                        progress.start_stage("build_tree", "Building τ-tree", total=None)
                    progress.update(
                        advance=1,
                        message=f"BFS batch {tree_batches}",
                        nodes=nodes,
                    )

            build_result = build_tau_tree(
                hf_runner,
                root_prefix=root_prefix,
                tau=tau,
                max_depth=cfg.max_tree_depth,
                breadth_warning_threshold=cfg.breadth_warning_threshold,
                numerical_floor=cfg.numerical_floor,
                batch_size=model_spec.hf_batch_size or 4,
                capture_hidden_states=cfg.capture_hidden_states,
                top_k_logprobs=cfg.top_k_logprobs,
                on_progress=on_tree_progress if progress is not None else None,
            )
            tree = build_result.tree
            tree_metrics = compute_tree_metrics(tree)
            if progress is not None:
                progress.start_stage("save_embeddings", "Saving node embeddings", total=1)
            embeddings_manifest = self._save_embeddings(tree_key, build_result)
            if progress is not None:
                progress.update(
                    advance=1,
                    message=(
                        f"Saved embeddings for {tree_metrics['total_nodes']} nodes"
                        if embeddings_manifest
                        else "No embeddings captured"
                    ),
                    nodes=tree_metrics["total_nodes"],
                    leaves=tree_metrics["leaf_count"],
                )
        finally:
            hf_runner.unload()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        report("load_vllm", f"Loading {model_spec.id} (vLLM)")
        vllm_runner = VllmRunner(model_spec, vllm_cfg)
        vllm_runner.load()
        try:
            all_leaves = tree.leaves
            leaf_total = len(all_leaves)
            if progress is not None:
                progress.start_stage(
                    "complete_leaves",
                    f"Completing {leaf_total} leaves",
                    total=leaf_total,
                )
                progress.update(leaves=leaf_total)

            leaf_completions = []
            for start in range(0, leaf_total, LEAF_COMPLETION_CHUNK_SIZE):
                chunk = all_leaves[start : start + LEAF_COMPLETION_CHUNK_SIZE]
                chunk_ids = [leaf.id for leaf in chunk]
                leaf_completions.extend(
                    complete_leaves(
                        vllm_runner,
                        tree,
                        leaf_ids=chunk_ids,
                        max_tokens=cfg.max_completion_tokens,
                        temperature=cfg.temperature,
                        accepted_capitals=accepted_answers,
                        answer_mode=resolved_answer_mode,
                        expected_answers_raw=expected_answers,
                    )
                )
                if progress is not None:
                    done = min(start + len(chunk), leaf_total)
                    progress.update(
                        current=done,
                        message=f"Completed {done}/{leaf_total} leaves",
                        leaves=leaf_total,
                    )

            if accepted_answers is None:
                for completion in leaf_completions:
                    completion.answer_correct = completion.reasoning_complete
        finally:
            vllm_runner.unload()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if progress is not None:
            progress.start_stage("analyze", "Classifying good/bad nodes", total=1)
            progress.update(advance=0, message="Evaluating candidate nodes")

        nodes_by_id = build_node_index(tree)
        leaf_cache = {item.leaf_id: item for item in leaf_completions}
        candidate_results = evaluate_candidates(nodes_by_id, leaf_cache)
        summary = summarize_bad_nodes(leaf_cache, candidate_results)

        if progress is not None:
            progress.update(
                advance=1,
                message=(
                    f"{summary['exclusively_bad_count']} exclusively bad · "
                    f"{summary['total_leaves']} leaves"
                ),
            )

        compact_nodes = compact_tree(tree.to_dict()["nodes"], root_prefix=root_prefix)
        return build_run_entry(
            tree_key=tree_key,
            prompt=prompt,
            model_id=model_id,
            tau=tau,
            tree_nodes=compact_nodes,
            tree_metrics=tree_metrics,
            leaf_completions=[item.to_dict() for item in leaf_cache.values()],
            candidate_nodes=[item.to_dict() for item in candidate_results],
            summary=summary,
            expected_answers=expected_answers if isinstance(expected_answers, str) else (
                ", ".join(expected_answers) if expected_answers else None
            ),
            answer_mode=resolved_answer_mode if accepted_answers else None,
            embeddings=embeddings_manifest,
        )
