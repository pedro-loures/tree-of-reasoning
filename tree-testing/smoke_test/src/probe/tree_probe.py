"""BFS tree probe: expand until cumulative path probability falls below threshold."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.models.vllm_runner import ModelSpec, VllmConfig, VllmRunner
from src.probe.distributions import (
    TokenProb,
    distribution_mass,
    filter_distribution,
    logprobs_to_distribution,
    tau_branch_tokens,
    top_branch_tokens,
)


@dataclass
class ProbeNode:
    id: str
    depth: int
    prefix_text: str
    path_prob: float = 1.0
    branch_tokens: list[str] = field(default_factory=list)
    distribution: list[TokenProb] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "depth": self.depth,
            "prefix_text": self.prefix_text,
            "path_prob": self.path_prob,
            "branch_tokens": self.branch_tokens,
            "distribution": [t.to_dict() for t in self.distribution],
            "distribution_mass": distribution_mass(self.distribution),
        }


@dataclass
class ProbeConfig:
    prompt: str
    logprobs_limit: int = 200
    min_prob: float = 0.01
    min_count: int = 2
    branch_mode: str = "tau"
    path_prob_threshold: float = 0.05
    branch_factor: int = 2
    tree_depth: int | None = None
    max_tree_depth: int = 128
    temperature: float = 0.0
    reasoning_probe_max_tokens: int = 8
    results_dir: str = "results"


def load_config(config_path: Path) -> tuple[ProbeConfig, list[ModelSpec], VllmConfig]:
    with config_path.open() as f:
        raw = yaml.safe_load(f)

    probe = ProbeConfig(
        prompt=raw["prompt"],
        logprobs_limit=raw["probe"]["logprobs_limit"],
        min_prob=raw["probe"]["min_prob"],
        min_count=raw["probe"]["min_count"],
        branch_mode=raw["probe"].get("branch_mode", "tau"),
        path_prob_threshold=raw["probe"].get("path_prob_threshold", 0.05),
        branch_factor=raw["probe"]["branch_factor"],
        tree_depth=raw["probe"].get("tree_depth"),
        max_tree_depth=raw["probe"].get("max_tree_depth", 128),
        temperature=raw["probe"]["temperature"],
        reasoning_probe_max_tokens=raw["probe"]["reasoning_probe_max_tokens"],
        results_dir=raw.get("results_dir", "results"),
    )
    models = [
        ModelSpec(
            id=m["id"],
            hf_id=m["hf_id"],
            reasoning_parser=m.get("reasoning_parser", "deepseek_r1"),
            max_model_len=m.get("max_model_len"),
            gpu_memory_utilization=m.get("gpu_memory_utilization"),
        )
        for m in raw["models"]
    ]
    vllm_cfg = VllmConfig(
        max_model_len=raw["vllm"]["max_model_len"],
        gpu_memory_utilization=raw["vllm"]["gpu_memory_utilization"],
        trust_remote_code=raw["vllm"]["trust_remote_code"],
        enforce_eager=raw["vllm"].get("enforce_eager", False),
    )
    return probe, models, vllm_cfg


def _next_token_distribution(
    runner: VllmRunner,
    prefix: str,
    probe: ProbeConfig,
) -> list[TokenProb]:
    outputs = runner.generate(
        [prefix],
        max_tokens=1,
        temperature=probe.temperature,
        logprobs=probe.logprobs_limit,
    )
    out = outputs[0].outputs[0]
    if not out.logprobs or out.logprobs[0] is None:
        raise RuntimeError("vLLM did not return logprobs. Ensure logprobs_limit is set.")
    raw = logprobs_to_distribution(out.logprobs[0])
    return filter_distribution(raw, min_prob=probe.min_prob, min_count=probe.min_count)


def _select_branches(
    raw_distribution: list[TokenProb],
    probe: ProbeConfig,
    parent_path_prob: float,
) -> list[TokenProb]:
    if probe.branch_mode == "top_k":
        return top_branch_tokens(raw_distribution, probe.branch_factor)
    return tau_branch_tokens(raw_distribution, parent_path_prob, probe.path_prob_threshold)


def run_tree_probe(
    runner: VllmRunner,
    probe: ProbeConfig,
) -> dict[str, Any]:
    root_prefix, reasoning_suffix = runner.find_reasoning_root_prefix(
        probe.prompt,
        probe_max_tokens=probe.reasoning_probe_max_tokens,
    )

    nodes: list[ProbeNode] = []
    # (node_id, prefix, depth, path_prob)
    frontier: list[tuple[str, str, int, float]] = [("root", root_prefix, 0, 1.0)]
    node_counter = 0

    while frontier:
        batch = frontier
        frontier = []

        prefixes = [prefix for _, prefix, _, _ in batch]
        outputs = runner.generate(
            prefixes,
            max_tokens=1,
            temperature=probe.temperature,
            logprobs=probe.logprobs_limit,
        )

        for (node_id, prefix, depth, path_prob), output in zip(batch, outputs):
            out = output.outputs[0]
            if not out.logprobs or out.logprobs[0] is None:
                raise RuntimeError(f"No logprobs returned for node {node_id}")

            raw = logprobs_to_distribution(out.logprobs[0])
            distribution = filter_distribution(
                raw,
                min_prob=probe.min_prob,
                min_count=probe.min_count,
            )
            branches = _select_branches(raw, probe, path_prob)
            branch_token_texts = [b.token for b in branches]

            nodes.append(
                ProbeNode(
                    id=node_id,
                    depth=depth,
                    prefix_text=prefix,
                    path_prob=path_prob,
                    branch_tokens=branch_token_texts,
                    distribution=distribution,
                )
            )

            if probe.tree_depth is not None and depth >= probe.tree_depth:
                continue
            if depth >= probe.max_tree_depth:
                continue

            for branch in branches:
                node_counter += 1
                child_id = f"d{depth + 1}_{node_counter}"
                child_prefix = prefix + branch.token
                child_path_prob = path_prob * branch.prob
                frontier.append((child_id, child_prefix, depth + 1, child_path_prob))

    return {
        "model": runner.model_spec.hf_id,
        "model_id": runner.model_spec.id,
        "prompt": probe.prompt,
        "reasoning_root_prefix_suffix": reasoning_suffix,
        "logprobs_limit": probe.logprobs_limit,
        "min_prob": probe.min_prob,
        "min_count": probe.min_count,
        "branch_mode": probe.branch_mode,
        "path_prob_threshold": probe.path_prob_threshold,
        "branch_factor": probe.branch_factor,
        "tree_depth": probe.tree_depth,
        "max_tree_depth": probe.max_tree_depth,
        "max_depth_reached": max((n.depth for n in nodes), default=0),
        "node_count": len(nodes),
        "nodes": [n.to_dict() for n in nodes],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def save_probe_result(result: dict[str, Any], results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    model_slug = result["model_id"].replace("/", "_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = results_dir / f"{model_slug}_{ts}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    return out_path


def format_probe_report(result: dict[str, Any]) -> str:
    lines = [
        f"Model: {result['model']}",
        f"Prompt: {result['prompt'][:80]}...",
        f"Nodes: {result['node_count']}",
        "",
    ]
    for node in result["nodes"]:
        lines.append(f"--- {node['id']} (depth {node['depth']}) ---")
        lines.append(f"Branch tokens: {node['branch_tokens']}")
        lines.append(f"Distribution mass (returned): {node['distribution_mass']:.4f}")
        for entry in node["distribution"]:
            lines.append(f"  {entry['token']!r}: {entry['prob']:.6f}")
        lines.append("")
    return "\n".join(lines)


def run_probe_for_model(
    model_spec: ModelSpec,
    vllm_cfg: VllmConfig,
    probe: ProbeConfig,
    results_dir: Path,
) -> dict[str, Any]:
    runner = VllmRunner(model_spec, vllm_cfg)
    try:
        runner.load()
        result = run_tree_probe(runner, probe)
        save_probe_result(result, results_dir)
        return result
    finally:
        runner.unload()
