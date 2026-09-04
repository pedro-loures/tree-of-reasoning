"""Shared pytest fixtures for smoke_test."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.vllm_runner import ModelSpec, VllmConfig  # noqa: E402
from src.probe.tree_probe import ProbeConfig  # noqa: E402


@pytest.fixture(scope="session")
def probe_config_path() -> Path:
    return ROOT / "configs" / "probe.yaml"


@pytest.fixture(scope="session")
def loaded_config(probe_config_path: Path):
    with probe_config_path.open() as f:
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


@pytest.fixture(scope="session")
def results_dir(loaded_config) -> Path:
    probe, _, _ = loaded_config
    path = ROOT / probe.results_dir
    path.mkdir(parents=True, exist_ok=True)
    return path
