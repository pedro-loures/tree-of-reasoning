"""Shared model types and reasoning delimiter helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.data.countries import CountrySpec, instruction_from_template, load_countries

_THINK_OPEN = "<" + "think" + ">"
_REDACTED_THINK_OPEN = "<" + "redacted_thinking" + ">"
_THINK_CLOSE = "<" + "/" + "think" + ">"
_REDACTED_THINK_CLOSE = "<" + "/" + "redacted_thinking" + ">"

REASONING_START_MARKERS = (_THINK_OPEN, _REDACTED_THINK_OPEN)
REASONING_END_MARKERS = (_THINK_CLOSE, _REDACTED_THINK_CLOSE)


@dataclass
class ModelSpec:
    id: str
    hf_id: str
    reasoning_parser: str = "deepseek_r1"
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    hf_batch_size: int | None = None


@dataclass
class VllmConfig:
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    trust_remote_code: bool = True
    enforce_eager: bool = False


@dataclass
class InstructionSweep:
    instruction: str
    prefix_lengths: list[int]
    seeds: list[int]
    country_id: str | None = None
    country_name: str | None = None
    accepted_capitals: list[str] | None = None


@dataclass
class Condition:
    instruction: str
    prefix_length: int
    seed: int
    country_id: str | None = None
    country_name: str | None = None
    accepted_capitals: list[str] | None = None


@dataclass
class ExperimentConfig:
    prefix_lengths: list[int]
    seeds: list[int]
    tau: float
    max_tree_depth: int
    top_k_leaves: int
    breadth_warning_threshold: int
    instructions: list[str]
    temperature: float
    max_completion_tokens: int
    reasoning_probe_max_tokens: int
    logprobs_limit: int
    hf_batch_size: int
    numerical_floor: float
    results_dir: str
    instruction_sweeps: list[InstructionSweep] | None = None
    capture_hidden_states: bool = True
    embeddings_dir: str = "embeddings"
    top_k_logprobs: int = 20
    countries: list[CountrySpec] | None = None
    politician_registry_path: str | None = None

    def iter_conditions(self) -> list[Condition]:
        if self.instruction_sweeps:
            conditions: list[Condition] = []
            for sweep in self.instruction_sweeps:
                for prefix_length in sweep.prefix_lengths:
                    for seed in sweep.seeds:
                        conditions.append(
                            Condition(
                                instruction=sweep.instruction,
                                prefix_length=prefix_length,
                                seed=seed,
                                country_id=sweep.country_id,
                                country_name=sweep.country_name,
                                accepted_capitals=sweep.accepted_capitals,
                            )
                        )
            return conditions
        return [
            Condition(
                instruction=instruction,
                prefix_length=prefix_length,
                seed=seed,
            )
            for instruction in self.instructions
            for prefix_length in self.prefix_lengths
            for seed in self.seeds
        ]


def split_reasoning_and_answer(text: str) -> tuple[str, str]:
    for marker in REASONING_END_MARKERS:
        if marker in text:
            parts = text.split(marker, 1)
            return parts[0], parts[1] if len(parts) > 1 else ""
    return text, ""


def reasoning_is_complete(text: str) -> bool:
    return any(marker in text for marker in REASONING_END_MARKERS)


def find_reasoning_root_from_generated(base_prompt: str, generated: str) -> tuple[str, str]:
    for marker in REASONING_START_MARKERS:
        if marker in generated:
            idx = generated.index(marker) + len(marker)
            return base_prompt + generated[:idx], generated[idx:]
    if generated:
        return base_prompt, generated
    return base_prompt, ""


def _expand_instruction_sweeps(
    exp: dict[str, Any],
    countries: list[CountrySpec] | None,
) -> tuple[list[InstructionSweep], list[str]]:
    if "instruction_sweeps" not in exp:
        return [], list(exp.get("instructions", []))

    instruction_sweeps: list[InstructionSweep] = []
    instructions: list[str] = list(exp.get("instructions", []))

    for sweep in exp["instruction_sweeps"]:
        if "template" in sweep:
            if not countries:
                raise ValueError("instruction_sweeps with template require countries_file")
            for country in countries:
                instruction = instruction_from_template(country, sweep["template"])
                instruction_sweeps.append(
                    InstructionSweep(
                        instruction=instruction,
                        prefix_lengths=list(sweep["prefix_lengths"]),
                        seeds=list(sweep["seeds"]),
                        country_id=country.id,
                        country_name=country.name,
                        accepted_capitals=country.normalized_capitals,
                    )
                )
                instructions.append(instruction)
        else:
            instruction_sweeps.append(
                InstructionSweep(
                    instruction=sweep["instruction"],
                    prefix_lengths=list(sweep["prefix_lengths"]),
                    seeds=list(sweep.get("seeds", [0])),
                )
            )
            instructions.append(sweep["instruction"])

    if not instructions and instruction_sweeps:
        instructions = [sweep.instruction for sweep in instruction_sweeps]
    return instruction_sweeps, instructions


def load_experiment_config(
    config_path: Path,
    repo_root: Path | None = None,
) -> tuple[ExperimentConfig, list[ModelSpec], VllmConfig]:
    with config_path.open() as f:
        raw = yaml.safe_load(f)

    exp = raw["experiment"]
    countries: list[CountrySpec] | None = None
    if "countries_file" in raw:
        countries_path = Path(raw["countries_file"])
        if not countries_path.is_absolute():
            base = repo_root or config_path.parent.parent
            countries_path = (base / countries_path).resolve()
        countries = load_countries(countries_path)

    if "instructions" in exp:
        instructions = list(exp["instructions"])
    elif "instruction" in exp:
        instructions = [exp["instruction"]]
    else:
        instructions = []

    instruction_sweeps: list[InstructionSweep] | None = None
    if "instruction_sweeps" in exp:
        instruction_sweeps, sweep_instructions = _expand_instruction_sweeps(exp, countries)
        if not instructions:
            instructions = sweep_instructions

    experiment = ExperimentConfig(
        prefix_lengths=exp.get("prefix_lengths", [0]),
        seeds=exp.get("seeds", [0]),
        tau=exp["tau"],
        max_tree_depth=exp["max_tree_depth"],
        top_k_leaves=exp["top_k_leaves"],
        breadth_warning_threshold=exp["breadth_warning_threshold"],
        instructions=instructions,
        temperature=exp.get("temperature", 0.0),
        max_completion_tokens=exp.get("max_completion_tokens", 4096),
        reasoning_probe_max_tokens=exp.get("reasoning_probe_max_tokens", 8),
        logprobs_limit=exp.get("logprobs_limit", 20),
        hf_batch_size=exp.get("hf_batch_size", 4),
        numerical_floor=exp.get("numerical_floor", 1e-12),
        results_dir=raw.get("results_dir", "results"),
        instruction_sweeps=instruction_sweeps,
        capture_hidden_states=exp.get("capture_hidden_states", True),
        embeddings_dir=exp.get("embeddings_dir", "embeddings"),
        top_k_logprobs=exp.get("top_k_logprobs", 20),
        countries=countries,
        politician_registry_path=raw.get("politician_registry"),
    )
    models = [
        ModelSpec(
            id=m["id"],
            hf_id=m["hf_id"],
            reasoning_parser=m.get("reasoning_parser", "deepseek_r1"),
            max_model_len=m.get("max_model_len"),
            gpu_memory_utilization=m.get("gpu_memory_utilization"),
            hf_batch_size=m.get("hf_batch_size"),
        )
        for m in raw["models"]
    ]
    vllm_raw = raw.get("vllm", {})
    vllm_cfg = VllmConfig(
        max_model_len=vllm_raw.get("max_model_len", 8192),
        gpu_memory_utilization=vllm_raw.get("gpu_memory_utilization", 0.90),
        trust_remote_code=vllm_raw.get("trust_remote_code", True),
        enforce_eager=vllm_raw.get("enforce_eager", False),
    )
    return experiment, models, vllm_cfg
