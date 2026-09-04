"""Greedy path trace with per-token confidence metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.models.common import REASONING_END_MARKERS, REASONING_START_MARKERS, split_reasoning_and_answer
from src.models.vllm_runner import VllmRunner
from src.utils.answer import extract_answer_text, is_answer_correct, mentions_lorem


@dataclass
class TokenMetric:
    position: int
    token_text: str
    token_id: int | None
    logprob_selected: float | None
    prob_selected: float | None
    entropy: float | None
    rank_selected: int | None
    phase: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "token_text": self.token_text,
            "token_id": self.token_id,
            "logprob_selected": self.logprob_selected,
            "prob_selected": self.prob_selected,
            "entropy": self.entropy,
            "rank_selected": self.rank_selected,
            "phase": self.phase,
        }


def _entropy_from_logprobs(logprob_dict: dict) -> float:
    probs = [math.exp(entry.logprob) for entry in logprob_dict.values()]
    total = sum(probs)
    if total <= 0:
        return 0.0
    normalized = [prob / total for prob in probs]
    return -sum(prob * math.log(prob) for prob in normalized if prob > 0)


def _rank_selected(logprob_dict: dict, selected_id: int | None) -> int | None:
    if selected_id is None:
        return None
    sorted_ids = sorted(logprob_dict.keys(), key=lambda token_id: logprob_dict[token_id].logprob, reverse=True)
    try:
        return sorted_ids.index(selected_id) + 1
    except ValueError:
        return None


def prompt_opens_reasoning(prompt: str) -> bool:
    """True when the prompt ends inside an unclosed thinking block."""
    last_start = -1
    last_end = -1
    for marker in REASONING_START_MARKERS:
        pos = prompt.rfind(marker)
        if pos > last_start:
            last_start = pos
    for marker in REASONING_END_MARKERS:
        pos = prompt.rfind(marker)
        if pos > last_end:
            last_end = pos
    return last_start > last_end


def classify_generation_phases(token_texts: list[str], starts_in_reasoning: bool) -> list[str]:
    """Tag each generated token as preface, reasoning, or answer."""
    in_reasoning = starts_in_reasoning
    saw_reasoning = starts_in_reasoning
    phases: list[str] = []

    for token_text in token_texts:
        for marker in REASONING_START_MARKERS:
            if marker in token_text:
                in_reasoning = True
                saw_reasoning = True

        if in_reasoning:
            phase = "reasoning"
        elif saw_reasoning:
            phase = "answer"
        else:
            phase = "preface"

        for marker in REASONING_END_MARKERS:
            if marker in token_text:
                phase = "reasoning"
                in_reasoning = False

        phases.append(phase)

    return phases


def summarize_reasoning_metrics(token_metrics: list[TokenMetric]) -> dict[str, Any]:
    reasoning_tokens = [metric for metric in token_metrics if metric.phase == "reasoning"]
    reasoning_entropies = [metric.entropy for metric in reasoning_tokens if metric.entropy is not None]
    reasoning_probs = [metric.prob_selected for metric in reasoning_tokens if metric.prob_selected is not None]
    return {
        "reasoning_token_count": len(reasoning_tokens),
        "mean_entropy_reasoning": (
            sum(reasoning_entropies) / len(reasoning_entropies) if reasoning_entropies else None
        ),
        "mean_logprob_selected": (
            sum(reasoning_probs) / len(reasoning_probs) if reasoning_probs else None
        ),
    }


def retag_token_metrics(
    token_metrics: list[dict[str, Any]],
    prompt_prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recompute per-token phases and reasoning aggregates from saved trace rows."""
    starts_in_reasoning = prompt_opens_reasoning(prompt_prefix)
    phases = classify_generation_phases([metric["token_text"] for metric in token_metrics], starts_in_reasoning)
    retagged: list[TokenMetric] = []
    for metric, phase in zip(token_metrics, phases):
        retagged.append(
            TokenMetric(
                position=metric["position"],
                token_text=metric["token_text"],
                token_id=metric.get("token_id"),
                logprob_selected=metric.get("logprob_selected"),
                prob_selected=metric.get("prob_selected"),
                entropy=metric.get("entropy"),
                rank_selected=metric.get("rank_selected"),
                phase=phase,
            )
        )
    return [metric.to_dict() for metric in retagged], summarize_reasoning_metrics(retagged)


def run_path_probe(
    vllm_runner: VllmRunner,
    user_message: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    logprobs_limit: int = 20,
    accepted_capitals: list[str] | None = None,
) -> dict[str, Any]:
    base_prompt = vllm_runner.format_chat_prompt(user_message)
    outputs = vllm_runner.generate(
        [base_prompt],
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=logprobs_limit,
    )
    output = outputs[0].outputs[0]
    generated_text = output.text

    reasoning_text, answer_text_raw = split_reasoning_and_answer(generated_text)
    answer_text = extract_answer_text(answer_text_raw or generated_text)

    starts_in_reasoning = prompt_opens_reasoning(base_prompt)
    token_texts = [
        vllm_runner.tokenizer.decode([step], skip_special_tokens=False) for step in output.token_ids
    ]
    phases = classify_generation_phases(token_texts, starts_in_reasoning)
    token_metrics: list[TokenMetric] = []

    for position, (step, token_text, phase) in enumerate(zip(output.token_ids, token_texts, phases)):

        step_logprobs = None
        if output.logprobs and position < len(output.logprobs):
            step_logprobs = output.logprobs[position]

        logprob_selected = None
        prob_selected = None
        entropy = None
        rank = None
        if step_logprobs is not None:
            entropy = _entropy_from_logprobs(step_logprobs)
            if step in step_logprobs:
                logprob_selected = step_logprobs[step].logprob
                prob_selected = math.exp(logprob_selected)
            rank = _rank_selected(step_logprobs, step)

        token_metrics.append(
            TokenMetric(
                position=position,
                token_text=token_text,
                token_id=step,
                logprob_selected=logprob_selected,
                prob_selected=prob_selected,
                entropy=entropy,
                rank_selected=rank,
                phase=phase,
            )
        )

    reasoning_summary = summarize_reasoning_metrics(token_metrics)

    return {
        "generated_text": generated_text,
        "reasoning_text": reasoning_text,
        "answer_text": answer_text,
        "answer_correct": is_answer_correct(answer_text, accepted_capitals),
        "mentions_lorem": mentions_lorem(generated_text),
        **reasoning_summary,
        "token_metrics": [metric.to_dict() for metric in token_metrics],
    }
