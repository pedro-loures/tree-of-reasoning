"""Standalone trace metric retagging for backfill (no dependency on tree-of-reasoning)."""

from __future__ import annotations

from typing import Any

_THINK_OPEN = "<" + "think" + ">"
_REDACTED_THINK_OPEN = "<" + "redacted_thinking" + ">"
_THINK_CLOSE = "<" + "/" + "think" + ">"
_REDACTED_THINK_CLOSE = "<" + "/" + "redacted_thinking" + ">"

REASONING_START_MARKERS = (_THINK_OPEN, _REDACTED_THINK_OPEN)
REASONING_END_MARKERS = (_THINK_CLOSE, _REDACTED_THINK_CLOSE)


def prompt_opens_reasoning(prompt: str) -> bool:
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


def summarize_reasoning_metrics(token_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    reasoning_tokens = [metric for metric in token_metrics if metric["phase"] == "reasoning"]
    reasoning_entropies = [
        metric["entropy"] for metric in reasoning_tokens if metric.get("entropy") is not None
    ]
    reasoning_probs = [
        metric["prob_selected"] for metric in reasoning_tokens if metric.get("prob_selected") is not None
    ]
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
    starts_in_reasoning = prompt_opens_reasoning(prompt_prefix)
    phases = classify_generation_phases([metric["token_text"] for metric in token_metrics], starts_in_reasoning)
    retagged: list[dict[str, Any]] = []
    for metric, phase in zip(token_metrics, phases):
        updated = dict(metric)
        updated["phase"] = phase
        retagged.append(updated)
    return retagged, summarize_reasoning_metrics(retagged)
