"""Complete top-k tau-tree leaves with vLLM and evaluate answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models.common import reasoning_is_complete, split_reasoning_and_answer
from src.models.vllm_runner import VllmRunner
from src.tree.tau_builder import TauTreeNode, TauTreeResult
from src.utils.answer import extract_answer_text, is_answer_correct, mentions_lorem
from src.utils.answer_scoring import AnswerMode, evaluate_expected_answers, normalize_answer_mode
from src.utils.politician_mentions import (
    PoliticianRegistry,
    analyze_text,
    is_good_leaf_no_candidates,
)


@dataclass
class LeafCompletionResult:
    leaf_id: str
    path_prob: float
    completion_text: str
    answer_text: str
    answer_correct: bool
    mentions_lorem: bool
    reasoning_complete: bool
    answer_mode: str | None = None
    answer_matches: dict[str, bool] | None = None
    mention_category: str | None = None
    mentions: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "leaf_id": self.leaf_id,
            "path_prob": self.path_prob,
            "completion_text": self.completion_text,
            "answer_text": self.answer_text,
            "answer_correct": self.answer_correct,
            "mentions_lorem": self.mentions_lorem,
            "reasoning_complete": self.reasoning_complete,
        }
        if self.answer_mode is not None:
            payload["answer_mode"] = self.answer_mode
        if self.answer_matches is not None:
            payload["answer_matches"] = self.answer_matches
        if self.mention_category is not None:
            payload["mention_category"] = self.mention_category
        if self.mentions is not None:
            payload["mentions"] = self.mentions
        return payload


def parse_leaf_completion(prefix_text: str, generated_text: str) -> tuple[str, str, bool]:
    """Parse greedy continuation from a tree leaf prefix.

    Returns (answer_text, completion_text, reasoning_complete). The answer is
    only evaluated after the reasoning block closes — unfinished reasoning does
    not count as a completed line of thought.
    """
    completion_text = generated_text
    full_text = prefix_text + generated_text
    complete = reasoning_is_complete(full_text)
    if not complete:
        return "", completion_text, False

    _, answer_part = split_reasoning_and_answer(full_text)
    answer_text = extract_answer_text(answer_part or "")
    return answer_text, completion_text, True


def leaf_completion_from_dict(data: dict[str, Any]) -> LeafCompletionResult:
    return LeafCompletionResult(
        leaf_id=data["leaf_id"],
        path_prob=float(data["path_prob"]),
        completion_text=data.get("completion_text", ""),
        answer_text=data.get("answer_text", ""),
        answer_correct=bool(data.get("answer_correct", False)),
        mentions_lorem=bool(data.get("mentions_lorem", False)),
        reasoning_complete=bool(data.get("reasoning_complete", False)),
        answer_mode=data.get("answer_mode"),
        answer_matches=data.get("answer_matches"),
        mention_category=data.get("mention_category"),
        mentions=data.get("mentions"),
    )


def seed_cache_from_top_k(top_k_metrics: dict[str, Any]) -> dict[str, LeafCompletionResult]:
    cache: dict[str, LeafCompletionResult] = {}
    for item in top_k_metrics.get("top_k_completions", []):
        result = leaf_completion_from_dict(item)
        cache[result.leaf_id] = result
    return cache


def rescore_leaf_cache(
    cache: dict[str, LeafCompletionResult],
    accepted_capitals: list[str] | None,
    politician_registry: PoliticianRegistry | None = None,
    leaf_prefixes: dict[str, str] | None = None,
    answer_mode: str | None = None,
    expected_answers_raw: str | list[str] | None = None,
) -> None:
    prefixes = leaf_prefixes or {}
    mode = normalize_answer_mode(answer_mode)
    for result in cache.values():
        if politician_registry is not None:
            prefix = prefixes.get(result.leaf_id, "")
            analysis = analyze_text(politician_registry, prefix + result.completion_text)
            if result.reasoning_complete:
                result.mention_category = analysis["category"]
                result.mentions = analysis["mentions"]
                result.answer_correct = is_good_leaf_no_candidates(
                    result.mentions,
                    reasoning_complete=result.reasoning_complete,
                    mention_category=result.mention_category,
                )
            else:
                result.mention_category = "incomplete"
                result.mentions = []
                result.answer_correct = False
        elif result.reasoning_complete:
            if expected_answers_raw is not None:
                correct, hits, resolved_mode = evaluate_expected_answers(
                    result.answer_text,
                    expected_answers_raw,
                    mode,
                )
                result.answer_correct = correct
                result.answer_mode = resolved_mode
                result.answer_matches = hits
            else:
                result.answer_correct = is_answer_correct(result.answer_text, accepted_capitals)
                result.answer_mode = None
                result.answer_matches = None


def _completion_from_leaf(
    leaf: TauTreeNode,
    generated_text: str,
    accepted_capitals: list[str] | None = None,
    politician_registry: PoliticianRegistry | None = None,
    answer_mode: str | None = None,
    expected_answers_raw: str | list[str] | None = None,
) -> LeafCompletionResult:
    answer_text, completion_text, complete = parse_leaf_completion(
        leaf.prefix_text, generated_text
    )
    result = LeafCompletionResult(
        leaf_id=leaf.id,
        path_prob=leaf.path_prob,
        completion_text=completion_text,
        answer_text=answer_text,
        answer_correct=False,
        mentions_lorem=mentions_lorem(completion_text),
        reasoning_complete=complete,
    )
    if politician_registry is not None:
        if complete:
            analysis = analyze_text(politician_registry, leaf.prefix_text + completion_text)
            result.mention_category = analysis["category"]
            result.mentions = analysis["mentions"]
            result.answer_correct = is_good_leaf_no_candidates(
                result.mentions,
                reasoning_complete=result.reasoning_complete,
                mention_category=result.mention_category,
            )
        else:
            result.mention_category = "incomplete"
            result.mentions = []
    elif complete:
        if expected_answers_raw is not None:
            correct, hits, resolved_mode = evaluate_expected_answers(
                answer_text,
                expected_answers_raw,
                answer_mode,
            )
            result.answer_correct = correct
            result.answer_mode = resolved_mode
            result.answer_matches = hits
        else:
            result.answer_correct = is_answer_correct(answer_text, accepted_capitals)
    return result


def complete_leaves(
    vllm_runner: VllmRunner,
    tree: TauTreeResult,
    leaf_ids: list[str] | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    accepted_capitals: list[str] | None = None,
    politician_registry: PoliticianRegistry | None = None,
    answer_mode: str | None = None,
    expected_answers_raw: str | list[str] | None = None,
) -> list[LeafCompletionResult]:
    """Complete the given tau-tree leaves (or all leaves when leaf_ids is None)."""
    by_id = {node.id: node for node in tree.nodes}
    if leaf_ids is None:
        target_leaves = tree.leaves
    else:
        target_leaves = [by_id[leaf_id] for leaf_id in leaf_ids]

    if not target_leaves:
        return []

    prompts = [leaf.prefix_text for leaf in target_leaves]
    outputs = vllm_runner.generate(prompts, max_tokens=max_tokens, temperature=temperature)
    return [
        _completion_from_leaf(
            leaf,
            output.outputs[0].text,
            accepted_capitals,
            politician_registry=politician_registry,
            answer_mode=answer_mode,
            expected_answers_raw=expected_answers_raw,
        )
        for leaf, output in zip(target_leaves, outputs)
    ]


def complete_top_k_leaves(
    vllm_runner: VllmRunner,
    tree: TauTreeResult,
    k: int,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    accepted_capitals: list[str] | None = None,
) -> list[LeafCompletionResult]:
    leaves = sorted(tree.leaves, key=lambda node: node.log_path_prob, reverse=True)[:k]
    if not leaves:
        return []

    leaf_ids = [leaf.id for leaf in leaves]
    return complete_leaves(
        vllm_runner,
        tree,
        leaf_ids=leaf_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        accepted_capitals=accepted_capitals,
    )


def summarize_top_k_completions(completions: list[LeafCompletionResult]) -> dict[str, Any]:
    if not completions:
        return {
            "top_1_correct": False,
            "top_k_any_correct": False,
            "top_k_answers": [],
            "top_k_leaf_paths": [],
            "top_k_completions": [],
            "top_1_reasoning_complete": False,
            "top_k_any_reasoning_complete": False,
        }

    return {
        "top_1_correct": completions[0].answer_correct,
        "top_k_any_correct": any(item.answer_correct for item in completions),
        "top_k_answers": [item.answer_text for item in completions],
        "top_k_leaf_paths": [item.path_prob for item in completions],
        "top_k_completions": [item.to_dict() for item in completions],
        "top_1_reasoning_complete": completions[0].reasoning_complete,
        "top_k_any_reasoning_complete": any(item.reasoning_complete for item in completions),
    }
