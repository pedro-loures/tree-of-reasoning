"""Combine multiple expected answers with AND / OR / XOR / category logic."""

from __future__ import annotations

from typing import Literal

from src.utils.answer import normalize_text

AnswerMode = Literal["or", "and", "xor", "categories"]
ANSWER_MODES: tuple[AnswerMode, ...] = ("or", "and", "xor", "categories")


def normalize_answer_mode(mode: str | None) -> AnswerMode:
    if mode in ANSWER_MODES:
        return mode
    return "or"


def text_matches_term(text: str, term: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    return normalize_text(term) in normalized


def parse_answer_terms(raw: str | list[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [item.strip() for item in raw if item and item.strip()]
    return [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]


def parse_answer_categories(raw: str | list[str] | None) -> list[list[str]]:
    """Each comma-separated segment is a category; ``|`` gives synonyms within one."""
    if raw is None:
        return []
    if isinstance(raw, list):
        segments = raw
    else:
        segments = raw.replace("\n", ",").split(",")
    categories: list[list[str]] = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        alternatives = [item.strip() for item in segment.split("|") if item.strip()]
        if alternatives:
            categories.append(alternatives)
    return categories


def category_label(alternatives: list[str]) -> str:
    if len(alternatives) == 1:
        return alternatives[0]
    return " | ".join(alternatives)


def score_flat_answers(text: str, answers: list[str], mode: AnswerMode) -> tuple[bool, dict[str, bool]]:
    if not answers:
        return False, {}
    hits = {answer: text_matches_term(text, answer) for answer in answers}
    hit_count = sum(hits.values())
    if mode == "or":
        correct = hit_count >= 1
    elif mode == "and":
        correct = hit_count == len(answers)
    elif mode == "xor":
        correct = hit_count == 1
    else:
        correct = hit_count == len(answers)
    return correct, hits


def score_category_answers(text: str, categories: list[list[str]]) -> tuple[bool, dict[str, bool]]:
    if not categories:
        return False, {}
    hits = {
        category_label(alternatives): any(text_matches_term(text, alt) for alt in alternatives)
        for alternatives in categories
    }
    return all(hits.values()), hits


def evaluate_expected_answers(
    text: str,
    raw_answers: str | list[str] | None,
    mode: str | None = "or",
) -> tuple[bool, dict[str, bool], AnswerMode]:
    answer_mode = normalize_answer_mode(mode)
    if answer_mode == "categories":
        categories = parse_answer_categories(raw_answers)
        if not categories:
            return False, {}, answer_mode
        correct, hits = score_category_answers(text, categories)
        return correct, hits, answer_mode

    answers = parse_answer_terms(raw_answers)
    if not answers:
        return False, {}, answer_mode
    if len(answers) == 1:
        hit = text_matches_term(text, answers[0])
        return hit, {answers[0]: hit}, answer_mode
    correct, hits = score_flat_answers(text, answers, answer_mode)
    return correct, hits, answer_mode
