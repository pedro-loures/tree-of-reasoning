"""Tests for multi-answer scoring modes."""

from __future__ import annotations

from src.utils.answer_scoring import (
    evaluate_expected_answers,
    parse_answer_categories,
    score_category_answers,
    score_flat_answers,
)


def test_or_mode_any_match() -> None:
    correct, hits = score_flat_answers("The capital is Brasília", ["Brasília", "Rio"], "or")
    assert correct is True
    assert hits["Brasília"] is True
    assert hits["Rio"] is False


def test_and_mode_requires_all() -> None:
    correct, hits = score_flat_answers("Brasília is in Brazil", ["Brasília", "Brazil"], "and")
    assert correct is True
    assert all(hits.values())

    correct, hits = score_flat_answers("Brasília only", ["Brasília", "Brazil"], "and")
    assert correct is False
    assert hits["Brasília"] is True
    assert hits["Brazil"] is False


def test_xor_mode_exactly_one() -> None:
    correct, _ = score_flat_answers("Brasília", ["Brasília", "Rio"], "xor")
    assert correct is True

    correct, _ = score_flat_answers("Brasília and Rio", ["Brasília", "Rio"], "xor")
    assert correct is False

    correct, _ = score_flat_answers("Neither city", ["Brasília", "Rio"], "xor")
    assert correct is False


def test_categories_with_synonyms() -> None:
    categories = parse_answer_categories("Brasília|Brasilia, Brazil")
    assert categories == [["Brasília", "Brasilia"], ["Brazil"]]

    correct, hits = score_category_answers("Capital: Brasilia, country Brazil", categories)
    assert correct is True
    assert hits["Brasília | Brasilia"] is True
    assert hits["Brazil"] is True


def test_evaluate_expected_answers_categories_mode() -> None:
    correct, hits, mode = evaluate_expected_answers(
        "Brasília",
        "Brasília|Brasilia, Rio",
        "categories",
    )
    assert mode == "categories"
    assert correct is False
    assert hits["Brasília | Brasilia"] is True
    assert hits["Rio"] is False
