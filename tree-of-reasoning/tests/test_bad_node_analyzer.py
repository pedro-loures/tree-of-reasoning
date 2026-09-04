"""Tests for bad-node leaf subtype classification."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.completion.bad_node_analyzer import classify_bad_subtypes  # noqa: E402
from src.completion.leaf_completer import LeafCompletionResult  # noqa: E402


def _leaf(**kwargs) -> LeafCompletionResult:
    defaults = {
        "leaf_id": "d0_0",
        "path_prob": 0.1,
        "completion_text": "",
        "answer_text": "",
        "answer_correct": False,
        "mentions_lorem": False,
        "reasoning_complete": True,
    }
    defaults.update(kwargs)
    return LeafCompletionResult(**defaults)


def test_wrong_city_and_lorem_drift_can_coexist():
    result = _leaf(
        answer_text="The capital of Brazil is Rio de Janeiro.",
        mentions_lorem=True,
    )
    assert classify_bad_subtypes(result) == ["lorem_drift", "wrong_city"]


def test_correct_leaf_has_no_failure_subtypes():
    result = _leaf(
        answer_text="The capital of Brazil is Brasília.",
        answer_correct=True,
        mentions_lorem=True,
    )
    assert classify_bad_subtypes(result) == []


def test_incomplete_and_empty_can_coexist():
    result = _leaf(reasoning_complete=False, answer_text="")
    assert classify_bad_subtypes(result) == ["incomplete", "empty"]
