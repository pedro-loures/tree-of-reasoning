"""Unit tests for leaf completion parsing (no GPU required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.completion.leaf_completer import parse_leaf_completion  # noqa: E402

_THINK_CLOSE = "<" + "/" + "think" + ">"
_REDACTED_THINK_CLOSE = "<" + "/" + "redacted_thinking" + ">"


def test_parse_leaf_completion_requires_reasoning_end():
    prefix = "system prompt<think>\nLet me work through this."
    generated = " Still thinking about capitals."
    answer, _, complete = parse_leaf_completion(prefix, generated)
    assert not complete
    assert answer == ""


def test_parse_leaf_completion_extracts_answer_after_reasoning():
    prefix = "system<think>\n"
    generated = "Brazil's capital is Brasilia." + _REDACTED_THINK_CLOSE + "\nBrasilia"
    answer, _, complete = parse_leaf_completion(prefix, generated)
    assert complete
    assert "Brasilia" in answer or "Brasília" in answer


def test_parse_leaf_completion_supports_think_close_marker():
    prefix = "prompt<think>\n"
    generated = "done" + _THINK_CLOSE + "\nThe answer is Brasilia."
    answer, _, complete = parse_leaf_completion(prefix, generated)
    assert complete
    assert "Brasilia" in answer
