"""Unit tests for tau pruning logic (no GPU required)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tree.tau_builder import _children_from_logprobs  # noqa: E402


def test_children_from_logprobs_respects_tau():
    logprobs = torch.log(torch.tensor([0.5, 0.4, 0.05, 0.04, 0.01]))
    children = _children_from_logprobs(0.0, logprobs, math.log(0.1), numerical_floor=1e-12)
    token_ids = [token_id for token_id, _ in children]
    assert 0 in token_ids
    assert 1 in token_ids
    assert 2 not in token_ids


def test_children_from_logprobs_uses_cumulative_threshold():
    logprobs = torch.log(torch.tensor([0.5, 0.2, 0.2, 0.05, 0.05]))
    parent_log_p = math.log(0.2)
    children = _children_from_logprobs(parent_log_p, logprobs, math.log(0.05), numerical_floor=1e-12)
    token_ids = [token_id for token_id, _ in children]
    assert 0 in token_ids
    assert 1 not in token_ids
