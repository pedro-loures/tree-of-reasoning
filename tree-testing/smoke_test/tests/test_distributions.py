"""Unit tests for distribution filtering (no GPU required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.probe.distributions import TokenProb, filter_distribution, tau_branch_tokens, top_branch_tokens  # noqa: E402


def test_filter_distribution_min_prob_and_count():
    dist = [
        TokenProb("a", 0.5),
        TokenProb("b", 0.3),
        TokenProb("c", 0.005),
        TokenProb("d", 0.195),
    ]
    filtered = filter_distribution(dist, min_prob=0.01, min_count=2)
    tokens = [t.token for t in filtered]
    assert "a" in tokens
    assert "b" in tokens
    assert "d" in tokens
    assert "c" not in tokens


def test_filter_distribution_pads_to_min_count():
    dist = [
        TokenProb("a", 0.9),
        TokenProb("b", 0.05),
        TokenProb("c", 0.05),
    ]
    filtered = filter_distribution(dist, min_prob=0.01, min_count=2)
    assert len(filtered) >= 2


def test_filter_distribution_pads_when_only_one_above_threshold():
    dist = [
        TokenProb("a", 0.99),
        TokenProb("b", 0.005),
        TokenProb("c", 0.003),
        TokenProb("d", 0.002),
    ]
    filtered = filter_distribution(dist, min_prob=0.01, min_count=2)
    assert len(filtered) == 2
    assert filtered[0].token == "a"


def test_tau_branch_tokens():
    dist = [
        TokenProb("a", 0.5),
        TokenProb("b", 0.3),
        TokenProb("c", 0.15),
        TokenProb("d", 0.05),
    ]
    branches = tau_branch_tokens(dist, parent_path_prob=0.2, threshold=0.05)
    assert [t.token for t in branches] == ["a", "b"]
    # a: 0.1, b: 0.06 >= 0.05; c: 0.03, d: 0.01 pruned

    root_branches = tau_branch_tokens(dist, parent_path_prob=1.0, threshold=0.05)
    assert [t.token for t in root_branches] == ["a", "b", "c", "d"]


def test_top_branch_tokens():
    dist = [
        TokenProb("a", 0.5),
        TokenProb("b", 0.3),
        TokenProb("c", 0.2),
    ]
    branches = top_branch_tokens(dist, branch_factor=2)
    assert [t.token for t in branches] == ["a", "b"]
