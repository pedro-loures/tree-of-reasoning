"""Probability distribution filtering and branching helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TokenProb:
    token: str
    prob: float
    token_id: int | None = None

    def to_dict(self) -> dict:
        result = {"token": self.token, "prob": self.prob}
        if self.token_id is not None:
            result["token_id"] = self.token_id
        return result


def logprobs_to_distribution(logprob_dict: dict) -> list[TokenProb]:
    """Convert vLLM logprob dict (token_id -> Logprob) to sorted TokenProb list."""
    entries: list[TokenProb] = []
    for token_id, entry in logprob_dict.items():
        prob = math.exp(entry.logprob)
        token_text = entry.decoded_token if entry.decoded_token is not None else str(token_id)
        entries.append(TokenProb(token=token_text, prob=prob, token_id=int(token_id)))
    entries.sort(key=lambda x: x.prob, reverse=True)
    return entries


def filter_distribution(
    distribution: list[TokenProb],
    min_prob: float = 0.01,
    min_count: int = 2,
) -> list[TokenProb]:
    """Keep tokens with p >= min_prob; pad to min_count with top tokens if needed."""
    filtered = [t for t in distribution if t.prob >= min_prob]
    if len(filtered) >= min_count:
        return filtered

    seen = {t.token for t in filtered}
    for token in distribution:
        if token.token not in seen:
            filtered.append(token)
            seen.add(token.token)
        if len(filtered) >= min_count:
            break
    return filtered


def tau_branch_tokens(
    distribution: list[TokenProb],
    parent_path_prob: float,
    threshold: float,
) -> list[TokenProb]:
    """Keep tokens whose cumulative path probability stays at or above threshold."""
    branches: list[TokenProb] = []
    for token in distribution:
        path_prob = parent_path_prob * token.prob
        if path_prob >= threshold:
            branches.append(token)
    return branches


def top_branch_tokens(distribution: list[TokenProb], branch_factor: int = 2) -> list[TokenProb]:
    """Select top branch_factor tokens for tree expansion."""
    if not distribution:
        return []
    return distribution[:branch_factor]


def distribution_mass(distribution: list[TokenProb]) -> float:
    return sum(t.prob for t in distribution)
