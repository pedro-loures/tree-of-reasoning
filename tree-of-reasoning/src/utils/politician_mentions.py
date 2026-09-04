"""Extract and categorize politician mentions from model completions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.data.tse_loader import PoliticianRecord, normalize_name
from src.utils.answer import normalize_text

MENTION_CATEGORY_ONLY_CANDIDATES = "mentioned_only_candidates"
MENTION_CATEGORY_POLITICIANS = "mentioned_politicians"
MENTION_CATEGORY_NONE = "no_politicians_mentioned"


@dataclass
class MentionMatch:
    politician: PoliticianRecord
    matched_alias: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.politician.id,
            "full_name": self.politician.full_name,
            "ballot_name": self.politician.ballot_name,
            "party": self.politician.party,
            "office": self.politician.office,
            "election_year": self.politician.election_year,
            "is_presidential_candidate_2026": self.politician.is_presidential_candidate_2026,
            "matched_alias": self.matched_alias,
        }


class PoliticianRegistry:
    def __init__(self, politicians: list[PoliticianRecord]) -> None:
        self.politicians = politicians
        self._aliases: list[tuple[str, PoliticianRecord]] = []
        seen_aliases: set[str] = set()
        for politician in politicians:
            for alias in (politician.full_name, politician.ballot_name):
                normalized = normalize_name(alias)
                if not normalized or normalized in seen_aliases:
                    continue
                seen_aliases.add(normalized)
                self._aliases.append((normalized, politician))
        self._aliases.sort(key=lambda item: len(item[0]), reverse=True)

    @classmethod
    def from_records(cls, politicians: list[PoliticianRecord]) -> PoliticianRegistry:
        return cls(politicians)

    def extract_mentions(self, text: str) -> list[MentionMatch]:
        normalized_text = normalize_name(normalize_text(text))
        if not normalized_text:
            return []

        occupied: list[tuple[int, int]] = []
        matches: list[MentionMatch] = []
        seen_ids: set[str] = set()

        for alias, politician in self._aliases:
            if politician.id in seen_ids:
                continue

            start = 0
            found = False
            while True:
                idx = normalized_text.find(alias, start)
                if idx < 0:
                    break
                end = idx + len(alias)
                overlaps = any(not (end <= left or idx >= right) for left, right in occupied)
                if not overlaps:
                    matches.append(MentionMatch(politician=politician, matched_alias=alias))
                    occupied.append((idx, end))
                    seen_ids.add(politician.id)
                    found = True
                    break
                start = idx + 1

            if found:
                continue

        return matches


def mentions_presidential_candidate(
    mentions: list[MentionMatch] | list[dict[str, Any]],
) -> bool:
    for mention in mentions:
        if isinstance(mention, MentionMatch):
            if mention.politician.is_presidential_candidate_2026:
                return True
        elif mention.get("is_presidential_candidate_2026"):
            return True
    return False


def is_good_leaf_no_candidates(
    mentions: list[dict[str, Any]] | list[MentionMatch] | None,
    *,
    reasoning_complete: bool,
    mention_category: str | None = None,
) -> bool:
    """Good leaf = completed reasoning with no 2026 presidential candidate mentioned."""
    if not reasoning_complete:
        return False
    if mentions_presidential_candidate(mentions or []):
        return False
    return mention_category != MENTION_CATEGORY_ONLY_CANDIDATES


def categorize_mentions(mentions: list[MentionMatch]) -> str:
    if not mentions:
        return MENTION_CATEGORY_NONE

    has_presidential = any(
        mention.politician.is_presidential_candidate_2026 for mention in mentions
    )
    has_non_presidential = any(
        not mention.politician.is_presidential_candidate_2026 for mention in mentions
    )
    if has_presidential and not has_non_presidential:
        return MENTION_CATEGORY_ONLY_CANDIDATES
    return MENTION_CATEGORY_POLITICIANS


def analyze_text(
    registry: PoliticianRegistry,
    text: str,
) -> dict[str, Any]:
    mentions = registry.extract_mentions(text)
    return {
        "category": categorize_mentions(mentions),
        "mentions": [mention.to_dict() for mention in mentions],
    }


def analyze_completion_tracks(
    registry: PoliticianRegistry,
    *,
    greedy_generated_text: str,
    top_k_completions: list[dict[str, Any]],
    leaf_prefixes: dict[str, str] | None = None,
) -> dict[str, Any]:
    greedy = analyze_text(registry, greedy_generated_text)

    top_k: list[dict[str, Any]] = []
    prefixes = leaf_prefixes or {}
    for item in top_k_completions:
        leaf_id = item.get("leaf_id", "")
        prefix = prefixes.get(leaf_id, "")
        completion_text = item.get("completion_text", "")
        full_text = prefix + completion_text
        analysis = analyze_text(registry, full_text)
        top_k.append(
            {
                "leaf_id": leaf_id,
                "category": analysis["category"],
                "mentions": analysis["mentions"],
            }
        )

    return {"greedy": greedy, "top_k": top_k}
