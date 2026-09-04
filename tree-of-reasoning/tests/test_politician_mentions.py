"""Tests for politician mention extraction and categorization."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.tse_loader import build_registry  # noqa: E402
from src.utils.politician_mentions import (  # noqa: E402
    MENTION_CATEGORY_NONE,
    MENTION_CATEGORY_ONLY_CANDIDATES,
    MENTION_CATEGORY_POLITICIANS,
    PoliticianRegistry,
    analyze_completion_tracks,
    categorize_mentions,
    is_good_leaf_no_candidates,
    mentions_presidential_candidate,
)

FIXTURE_DIR = ROOT / "tests" / "fixtures" / "tse"


def _registry() -> PoliticianRegistry:
    records = build_registry(
        FIXTURE_DIR / "consulta_cand_2026_BRASIL.csv",
        FIXTURE_DIR / "consulta_cand_2022_BRASIL.csv",
    )
    return PoliticianRegistry.from_records(records)


def test_extract_presidential_candidate_by_ballot_name():
    registry = _registry()
    mentions = registry.extract_mentions("I think Lula is likely to win.")
    assert len(mentions) == 1
    assert mentions[0].politician.party == "PT"
    assert mentions[0].politician.is_presidential_candidate_2026


def test_extract_with_accent_variant():
    registry = _registry()
    mentions = registry.extract_mentions("Flávio Bolsonaro could be a factor.")
    assert len(mentions) == 1
    assert mentions[0].politician.full_name == "FLAVIO NANTES BOLSONARO"


def test_categorize_only_candidates():
    registry = _registry()
    mentions = registry.extract_mentions("Lula and Bolsonaro are the front-runners.")
    assert categorize_mentions(mentions) == MENTION_CATEGORY_ONLY_CANDIDATES


def test_categorize_mixed_politicians():
    registry = _registry()
    mentions = registry.extract_mentions("Lula may face competition from Tarcísio.")
    assert categorize_mentions(mentions) == MENTION_CATEGORY_POLITICIANS


def test_categorize_non_presidential_politician_only():
    registry = _registry()
    mentions = registry.extract_mentions("Ciro Gomes remains influential.")
    assert categorize_mentions(mentions) == MENTION_CATEGORY_POLITICIANS


def test_no_politicians_mentioned():
    registry = _registry()
    mentions = registry.extract_mentions("The economy will decide the outcome.")
    assert categorize_mentions(mentions) == MENTION_CATEGORY_NONE


def test_is_good_leaf_no_candidates():
    registry = _registry()
    candidate_mentions = registry.extract_mentions("Lula is likely.")
    politician_only = registry.extract_mentions("Tarcísio could run.")
    none = registry.extract_mentions("The economy matters.")

    assert mentions_presidential_candidate(candidate_mentions)
    assert not mentions_presidential_candidate(politician_only)
    assert is_good_leaf_no_candidates(none, reasoning_complete=True)
    assert not is_good_leaf_no_candidates(candidate_mentions, reasoning_complete=True)
    assert is_good_leaf_no_candidates(politician_only, reasoning_complete=True)
    assert not is_good_leaf_no_candidates(none, reasoning_complete=False)


def test_analyze_completion_tracks_includes_leaf_prefix():
    registry = _registry()
    analysis = analyze_completion_tracks(
        registry,
        greedy_generated_text="No names here.",
        top_k_completions=[
            {
                "leaf_id": "d5_1",
                "completion_text": " I believe Lula will win.",
            }
        ],
        leaf_prefixes={"d5_1": "Reasoning about "},
    )
    assert analysis["greedy"]["category"] == MENTION_CATEGORY_NONE
    assert analysis["top_k"][0]["category"] == MENTION_CATEGORY_ONLY_CANDIDATES
