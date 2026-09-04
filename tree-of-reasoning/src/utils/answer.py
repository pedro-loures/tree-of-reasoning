"""Answer parsing and correctness checks."""

from __future__ import annotations

import re

BRAZIL_CAPITALS = ["brasilia", "brasília"]

CORRECT_ANSWERS = {
    "brasilia",
    "brasília",
    "brazil's capital is brasilia",
    "brazil's capital is brasília",
}

LOREM_MARKERS = ("lorem", "ipsum", "dolor", "amet", "consectetur", "adipiscing")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_answer_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return cleaned


def is_answer_correct(text: str, accepted_capitals: list[str] | None = None) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False

    capitals = accepted_capitals if accepted_capitals is not None else BRAZIL_CAPITALS
    normalized_capitals = [normalize_text(capital) for capital in capitals]
    if normalized in CORRECT_ANSWERS and accepted_capitals is None:
        return True
    return any(capital in normalized for capital in normalized_capitals)


def mentions_lorem(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in LOREM_MARKERS)
