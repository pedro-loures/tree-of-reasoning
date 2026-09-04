"""Deterministic Lorem Ipsum prefix generation by word count."""

from __future__ import annotations

LOREM_WORDS = """
Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua Ut enim ad minim veniam quis
nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore
eu fugiat nulla pariatur Excepteur sint occaecat cupidatat non proident sunt
in culpa qui officia deserunt mollit anim id est laborum
""".split()


def lorem_prefix(word_count: int) -> str:
    """Return the first ``word_count`` words of Lorem ipsum in fixed order.

    Longer prefixes extend shorter ones so the filler text stays predictable.
    Words repeat from the start when ``word_count`` exceeds the corpus length.
    """
    if word_count <= 0:
        return ""
    words: list[str] = []
    while len(words) < word_count:
        words.extend(LOREM_WORDS)
    return " ".join(words[:word_count])


def sample_lorem_prefix(word_count: int, seed: int = 0) -> str:
    """Compatibility wrapper; ``seed`` is ignored (prefix is deterministic)."""
    del seed
    return lorem_prefix(word_count)


def build_prompt(prefix: str, instruction: str) -> str:
    prefix = prefix.strip()
    if prefix:
        return f"{prefix} {instruction}"
    return instruction
