from src.trace.path_probe import classify_generation_phases, prompt_opens_reasoning

_THINK_OPEN = "<" + "think" + ">"
_THINK_CLOSE = "<" + "/" + "think" + ">"
_REDACTED_OPEN = "<" + "redacted_thinking" + ">"
_REDACTED_CLOSE = "<" + "/" + "redacted_thinking" + ">"


def test_prompt_opens_reasoning_when_block_unclosed():
    prompt = f"user message{_REDACTED_OPEN}\n"
    assert prompt_opens_reasoning(prompt) is True


def test_prompt_opens_reasoning_false_after_close():
    prompt = f"user{_REDACTED_OPEN}thoughts{_REDACTED_CLOSE}\nanswer"
    assert prompt_opens_reasoning(prompt) is False


def test_classify_generation_phases_starts_in_reasoning():
    phases = classify_generation_phases(["Okay", ",", " so"], starts_in_reasoning=True)
    assert phases == ["reasoning", "reasoning", "reasoning"]


def test_classify_generation_phases_closes_reasoning():
    phases = classify_generation_phases(
        ["done", _THINK_CLOSE, "Bras", "ília"],
        starts_in_reasoning=True,
    )
    assert phases == ["reasoning", "reasoning", "answer", "answer"]


def test_classify_generation_phases_opens_in_generation():
    phases = classify_generation_phases(
        ["preface", _THINK_OPEN, "thought", _THINK_CLOSE, "answer"],
        starts_in_reasoning=False,
    )
    assert phases == ["preface", "reasoning", "reasoning", "reasoning", "answer"]
