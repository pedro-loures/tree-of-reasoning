"""Smoke tests: verify each model loads and generates in reasoning mode."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.vllm_runner import VllmRunner  # noqa: E402


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("model_id", ["deepseek-r1-7b", "qwq-32b-awq"])
def test_model_smoke(model_id: str, loaded_config):
    probe, models, vllm_cfg = loaded_config
    model_spec = next(m for m in models if m.id == model_id)

    runner = VllmRunner(model_spec, vllm_cfg)
    try:
        runner.load()
        output = runner.smoke_generate("What is 2+2? Reply briefly.")
        assert output, f"Model {model_spec.hf_id} returned empty output"
    finally:
        runner.unload()
