"""vLLM model loading and generation for completions and path traces."""

from __future__ import annotations

import gc
import os
from typing import Any

os.environ.setdefault("VLLM_USE_V1", "0")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.models.common import (
    ModelSpec,
    VllmConfig,
    find_reasoning_root_from_generated,
)


class VllmRunner:
    """Load a reasoning model with vLLM and format chat prompts."""

    def __init__(self, model_spec: ModelSpec, config: VllmConfig):
        self.model_spec = model_spec
        self.config = config
        self.llm: LLM | None = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_spec.hf_id,
            trust_remote_code=config.trust_remote_code,
        )

    def load(self) -> None:
        kwargs: dict[str, Any] = {
            "model": self.model_spec.hf_id,
            "max_model_len": self.model_spec.max_model_len or self.config.max_model_len,
            "gpu_memory_utilization": (
                self.model_spec.gpu_memory_utilization or self.config.gpu_memory_utilization
            ),
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.enforce_eager:
            kwargs["enforce_eager"] = True
        self.llm = LLM(**kwargs)

    def unload(self) -> None:
        if self.llm is not None:
            del self.llm
            self.llm = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _ensure_loaded(self) -> LLM:
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.llm

    def format_chat_prompt(self, user_message: str, enable_thinking: bool = True) -> str:
        messages = [{"role": "user", "content": user_message}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 1,
        temperature: float = 0.0,
        logprobs: int | None = None,
    ):
        llm = self._ensure_loaded()
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
        )
        return llm.generate(prompts, params)

    def find_reasoning_root_prefix(
        self, user_message: str, probe_max_tokens: int = 8
    ) -> tuple[str, str]:
        base_prompt = self.format_chat_prompt(user_message)
        outputs = self.generate([base_prompt], max_tokens=probe_max_tokens, temperature=0.0)
        generated = outputs[0].outputs[0].text
        return find_reasoning_root_from_generated(base_prompt, generated)
