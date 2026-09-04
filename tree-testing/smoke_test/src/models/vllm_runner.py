"""vLLM model loading and reasoning-aware prompt formatting."""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Any

# Legacy v0 engine is required for this CUDA/driver stack.
os.environ.setdefault("VLLM_USE_V1", "0")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


_THINK_OPEN = "<" + "think" + ">"
_REDACTED_THINK_OPEN = "<" + "redacted_thinking" + ">"
_THINK_CLOSE = "<" + "/" + "think" + ">"
_REDACTED_THINK_CLOSE = "<" + "/" + "redacted_thinking" + ">"

REASONING_START_MARKERS = (_THINK_OPEN, _REDACTED_THINK_OPEN)
REASONING_END_MARKERS = (_THINK_CLOSE, _REDACTED_THINK_CLOSE)


@dataclass
class VllmConfig:
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    trust_remote_code: bool = True
    enforce_eager: bool = False


@dataclass
class ModelSpec:
    id: str
    hf_id: str
    reasoning_parser: str = "deepseek_r1"
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None


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

    def smoke_generate(self, user_message: str = "What is 2+2?") -> str:
        prompt = self.format_chat_prompt(user_message)
        outputs = self.generate([prompt], max_tokens=32, temperature=0.6)
        return outputs[0].outputs[0].text

    def find_reasoning_root_prefix(self, user_message: str, probe_max_tokens: int = 8) -> tuple[str, str]:
        """
        Return (root_prefix, reasoning_suffix) where root_prefix ends right before
        the first in-reasoning content token.
        """
        base_prompt = self.format_chat_prompt(user_message)
        outputs = self.generate([base_prompt], max_tokens=probe_max_tokens, temperature=0.0)
        generated = outputs[0].outputs[0].text

        for marker in REASONING_START_MARKERS:
            if marker in generated:
                idx = generated.index(marker) + len(marker)
                reasoning_suffix = generated[idx:]
                root_prefix = base_prompt + generated[:idx]
                return root_prefix, reasoning_suffix

        # No explicit delimiter: treat first generated token as reasoning start.
        if generated:
            root_prefix = base_prompt
            return root_prefix, generated

        return base_prompt, ""

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], skip_special_tokens=False)

    def encode_token_text(self, token_text: str) -> str:
        """Append token text to a running prefix."""
        return token_text
