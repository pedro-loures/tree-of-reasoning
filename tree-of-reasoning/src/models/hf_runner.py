"""HuggingFace model loading, forward features, and full-vocabulary logits."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.common import ModelSpec, find_reasoning_root_from_generated


@dataclass
class ForwardFeatures:
    logprobs: torch.Tensor
    hidden_by_layer: dict[int, np.ndarray]
    top_k_token_ids: np.ndarray
    top_k_logprobs: np.ndarray


def snapshot_layer_indices(num_hidden_layers: int) -> list[int]:
    """Return early, 1/3, 2/3, and final layer indices for hidden_states tuples."""
    if num_hidden_layers <= 0:
        return []
    last = num_hidden_layers
    early = min(4, last)
    third = max(1, num_hidden_layers // 3)
    two_third = max(third, (2 * num_hidden_layers) // 3)
    return sorted({early, third, two_third, last})


class HfRunner:
    """Load a causal LM with transformers for tau-tree building."""

    def __init__(self, model_spec: ModelSpec, trust_remote_code: bool = True):
        self.model_spec = model_spec
        self.trust_remote_code = trust_remote_code
        self.model: AutoModelForCausalLM | None = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_spec.hf_id,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load(self) -> None:
        is_awq = "awq" in self.model_spec.hf_id.lower()
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": torch.float16 if is_awq else torch.bfloat16,
            "device_map": "auto",
        }
        self.model = AutoModelForCausalLM.from_pretrained(self.model_spec.hf_id, **load_kwargs)
        self.model.eval()

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_loaded(self) -> AutoModelForCausalLM:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model

    @property
    def num_hidden_layers(self) -> int:
        model = self._ensure_loaded()
        config = model.config
        return int(getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0)))

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

    def tokenize_prefix(self, prefix_text: str) -> torch.Tensor:
        encoded = self.tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
        return encoded["input_ids"]

    def next_token_logprobs(self, prefix_text: str) -> torch.Tensor:
        return self.forward_batch([prefix_text])[0].logprobs

    def next_token_logprobs_batch(self, prefix_texts: list[str]) -> list[torch.Tensor]:
        return [features.logprobs for features in self.forward_batch(prefix_texts)]

    def forward_batch(
        self,
        prefix_texts: list[str],
        capture_layers: list[int] | None = None,
        top_k: int = 20,
    ) -> list[ForwardFeatures]:
        if not prefix_texts:
            return []
        model = self._ensure_loaded()
        layer_indices = capture_layers or snapshot_layer_indices(self.num_hidden_layers)
        encoded = self.tokenizer(
            prefix_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=bool(layer_indices),
            )
            last_indices = attention_mask.sum(dim=1) - 1
            results: list[ForwardFeatures] = []
            for row, last_idx in enumerate(last_indices):
                logits = outputs.logits[row, last_idx]
                logprobs = F.log_softmax(logits, dim=-1)
                probs = torch.exp(logprobs)
                top_values, top_ids = torch.topk(probs, k=min(top_k, probs.numel()))
                hidden_by_layer: dict[int, np.ndarray] = {}
                if layer_indices and outputs.hidden_states is not None:
                    for layer_idx in layer_indices:
                        vector = outputs.hidden_states[layer_idx][row, last_idx]
                        hidden_by_layer[layer_idx] = vector.to(torch.float16).cpu().numpy()
                results.append(
                    ForwardFeatures(
                        logprobs=logprobs,
                        hidden_by_layer=hidden_by_layer,
                        top_k_token_ids=top_ids.to(torch.int32).cpu().numpy(),
                        top_k_logprobs=torch.log(top_values).to(torch.float32).cpu().numpy(),
                    )
                )
        return results

    def find_reasoning_root_prefix(
        self, user_message: str, probe_max_tokens: int = 8
    ) -> tuple[str, str]:
        model = self._ensure_loaded()
        base_prompt = self.format_chat_prompt(user_message)
        input_ids = self.tokenize_prefix(base_prompt).to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=probe_max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = generated_ids[0, input_ids.shape[1] :]
        generated = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return find_reasoning_root_from_generated(base_prompt, generated)

    def decode_token_id(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], skip_special_tokens=False)

    def get_hidden_state(
        self,
        prefix_text: str,
        layer_index: int,
        token_position: int = -1,
    ) -> np.ndarray:
        """Fetch a single hidden vector for on-demand patching workflows."""
        features = self.forward_batch([prefix_text], capture_layers=[layer_index])
        return features[0].hidden_by_layer[layer_index]
