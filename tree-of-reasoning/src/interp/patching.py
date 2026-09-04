"""On-demand activation patching for targeted mech-interp (phase 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from src.models.hf_runner import HfRunner


@dataclass
class PatchResult:
    prefix_text: str
    layer_index: int
    token_position: int
    baseline_top_token_id: int
    patched_top_token_id: int
    baseline_top_logprob: float
    patched_top_logprob: float
    logprobs: torch.Tensor

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix_text": self.prefix_text,
            "layer_index": self.layer_index,
            "token_position": self.token_position,
            "baseline_top_token_id": self.baseline_top_token_id,
            "patched_top_token_id": self.patched_top_token_id,
            "baseline_top_logprob": self.baseline_top_logprob,
            "patched_top_logprob": self.patched_top_logprob,
        }


def _resolve_token_position(seq_len: int, token_position: int) -> int:
    if token_position < 0:
        return seq_len + token_position
    return token_position


def patch_and_forward(
    hf_runner: HfRunner,
    prefix_text: str,
    layer_index: int,
    patch_vector: torch.Tensor,
    token_position: int = -1,
) -> PatchResult:
    """Patch the residual stream at ``layer_index`` and return next-token logprobs."""
    model = hf_runner._ensure_loaded()
    input_ids = hf_runner.tokenize_prefix(prefix_text).to(model.device)
    seq_len = int(input_ids.shape[1])
    position = _resolve_token_position(seq_len, token_position)

    with torch.no_grad():
        baseline_outputs = model(input_ids=input_ids)
        baseline_logits = baseline_outputs.logits[0, -1]
        baseline_logprobs = F.log_softmax(baseline_logits, dim=-1)
        baseline_top = int(torch.argmax(baseline_logprobs).item())

    patch = patch_vector.to(model.device, dtype=next(model.parameters()).dtype)
    if patch.ndim != 1:
        raise ValueError("patch_vector must be a 1-D hidden state vector")

    captured: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        patched = hidden.clone()
        patched[0, position, :] = patch
        captured["hidden"] = patched
        return (patched, *output[1:]) if isinstance(output, tuple) else patched

    layers = getattr(model, "model", model)
    if not hasattr(layers, "layers"):
        raise RuntimeError("Unsupported model architecture for activation patching")
    target_layer = layers.layers[layer_index]
    handle = target_layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            patched_outputs = model(input_ids=input_ids)
            patched_logits = patched_outputs.logits[0, -1]
            patched_logprobs = F.log_softmax(patched_logits, dim=-1)
            patched_top = int(torch.argmax(patched_logprobs).item())
    finally:
        handle.remove()

    return PatchResult(
        prefix_text=prefix_text,
        layer_index=layer_index,
        token_position=position,
        baseline_top_token_id=baseline_top,
        patched_top_token_id=patched_top,
        baseline_top_logprob=float(baseline_logprobs[baseline_top].item()),
        patched_top_logprob=float(patched_logprobs[patched_top].item()),
        logprobs=patched_logprobs,
    )


def patch_from_source_node(
    hf_runner: HfRunner,
    target_prefix: str,
    source_hidden: torch.Tensor,
    layer_index: int,
    token_position: int = -1,
) -> PatchResult:
    """Copy ``source_hidden`` into ``target_prefix`` at the given layer/token."""
    return patch_and_forward(
        hf_runner,
        prefix_text=target_prefix,
        layer_index=layer_index,
        patch_vector=source_hidden,
        token_position=token_position,
    )
