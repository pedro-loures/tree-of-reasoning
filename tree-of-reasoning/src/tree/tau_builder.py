"""Tau-pruned tree builder using full-vocabulary HuggingFace logits."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from src.models.hf_runner import ForwardFeatures, HfRunner, snapshot_layer_indices


@dataclass
class TauTreeNode:
    id: str
    depth: int
    prefix_text: str
    log_path_prob: float
    path_prob: float
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)
    child_token_ids: list[int] = field(default_factory=list)
    child_tokens: list[str] = field(default_factory=list)
    breadth: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "depth": self.depth,
            "prefix_text": self.prefix_text,
            "log_path_prob": self.log_path_prob,
            "path_prob": self.path_prob,
            "parent_id": self.parent_id,
            "breadth": self.breadth,
            "child_ids": self.child_ids,
            "child_token_ids": self.child_token_ids,
            "child_tokens": self.child_tokens,
        }


@dataclass
class BreadthWarning:
    node_id: str
    depth: int
    breadth: int

    def to_dict(self) -> dict[str, str | int]:
        return {"node_id": self.node_id, "depth": self.depth, "breadth": self.breadth}


@dataclass
class TauTreeResult:
    nodes: list[TauTreeNode]
    breadth_warnings: list[BreadthWarning]
    breadth_warning_count: int
    tau: float
    max_tree_depth: int
    root_prefix: str

    @property
    def leaves(self) -> list[TauTreeNode]:
        return [node for node in self.nodes if not node.child_ids]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau": self.tau,
            "max_tree_depth": self.max_tree_depth,
            "root_prefix": self.root_prefix,
            "node_count": len(self.nodes),
            "breadth_warning_count": self.breadth_warning_count,
            "breadth_warnings": [warning.to_dict() for warning in self.breadth_warnings],
            "nodes": [node.to_dict() for node in self.nodes],
        }


@dataclass
class TauTreeBuildResult:
    tree: TauTreeResult
    node_features: dict[str, ForwardFeatures]
    capture_layers: list[int]


def _children_from_logprobs(
    log_path_prob: float,
    logprobs: torch.Tensor,
    log_tau: float,
    numerical_floor: float,
) -> list[tuple[int, float]]:
    import math

    child_log_probs = log_path_prob + logprobs
    mask = child_log_probs >= log_tau
    if not mask.any():
        return []
    token_ids = torch.nonzero(mask, as_tuple=False).flatten()
    results: list[tuple[int, float]] = []
    for token_id_tensor in token_ids:
        token_id = int(token_id_tensor.item())
        prob = math.exp(log_path_prob + float(logprobs[token_id].item()))
        if prob < numerical_floor:
            continue
        results.append((token_id, prob))
    results.sort(key=lambda item: item[1], reverse=True)
    return results


def build_tau_tree(
    hf_runner: HfRunner,
    root_prefix: str,
    tau: float,
    max_depth: int,
    breadth_warning_threshold: int = 20,
    numerical_floor: float = 1e-12,
    batch_size: int = 4,
    capture_hidden_states: bool = True,
    top_k_logprobs: int = 20,
    on_progress: Callable[[int, int], None] | None = None,
) -> TauTreeBuildResult:
    import math

    log_tau = math.log(tau)
    capture_layers = snapshot_layer_indices(hf_runner.num_hidden_layers) if capture_hidden_states else []
    nodes: list[TauTreeNode] = []
    warnings: list[BreadthWarning] = []
    node_lookup: dict[str, TauTreeNode] = {}
    node_features: dict[str, ForwardFeatures] = {}

    root = TauTreeNode(
        id="root",
        depth=0,
        prefix_text=root_prefix,
        log_path_prob=0.0,
        path_prob=1.0,
        parent_id=None,
    )
    nodes.append(root)
    node_lookup[root.id] = root

    frontier: list[TauTreeNode] = [root]
    node_counter = 0

    while frontier:
        batch_nodes = frontier
        frontier = []
        prefixes = [node.prefix_text for node in batch_nodes]

        for start in range(0, len(prefixes), batch_size):
            batch = batch_nodes[start : start + batch_size]
            batch_prefixes = prefixes[start : start + batch_size]
            feature_batch = hf_runner.forward_batch(
                batch_prefixes,
                capture_layers=capture_layers,
                top_k=top_k_logprobs,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for parent, features in zip(batch, feature_batch):
                node_features[parent.id] = features
                logprobs = features.logprobs
                if parent.depth >= max_depth:
                    continue

                children = _children_from_logprobs(
                    parent.log_path_prob,
                    logprobs,
                    log_tau,
                    numerical_floor,
                )
                parent.breadth = len(children)

                if parent.breadth >= breadth_warning_threshold:
                    warnings.append(
                        BreadthWarning(
                            node_id=parent.id,
                            depth=parent.depth,
                            breadth=parent.breadth,
                        )
                    )

                for token_id, path_prob in children:
                    token_text = hf_runner.decode_token_id(token_id)
                    child_prefix = parent.prefix_text + token_text
                    node_counter += 1
                    child = TauTreeNode(
                        id=f"d{parent.depth + 1}_{node_counter}",
                        depth=parent.depth + 1,
                        prefix_text=child_prefix,
                        log_path_prob=parent.log_path_prob + float(logprobs[token_id].item()),
                        path_prob=path_prob,
                        parent_id=parent.id,
                    )
                    nodes.append(child)
                    node_lookup[child.id] = child
                    parent.child_ids.append(child.id)
                    parent.child_token_ids.append(token_id)
                    parent.child_tokens.append(token_text)
                    frontier.append(child)

            if on_progress is not None:
                on_progress(len(nodes), len(frontier))

    tree = TauTreeResult(
        nodes=nodes,
        breadth_warnings=warnings,
        breadth_warning_count=len(warnings),
        tau=tau,
        max_tree_depth=max_depth,
        root_prefix=root_prefix,
    )
    return TauTreeBuildResult(
        tree=tree,
        node_features=node_features,
        capture_layers=capture_layers,
    )


def stack_node_embeddings(
    tree: TauTreeResult,
    node_features: dict[str, ForwardFeatures],
    capture_layers: list[int],
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    node_ids: list[str] = []
    parent_ids: list[str] = []
    hidden_rows: list[np.ndarray] = []
    top_k_token_ids: list[np.ndarray] = []
    top_k_logprobs: list[np.ndarray] = []

    for node in tree.nodes:
        features = node_features.get(node.id)
        if features is None or not capture_layers:
            continue
        layer_vectors = [features.hidden_by_layer[layer_idx] for layer_idx in capture_layers]
        node_ids.append(node.id)
        parent_ids.append(node.parent_id or "")
        hidden_rows.append(np.stack(layer_vectors, axis=0))
        top_k_token_ids.append(features.top_k_token_ids)
        top_k_logprobs.append(features.top_k_logprobs)

    if not hidden_rows:
        empty_hidden = np.empty((0, len(capture_layers), 0), dtype=np.float16)
        empty_top = np.empty((0, 0), dtype=np.float32)
        return node_ids, parent_ids, empty_hidden, empty_top, empty_top

    return (
        node_ids,
        parent_ids,
        np.stack(hidden_rows, axis=0).astype(np.float16),
        np.stack(top_k_token_ids, axis=0),
        np.stack(top_k_logprobs, axis=0),
    )
