"""Compute cosine shift and predictability features per internal node."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.io import RunRecord


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64)
    right = right.astype(np.float64)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    similarity = float(np.dot(left, right) / denom)
    return 1.0 - similarity


def topk_entropy(top_k_logprobs: np.ndarray) -> float:
    probs = np.exp(top_k_logprobs.astype(np.float64))
    total = probs.sum()
    if total <= 0.0:
        return 0.0
    normalized = probs / total
    positive = normalized[normalized > 0]
    return float(-(positive * np.log(positive)).sum())


def _incoming_token(node: dict[str, Any], parent: dict[str, Any] | None) -> str:
    if parent is None:
        return ""
    child_ids = parent.get("child_ids", [])
    child_tokens = parent.get("child_tokens", [])
    node_id = node["id"]
    if node_id in child_ids:
        index = child_ids.index(node_id)
        if index < len(child_tokens):
            return child_tokens[index]
    node_prefix = node.get("prefix_text", "")
    parent_prefix = parent.get("prefix_text", "")
    if node_prefix.startswith(parent_prefix):
        return node_prefix[len(parent_prefix) :]
    return ""


def build_node_rows(run: RunRecord, min_breadth: int = 1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    node_by_id = run.node_by_id

    for node in run.tree_nodes:
        breadth = int(node.get("breadth", 0))
        if breadth < min_breadth:
            continue

        node_id = node["id"]
        if node_id not in run.embeddings_by_node:
            continue

        node_embedding = run.embeddings_by_node[node_id]
        parent_id = node.get("parent_id") or node_embedding.parent_id
        if not parent_id:
            continue
        if parent_id not in run.embeddings_by_node:
            continue

        parent_embedding = run.embeddings_by_node[parent_id]
        parent_node = node_by_id.get(parent_id)

        row: dict[str, Any] = {
            "run_key": run.run_key,
            "model_id": run.model_id,
            "instruction_variant": run.instruction_variant,
            "prefix_length": run.prefix_length,
            "seed": run.seed,
            "country_id": run.country_id,
            "region_id": run.region_id,
            "node_id": node_id,
            "parent_id": parent_id,
            "depth": int(node.get("depth", 0)),
            "breadth": breadth,
            "incoming_token": _incoming_token(node, parent_node),
            "top1_prob": float(np.exp(node_embedding.top_k_logprobs[0])),
            "topk_entropy": topk_entropy(node_embedding.top_k_logprobs),
        }

        for layer in run.layers:
            child_hidden = node_embedding.hidden_by_layer[layer]
            parent_hidden = parent_embedding.hidden_by_layer[layer]
            row[f"cos_dist_l{layer}"] = cosine_distance(child_hidden, parent_hidden)

        rows.append(row)
    return rows


def build_features_dataframe(runs: list[RunRecord], min_breadth: int = 1) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    for run in runs:
        all_rows.extend(build_node_rows(run, min_breadth=min_breadth))
    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


def layer_columns(df: pd.DataFrame) -> list[str]:
    return sorted(column for column in df.columns if column.startswith("cos_dist_l"))
