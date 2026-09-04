"""Load experiment JSONL and embedding npz files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from src.regions import region_for_country


def instruction_variant(row: dict[str, Any]) -> str:
    instruction = row.get("instruction", "")
    if instruction.strip().lower().startswith("ignore the previous text"):
        return "legacy"
    return "plain"


@dataclass
class NodeEmbedding:
    node_id: str
    parent_id: str
    hidden_by_layer: dict[int, np.ndarray]
    top_k_logprobs: np.ndarray


@dataclass
class RunRecord:
    run_key: str
    model_id: str
    instruction_variant: str
    prefix_length: int
    seed: int
    tree_nodes: list[dict[str, Any]]
    layers: list[int]
    embeddings_by_node: dict[str, NodeEmbedding]
    country_id: str | None = None
    region_id: str | None = None

    @property
    def node_by_id(self) -> dict[str, dict[str, Any]]:
        return {node["id"]: node for node in self.tree_nodes}


def _run_key(
    model_id: str,
    variant: str,
    prefix_length: int,
    seed: int,
    country_id: str | None = None,
) -> str:
    if country_id:
        return f"{model_id}:{country_id}:{variant}:{prefix_length}:{seed}"
    return f"{model_id}:{variant}:{prefix_length}:{seed}"


def load_embeddings_npz(npz_path: Path, layers: list[int] | None = None) -> tuple[list[int], dict[str, NodeEmbedding]]:
    with np.load(npz_path, allow_pickle=True) as data:
        node_ids = [str(node_id) for node_id in data["node_ids"]]
        parent_ids = [str(parent_id) for parent_id in data["parent_ids"]]
        stored_layers = [int(layer) for layer in data["layers"]]
        hidden_states = data["hidden_states"]
        top_k_logprobs = data["top_k_logprobs"]

    active_layers = stored_layers if layers is None else [layer for layer in layers if layer in stored_layers]
    layer_indices = [stored_layers.index(layer) for layer in active_layers]

    embeddings: dict[str, NodeEmbedding] = {}
    for row_idx, node_id in enumerate(node_ids):
        hidden_by_layer = {
            active_layers[col]: hidden_states[row_idx, layer_indices[col]]
            for col in range(len(active_layers))
        }
        embeddings[node_id] = NodeEmbedding(
            node_id=node_id,
            parent_id=parent_ids[row_idx],
            hidden_by_layer=hidden_by_layer,
            top_k_logprobs=top_k_logprobs[row_idx],
        )
    return active_layers, embeddings


def iter_runs(
    results_dir: Path,
    layers: list[int] | None = None,
) -> Iterator[tuple[RunRecord | None, str | None]]:
    """Yield (RunRecord, None) for successful loads or (None, skip_reason) for skipped rows."""
    embeddings_dir = results_dir / "embeddings"
    for path in sorted(results_dir.glob("*.jsonl")):
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                model_id = row["model_id"]
                variant = instruction_variant(row)
                prefix_length = int(row["prefix_length"])
                seed = int(row["seed"])
                country_id = row.get("country_id")
                run_key = _run_key(model_id, variant, prefix_length, seed, country_id=country_id)

                if "embeddings" not in row:
                    yield None, f"{run_key} line {line_number}: missing embeddings"
                    continue

                manifest = row["embeddings"]
                npz_path = embeddings_dir / manifest["path"]
                if not npz_path.exists():
                    yield None, f"{run_key}: embeddings file not found ({npz_path})"
                    continue

                active_layers, embeddings_by_node = load_embeddings_npz(npz_path, layers=layers)
                yield RunRecord(
                    run_key=run_key,
                    model_id=model_id,
                    instruction_variant=variant,
                    prefix_length=prefix_length,
                    seed=seed,
                    tree_nodes=row["tree"]["nodes"],
                    layers=active_layers,
                    embeddings_by_node=embeddings_by_node,
                    country_id=country_id,
                    region_id=region_for_country(country_id) if country_id else None,
                ), None


def load_all_runs(results_dir: Path, layers: list[int] | None = None) -> tuple[list[RunRecord], list[str]]:
    runs: list[RunRecord] = []
    skipped: list[str] = []
    for record, skip_reason in iter_runs(results_dir, layers=layers):
        if record is not None:
            runs.append(record)
        elif skip_reason is not None:
            skipped.append(skip_reason)
    return runs, skipped
