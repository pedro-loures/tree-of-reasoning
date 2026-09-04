#!/usr/bin/env python3
"""Run on-demand activation patching for a tree node (phase 3 mech-interp)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.interp.patching import patch_from_source_node  # noqa: E402
from src.models.common import load_experiment_config  # noqa: E402
from src.models.hf_runner import HfRunner  # noqa: E402


def _load_manifest(embeddings_dir: Path, run_key: str) -> dict:
    manifest_path = embeddings_dir / f"{run_key}.json"
    return json.loads(manifest_path.read_text())


def _node_hidden(embeddings_dir: Path, run_key: str, node_id: str, layer_index: int) -> torch.Tensor:
    npz_path = embeddings_dir / f"{run_key}.npz"
    with np.load(npz_path, allow_pickle=True) as data:
        node_ids = list(data["node_ids"])
        layers = list(data["layers"])
        hidden_states = data["hidden_states"]
    row = node_ids.index(node_id)
    layer_row = layers.index(layer_index)
    return torch.from_numpy(hidden_states[row, layer_row].astype(np.float32))


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch a target prefix with a source node hidden state")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--model", required=True, help="Model id from experiment config")
    parser.add_argument("--embeddings-dir", type=Path, default=ROOT / "embeddings")
    parser.add_argument("--run-key", required=True, help="Embedding run key (see embeddings/*.json)")
    parser.add_argument("--source-node", required=True, help="Node id to copy hidden state from")
    parser.add_argument("--target-prefix", required=True, help="Prefix text to patch during forward pass")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to patch")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    _, models, _ = load_experiment_config(args.config)
    model_spec = next(model for model in models if model.id == args.model)

    source_hidden = _node_hidden(args.embeddings_dir, args.run_key, args.source_node, args.layer)

    hf_runner = HfRunner(model_spec)
    hf_runner.load()
    try:
        result = patch_from_source_node(
            hf_runner,
            target_prefix=args.target_prefix,
            source_hidden=source_hidden,
            layer_index=args.layer,
        )
    finally:
        hf_runner.unload()

    payload = result.to_dict()
    payload["source_node"] = args.source_node
    payload["run_key"] = args.run_key
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
