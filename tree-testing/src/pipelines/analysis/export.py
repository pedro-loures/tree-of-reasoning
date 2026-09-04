"""Export canvas JSON artifacts and tree text files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from src.pipelines.analysis.io import RunRecord
from src.pipelines.analysis.metrics import METRIC_KEYS, METRIC_LABELS, aggregate, run_metrics
from src.pipelines.analysis.tree_parser import build_tree_text, compact_tree


@dataclass
class ExportResult:
    datasets: list[dict[str, str]]
    models: list[str]
    instruction_variants: list[str]
    prefix_lengths: list[int]
    seeds: list[int]
    runs: list[dict[str, Any]]
    trees: dict[str, list[dict]]
    base_payload: dict[str, Any]


def export_runs(
    records: list[RunRecord],
    output_dir: Path,
    source: str,
    dataset_labels: dict[str, str] | None = None,
) -> ExportResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    trees_dir = output_dir / "trees"
    trees_dir.mkdir(parents=True, exist_ok=True)
    for old_file in trees_dir.glob("*.txt"):
        old_file.unlink()

    datasets: list[dict[str, str]] = []
    models: list[str] = []
    instruction_variants: list[str] = []
    prefix_lengths: set[int] = set()
    seeds: set[int] = set()
    all_runs: list[dict[str, Any]] = []
    trees: dict[str, list[dict]] = {}

    labels = dataset_labels or {}

    for record in records:
        if record.dataset_id not in {item["id"] for item in datasets}:
            datasets.append(
                {
                    "id": record.dataset_id,
                    "label": labels.get(record.dataset_id, record.dataset_id),
                }
            )
        if record.model_id not in models:
            models.append(record.model_id)
        if record.instruction_variant not in instruction_variants:
            instruction_variants.append(record.instruction_variant)
        prefix_lengths.add(record.prefix_length)
        seeds.add(record.seed)

        root_prefix = str(record.raw.get("root_prefix", ""))
        compact = compact_tree(record.tree_nodes, root_prefix=root_prefix)
        trees[record.tree_key] = compact
        tree_text = build_tree_text(compact)
        tree_path = trees_dir / f"{record.tree_key.replace(':', '_')}.txt"
        tree_path.write_text(tree_text + "\n")

        all_runs.append(
            run_metrics(
                record.raw,
                record.dataset_id,
                record.model_id,
                record.prefix_length,
                record.seed,
                record.tree_key,
            )
        )

    base = {
        "source": source,
        "generated_at": date.today().isoformat(),
        "datasets": datasets,
        "prefix_lengths": sorted(prefix_lengths),
        "models": models,
        "metrics": [
            {"key": key, "label": METRIC_LABELS[key][0], "unit": METRIC_LABELS[key][1]}
            for key in METRIC_KEYS
        ],
        "runs": all_runs,
        "aggregates": aggregate(all_runs),
    }

    full_payload = {**base, "trees": trees}
    metrics_payload = base
    trees_payload = {
        "source": base["source"],
        "generated_at": base["generated_at"],
        "root_prefix": str(records[0].raw.get("root_prefix", "")) if records else "",
        "datasets": datasets,
        "models": models,
        "prefix_lengths": sorted(prefix_lengths),
        "runs": [
            {
                "dataset_id": run["dataset_id"],
                "tree_key": run["tree_key"],
                "model_id": run["model_id"],
                "prefix_length": run["prefix_length"],
                "seed": run["seed"],
                "total_nodes": run["total_nodes"],
                "mass_above_tau": run["mass_above_tau"],
                "top_k_completions": run.get("top_k_completions", []),
            }
            for run in all_runs
        ],
        "trees": trees,
    }

    (output_dir / "canvas_data.json").write_text(json.dumps(full_payload, separators=(",", ":")))
    (output_dir / "canvas_metrics.json").write_text(json.dumps(metrics_payload, separators=(",", ":")))
    (output_dir / "canvas_trees.json").write_text(json.dumps(trees_payload, separators=(",", ":")))

    return ExportResult(
        datasets=datasets,
        models=models,
        instruction_variants=instruction_variants,
        prefix_lengths=sorted(prefix_lengths),
        seeds=sorted(seeds),
        runs=all_runs,
        trees=trees,
        base_payload=metrics_payload,
    )
