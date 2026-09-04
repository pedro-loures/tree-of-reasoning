"""Build the HTML results viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _copy_template(output_dir: Path, template_path: Path, output_name: str) -> Path:
    out_path = output_dir / output_name
    out_path.write_text(template_path.read_text())
    return out_path


def build_viewer(
    output_dir: Path,
    template_path: Path,
    metrics_payload: dict[str, Any] | None = None,
) -> Path:
    out_path = _copy_template(output_dir, template_path, "results_viewer.html")

    if metrics_payload is not None:
        import json

        metrics_path = output_dir / "canvas_metrics.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    return out_path


def build_graph_viewer(
    output_dir: Path,
    template_path: Path,
) -> Path:
    return _copy_template(output_dir, template_path, "tree_graph.html")
