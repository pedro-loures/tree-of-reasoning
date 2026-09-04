#!/usr/bin/env python3
"""Build a tau-tree and export an interactive D3 graph viewer."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.lorem_sampler import build_prompt, lorem_prefix  # noqa: E402
from src.models.common import load_experiment_config  # noqa: E402
from src.models.hf_runner import HfRunner  # noqa: E402
from src.tree.metrics import compute_tree_metrics  # noqa: E402
from src.tree.tau_builder import build_tau_tree  # noqa: E402

TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "tree-testing"
    / "templates"
    / "tree_graph.template.html"
)
LEGACY_INSTRUCTION = "ignore the previous text, what is the capital of brazil"


def build_incoming_token_map(nodes: list[dict]) -> dict[str, str]:
    incoming: dict[str, str] = {}
    for node in nodes:
        for child_id, token in zip(node.get("child_ids", []), node.get("child_tokens", [])):
            incoming[child_id] = token
    return incoming


def compact_tree(nodes: list[dict], root_prefix: str | None = None) -> list[dict]:
    parent_token = build_incoming_token_map(nodes)
    if root_prefix is None:
        root_node = next((n for n in nodes if n.get("id") == "root"), None)
        root_prefix = root_node.get("prefix_text", "") if root_node else ""

    compact: list[dict] = []
    for node in nodes:
        entry: dict = {
            "id": node["id"],
            "d": node["depth"],
            "p": round(node["path_prob"], 6),
            "c": node["child_ids"],
        }
        if node["id"] != "root":
            entry["tok"] = parent_token.get(node["id"], "?")
        if node.get("child_tokens"):
            entry["ct"] = node["child_tokens"]
        prefix_text = node.get("prefix_text")
        if prefix_text and node["id"] != "root":
            suffix = prefix_text[len(root_prefix):] if prefix_text.startswith(root_prefix) else prefix_text
            if suffix:
                entry["suffix"] = suffix
        compact.append(entry)
    return compact


def build_tree_text(nodes: list[dict]) -> str:
    by_id = {node["id"]: node for node in nodes}

    def display_token(token: str | None) -> str:
        if not token:
            return "?"
        if token == "\n":
            return "\\n"
        if token == "\t":
            return "\\t"
        if token == " ":
            return "·"
        return token

    lines: list[str] = []

    def walk(node_id: str, prefix: str, is_last: bool, depth: int) -> None:
        node = by_id[node_id]
        connector = "" if depth == 0 else ("└─ " if is_last else "├─ ")
        label = "⟨prompt⟩" if node_id == "root" else display_token(node.get("tok"))
        prob = node["p"]
        prob_text = f"{prob:.4f}" if prob >= 0.0001 else f"{prob:.2e}"
        lines.append(f"{prefix}{connector}{label} ({prob_text})")
        child_prefix = "" if depth == 0 else prefix + ("   " if is_last else "│  ")
        for index, child_id in enumerate(node["c"]):
            walk(child_id, child_prefix, index == len(node["c"]) - 1, depth + 1)

    walk("root", "", True, 0)
    return "\n".join(lines)


def export_viewer(
    output_dir: Path,
    record: dict,
    tree_key: str,
    dataset_id: str = "tau001",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes = record["tree"]["nodes"]
    root_prefix = record.get("root_prefix", "")
    compact = compact_tree(nodes, root_prefix=root_prefix)
    tree_metrics = record["tree_metrics"]

    run_meta = {
        "dataset_id": dataset_id,
        "tree_key": tree_key,
        "model_id": record["model_id"],
        "prefix_length": record["prefix_length"],
        "seed": record["seed"],
        "total_nodes": tree_metrics["total_nodes"],
        "mass_above_tau": tree_metrics["mass_above_tau"],
        "top_k_completions": record.get("top_k_metrics", {}).get("top_k_completions", []),
    }

    trees_payload = {
        "source": f"tau={record['tree']['tau']} tree build",
        "generated_at": date.today().isoformat(),
        "root_prefix": root_prefix,
        "datasets": [{"id": dataset_id, "label": f"τ={record['tree']['tau']}"}],
        "models": [record["model_id"]],
        "prefix_lengths": [record["prefix_length"]],
        "runs": [run_meta],
        "trees": {tree_key: compact},
    }

    (output_dir / "canvas_trees.json").write_text(json.dumps(trees_payload, separators=(",", ":")))

    trees_dir = output_dir / "trees"
    trees_dir.mkdir(exist_ok=True)
    (trees_dir / f"{tree_key.replace(':', '_')}.txt").write_text(build_tree_text(compact) + "\n")

    html = TEMPLATE.read_text()
    html = html.replace("<title>τ-Tree Graph Viewer</title>", "<title>τ=0.001 Tree · DeepSeek pl=1000</title>")
    html = html.replace(
        "<h1>τ-Pruned Reasoning Tree</h1>",
        f"<h1>τ={record['tree']['tau']} Reasoning Tree</h1>",
    )
    html = html.replace(
        '<option value="0.01">0.01 (τ)</option>',
        '<option value="0.01">0.01</option>',
    )
    html = html.replace(
        '<option value="0.001">0.001</option>',
        '<option value="0.001" selected>0.001 (τ)</option>',
    )
    viewer_path = output_dir / "tree_graph.html"
    viewer_path.write_text(html)

    # Standalone copy with embedded JSON (works without HTTP server / remote localhost)
    embedded_html = html.replace(
        'fetch("canvas_trees.json")\n  .then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); })\n  .then(data => { DATA = data; init(); })\n  .catch(err => {\n    document.getElementById("source").innerHTML =\n      `<span class="error">Failed to load canvas_trees.json: ${err.message}. Serve output/ via HTTP.</span>`;\n  });',
        f"const EMBEDDED_DATA = {json.dumps(trees_payload, separators=(',', ':'))};\nDATA = EMBEDDED_DATA;\ninit();",
    )
    standalone_path = output_dir / "tree_graph_standalone.html"
    standalone_path.write_text(embedded_html)

    d3_src = output_dir / "d3.min.js"
    if not d3_src.exists():
        import urllib.request
        urllib.request.urlretrieve(
            "https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js",
            d3_src,
        )
    self_contained = embedded_html.replace(
        '<script src="d3.min.js"></script>',
        f"<script>{d3_src.read_text()}</script>",
    )
    (output_dir / "tree_graph_self_contained.html").write_text(self_contained)

    return output_dir / "tree_graph_self_contained.html"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build tau-tree and D3 graph viewer")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--model", default="deepseek-r1-7b")
    parser.add_argument("--prefix-length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.001)
    parser.add_argument(
        "--instruction",
        default=LEGACY_INSTRUCTION,
        help="User instruction appended after Lorem prefix",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "tau001_viz",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse existing JSONL in output dir",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    jsonl_path = output_dir / f"{args.model}.jsonl"
    tree_key = f"tau001:{args.model}:legacy:{args.prefix_length}:{args.seed}"

    if args.skip_build and jsonl_path.exists():
        record = json.loads(jsonl_path.read_text().strip())
    else:
        _, models, _ = load_experiment_config(args.config)
        model_spec = next(m for m in models if m.id == args.model)

        prompt = build_prompt(lorem_prefix(args.prefix_length), args.instruction)
        hf = HfRunner(model_spec)
        hf.load()
        root_prefix, _ = hf.find_reasoning_root_prefix(prompt, probe_max_tokens=8)
        build_result = build_tau_tree(
            hf,
            root_prefix=root_prefix,
            tau=args.tau,
            max_depth=512,
            breadth_warning_threshold=20,
            numerical_floor=1e-12,
            batch_size=model_spec.hf_batch_size or 4,
            capture_hidden_states=False,
            top_k_logprobs=20,
        )
        hf.unload()

        tree = build_result.tree
        tree_metrics = compute_tree_metrics(tree)
        record = {
            "model_id": args.model,
            "instruction": args.instruction,
            "prefix_length": args.prefix_length,
            "seed": args.seed,
            "prompt": prompt,
            "root_prefix": root_prefix,
            "tree": tree.to_dict(),
            "tree_metrics": tree_metrics,
            "top_k_metrics": {"top_k_completions": []},
            "trace_metrics": {},
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text(json.dumps(record) + "\n")

    viewer_path = export_viewer(output_dir, record, tree_key)
    print(f"Nodes: {record['tree_metrics']['total_nodes']}")
    print(f"Leaves: {record['tree_metrics']['leaf_count']}")
    print(f"JSONL: {jsonl_path}")
    print(f"Viewer: {viewer_path}")
    print(f"Serve: python -m http.server 8765 --directory {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
