"""Export sharded dashboard payloads for GitHub Pages."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from src.pipelines.analysis.capitals_viewer import export_capitals_dashboard
from src.pipelines.analysis.president_viewer import export_president_dashboard
from src.pipelines.analysis.tree_parser import compact_tree

COMPLETION_PREVIEW_CHARS = 300

# Heavy per-tree fields belong in shard JSON files, not the manifest.
_RUN_MANIFEST_DROP_KEYS = frozenset(
    {
        "candidate_nodes",
        "embeddings",
        "exclusively_bad_nodes",
        "tree_nodes",
        "node_status",
        "leaf_completions",
        "node_stats",
        "node_expansions",
        "tree_summary",
    }
)


def compact_run_for_manifest(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key not in _RUN_MANIFEST_DROP_KEYS}


def compact_runs_for_manifest(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [compact_run_for_manifest(run) for run in runs]


def tree_key_to_slug(tree_key: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", tree_key)
    return cleaned[:180] or "tree"


def compact_leaf_for_publish(leaf: dict[str, Any], *, preview_chars: int = COMPLETION_PREVIEW_CHARS) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "leaf_id",
        "path_prob",
        "answer_text",
        "answer_correct",
        "reasoning_complete",
        "mention_category",
        "mentions",
        "mentions_lorem",
        "answer_matches",
    ):
        if key in leaf:
            compact[key] = leaf[key]
    completion = leaf.get("completion_text")
    if completion and preview_chars > 0:
        compact["completion_preview"] = completion[:preview_chars]
    return compact


def compact_leaf_map_for_publish(leaf_map: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {leaf_id: compact_leaf_for_publish(leaf) for leaf_id, leaf in leaf_map.items()}


def _strip_absolute_paths(source: str) -> str:
    return re.sub(r"/[^\s+]+\.(jsonl|json)", lambda m: m.group(0).rsplit("/", 1)[-1], source)


def build_tree_shard(
    tree_key: str,
    payload: dict[str, Any],
    *,
    compact_leaves: bool = True,
) -> dict[str, Any]:
    leaf_map = payload.get("leaf_completions", {}).get(tree_key, {})
    if compact_leaves and leaf_map:
        leaf_map = compact_leaf_map_for_publish(leaf_map)
    shard: dict[str, Any] = {
        "tree_key": tree_key,
        "tree_nodes": payload.get("trees", {}).get(tree_key, []),
        "node_status": payload.get("node_status", {}).get(tree_key, {}),
        "node_stats": payload.get("node_stats", {}).get(tree_key, {}),
        "leaf_completions": leaf_map,
    }
    expansions = payload.get("node_expansions", {}).get(tree_key)
    if expansions:
        shard["node_expansions"] = expansions
    candidate_probs = payload.get("candidate_mention_probs", {}).get(tree_key)
    if candidate_probs:
        shard["candidate_mention_probs"] = candidate_probs
    return shard


def split_payload_to_shards(
    payload: dict[str, Any],
    output_dir: Path,
    experiment: str,
    *,
    compact_leaves: bool = True,
    view_only: bool = False,
) -> dict[str, Any]:
    trees_dir = output_dir / "trees"
    trees_dir.mkdir(parents=True, exist_ok=True)

    tree_keys = list(payload.get("trees", {}).keys())
    shards: list[dict[str, str]] = []
    for tree_key in tree_keys:
        slug = tree_key_to_slug(tree_key)
        shard_path = trees_dir / f"{slug}.json"
        shard = build_tree_shard(tree_key, payload, compact_leaves=compact_leaves)
        shard_path.write_text(json.dumps(shard, separators=(",", ":")), encoding="utf-8")
        shards.append({"tree_key": tree_key, "slug": slug, "file": f"trees/{slug}.json"})

    manifest_keys = (
        "source",
        "generated_at",
        "experiment_type",
        "mech_interp_trees",
        "bad_nodes_trees",
        "expanded_bad_nodes_trees",
        "viewer_runs",
        "tree_summaries",
        "prefix_lengths",
        "instruction_variants",
        "models",
        "taus",
        "mention_categories",
        "regions",
        "presidential_candidates",
    )
    manifest: dict[str, Any] = {
        "experiment": experiment,
        "view_only": view_only,
        "shards": shards,
    }
    for key in manifest_keys:
        if key not in payload:
            continue
        manifest[key] = payload[key]
    if "source" in manifest:
        manifest["source"] = _strip_absolute_paths(str(manifest["source"]))
    if "generated_at" not in manifest:
        manifest["generated_at"] = date.today().isoformat()

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    return manifest


def _unwrap_interactive_session(payload: dict[str, Any]) -> dict[str, Any]:
    if "run" in payload:
        return payload["run"]
    return payload


def load_interactive_runs(interactive_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not interactive_dir.exists():
        return runs
    for path in sorted(interactive_dir.glob("*.json")):
        if path.name.endswith(".log"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            run = _unwrap_interactive_session(payload)
            if "tree_key" in run and "tree_nodes" in run:
                runs.append(run)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return runs


def export_interactive_payload(
    interactive_dir: Path,
    *,
    merge_runs_fn: Any | None = None,
) -> dict[str, Any]:
    runs = load_interactive_runs(interactive_dir)
    if not runs:
        return {
            "source": "interactive dashboard sessions",
            "generated_at": date.today().isoformat(),
            "experiment_type": "interactive",
            "models": [],
            "taus": [],
            "tree_summaries": [],
            "runs": [],
            "trees": {},
            "node_status": {},
            "leaf_completions": {},
            "node_stats": {},
        }
    if merge_runs_fn is None:
        raise ValueError("merge_runs_fn is required when interactive runs are present")
    return merge_runs_fn(runs)


def build_tutorial_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    runs = payload.get("runs") or []
    if not runs:
        return None
    smallest = min(
        runs,
        key=lambda run: (run.get("summary") or {}).get("total_leaves", 10**9),
    )
    tree_key = smallest["tree_key"]
    summary = next(
        (row for row in payload.get("tree_summaries", []) if row.get("tree_key") == tree_key),
        None,
    )
    return {
        "tree_key": tree_key,
        "prompt": smallest.get("prompt", ""),
        "model_id": smallest.get("model_id", ""),
        "tau": smallest.get("tau"),
        "expected_answers": smallest.get("expected_answers"),
        "answer_mode": smallest.get("answer_mode"),
        "summary": summary,
    }


def filter_payload_variant(payload: dict[str, Any], variant: str = "plain") -> dict[str, Any]:
    summaries = [s for s in payload.get("tree_summaries", []) if s.get("instruction_variant") == variant]
    keys = {s["tree_key"] for s in summaries}
    filtered = dict(payload)
    filtered["tree_summaries"] = summaries
    filtered["instruction_variants"] = [variant]
    filtered["viewer_runs"] = len(summaries)
    for key in ("trees", "node_status", "leaf_completions", "node_stats", "node_expansions", "candidate_mention_probs"):
        if key in filtered and isinstance(filtered[key], dict):
            filtered[key] = {tree_key: value for tree_key, value in filtered[key].items() if tree_key in keys}
    if "runs" in filtered:
        filtered["runs"] = [run for run in filtered["runs"] if run.get("tree_key") in keys]
    filtered["prefix_lengths"] = sorted({
        int(row["prefix_length"]) for row in summaries if row.get("prefix_length") is not None
    })
    return filtered


def export_capitals_sharded(
    mech_interp_path: Path,
    bad_nodes_path: Path,
    output_dir: Path,
    *,
    expanded_bad_nodes_path: Path | None = None,
) -> dict[str, Any]:
    temp_json = output_dir / "_full_payload.json"
    payload = export_capitals_dashboard(
        mech_interp_path,
        bad_nodes_path,
        temp_json,
        expanded_bad_nodes_path=expanded_bad_nodes_path,
    )
    if temp_json.exists():
        temp_json.unlink()
    payload = filter_payload_variant(payload, "plain")
    return split_payload_to_shards(payload, output_dir, "capitals")


def export_elections_sharded(
    mech_interp_dir: Path,
    bad_nodes_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    temp_json = output_dir / "_full_payload.json"
    payload = export_president_dashboard(mech_interp_dir, bad_nodes_dir, temp_json)
    if temp_json.exists():
        temp_json.unlink()
    return split_payload_to_shards(payload, output_dir, "elections")


def _load_merge_runs(repo_root: Path):
    tor_root = repo_root if (repo_root / "src" / "dashboard" / "payload.py").exists() else repo_root / "tree-of-reasoning"
    payload_path = tor_root / "src" / "dashboard" / "payload.py"
    if not payload_path.exists():
        raise FileNotFoundError(f"Cannot find dashboard payload module at {payload_path}")
    import importlib.util

    spec = importlib.util.spec_from_file_location("tor_dashboard_payload", payload_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {payload_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.merge_runs


def export_interactive_sharded(
    interactive_dir: Path,
    output_dir: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root or Path(__file__).resolve().parents[4]
    merge_runs = _load_merge_runs(repo_root)

    payload = export_interactive_payload(interactive_dir, merge_runs_fn=merge_runs)
    manifest = split_payload_to_shards(
        payload,
        output_dir,
        "interactive",
        compact_leaves=True,
        view_only=True,
    )
    tutorial = build_tutorial_payload(payload)
    if tutorial:
        (output_dir / "tutorial.json").write_text(
            json.dumps(tutorial, separators=(",", ":")),
            encoding="utf-8",
        )
    return manifest


def copy_site_assets(site_dir: Path, output_dir: Path) -> None:
    import shutil

    for item in site_dir.iterdir():
        dest = output_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
