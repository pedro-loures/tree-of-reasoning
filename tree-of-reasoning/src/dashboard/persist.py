"""Persist interactive dashboard tree runs to disk."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SESSION_VERSION = 1
DEFAULT_SAVE_DIR = Path(__file__).resolve().parents[2] / "results" / "dashboard_sessions"

_RUN_FIELDS = (
    "tree_key",
    "prompt",
    "model_id",
    "tau",
    "summary",
    "candidate_nodes",
    "tree_summary",
    "tree_nodes",
    "node_status",
    "leaf_completions",
    "node_stats",
)


def default_save_dir() -> Path:
    return DEFAULT_SAVE_DIR


def safe_filename(tree_key: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", tree_key)
    return cleaned[:180] or "tree"


def session_file_path(save_dir: Path, tree_key: str) -> Path:
    return save_dir / f"{safe_filename(tree_key)}.json"


def wrap_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": SESSION_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "run": run,
    }


def unwrap_run(payload: dict[str, Any]) -> dict[str, Any]:
    if "run" in payload:
        run = payload["run"]
    else:
        run = payload
    missing = [field for field in _RUN_FIELDS if field not in run]
    if missing:
        raise ValueError(f"Saved tree is missing fields: {', '.join(missing)}")
    return run


def save_run(run: dict[str, Any], save_dir: Path | None = None) -> Path:
    save_dir = save_dir or default_save_dir()
    save_dir.mkdir(parents=True, exist_ok=True)
    path = session_file_path(save_dir, run["tree_key"])
    path.write_text(json.dumps(wrap_run(run), indent=2), encoding="utf-8")
    return path


def load_run_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return unwrap_run(payload)


def list_saved_runs(save_dir: Path | None = None) -> list[dict[str, Any]]:
    save_dir = save_dir or default_save_dir()
    if not save_dir.exists():
        return []

    entries: list[dict[str, Any]] = []
    for path in sorted(save_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            run = unwrap_run(payload)
        except (json.JSONDecodeError, ValueError, KeyError):
            continue
        summary = run.get("tree_summary") or {}
        entries.append(
            {
                "filename": path.name,
                "path": str(path),
                "tree_key": run["tree_key"],
                "prompt_preview": summary.get("prompt_preview") or run.get("prompt", "")[:72],
                "model_id": run.get("model_id"),
                "tau": run.get("tau"),
                "saved_at": payload.get("saved_at"),
                "total_nodes": summary.get("total_nodes"),
                "leaf_count": summary.get("leaf_count"),
            }
        )
    return entries


def runs_from_session_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild full run dicts from a merged dashboard payload."""
    runs: list[dict[str, Any]] = []
    for run_meta in payload.get("runs") or []:
        tree_key = run_meta["tree_key"]
        summary = next(
            (row for row in payload.get("tree_summaries") or [] if row.get("tree_key") == tree_key),
            None,
        )
        if summary is None:
            continue
        runs.append(
            {
                "tree_key": tree_key,
                "prompt": run_meta["prompt"],
                "model_id": run_meta["model_id"],
                "tau": run_meta["tau"],
                "expected_answers": run_meta.get("expected_answers"),
                "answer_mode": run_meta.get("answer_mode"),
                "summary": run_meta["summary"],
                "candidate_nodes": run_meta["candidate_nodes"],
                "tree_summary": summary,
                "tree_nodes": payload["trees"][tree_key],
                "node_status": payload["node_status"][tree_key],
                "leaf_completions": payload["leaf_completions"][tree_key],
                "node_stats": payload["node_stats"][tree_key],
                "embeddings": run_meta.get("embeddings"),
            }
        )
    return runs
