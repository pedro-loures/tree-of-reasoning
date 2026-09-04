#!/usr/bin/env python3
"""Backfill node embeddings for saved interactive dashboard trees."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dashboard.persist import default_save_dir, load_run_file, save_run  # noqa: E402
from src.dashboard.service import DashboardConfig, DashboardService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill embeddings for saved dashboard trees")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiment.yaml",
        help="Experiment config with model definitions",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Dashboard sessions directory",
    )
    parser.add_argument(
        "--session",
        action="append",
        dest="sessions",
        help="Specific session JSON filename (repeatable). Default: all missing embeddings.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if embeddings exist")
    args = parser.parse_args()

    save_dir = args.save_dir or default_save_dir()
    if args.sessions:
        session_paths = [save_dir / name for name in args.sessions]
    else:
        session_paths = sorted(save_dir.glob("*.json"))

    if not session_paths:
        print(f"No session files found in {save_dir}")
        return 1

    service = DashboardService(
        DashboardConfig(config_path=args.config, repo_root=ROOT),
    )

    updated = 0
    skipped = 0
    for path in session_paths:
        if not path.exists():
            print(f"Skip missing file: {path.name}")
            skipped += 1
            continue

        run = load_run_file(path)
        if run.get("embeddings") and not args.force:
            print(f"Skip {path.name}: embeddings already present")
            skipped += 1
            continue

        print(f"Backfilling {path.name}")
        print(f"  prompt: {run['prompt']}")
        manifest = service._save_embeddings_for_prompt(
            prompt=run["prompt"],
            model_id=run["model_id"],
            tau=float(run["tau"]),
            tree_key=run["tree_key"],
        )
        if manifest is None:
            print(f"  warning: no embeddings captured for {path.name}")
            skipped += 1
            continue

        run["embeddings"] = manifest
        save_run(run, save_dir)
        print(f"  saved embeddings: {manifest['path']} ({len(manifest['node_ids'])} nodes)")
        updated += 1

    print(f"Done. updated={updated} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
