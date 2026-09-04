#!/usr/bin/env python3
"""Serve the interactive tree dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

from src.dashboard.server import create_app  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve interactive tree dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiment.yaml",
        help="Experiment config with model definitions",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Override dashboard HTML template path",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=ROOT / "results" / "dashboard_sessions",
        help="Directory for saved dashboard trees",
    )
    args = parser.parse_args()

    app = create_app(config_path=args.config, template_path=args.template, save_dir=args.save_dir)
    print(f"Dashboard: http://{args.host}:{args.port}/")
    print("Enter a prompt, pick model + τ, then Generate tree.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
