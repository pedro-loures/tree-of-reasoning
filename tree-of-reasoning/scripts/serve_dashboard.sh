#!/usr/bin/env bash
# Serve the interactive tree dashboard (prompt → τ-tree → good/bad nodes).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python3 scripts/serve_dashboard.py "$@"
