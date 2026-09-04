#!/usr/bin/env bash
# Rebuild capitals dashboard; optionally wait for bad-nodes pipeline to finish.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BAD_NODES="../tree-of-reasoning/results/capitals_bad_nodes/deepseek-r1-7b.jsonl"
TARGET=210

if [[ "${1:-}" == "--wait" ]]; then
  echo "Waiting for bad-nodes pipeline to reach ${TARGET} trees..."
  while true; do
    count=$(wc -l < "$BAD_NODES" 2>/dev/null || echo 0)
    echo "  $(date -Iseconds): ${count}/${TARGET}"
    if [[ "$count" -ge "$TARGET" ]]; then
      break
    fi
    sleep 120
  done
fi

.venv/bin/python scripts/build_capitals_viewer.py
echo "Open: http://localhost:8768/output/capitals/capitals_dashboard.html"
echo "Serve: cd $ROOT && python3 -m http.server 8768"
