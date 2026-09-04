#!/usr/bin/env bash
# Serve viewer + data from tree-testing root.
# Open via Cursor Ports → forward 8766 → http://localhost:8766/viewer/index.html
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "Serving from: $ROOT"
echo "Open: http://localhost:8766/viewer/index.html"
echo "Bad-nodes viewer: http://localhost:8766/output/bad_nodes_graph.html"
echo "President experiment: http://localhost:8766/output/president/president_dashboard.html"
exec python3 -m http.server 8766 --bind 0.0.0.0
