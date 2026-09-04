#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOCS="${DOCS:-$ROOT/docs}"
PORT="${PORT:-8080}"
if [[ ! -d "$DOCS" ]]; then
  echo "docs/ not found. Run: ./scripts/deploy_pages.sh --build-only" >&2
  exit 1
fi
cd "$DOCS"
echo "Serving $DOCS at http://localhost:$PORT/"
exec python3 -m http.server "$PORT" --bind 0.0.0.0
