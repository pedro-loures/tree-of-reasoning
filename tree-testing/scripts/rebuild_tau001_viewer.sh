#!/usr/bin/env bash
# Rebuild tau001 bad-nodes viewer artifacts.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOR="$(cd "$ROOT/../tree-of-reasoning" && pwd)"
OUT="$ROOT/output/tau001_viz"
mkdir -p "$OUT"

cd "$ROOT"
.venv/bin/python scripts/build_bad_nodes_viewer.py \
  --canvas-trees "$OUT/canvas_trees.json" \
  --bad-nodes-dir "$TOR/results/bad_nodes_tau001" \
  --mech-interp-dir "$TOR/results/tau001_mech_interp" \
  --dataset-id tau001 \
  --output-dir "$OUT"

python3 <<'PY'
import json
from pathlib import Path

out = Path("/scratch2/pedro.loures/tree_of_reasoning/tree-testing/output/tau001_viz")
template = (out / "bad_nodes_graph.html").read_text()
payload = json.loads((out / "bad_nodes_canvas.json").read_text())
d3 = (out / "d3.min.js")
if not d3.exists():
    src = Path("/scratch2/pedro.loures/tree_of_reasoning/tree-of-reasoning/results/tau001_viz/d3.min.js")
    d3.write_bytes(src.read_bytes())

embedded = template.replace(
    'fetch("bad_nodes_canvas.json")\n  .then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); })\n  .then(data => { DATA = data; init(); })\n  .catch(err => {\n    document.getElementById("source").innerHTML =\n      `<span class="error">Failed to load bad_nodes_canvas.json: ${err.message}. Serve output/ via HTTP.</span>`;\n  });',
    f"const EMBEDDED_DATA = {json.dumps(payload, separators=(',', ':'))};\nDATA = EMBEDDED_DATA;\ninit();",
)
if "cdn.jsdelivr.net" in embedded:
    embedded = embedded.replace(
        '<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>',
        f"<script>{d3.read_text()}</script>",
    )
(out / "bad_nodes_graph_self_contained.html").write_text(embedded)
print("Wrote", out / "bad_nodes_graph_self_contained.html")
PY
