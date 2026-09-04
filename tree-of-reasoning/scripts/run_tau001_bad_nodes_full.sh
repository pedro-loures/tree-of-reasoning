#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_BUILD="$ROOT/results/tau001_trees_build.log"
LOG_BAD="$ROOT/results/bad_nodes_tau001_full.log"

mkdir -p "$ROOT/results/tau001_mech_interp" "$ROOT/results/bad_nodes_tau001"

# Seed output with the already-built DeepSeek pl=1000 tree if missing.
python3 <<'PY'
import json
from pathlib import Path

root = Path("/scratch2/pedro.loures/tree_of_reasoning/tree-of-reasoning")
src = root / "results/tau001_viz/deepseek-r1-7b.jsonl"
dst = root / "results/tau001_mech_interp/deepseek-r1-7b.jsonl"
if not src.exists():
    raise SystemExit(0)

record = json.loads(src.read_text().strip())
key = (record["instruction"], int(record["prefix_length"]), int(record["seed"]))
existing = set()
if dst.exists():
    for line in dst.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        existing.add((row["instruction"], int(row["prefix_length"]), int(row["seed"])))
if key not in existing:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(f"Seeded {dst} with prefix_length={record['prefix_length']}", flush=True)
PY

echo "=== Phase 1: build tau=0.001 trees ===" | tee -a "$LOG_BUILD"
.venv/bin/python scripts/build_tau001_trees.py 2>&1 | tee -a "$LOG_BUILD"

echo "=== Phase 2: bad-nodes leaf completions ===" | tee -a "$LOG_BAD"
.venv/bin/python scripts/run_bad_nodes.py --config configs/bad_nodes_tau001.yaml 2>&1 | tee -a "$LOG_BAD"

echo "=== Done ===" | tee -a "$LOG_BAD"
