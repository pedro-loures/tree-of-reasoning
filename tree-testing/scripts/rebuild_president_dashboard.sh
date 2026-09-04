#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MECH="${ROOT}/../tree-of-reasoning/results/president_mech_interp"
BAD="${ROOT}/../tree-of-reasoning/results/president_bad_nodes"
OUTPUT="${ROOT}/output/president"

cd "${ROOT}"
.venv/bin/python ../tree-of-reasoning/scripts/run_bad_nodes.py \
  --config ../tree-of-reasoning/configs/president_bad_nodes.yaml
.venv/bin/python scripts/build_president_dashboard.py \
  --mech-dir "${MECH}" \
  --bad-nodes-dir "${BAD}" \
  --output-dir "${OUTPUT}"
echo "Dashboard: ${OUTPUT}/president_dashboard.html"
