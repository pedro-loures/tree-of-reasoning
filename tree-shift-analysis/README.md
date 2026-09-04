# tree-shift-analysis

Secondary analysis on [tree-of-reasoning](../tree-of-reasoning) mech-interp runs that store hidden states.

## Question

When the model branches (high **breadth** at an internal node), does the child’s hidden state diverge sharply from its parent? We measure **cosine distance** between consecutive node representations at four layer snapshots and correlate it with branching.

## Prerequisite

Runs with `capture_hidden_states: true` in `results/mech_interp/` (Brazil prefix sweep config). Re-run analysis incrementally as more conditions finish.

## Setup & run

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/run_analysis.py
.venv/bin/python scripts/run_analysis.py --model deepseek-r1-7b --no-plots
```

## Outputs

| Path | Description |
|------|-------------|
| `output/node_features.parquet` | One row per internal node |
| `output/correlations.json` | Spearman ρ overall and by stratum |
| `output/plots/` | cos_dist vs breadth scatter panels |

## Metrics

- **cos_dist_l{L}** — `1 − cosine_similarity(h_node, h_parent)` at layer L
- **breadth** — child edges above τ at this node
- Population: internal nodes only (`breadth ≥ 1`)

Positive ρ suggests larger representational shifts coincide with wider branching; ρ ≈ 0 means breadth is not explained by consecutive-step shift alone.

## Tests

```bash
.venv/bin/pytest tests/ -v
```

Not included in the public GitHub Pages dashboard.
