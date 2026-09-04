# tree-testing

Analysis and dashboards for [tree-of-reasoning](../tree-of-reasoning) experiment outputs.

## What each dashboard shows

| Dashboard | Experiment | What you explore |
|-----------|------------|------------------|
| **Capitals** | 30-country capital QA | Per-country τ-trees, exclusively-bad nodes, P(good) vs P(bad) mass by branch |
| **Elections** | Brazil 2026 president | Three prompt framings, candidate mentions per leaf, bad-node map |
| **Interactive** | Ad-hoc prompts | Saved τ-trees from the local dashboard (view-only on Pages) |
| **Legacy viewers** | Brazil prefix sweep | Word-edge stats, τ-tree graphs, bad-nodes PNG/HTML in `output/` |

## GitHub Pages (unified site)

Single static site with three tabs. Source in `site/`; built output in `docs/` (gitignored on `main`).

```bash
# From repo root
./scripts/deploy_pages.sh --build-only
./tree-testing/scripts/preview_pages_site.sh   # http://localhost:8080/

./scripts/deploy_pages.sh   # push to gh-pages only
```

Or build manually:

```bash
.venv/bin/python scripts/build_pages_site.py --output-dir ../docs
```

Trees are sharded per run (`data/<experiment>/trees/*.json`) for lazy loading. Interactive tab includes a view-only mock of the generate UI.

## Local dashboards (legacy)

**Capitals** — summary tables + tree explorer:

```bash
.venv/bin/python scripts/build_capitals_viewer.py
# output/capitals/capitals_dashboard.html
```

**Elections** — president mention categories + tree explorer:

```bash
.venv/bin/python scripts/build_president_dashboard.py
# output/president/president_dashboard.html
```

**Brazil prefix sweep** — metrics and τ-tree graph:

```bash
.venv/bin/python scripts/analyze_results.py
./viewer/serve.sh   # http://localhost:8766/viewer/index.html
```

## Other analysis

```bash
.venv/bin/python scripts/summarize_bad_nodes.py \
  --results-dir ../tree-of-reasoning/results/capitals_bad_nodes

.venv/bin/python scripts/build_bad_nodes_viewer.py
.venv/bin/python scripts/analyze_token_path_motifs.py
```

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Tests

```bash
.venv/bin/pytest tests/ -v
```

## Key outputs

| Path | Description |
|------|-------------|
| `output/capitals/` | Capitals dashboard JSON + HTML |
| `output/president/` | Elections dashboard JSON + HTML |
| `output/word_edges.json` | Token/depth edge statistics (Brazil sweep) |
| `output/plots/` | Tree PNGs and comparison grids |
| `site/` | Unified Pages site source |
