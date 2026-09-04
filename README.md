# tree_of_reasoning

Research monorepo for studying **how language models branch during chain-of-thought reasoning**.

Models are probed token-by-token; branches above a probability threshold τ form a **τ-tree**. We then ask which internal nodes are *exclusively bad* (every completion below them is wrong) and how tree shape changes with prompt design.

| Package | Role |
|---------|------|
| [tree-of-reasoning](tree-of-reasoning/) | Run GPU experiments, build τ-trees, interactive dashboard backend |
| [tree-testing](tree-testing/) | Analysis, dashboards, GitHub Pages site |
| [tree-shift-analysis](tree-shift-analysis/) | Hidden-state shift vs branching correlations |

## Experiments (summary)

| Experiment | Question | Published dashboard tab |
|------------|----------|-------------------------|
| **Capitals** | Across 30 countries, where does the model branch—and which branches always lead to a wrong capital? | Capitals |
| **Elections** | For 2026 Brazilian president prompts (left / right / neutral), does the model name real candidates, and on which paths? | Elections |
| **Interactive** | Ad-hoc prompts: explore a freshly generated τ-tree with good/bad node labels | Interactive (view-only demo) |
| **Brazil prefix sweep** | How does a Lorem ipsum prefix length change Brazil capital reasoning? | — (legacy local viewers) |
| **Shift analysis** | Does representational shift between parent and child predict branching? | — |

Details and run commands: see each package README.

## GitHub Pages

Public site with **Capitals**, **Elections**, and **Interactive** tabs. Tree data lives on the `gh-pages` branch only—not on `main`.

```bash
./scripts/deploy_pages.sh --build-only   # preview
./scripts/deploy_pages.sh                # push to gh-pages
```

Enable **Settings → Pages → branch `gh-pages` / root**.

## Branch policy

| Branch | Contents |
|--------|----------|
| `main` | Source, tests, site templates — no `results/`, `output/`, or `docs/` |
| `gh-pages` | Built static site + tree JSON shards |
