"""Find repeated token subsequences across root-to-leaf branches in tau trees."""

from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from src.pipelines.analysis.io import RunRecord, load_all
from src.pipelines.analysis.tree_parser import build_incoming_token_map

PILOT_COUNTRIES = frozenset({"brazil", "fiji", "india", "morocco", "new_zealand"})


@dataclass(frozen=True)
class TokenMotif:
    tokens: tuple[str, ...]
    branch_count: int


def display_token(token: str) -> str:
    if token == "\n":
        return "\\n"
    if token == "\t":
        return "\\t"
    if token == " ":
        return "·"
    return token


def format_motif_display(tokens: tuple[str, ...]) -> str:
    return "".join(display_token(token) for token in tokens)


def enumerate_leaf_paths(nodes: list[dict[str, Any]]) -> list[tuple[str, ...]]:
    """Return root-to-leaf token sequences for every leaf in the tree."""
    by_id = {node["id"]: node for node in nodes}
    incoming = build_incoming_token_map(nodes)
    paths: list[tuple[str, ...]] = []

    def walk(node_id: str = "root") -> None:
        node = by_id[node_id]
        child_ids = node.get("child_ids") or []
        if not child_ids:
            tokens: list[str] = []
            current_id: str | None = node_id
            while current_id and current_id != "root":
                tokens.append(incoming.get(current_id, "?"))
                current_id = by_id[current_id].get("parent_id")
            paths.append(tuple(reversed(tokens)))
            return
        for child_id in child_ids:
            walk(child_id)

    walk()
    return paths


def find_repeated_motifs(
    paths: list[tuple[str, ...]],
    *,
    min_length: int = 6,
    min_branches: int = 2,
) -> list[TokenMotif]:
    """Return every contiguous token substring on at least min_branches distinct paths."""
    motif_paths: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for path_index, path in enumerate(paths):
        path_len = len(path)
        for start in range(path_len):
            for end in range(start + min_length, path_len + 1):
                motif_paths[path[start:end]].add(path_index)

    motifs = [
        TokenMotif(tokens=tokens, branch_count=len(path_indices))
        for tokens, path_indices in motif_paths.items()
        if len(path_indices) >= min_branches
    ]
    motifs.sort(key=lambda motif: (-len(motif.tokens), -motif.branch_count, "".join(motif.tokens)))
    return motifs


def motifs_to_rows(motifs: list[TokenMotif]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, motif in enumerate(motifs, start=1):
        rows.append(
            {
                "rank": rank,
                "token_length": len(motif.tokens),
                "branch_count": motif.branch_count,
                "token_text": "".join(motif.tokens),
                "token_text_display": format_motif_display(motif.tokens),
            }
        )
    return rows


def analyze_tree_nodes(
    nodes: list[dict[str, Any]],
    *,
    min_length: int = 6,
    min_branches: int = 2,
) -> tuple[list[tuple[str, ...]], list[TokenMotif]]:
    paths = enumerate_leaf_paths(nodes)
    motifs = find_repeated_motifs(
        paths,
        min_length=min_length,
        min_branches=min_branches,
    )
    return paths, motifs


def write_motif_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rank", "token_length", "branch_count", "token_text", "token_text_display"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class TreeMotifSummary:
    tree_key: str
    country_id: str | None
    country_name: str | None
    model_id: str
    instruction_variant: str
    prefix_length: int
    seed: int
    leaf_count: int
    motif_count: int
    top_motif_display: str | None
    top_motif_length: int | None
    top_motif_branch_count: int | None
    csv_path: str


def analyze_record(
    record: RunRecord,
    *,
    min_length: int = 6,
    min_branches: int = 2,
) -> tuple[list[TokenMotif], list[dict[str, Any]]]:
    paths, motifs = analyze_tree_nodes(
        record.tree_nodes,
        min_length=min_length,
        min_branches=min_branches,
    )
    rows = motifs_to_rows(motifs)
    return motifs, rows


def filter_records(
    records: list[RunRecord],
    *,
    model_id: str | None = None,
    instruction_variants: list[str] | None = None,
    countries: list[str] | None = None,
    pilot_only: bool = False,
) -> list[RunRecord]:
    filtered: list[RunRecord] = []
    for record in records:
        if model_id and record.model_id != model_id:
            continue
        if instruction_variants and record.instruction_variant not in instruction_variants:
            continue
        if countries and record.country_id not in countries:
            continue
        if pilot_only and record.country_id not in PILOT_COUNTRIES:
            continue
        filtered.append(record)
    filtered.sort(key=lambda item: (item.country_id or "", item.prefix_length, item.seed))
    return filtered


def build_index_html(summaries: list[TreeMotifSummary], title: str) -> str:
    rows_html = []
    for summary in summaries:
        top = summary.top_motif_display or "—"
        if len(top) > 120:
            top = top[:117] + "..."
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(summary.country_name or summary.country_id or '—')}</td>"
            f"<td>{html.escape(summary.tree_key)}</td>"
            f"<td>{summary.leaf_count}</td>"
            f"<td>{summary.motif_count}</td>"
            f"<td>{summary.top_motif_length or '—'}</td>"
            f"<td>{summary.top_motif_branch_count or '—'}</td>"
            f"<td><code>{html.escape(top)}</code></td>"
            f'<td><a href="{html.escape(summary.csv_path)}">CSV</a></td>'
            "</tr>"
        )

    body_rows = "\n".join(rows_html) if rows_html else '<tr><td colspan="8">No trees matched.</td></tr>'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font: 14px/1.5 ui-sans-serif, system-ui, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f5f5f5; }}
    code {{ white-space: pre-wrap; word-break: break-word; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="muted">Generated {date.today().isoformat()}</p>
  <table>
    <thead>
      <tr>
        <th>Country</th>
        <th>Tree key</th>
        <th>Leaves</th>
        <th>Motifs</th>
        <th>Top length</th>
        <th>Top branches</th>
        <th>Top motif</th>
        <th>CSV</th>
      </tr>
    </thead>
    <tbody>
      {body_rows}
    </tbody>
  </table>
</body>
</html>
"""


def run_motif_analysis(
    records: list[RunRecord],
    output_dir: Path,
    *,
    min_length: int = 6,
    min_branches: int = 2,
    subdir: str,
) -> dict[str, Any]:
    target_dir = output_dir / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[TreeMotifSummary] = []

    for record in records:
        paths, motifs = analyze_tree_nodes(
            record.tree_nodes,
            min_length=min_length,
            min_branches=min_branches,
        )
        rows = motifs_to_rows(motifs)
        country_slug = record.country_id or "unknown"
        csv_name = f"{record.model_id}_{country_slug}.csv"
        csv_path = target_dir / csv_name
        write_motif_csv(csv_path, rows)

        top = motifs[0] if motifs else None
        summaries.append(
            TreeMotifSummary(
                tree_key=record.tree_key,
                country_id=record.country_id,
                country_name=record.raw.get("country_name"),
                model_id=record.model_id,
                instruction_variant=record.instruction_variant,
                prefix_length=record.prefix_length,
                seed=record.seed,
                leaf_count=len(paths),
                motif_count=len(motifs),
                top_motif_display=format_motif_display(top.tokens) if top else None,
                top_motif_length=len(top.tokens) if top else None,
                top_motif_branch_count=top.branch_count if top else None,
                csv_path=f"{subdir}/{csv_name}",
            )
        )

    manifest = {
        "generated_at": date.today().isoformat(),
        "subdir": subdir,
        "trees": len(summaries),
        "min_length": min_length,
        "min_branches": min_branches,
        "summaries": [summary.__dict__ for summary in summaries],
    }
    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    root_index_path = update_root_index(output_dir)
    return {
        "trees": len(summaries),
        "output_dir": str(output_dir),
        "index_html": str(root_index_path),
        "subdir_index_html": str(target_dir / "index.html"),
        "manifest": str(manifest_path),
        "summaries": summaries,
    }


def update_root_index(output_dir: Path) -> Path:
    """Build a root index linking pilot and plain subdirectories when present."""
    sections: list[str] = []
    for subdir in ("pilot", "plain"):
        manifest_path = output_dir / subdir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summaries = [TreeMotifSummary(**item) for item in manifest.get("summaries", [])]
        section_title = f"Repeated token paths ({subdir})"
        sections.append(build_index_html(summaries, section_title))
        sub_index = output_dir / subdir / "index.html"
        sub_index.write_text(build_index_html(summaries, section_title), encoding="utf-8")

    if not sections:
        root_index = output_dir / "index.html"
        root_index.write_text(build_index_html([], "Repeated token paths"), encoding="utf-8")
        return root_index

    combined = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8' />"
        "<title>Repeated token paths</title></head><body>"
        + "<hr />".join(sections)
        + "</body></html>"
    )
    root_index = output_dir / "index.html"
    root_index.write_text(combined, encoding="utf-8")
    return root_index


def load_capitals_records(mech_dir: Path, dataset_id: str = "capitals") -> list[RunRecord]:
    return load_all(mech_dir, dataset_id=dataset_id)
