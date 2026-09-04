"""Compact tree representations and text rendering."""

from __future__ import annotations


def build_incoming_token_map(nodes: list[dict]) -> dict[str, str]:
    incoming: dict[str, str] = {}
    for node in nodes:
        for child_id, token in zip(node.get("child_ids", []), node.get("child_tokens", [])):
            incoming[child_id] = token
    return incoming


def compact_tree(nodes: list[dict], root_prefix: str | None = None) -> list[dict]:
    parent_token = build_incoming_token_map(nodes)
    if root_prefix is None:
        root_node = next((n for n in nodes if n.get("id") == "root"), None)
        root_prefix = root_node.get("prefix_text", "") if root_node else ""

    compact: list[dict] = []
    for node in nodes:
        entry: dict = {
            "id": node["id"],
            "d": node["depth"],
            "p": round(node["path_prob"], 6),
            "c": node["child_ids"],
        }
        if node["id"] != "root":
            entry["tok"] = parent_token.get(node["id"], "?")
        if node.get("child_tokens"):
            entry["ct"] = node["child_tokens"]
        prefix_text = node.get("prefix_text")
        if prefix_text and node["id"] != "root":
            suffix = prefix_text[len(root_prefix):] if prefix_text.startswith(root_prefix) else prefix_text
            if suffix:
                entry["suffix"] = suffix
        compact.append(entry)
    return compact


def build_tree_text(nodes: list[dict]) -> str:
    by_id = {node["id"]: node for node in nodes}

    def display_token(token: str | None) -> str:
        if not token:
            return "?"
        if token == "\n":
            return "\\n"
        if token == "\t":
            return "\\t"
        if token == " ":
            return "·"
        return token

    lines: list[str] = []

    def walk(node_id: str, prefix: str, is_last: bool, depth: int) -> None:
        node = by_id[node_id]
        connector = "" if depth == 0 else ("└─ " if is_last else "├─ ")
        label = "⟨prompt⟩" if node_id == "root" else display_token(node.get("tok"))
        prob = node["p"]
        prob_text = f"{prob:.4f}" if prob >= 0.0001 else f"{prob:.2e}"
        lines.append(f"{prefix}{connector}{label} ({prob_text})")
        child_prefix = "" if depth == 0 else prefix + ("   " if is_last else "│  ")
        for index, child_id in enumerate(node["c"]):
            walk(child_id, child_prefix, index == len(node["c"]) - 1, depth + 1)

    walk("root", "", True, 0)
    return "\n".join(lines)


def prune_tree_by_path_prob(nodes: list[dict], min_path_prob: float) -> list[dict]:
    """Keep root-to-node paths for nodes whose path probability meets the threshold."""
    by_id = {node["id"]: node for node in nodes}
    keep: set[str] = set()

    def mark(node_id: str) -> None:
        keep.add(node_id)
        for child_id in by_id[node_id]["c"]:
            if by_id[child_id]["p"] >= min_path_prob:
                mark(child_id)

    mark("root")
    pruned: list[dict] = []
    for node in nodes:
        if node["id"] not in keep:
            continue
        pruned.append({**node, "c": [child_id for child_id in node["c"] if child_id in keep]})
    return pruned
