"""Tests for repeated token path motif analysis."""

from __future__ import annotations

from src.pipelines.analysis.token_path_motifs import (
    enumerate_leaf_paths,
    find_repeated_motifs,
    format_motif_display,
    motifs_to_rows,
)


def _shared_prefix_tree() -> list[dict]:
    shared = ["a", "b", "c", "d", "e", "f", "g"]
    nodes = [
        {
            "id": "root",
            "depth": 0,
            "parent_id": None,
            "child_ids": ["l1", "l2", "l3"],
            "child_tokens": ["a", "a", "a"],
        },
        {
            "id": "l1",
            "depth": 1,
            "parent_id": "root",
            "child_ids": [],
            "child_tokens": [],
        },
        {
            "id": "l2",
            "depth": 1,
            "parent_id": "root",
            "child_ids": [],
            "child_tokens": [],
        },
        {
            "id": "l3",
            "depth": 1,
            "parent_id": "root",
            "child_ids": [],
            "child_tokens": [],
        },
    ]
    incoming = {leaf_id: token for leaf_id, token in zip(["l1", "l2", "l3"], shared)}
    for node in nodes:
        if node["id"] in incoming:
            node["tok"] = incoming[node["id"]]
    return nodes, incoming


def _tree_with_incoming_tokens(nodes: list[dict], incoming: dict[str, list[str]]) -> list[dict]:
    rebuilt: list[dict] = []
    for node in nodes:
        child_ids = node.get("child_ids", [])
        child_tokens = incoming.get(node["id"], [""] * len(child_ids))
        rebuilt.append(
            {
                **node,
                "child_ids": child_ids,
                "child_tokens": child_tokens,
            }
        )
    return rebuilt


def test_enumerate_leaf_paths_returns_root_to_leaf_tokens() -> None:
    base_nodes, incoming = _shared_prefix_tree()
    nodes = _tree_with_incoming_tokens(
        [
            {
                "id": "root",
                "depth": 0,
                "parent_id": None,
                "child_ids": ["n1", "n2"],
            },
            {"id": "n1", "depth": 1, "parent_id": "root", "child_ids": ["l1"]},
            {"id": "n2", "depth": 1, "parent_id": "root", "child_ids": ["l2"]},
            {"id": "l1", "depth": 2, "parent_id": "n1", "child_ids": []},
            {"id": "l2", "depth": 2, "parent_id": "n2", "child_ids": []},
        ],
        {
            "root": ["x", "y"],
            "n1": ["mid"],
            "n2": ["mid"],
        },
    )
    paths = enumerate_leaf_paths(nodes)
    assert paths == [("x", "mid"), ("y", "mid")]


def test_find_repeated_motifs_keeps_nested_substrings() -> None:
    nodes = _tree_with_incoming_tokens(
        [
            {
                "id": "root",
                "depth": 0,
                "parent_id": None,
                "child_ids": ["l1", "l2", "l3"],
            },
            {"id": "l1", "depth": 1, "parent_id": "root", "child_ids": []},
            {"id": "l2", "depth": 1, "parent_id": "root", "child_ids": []},
            {"id": "l3", "depth": 1, "parent_id": "root", "child_ids": []},
        ],
        {
            "root": ["a", "a", "a"],
            "l1": [],
            "l2": [],
            "l3": [],
        },
    )
    # Override incoming map path: each leaf gets full shared prefix as single-step path
    paths = [
        ("a", "b", "c", "d", "e", "f", "g"),
        ("a", "b", "c", "d", "e", "f", "g"),
        ("a", "b", "c", "d", "e", "f", "g"),
    ]
    motifs = find_repeated_motifs(paths, min_length=6, min_branches=2)
    lengths = sorted({len(motif.tokens) for motif in motifs}, reverse=True)
    assert lengths == [7, 6]
    assert all(motif.branch_count == 3 for motif in motifs)
    assert motifs[0].tokens == ("a", "b", "c", "d", "e", "f", "g")


def test_find_repeated_motifs_preserves_shorter_higher_branch_counts() -> None:
    paths = [
        ("a", "b", "c", "d", "e", "f", "g", "1"),
        ("a", "b", "c", "d", "e", "f", "g", "2"),
        ("a", "b", "c", "d", "e", "f", "h", "3"),
    ]
    motifs = find_repeated_motifs(paths, min_length=6, min_branches=2)
    by_text = {"".join(motif.tokens): motif for motif in motifs}
    assert by_text["abcdefg"].branch_count == 2
    assert by_text["abcdef"].branch_count == 3


def test_format_motif_display_escapes_whitespace_tokens() -> None:
    assert format_motif_display(("a", "\n", "b")) == "a\\nb"


def test_motifs_to_rows_assigns_rank() -> None:
    motifs = find_repeated_motifs(
        [
            ("a", "b", "c", "d", "e", "f"),
            ("a", "b", "c", "d", "e", "f"),
        ],
        min_length=6,
        min_branches=2,
    )
    rows = motifs_to_rows(motifs)
    assert rows[0]["rank"] == 1
    assert rows[0]["token_length"] == 6
    assert rows[0]["branch_count"] == 2
