"""Tests for word edge analysis."""

from __future__ import annotations

from src.pipelines.analysis.io import RunRecord
from src.pipelines.analysis.word_edges import (
    all_edges_by_position,
    analyze_runs,
    count_edges_per_position,
    count_edges_per_word,
    top_k_positions,
    top_k_words,
)


def _make_record(
    model_id: str,
    prefix_length: int,
    seed: int,
    nodes: list[dict],
) -> RunRecord:
    return RunRecord(
        dataset_id="main",
        model_id=model_id,
        instruction_variant="legacy",
        prefix_length=prefix_length,
        seed=seed,
        tree_key=f"main:{model_id}:legacy:{prefix_length}:{seed}",
        raw={},
        tree_nodes=nodes,
    )


def test_count_edges_per_word_sums_breadth_by_incoming_token_internal_only() -> None:
    nodes = [
        {
            "id": "root",
            "depth": 0,
            "breadth": 2,
            "child_ids": ["a", "b"],
            "child_tokens": ["Okay", "The"],
        },
        {"id": "a", "depth": 1, "breadth": 1, "child_ids": ["a1"], "child_tokens": [","]},
        {"id": "b", "depth": 1, "breadth": 0, "child_ids": [], "child_tokens": []},
        {"id": "a1", "depth": 2, "breadth": 0, "child_ids": [], "child_tokens": []},
    ]
    counts = count_edges_per_word(nodes)
    assert counts["Okay"] == 1
    assert "The" not in counts
    assert "," not in counts


def test_count_edges_per_position_sums_breadth_by_depth_internal_only() -> None:
    nodes = [
        {
            "id": "root",
            "depth": 0,
            "breadth": 1,
            "child_ids": ["a"],
            "child_tokens": ["Okay"],
        },
        {"id": "a", "depth": 1, "breadth": 2, "child_ids": ["a1", "a2"], "child_tokens": ["x", "y"]},
        {"id": "a1", "depth": 2, "breadth": 3, "child_ids": ["a1a"], "child_tokens": ["z"]},
        {"id": "a2", "depth": 2, "breadth": 0, "child_ids": [], "child_tokens": []},
        {"id": "a1a", "depth": 3, "breadth": 0, "child_ids": [], "child_tokens": []},
    ]
    counts = count_edges_per_position(nodes)
    assert counts[1] == 2
    assert counts[2] == 3
    assert 3 not in counts


def test_all_edges_by_position_lists_every_depth() -> None:
    counts = {1: 2, 2: 3, 5: 1}
    assert all_edges_by_position(counts) == [
        {"position": 1, "edge_count": 2},
        {"position": 2, "edge_count": 3},
        {"position": 5, "edge_count": 1},
    ]


def test_top_k_positions_most_and_least() -> None:
    counts = {1: 5, 2: 1, 4: 3}
    most, least = top_k_positions(counts, 2)
    assert most == [{"position": 1, "edge_count": 5}, {"position": 4, "edge_count": 3}]
    assert least == [{"position": 2, "edge_count": 1}, {"position": 4, "edge_count": 3}]


def test_top_k_words_most_and_least() -> None:
    counts = {"Okay": 5, "The": 1, "So": 3}
    most, least = top_k_words(counts, 2)
    assert most == [{"word": "Okay", "edge_count": 5}, {"word": "So", "edge_count": 3}]
    assert least == [{"word": "The", "edge_count": 1}, {"word": "So", "edge_count": 3}]


def test_aggregate_across_seeds() -> None:
    nodes = [
        {
            "id": "root",
            "depth": 0,
            "breadth": 1,
            "child_ids": ["a"],
            "child_tokens": ["Okay"],
        },
        {"id": "a", "depth": 1, "breadth": 2, "child_ids": ["a1", "a2"], "child_tokens": ["x", "y"]},
        {"id": "a1", "depth": 2, "breadth": 0, "child_ids": [], "child_tokens": []},
        {"id": "a2", "depth": 2, "breadth": 0, "child_ids": [], "child_tokens": []},
    ]
    runs = [
        _make_record("deepseek-r1-7b", 0, 0, nodes),
        _make_record("deepseek-r1-7b", 0, 1, nodes),
    ]
    result = analyze_runs(runs, top_k=2)
    assert len(result.per_execution) == 2
    assert result.aggregated[0]["runs"] == 2
    assert result.aggregated[0]["most_edges"][0]["word"] == "Okay"
    assert result.aggregated[0]["most_edges"][0]["edge_count"] == 4
    assert result.aggregated[0]["most_edges_by_position"][0]["position"] == 1
    assert result.aggregated[0]["most_edges_by_position"][0]["edge_count"] == 4
    assert result.aggregated[0]["all_edges_by_position"] == [{"position": 1, "edge_count": 4}]
    assert result.per_execution[0]["all_edges_by_position"] == [{"position": 1, "edge_count": 2}]
