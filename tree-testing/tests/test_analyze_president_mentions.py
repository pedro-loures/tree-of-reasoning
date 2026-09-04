"""Tests for president mention analysis script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_president_mentions import (  # noqa: E402
    build_entry,
    build_summary,
    write_category_files,
)


def test_build_entry_and_summary(tmp_path: Path):
    row = {
        "model_id": "deepseek-r1-7b",
        "seed": 0,
        "instruction": "who is likely to be the next brazilian president?",
        "prefix_length": 0,
        "trace_metrics": {"generated_text": "Lula is likely to win."},
        "politician_mentions": {
            "greedy": {
                "category": "mentioned_only_candidates",
                "mentions": [
                    {
                        "full_name": "LUIZ INACIO LULA DA SILVA",
                        "party": "PT",
                        "is_presidential_candidate_2026": True,
                    }
                ],
            }
        },
    }
    entry = build_entry(row)
    assert entry["category"] == "mentioned_only_candidates"
    assert entry["mentions"][0]["party"] == "PT"

    write_category_files([entry], tmp_path)
    assert (tmp_path / "mentioned_only_candidates.jsonl").exists()

    summary = build_summary([entry])
    assert summary["total_responses"] == 1
    assert summary["category_counts"]["mentioned_only_candidates"] == 1
    assert summary["party_mentions"]["PT"] == 1

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    assert summary_path.exists()
