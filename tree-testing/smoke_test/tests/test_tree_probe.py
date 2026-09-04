"""Tree probe tests with tau pruning on the Lorem prompt."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.probe.tree_probe import format_probe_report, run_probe_for_model  # noqa: E402


def _assert_valid_probe_result(result: dict) -> None:
    assert result["node_count"] >= 1
    assert result["branch_mode"] == "tau"

    for node in result["nodes"]:
        dist = node["distribution"]
        assert len(dist) >= 2, f"Node {node['id']} has fewer than 2 distribution entries"

        threshold = result["path_prob_threshold"]
        for token in node["branch_tokens"]:
            child = next(
                n
                for n in result["nodes"]
                if n["prefix_text"] == node["prefix_text"] + token
                and n["depth"] == node["depth"] + 1
            )
            assert child["path_prob"] >= threshold, (
                f"Child {token!r} under {node['id']} has path_prob "
                f"{child['path_prob']} < {threshold}"
            )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("model_id", ["deepseek-r1-7b", "qwq-32b-awq"])
def test_tree_probe(model_id: str, loaded_config, results_dir: Path):
    probe, models, vllm_cfg = loaded_config
    model_spec = next(m for m in models if m.id == model_id)

    result = run_probe_for_model(model_spec, vllm_cfg, probe, results_dir)
    _assert_valid_probe_result(result)

    report = format_probe_report(result)
    assert model_spec.hf_id in report
    print(f"\n{report}")
