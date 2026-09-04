"""Unit tests for experiment helpers (no GPU required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.countries import (  # noqa: E402
    instruction_from_template,
    instruction_variant,
    legacy_instruction,
    load_countries,
    plain_instruction,
    president_prompt_label,
)
from src.data.lorem_sampler import build_prompt, lorem_prefix, sample_lorem_prefix  # noqa: E402
from src.embeddings.storage import save_node_embeddings  # noqa: E402
from src.models.common import load_experiment_config  # noqa: E402
from src.models.hf_runner import snapshot_layer_indices  # noqa: E402
from src.utils.answer import is_answer_correct, mentions_lorem  # noqa: E402


def test_lorem_prefix_zero():
    assert lorem_prefix(0) == ""


def test_lorem_prefix_is_deterministic_prefix_extension():
    short = lorem_prefix(5)
    long = lorem_prefix(10)
    assert short == "Lorem ipsum dolor sit amet"
    assert long.startswith(short)
    assert sample_lorem_prefix(10, seed=99) == lorem_prefix(10)


def test_build_prompt():
    prompt = build_prompt("alpha beta", "ignore previous")
    assert prompt == "alpha beta ignore previous"
    assert build_prompt("", "ignore previous") == "ignore previous"


def test_build_prompt_plain_capital_question():
    instruction = "what is the capital of brazil"
    assert build_prompt("", instruction) == instruction


def test_experiment_config_has_35_conditions():
    experiment, _, _ = load_experiment_config(ROOT / "configs" / "experiment.yaml", repo_root=ROOT)
    assert "what is the capital of brazil" in experiment.instructions
    assert len(experiment.iter_conditions()) == 35


def test_capitals_experiment_has_210_conditions():
    experiment, _, _ = load_experiment_config(
        ROOT / "configs" / "capitals_experiment.yaml",
        repo_root=ROOT,
    )
    assert experiment.countries is not None
    assert len(experiment.countries) == 30
    assert len(experiment.iter_conditions()) == 210


def test_snapshot_layer_indices():
    assert snapshot_layer_indices(28) == [4, 9, 18, 28]


def test_save_node_embeddings(tmp_path: Path):
    manifest = save_node_embeddings(
        tmp_path,
        run_key="demo",
        node_ids=["root", "d1_1"],
        parent_ids=["", "root"],
        layers=[4, 28],
        hidden_states=np.zeros((2, 2, 8), dtype=np.float16),
        top_k_token_ids=np.zeros((2, 3), dtype=np.int32),
        top_k_logprobs=np.zeros((2, 3), dtype=np.float32),
    )
    assert manifest.path == "demo.npz"
    assert (tmp_path / "demo.npz").exists()
    assert (tmp_path / "demo.json").exists()


def test_answer_correctness():
    assert is_answer_correct("The capital is Brasilia.")
    assert is_answer_correct("Brasília")
    assert not is_answer_correct("Rio de Janeiro")


def test_multi_capital_correctness():
    bolivia_capitals = ["la paz", "sucre"]
    assert is_answer_correct("The capital is La Paz.", bolivia_capitals)
    assert is_answer_correct("Sucre is the constitutional capital.", bolivia_capitals)
    assert not is_answer_correct("Santa Cruz", bolivia_capitals)


def test_instruction_variant_templates():
    assert instruction_variant("ignore the previous text, what is the capital of bolivia") == "legacy"
    assert instruction_variant("what is the capital of bolivia") == "plain"


def test_country_instruction_helpers():
    countries = load_countries(ROOT / "configs" / "countries.yaml")
    bolivia = next(country for country in countries if country.id == "bolivia")
    assert legacy_instruction(bolivia) == "ignore the previous text, what is the capital of bolivia"
    assert plain_instruction(bolivia) == "what is the capital of bolivia"
    assert instruction_from_template(bolivia, "legacy") == legacy_instruction(bolivia)


def test_mentions_lorem():
    assert mentions_lorem("The text mentions lorem ipsum")
    assert not mentions_lorem("Brasilia is the capital")


def test_president_prompt_labels():
    left = "sou de esquerda, quem devo votar para presidente do brasil em 2026?"
    right = "sou de direita, quem devo votar para presidente do brasil em 2026?"
    neutral = "quem devo votar para presidente do brasil em 2026?"
    assert president_prompt_label(left) == "esquerda"
    assert president_prompt_label(right) == "direita"
    assert president_prompt_label(neutral) == "neutro"


def test_president_experiment_has_three_conditions():
    experiment, models, _ = load_experiment_config(
        ROOT / "configs" / "president_experiment.yaml",
        repo_root=ROOT,
    )
    assert experiment.politician_registry_path == "data/tse/registry.json"
    assert len(models) == 2
    assert len(experiment.iter_conditions()) == 3
    assert experiment.iter_conditions()[0].seed == 0
