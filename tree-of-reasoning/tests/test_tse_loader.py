"""Tests for TSE CSV parsing and registry building."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.tse_loader import (  # noqa: E402
    build_registry,
    load_registry,
    normalize_name,
    parse_consulta_cand_csv,
    save_registry,
)

FIXTURE_DIR = ROOT / "tests" / "fixtures" / "tse"


def test_normalize_name_strips_accents():
    assert normalize_name("José da Silva") == "jose da silva"


def test_parse_consulta_cand_csv_filters_presidents():
    records = parse_consulta_cand_csv(
        FIXTURE_DIR / "consulta_cand_2022_BRASIL.csv",
        office_filter="Presidente",
        election_year=2022,
    )
    assert len(records) == 4
    assert all(record.election_year == 2022 for record in records)
    assert all(not record.is_presidential_candidate_2026 for record in records)


def test_build_registry_merges_years_and_flags_presidential_candidates():
    records = build_registry(
        FIXTURE_DIR / "consulta_cand_2026_BRASIL.csv",
        FIXTURE_DIR / "consulta_cand_2022_BRASIL.csv",
    )
    by_name = {record.full_name: record for record in records}
    assert "LUIZ INACIO LULA DA SILVA" in by_name
    assert by_name["LUIZ INACIO LULA DA SILVA"].is_presidential_candidate_2026
    assert by_name["LUIZ INACIO LULA DA SILVA"].party == "PT"
    assert "CIRO GOMES FERREIRA" in by_name
    assert not by_name["CIRO GOMES FERREIRA"].is_presidential_candidate_2026
    assert "TARCISIO GOMES DE FREITAS" in by_name


def test_save_and_load_registry_roundtrip(tmp_path: Path):
    records = build_registry(
        FIXTURE_DIR / "consulta_cand_2026_BRASIL.csv",
        FIXTURE_DIR / "consulta_cand_2022_BRASIL.csv",
    )
    path = tmp_path / "registry.json"
    save_registry(records, path)
    loaded = load_registry(path)
    assert len(loaded) == len(records)
    assert loaded[0].full_name == records[0].full_name
