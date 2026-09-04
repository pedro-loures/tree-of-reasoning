"""Load and build politician registries from TSE consulta_cand CSV files."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

PRESIDENT_OFFICE = "presidente"


@dataclass
class PoliticianRecord:
    id: str
    full_name: str
    ballot_name: str
    party: str
    office: str
    election_year: int
    is_presidential_candidate_2026: bool

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PoliticianRecord:
        return cls(
            id=str(data["id"]),
            full_name=str(data["full_name"]),
            ballot_name=str(data["ballot_name"]),
            party=str(data["party"]),
            office=str(data["office"]),
            election_year=int(data["election_year"]),
            is_presidential_candidate_2026=bool(data["is_presidential_candidate_2026"]),
        )


def normalize_name(text: str) -> str:
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_office(office: str) -> str:
    return normalize_name(office)


def _row_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        if key in row and row[key]:
            return row[key].strip()
    return ""


def parse_consulta_cand_csv(
    csv_path: Path,
    *,
    office_filter: str | None = None,
    election_year: int | None = None,
) -> list[PoliticianRecord]:
    """Parse a TSE consulta_cand CSV into politician records."""
    records: list[PoliticianRecord] = []
    with csv_path.open(encoding="latin-1", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            office = _row_value(row, "DS_CARGO")
            if office_filter and _normalize_office(office) != _normalize_office(office_filter):
                continue

            year_raw = _row_value(row, "ANO_ELEICAO")
            year = int(year_raw) if year_raw else (election_year or 0)
            if election_year is not None and year != election_year:
                continue

            candidate_id = _row_value(row, "SQ_CANDIDATO")
            full_name = _row_value(row, "NM_CANDIDATO")
            ballot_name = _row_value(row, "NM_URNA_CANDIDATO")
            party = _row_value(row, "SG_PARTIDO")
            if not candidate_id or not full_name:
                continue

            is_pres_2026 = year == 2026 and _normalize_office(office) == PRESIDENT_OFFICE
            records.append(
                PoliticianRecord(
                    id=candidate_id,
                    full_name=full_name,
                    ballot_name=ballot_name or full_name,
                    party=party,
                    office=office,
                    election_year=year,
                    is_presidential_candidate_2026=is_pres_2026,
                )
            )
    return records


def build_registry(
    csv_2026: Path,
    csv_2022_presidents: Path | None = None,
) -> list[PoliticianRecord]:
    """Build a deduplicated politician registry from TSE CSV files."""
    by_name: dict[str, PoliticianRecord] = {}

    for record in parse_consulta_cand_csv(csv_2026, election_year=2026):
        key = normalize_name(record.full_name)
        by_name[key] = record

    if csv_2022_presidents is not None and csv_2022_presidents.exists():
        for record in parse_consulta_cand_csv(
            csv_2022_presidents,
            office_filter="Presidente",
            election_year=2022,
        ):
            key = normalize_name(record.full_name)
            existing = by_name.get(key)
            if existing is None:
                by_name[key] = record
            elif record.is_presidential_candidate_2026 and not existing.is_presidential_candidate_2026:
                by_name[key] = record

    return sorted(by_name.values(), key=lambda item: (item.full_name.lower(), item.id))


def save_registry(records: list[PoliticianRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"politicians": [record.to_dict() for record in records]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def load_registry(path: Path) -> list[PoliticianRecord]:
    payload = json.loads(path.read_text())
    return [PoliticianRecord.from_dict(item) for item in payload["politicians"]]
