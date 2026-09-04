#!/usr/bin/env python3
"""Download TSE consulta_cand CSVs and build the politician registry."""

from __future__ import annotations

import argparse
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.tse_loader import build_registry, save_registry  # noqa: E402

TSE_BASE = "https://cdn.tse.jus.br/estatistica/sead/odsele/consulta_cand"
GITHUB_MIRROR_2026 = (
    "https://raw.githubusercontent.com/leofn/tse-candidatos-2026/main/dados/"
    "consulta_cand_2026_BRASIL.csv"
)
DEFAULT_DATA_DIR = ROOT / "data" / "tse"
REGISTRY_PATH = DEFAULT_DATA_DIR / "registry.json"
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; tree-of-reasoning/1.0; research)",
    "Accept": "*/*",
}


def _fetch_url(url: str) -> bytes:
    request = Request(url, headers=REQUEST_HEADERS)
    with urlopen(request, timeout=120) as response:
        return response.read()


def download_brasil_csv_from_cdn(year: int, data_dir: Path) -> Path:
    """Download consulta_cand zip for a year and extract the BRASIL CSV."""
    zip_url = f"{TSE_BASE}/consulta_cand_{year}.zip"
    csv_name = f"consulta_cand_{year}_BRASIL.csv"
    csv_path = data_dir / csv_name

    if csv_path.exists() and csv_path.stat().st_size > 1000:
        print(f"Using cached {csv_path}", flush=True)
        return csv_path

    print(f"Downloading {zip_url}", flush=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = _fetch_url(zip_url)

    with zipfile.ZipFile(BytesIO(payload)) as archive:
        member_names = archive.namelist()
        brasil_member = next(
            (name for name in member_names if name.endswith(csv_name)),
            None,
        )
        if brasil_member is None:
            raise FileNotFoundError(f"{csv_name} not found in {zip_url}")
        csv_path.write_bytes(archive.read(brasil_member))

    print(f"Wrote {csv_path}", flush=True)
    return csv_path


def download_brasil_csv_from_mirror(year: int, data_dir: Path) -> Path:
    """Fallback: full 2026 BRASIL CSV from a public GitHub mirror of TSE data."""
    if year != 2026:
        raise FileNotFoundError(f"No mirror configured for year {year}")
    csv_path = data_dir / f"consulta_cand_{year}_BRASIL.csv"
    print(f"Downloading mirror {GITHUB_MIRROR_2026}", flush=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(_fetch_url(GITHUB_MIRROR_2026))
    if csv_path.stat().st_size < 1000:
        raise ValueError(f"Mirror download too small: {csv_path}")
    print(f"Wrote {csv_path}", flush=True)
    return csv_path


def resolve_csv_2026(data_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    try:
        return download_brasil_csv_from_cdn(2026, data_dir)
    except Exception as cdn_exc:
        try:
            return download_brasil_csv_from_mirror(2026, data_dir)
        except Exception as mirror_exc:
            fixture = ROOT / "tests" / "fixtures" / "tse" / "consulta_cand_2026_BRASIL.csv"
            if fixture.exists():
                print(
                    f"CDN failed ({cdn_exc}); mirror failed ({mirror_exc}); "
                    f"using minimal test fixture {fixture}",
                    flush=True,
                )
                return fixture
            raise RuntimeError(
                f"Could not fetch 2026 TSE data (CDN: {cdn_exc}; mirror: {mirror_exc})"
            ) from mirror_exc


def resolve_csv_2022(data_dir: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    try:
        return download_brasil_csv_from_cdn(2022, data_dir)
    except Exception as cdn_exc:
        fixture = ROOT / "tests" / "fixtures" / "tse" / "consulta_cand_2022_BRASIL.csv"
        cached = data_dir / "consulta_cand_2022_BRASIL.csv"
        if cached.exists() and cached.stat().st_size > 100:
            print(f"Using cached {cached}", flush=True)
            return cached
        if fixture.exists():
            print(
                f"Warning: could not fetch 2022 data ({cdn_exc}); using fixture {fixture}",
                flush=True,
            )
            return fixture
        print(f"Warning: could not fetch 2022 data ({cdn_exc}); continuing with 2026 only", flush=True)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch TSE data and build politician registry")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory for raw CSVs and registry.json",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Output registry path (default: <data-dir>/registry.json)",
    )
    parser.add_argument(
        "--csv-2026",
        type=Path,
        default=None,
        help="Use a local 2026 consulta_cand CSV instead of downloading",
    )
    parser.add_argument(
        "--csv-2022",
        type=Path,
        default=None,
        help="Use a local 2022 consulta_cand CSV instead of downloading",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    registry_path = args.registry_path or (data_dir / "registry.json")

    csv_2026 = resolve_csv_2026(data_dir, args.csv_2026)
    csv_2022 = resolve_csv_2022(data_dir, args.csv_2022)

    records = build_registry(csv_2026, csv_2022)
    save_registry(records, registry_path)

    pres_count = sum(1 for record in records if record.is_presidential_candidate_2026)
    print(
        f"Built registry with {len(records)} politicians "
        f"({pres_count} presidential candidates for 2026) -> {registry_path}",
        flush=True,
    )
    if pres_count < 10:
        print(
            "Warning: expected ~13 presidential candidates for 2026; "
            "registry may be incomplete.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
