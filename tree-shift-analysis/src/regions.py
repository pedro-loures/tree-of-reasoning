"""Region groupings for the 30-country capitals experiment."""

from __future__ import annotations

COUNTRY_REGION: dict[str, str] = {
    "brazil": "south_america",
    "argentina": "south_america",
    "venezuela": "south_america",
    "bolivia": "south_america",
    "colombia": "south_america",
    "united_states": "north_america",
    "mexico": "north_america",
    "canada": "north_america",
    "cuba": "north_america",
    "jamaica": "north_america",
    "mozambique": "africa",
    "madagascar": "africa",
    "egypt": "africa",
    "morocco": "africa",
    "south_africa": "africa",
    "england": "europe",
    "italy": "europe",
    "albania": "europe",
    "portugal": "europe",
    "russia": "europe",
    "china": "asia",
    "japan": "asia",
    "india": "asia",
    "mongolia": "asia",
    "jordan": "asia",
    "australia": "oceania",
    "new_zealand": "oceania",
    "east_timor": "oceania",
    "papua_new_guinea": "oceania",
    "fiji": "oceania",
}


def region_for_country(country_id: str | None) -> str:
    if not country_id:
        return "unknown"
    return COUNTRY_REGION.get(country_id, "unknown")
