"""
Live geology -- best-effort via Macrostrat's free, no-key API
(https://macrostrat.org/api/v2/units?lat=...&lng=...).

Honest caveat, checked before writing this: Macrostrat's own coverage is
uneven outside North America -- a direct test against a point in Norway
(a well-studied, non-Himalayan region) returned zero mapped units, not
just remote Sikkim. So this is wired in as a genuine best-effort source,
same fail-soft contract as everything else here: if Macrostrat has data
for a given point, great; if not (plausible for much of the Himalayan
region right now), the caller falls back to the synthetic value, same as
before this module existed. Worth revisiting if Macrostrat's coverage
improves.

This project's existing geology categories (see CATEGORY_LABELS in
dashboard/app.py) are specific Himalayan crystalline/metamorphic rock
types -- Gneiss, Schist, Phyllite, Quartzite, Alluvium -- not a generic
sedimentary/igneous/metamorphic split, so this matches Macrostrat's
free-text lithology description against those specific keywords rather
than its broad lith_type field.
"""
import requests

UNITS_URL = "https://macrostrat.org/api/v2/units"

# This project's existing geology scheme: 1=Gneiss, 2=Schist,
# 3=Phyllite, 4=Quartzite, 5=Alluvium (see CATEGORY_LABELS in app.py).
# Matched in this order against Macrostrat's free-text lithology/unit
# description -- first keyword hit wins.
_KEYWORD_TO_CODE = [
    ("gneiss", 1),
    ("schist", 2),
    ("phyllite", 3),
    ("quartzite", 4),
    ("alluvium", 5),
    ("alluvial", 5),
]


def fetch_live_geology(lat: float, lon: float) -> dict | None:
    """Returns {"geology": <category code, 1-5>}, or None if Macrostrat
    has no mapped units for this point (common outside North America),
    none of the returned lithology text matches this project's specific
    rock-type scheme, or the API call fails."""
    try:
        resp = requests.get(
            UNITS_URL,
            params={"lat": lat, "lng": lon, "response": "long"},
            timeout=8,
        )
        resp.raise_for_status()
        units = resp.json().get("success", {}).get("data", [])
        if not units:
            return None

        # Units are typically returned oldest-first; check the
        # shallowest/youngest (last) entries first -- closest to what's
        # actually at the surface.
        for unit in reversed(units):
            text = " ".join(
                str(unit.get(field, "")) for field in ("lith", "name", "strat_name_long")
            ).lower()
            for keyword, code in _KEYWORD_TO_CODE:
                if keyword in text:
                    return {"geology": code}

        return None
    except Exception:
        return None
