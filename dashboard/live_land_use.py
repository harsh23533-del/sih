"""
Live land use via OpenStreetMap's free Overpass API (no key required).

Queries the nearest OSM way/relation carrying a `landuse`, `natural`, or
`leisure` tag within widening search radii, and maps its tag onto this
project's existing 4-class scheme (see CATEGORY_LABELS in
dashboard/app.py: 1=Forest, 2=Agriculture, 3=Urban, 4=Barren/Snow).

This is a real, per-point OSM lookup -- not a modelled classification
like Sentinel-2 land-cover would be -- so it's only as good as OSM's
mapping coverage in a given area. Sparse mapping in remote Himalayan
terrain means this can legitimately come back empty; callers should keep
falling back to the synthetic dataset value in that case, same contract
as the other live_* modules.
"""
import math

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
SEARCH_RADII_M = [3000, 10000, 30000]

# OSM tag value -> this project's category code.
_TAG_TO_CODE = {
    # Forest (1)
    "forest": 1, "wood": 1,
    # Agriculture (2)
    "farmland": 2, "farmyard": 2, "orchard": 2, "vineyard": 2, "meadow": 2,
    # Urban (3)
    "residential": 3, "commercial": 3, "industrial": 3, "retail": 3,
    "construction": 3, "urban": 3,
    # Barren/Snow (4)
    "bare_rock": 4, "scree": 4, "glacier": 4, "snow": 4, "quarry": 4,
}


def fetch_live_land_use(lat: float, lon: float) -> dict | None:
    """Returns {"land_use": <1-4 category code>}, or None if nothing
    OSM-tagged is found nearby or the API call fails."""
    try:
        for radius in SEARCH_RADII_M:
            query = (
                f'[out:json][timeout:10];'
                f'(way["landuse"](around:{radius},{lat},{lon});'
                f' way["natural"](around:{radius},{lat},{lon});'
                f' relation["landuse"](around:{radius},{lat},{lon}););'
                f'out center 20;'
            )
            resp = requests.get(OVERPASS_URL, params={"data": query}, timeout=12)
            resp.raise_for_status()
            elements = resp.json().get("elements", [])

            best_code = None
            best_dist = None
            for el in elements:
                tag_value = el.get("tags", {}).get("landuse") or el.get("tags", {}).get("natural")
                code = _TAG_TO_CODE.get(tag_value)
                if code is None:
                    continue
                center = el.get("center")
                if not center:
                    continue
                d = _haversine_km(lat, lon, center["lat"], center["lon"])
                if best_dist is None or d < best_dist:
                    best_dist, best_code = d, code

            if best_code is not None:
                return {"land_use": best_code}
        return None
    except Exception:
        return None


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
