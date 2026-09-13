"""
Live population density proxy via OpenStreetMap's free Overpass API
(no key required).

True gridded population density (WorldPop/GHSL-style, people per km2 on
a fine raster) needs a downloaded raster + a lookup, not a point API --
that stays out of scope here. This instead pools every OSM `place` node
(city/town/village/hamlet/suburb) with a mapped `population` tag within
30km, and combines them with inverse-square distance weighting into a
single "local population pressure" number:

    population_density = sum(population_i / (1 + distance_km_i)^2) / reference_area_km2

reference_area_km2 is the area of a 5km-radius circle (~78.5 km2), so the
result is roughly people-per-km2-equivalent, not an exact census figure.
This is a real, per-point estimate from actual mapped population figures
-- just a coarser proxy than a proper population raster, and it inherits
whatever gaps exist in OSM's population tagging for a given area (which
can mean a legitimate empty result in sparsely-mapped terrain).
"""
import math

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
SEARCH_RADIUS_M = 30000
REFERENCE_AREA_KM2 = math.pi * 5 ** 2  # 5km-radius circle


def fetch_live_population_density(lat: float, lon: float) -> dict | None:
    """Returns {"population_density": <estimate>}, or None if no OSM
    place nodes with a population tag are found nearby, or the API call
    fails."""
    try:
        query = (
            f'[out:json][timeout:10];'
            f'node["place"]["population"](around:{SEARCH_RADIUS_M},{lat},{lon});'
            f'out;'
        )
        resp = requests.get(OVERPASS_URL, params={"data": query}, timeout=12)
        resp.raise_for_status()
        elements = resp.json().get("elements", [])

        total = 0.0
        found = False
        for el in elements:
            pop_raw = el.get("tags", {}).get("population")
            try:
                pop = float(str(pop_raw).replace(",", ""))
            except (TypeError, ValueError):
                continue
            d_km = _haversine_km(lat, lon, el["lat"], el["lon"])
            total += pop / (1 + d_km) ** 2
            found = True

        if not found:
            return None
        return {"population_density": round(total / REFERENCE_AREA_KM2, 2)}
    except Exception:
        return None


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
