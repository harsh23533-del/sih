"""
Live historical-landslide density & distance -- no network call needed.

Unlike rainfall/soil/terrain, this doesn't need an external API: the real
historical inventory is already sitting in this repo
(data/historical_landslides/coolr_ner_labeled.csv + known_events_sikkim.csv).
So "live" here just means "computed exactly for the clicked point against
the real catalog," instead of inherited from the nearest synthetic sample
point -- and it works even with the sandbox's network fully blocked.

historical_landslide_density        = count of catalog events within
                                       radius_km of the point (default 5km,
                                       matching preprocessing/density_distance_features.py
                                       so live and training-time definitions agree)
distance_to_nearest_landslide_m     = distance to the single closest
                                       catalog event, in meters

The catalog is loaded once at import time and cached in memory -- it's only
~120 rows, so no need for a database or repeated disk reads.
"""
import math
import os

import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "historical_landslides")
_CATALOG_FILES = ["coolr_ner_labeled.csv", "known_events_sikkim.csv"]

_catalog_latlon = None  # lazy-loaded, cached


def _load_catalog() -> list:
    """Pools every catalog file's lat/lon into one list of (lat, lon)
    tuples, deduping exact repeats. Skips a file quietly if it's missing
    or malformed -- one bad file shouldn't break the others."""
    global _catalog_latlon
    if _catalog_latlon is not None:
        return _catalog_latlon

    points = set()
    for fname in _CATALOG_FILES:
        path = os.path.join(_DATA_DIR, fname)
        try:
            df = pd.read_csv(path)
            for lat, lon in zip(df["latitude"], df["longitude"]):
                if pd.notna(lat) and pd.notna(lon):
                    points.add((float(lat), float(lon)))
        except Exception:
            continue

    _catalog_latlon = list(points)
    return _catalog_latlon


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fetch_live_landslide_history(lat: float, lon: float, radius_km: float = 5.0) -> dict | None:
    """Real historical_landslide_density + distance_to_nearest_landslide_m
    for this exact point. Returns None only if the catalog couldn't be
    loaded at all (so callers fall back to the synthetic base row, same
    contract as the other live_* modules)."""
    catalog = _load_catalog()
    if not catalog:
        return None

    distances_km = [_haversine_km(lat, lon, clat, clon) for clat, clon in catalog]
    density = sum(1 for d in distances_km if d <= radius_km)
    nearest_m = min(distances_km) * 1000

    return {
        "historical_landslide_density": density,
        "distance_to_nearest_landslide_m": round(nearest_m, 1),
    }
