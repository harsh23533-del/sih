"""
Live glacial_lake_distance_km -- no network call needed at runtime.

data/geology/sikkim_glacial_lakes_real.csv is a small, real, sourced list
of 6 named Sikkim glacial lakes (South Lhonak, Lhonak, Shako Cho, Tso
Lhamo, Samiti, Tsomgo) with coordinates pulled from Wikipedia/published
sources -- see the notes/source_url columns in that file.

This replaces the repo's older sikkim_glacial_lakes.geojson, whose
points turned out to be placeholder data (named
"synthetic_glacial_lake_*"), not a real inventory -- see the commit that
added this module for that finding.

Six lakes is a small, incomplete list (a proper inventory like ICIMOD's
Hindu Kush Himalaya glacial lake dataset would have far more), so this
is a best-effort distance to the nearest *known, named* glacial lake,
not a comprehensive one -- worth noting in the UI wherever this feature
is explained.
"""
import csv
import math
import os

_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "geology", "sikkim_glacial_lakes_real.csv"
)

_lake_latlon = None  # lazy-loaded, cached


def _load_lakes() -> list:
    global _lake_latlon
    if _lake_latlon is not None:
        return _lake_latlon

    points = []
    try:
        with open(_DATA_PATH, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    points.append((float(row["latitude"]), float(row["longitude"])))
                except (KeyError, ValueError):
                    continue
    except Exception:
        pass

    _lake_latlon = points
    return _lake_latlon


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fetch_live_glacial_lake_distance(lat: float, lon: float) -> dict | None:
    """Returns {"glacial_lake_distance_km": <distance to nearest known
    named glacial lake>}, or None if the local list couldn't be loaded."""
    lakes = _load_lakes()
    if not lakes:
        return None

    nearest_km = min(_haversine_km(lat, lon, llat, llon) for llat, llon in lakes)
    return {"glacial_lake_distance_km": round(nearest_km, 3)}
