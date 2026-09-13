"""
Live terrain features from free, no-key public sources.

Elevation, slope, aspect, curvature, and ruggedness (TRI) are computed from
a small 3x3 elevation grid around the point, fetched from Open-Meteo's free
Elevation API (itself backed by a ~90m-resolution global DEM). Slope/aspect
use Horn's method (the standard GIS approach for a 3x3 window); curvature
uses a simple discrete Laplacian; TRI follows Riley et al. (1999). These are
solid real-world approximations, not a substitute for a proper full-DEM GIS
pipeline -- a single 3x3 window can't see terrain just outside it the way a
real watershed-scale analysis (e.g. TWI, which needs full flow-accumulation
and stays synthetic here) can.

road_distance and drainage_distance_km come from OpenStreetMap's Overpass
API: straight-line distance to the nearest mapped road / waterway, widening
the search radius if nothing is found nearby.

historical_landslide_density and distance_to_nearest_landslide_m are live
too, but computed locally against this repo's own catalog rather than
fetched from the network -- see live_landslide_history.py.

Left as synthetic (no simple free live source exists): geology, soil type,
land_use, NDVI, seismic_pga, fault_distance_km, insar_deformation_mm_yr,
twi, insolation_proxy, freeze_thaw_index, root_cohesion_proxy,
glacial_lake_distance_km, population_density, exposure_index.
"""
import math

import requests

ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
GRID_SPACING_M = 90  # matches the DEM's native ~90m resolution


def fetch_live_terrain(lat: float, lon: float) -> dict | None:
    """Best-effort real terrain values for a point. Returns whatever subset
    of {elevation, slope, aspect, curvature, ruggedness_tri, road_distance,
    drainage_distance_km} succeeds (each piece fails independently), or
    None if nothing could be fetched -- callers should merge this over the
    synthetic base row, not replace it wholesale."""
    out = {}
    try:
        grid = _fetch_elevation_grid(lat, lon)
        if grid:
            out.update(_terrain_from_elevations(grid, GRID_SPACING_M))
    except Exception:
        pass
    try:
        d = _fetch_osm_distance_km(lat, lon, 'way["highway"]', [3000, 10000, 30000])
        if d is not None:
            out["road_distance"] = round(d, 3)
    except Exception:
        pass
    try:
        d = _fetch_osm_distance_km(lat, lon, 'way["waterway"]', [5000, 15000, 40000])
        if d is not None:
            out["drainage_distance_km"] = round(d, 3)
    except Exception:
        pass
    return out or None


def _offsets_deg(lat: float, spacing_m: float) -> tuple:
    dlat = spacing_m / 110_540.0
    dlon = spacing_m / (111_320.0 * math.cos(math.radians(lat)) or 1e-6)
    return dlat, dlon


def _fetch_elevation_grid(lat: float, lon: float) -> list | None:
    dlat, dlon = _offsets_deg(lat, GRID_SPACING_M)
    # Row-major 3x3, north row first, so index 4 (z5) is the center point.
    points = [
        (lat + dlat, lon - dlon), (lat + dlat, lon), (lat + dlat, lon + dlon),
        (lat, lon - dlon),        (lat, lon),        (lat, lon + dlon),
        (lat - dlat, lon - dlon), (lat - dlat, lon), (lat - dlat, lon + dlon),
    ]
    lats = ",".join(f"{p[0]:.6f}" for p in points)
    lons = ",".join(f"{p[1]:.6f}" for p in points)
    resp = requests.get(ELEVATION_URL, params={"latitude": lats, "longitude": lons}, timeout=6)
    resp.raise_for_status()
    z = resp.json().get("elevation")
    if not z or len(z) != 9 or any(v is None for v in z):
        return None
    return z


def _terrain_from_elevations(z: list, cell_m: float) -> dict:
    """Pure math over a 3x3 elevation grid -- no network calls -- so this
    half of the logic can be unit-tested without hitting the DEM API."""
    z1, z2, z3, z4, z5, z6, z7, z8, z9 = z
    dz_dx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8 * cell_m)
    dz_dy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8 * cell_m)
    slope_deg = math.degrees(math.atan(math.hypot(dz_dx, dz_dy)))
    if dz_dx == 0 and dz_dy == 0:
        aspect_deg = 0.0  # flat -- no meaningful downslope direction
    else:
        raw = math.degrees(math.atan2(dz_dy, -dz_dx))
        aspect_deg = (90.0 - raw) % 360  # standard ESRI-style compass bearing
    curvature = ((z2 + z4 + z6 + z8) - 4 * z5) / (cell_m ** 2)
    tri = math.sqrt(sum((n - z5) ** 2 for n in [z1, z2, z3, z4, z6, z7, z8, z9]))
    return {
        "elevation": round(z5, 1),
        "slope": round(slope_deg, 2),
        "aspect": round(aspect_deg, 1),
        "curvature": round(curvature, 6),
        "ruggedness_tri": round(tri, 2),
    }


def _fetch_osm_distance_km(lat: float, lon: float, tag_filter: str, radii_m: list):
    for radius in radii_m:
        query = f'[out:json][timeout:10];({tag_filter}(around:{radius},{lat},{lon}););out center 10;'
        resp = requests.get(OVERPASS_URL, params={"data": query}, timeout=12)
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        best = None
        for el in elements:
            if "center" in el:
                elat, elon = el["center"]["lat"], el["center"]["lon"]
            elif "lat" in el:
                elat, elon = el["lat"], el["lon"]
            else:
                continue
            d = _haversine_km(lat, lon, elat, elon)
            if best is None or d < best:
                best = d
        if best is not None:
            return best
    return None


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
