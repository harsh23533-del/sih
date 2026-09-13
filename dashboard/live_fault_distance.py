"""
Live fault_distance_km -- no network call needed at runtime.

data/geology/sikkim_active_faults.geojson is a real subset (18 mapped
fault traces -- mostly Himalayan thrust/reverse faults, which matches
the region's known tectonics) of the GEM Foundation's Global Active
Faults Database (GAF-DB, CC-BY-SA 4.0), filtered to a bounding box
around Sikkim/the eastern Himalaya:
    https://github.com/GEMScienceTools/gem-global-active-faults

Distance is computed point-to-line-segment in a local equirectangular
projection (accurate to a few meters at this ~100km regional scale),
avoiding a geopandas/shapely dependency at dashboard runtime (the
project deliberately keeps those out of requirements-dashboard.txt --
see that file's header comment).
"""
import json
import math
import os

_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "geology", "sikkim_active_faults.geojson"
)

_fault_lines = None  # lazy-loaded, cached: list of [(lat, lon), ...] per fault


def _load_faults() -> list:
    global _fault_lines
    if _fault_lines is not None:
        return _fault_lines

    lines = []
    try:
        with open(_DATA_PATH) as f:
            geo = json.load(f)
        for feat in geo.get("features", []):
            geom = feat.get("geometry", {})
            if geom.get("type") != "LineString":
                continue
            # GeoJSON stores [lon, lat]; flip to (lat, lon) for consistency
            # with the rest of this project's haversine helpers.
            pts = [(pt[1], pt[0]) for pt in geom["coordinates"]]
            if len(pts) >= 2:
                lines.append(pts)
    except Exception:
        pass

    _fault_lines = lines
    return _fault_lines


def _to_local_xy(lat, lon, lat0):
    """Equirectangular projection to meters, centered near lat0. Good to
    a few meters of error at this region's scale (~100km)."""
    x = math.radians(lon) * 6371000.0 * math.cos(math.radians(lat0))
    y = math.radians(lat) * 6371000.0
    return x, y


def _point_to_segment_m(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = min(max(t, 0.0), 1.0)
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def fetch_live_fault_distance(lat: float, lon: float) -> dict | None:
    """Returns {"fault_distance_km": <distance to nearest mapped active
    fault>}, or None if the local fault file couldn't be loaded or no
    faults are known in this area at all (both would be unusual --
    callers fall back to the synthetic value either way)."""
    faults = _load_faults()
    if not faults:
        return None

    px, py = _to_local_xy(lat, lon, lat)
    best_m = None
    for line in faults:
        for (lat1, lon1), (lat2, lon2) in zip(line, line[1:]):
            ax, ay = _to_local_xy(lat1, lon1, lat)
            bx, by = _to_local_xy(lat2, lon2, lat)
            d = _point_to_segment_m(px, py, ax, ay, bx, by)
            if best_m is None or d < best_m:
                best_m = d

    if best_m is None:
        return None
    return {"fault_distance_km": round(best_m / 1000.0, 3)}
