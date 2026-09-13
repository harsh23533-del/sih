"""
Live root-cohesion proxy -- pure lookup, no network call needed.

Uses the project's own original formula from
preprocessing/environmental_features.py (so live and training-time
definitions -- and scales -- agree):

    base_cohesion = land_use mapped to {1: 0.8, 2: 0.35, 3: 0.15, 4: 0.05}
    root_cohesion_proxy = base_cohesion * (0.5 + 0.5 * NDVI)

land_use comes from live_land_use.py once mapped; NDVI stays the
synthetic dataset value (no free live source -- see live_terrain.py's
docstring), so this is a genuine mix of live + synthetic inputs, same
approach the dashboard already takes elsewhere.
"""

_LAND_USE_TO_BASE_COHESION = {1: 0.8, 2: 0.35, 3: 0.15, 4: 0.05}


def compute_root_cohesion_proxy(land_use_code: int, ndvi: float) -> float | None:
    """Returns the root-cohesion proxy (0-1) for a land_use category
    code (1-4) and an NDVI value (0-1), or None if the code isn't
    recognized."""
    base = _LAND_USE_TO_BASE_COHESION.get(land_use_code)
    if base is None:
        return None
    ndvi = min(max(ndvi if ndvi is not None else 0.5, 0.0), 1.0)
    return round(base * (0.5 + 0.5 * ndvi), 4)
