"""
Live insolation proxy -- pure geometry, no network call needed.

Uses the project's own original formula from
preprocessing/environmental_features.py (so the live and training-time
definitions -- and scales -- agree, which matters because the models
were trained on this 0-1 range):

    south_facing = cos(radians(aspect - 180))   # 1 = due south, -1 = due north
    slope_factor = clip(slope / 45, 0, 1)
    insolation_proxy = (south_facing + 1) / 2 * 0.5 + slope_factor * 0.5

Both inputs (slope, aspect) come from live_terrain.py, so this is "live"
for free once terrain is live -- no extra API call needed.
"""


def compute_insolation_proxy(slope_deg: float, aspect_deg: float) -> float:
    """Returns the insolation proxy (0-1) for this point. Never fails
    (pure arithmetic)."""
    import math

    south_facing = math.cos(math.radians(aspect_deg - 180))
    slope_factor = min(max(slope_deg / 45.0, 0.0), 1.0)
    value = (south_facing + 1) / 2 * 0.5 + slope_factor * 0.5
    return round(min(max(value, 0.0), 1.0), 4)
