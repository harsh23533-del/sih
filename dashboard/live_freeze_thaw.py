"""
Live freeze-thaw index -- pure math, no network call needed.

Uses the project's own original formula from
preprocessing/environmental_features.py (so live and training-time
definitions -- and scales -- agree):

    freeze_thaw_index = 1 / (1 + exp(-(elevation - 3500) / 300))

This is a logistic curve centered on ~3500m (the rough elevation band
where regular freeze-thaw cycling becomes significant in this region),
scaled to 0-1. elevation comes from live_terrain.py, so this is "live"
for free once terrain is live -- no extra API call needed.
"""
import math


def compute_freeze_thaw_index(elevation_m: float) -> float:
    """Returns the freeze-thaw index (0-1) for this elevation. Never
    fails (pure arithmetic)."""
    value = 1 / (1 + math.exp(-(elevation_m - 3500) / 300))
    return round(value, 4)
