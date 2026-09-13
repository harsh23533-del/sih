"""
Live exposure index -- pure math, no network call needed.

Uses the project's own original formula from
preprocessing/environmental_features.py (so live and training-time
definitions -- and scales -- agree):

    pop_norm = (population_density - pop_min) / (pop_max - pop_min)
    accessibility = 1 / (1 + road_distance_m / 1000)
    exposure_index = 0.7 * pop_norm + 0.3 * accessibility

population_density comes from live_population.py, road_distance from
live_terrain.py -- both already live, so this is "live" for free.
pop_min/pop_max are the training dataset's own population_density range
(passed in by the caller, e.g. from final_feature_table.csv), matching
how the synthetic version was originally normalized, so a live estimate
outside that historical range is clipped rather than distorting the 0-1
scale the model expects.
"""


def compute_exposure_index(population_density: float, road_distance_m: float,
                            pop_min: float, pop_max: float) -> float:
    """Returns the exposure index (0-1) for this point. Never fails
    (pure arithmetic, guards against pop_max == pop_min)."""
    span = pop_max - pop_min
    pop_norm = (population_density - pop_min) / span if span else 0.0
    pop_norm = min(max(pop_norm, 0.0), 1.0)
    accessibility = 1 / (1 + (road_distance_m or 0.0) / 1000.0)
    value = 0.7 * pop_norm + 0.3 * accessibility
    return round(min(max(value, 0.0), 1.0), 4)
