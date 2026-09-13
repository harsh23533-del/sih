"""
Hydrology-derived terrain features — Topographic Wetness Index (TWI) and
drainage distance — computed directly from the DEM via a simple D8 flow
accumulation, no extra external data needed. Landslide-susceptibility
literature generally ranks TWI/flow-accumulation-based wetness as at least
as informative as raw slope, since it captures *where water concentrates*
rather than just how steep the ground is.

This runs once at data-generation time (see generate_synthetic_data.py),
producing twi.tif / drainage_distance.tif alongside the DEM — the dashboard
and training pipeline just sample these like any other raster.

Requirements:
    pip install numpy rasterio scipy
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

# (row_offset, col_offset, distance_in_pixels)
_D8_NEIGHBORS = [
    (-1, -1, np.sqrt(2)), (-1, 0, 1), (-1, 1, np.sqrt(2)),
    (0, -1, 1), (0, 1, 1),
    (1, -1, np.sqrt(2)), (1, 0, 1), (1, 1, np.sqrt(2)),
]


def compute_flow_accumulation(elevation: np.ndarray) -> np.ndarray:
    """Simple D8 flow accumulation: every cell drains, in one step, to
    whichever of its 8 neighbors has the steepest downhill gradient; cells
    are then processed from highest to lowest elevation so accumulated
    flow propagates correctly downstream in a single pass."""
    size = elevation.shape[0]
    best_drop = np.full((size, size), -np.inf)
    target_row = np.full((size, size), -1, dtype=int)
    target_col = np.full((size, size), -1, dtype=int)

    for dr, dc, dist in _D8_NEIGHBORS:
        shifted = np.full((size, size), np.nan)
        r0, r1 = max(0, -dr), size - max(0, dr)
        c0, c1 = max(0, -dc), size - max(0, dc)
        shifted[r0:r1, c0:c1] = elevation[r0 + dr:r1 + dr, c0 + dc:c1 + dc]

        drop = (elevation - shifted) / dist
        drop = np.nan_to_num(drop, nan=-np.inf)
        better = drop > best_drop
        best_drop = np.where(better, drop, best_drop)
        rows, cols = np.where(better)
        target_row[rows, cols] = rows + dr
        target_col[rows, cols] = cols + dc

    order = np.argsort(-elevation.flatten())
    accum = np.ones(size * size, dtype=float)
    flat_tr, flat_tc = target_row.flatten(), target_col.flatten()
    for idx in order:
        tr, tc = flat_tr[idx], flat_tc[idx]
        if tr >= 0:
            accum[tr * size + tc] += accum[idx]
    return accum.reshape(size, size)


def compute_twi(elevation: np.ndarray, slope_deg: np.ndarray, flow_accum: np.ndarray,
                 pixel_size_m: float) -> np.ndarray:
    """Topographic Wetness Index: ln(specific catchment area / tan(slope)).
    Higher TWI = water tends to accumulate here (valley bottoms, hollows);
    lower TWI = water drains away quickly (ridges, steep convex slopes)."""
    slope_rad = np.radians(np.clip(slope_deg, 1.0, 89.0))  # avoid tan(0)
    specific_catchment_area = flow_accum * pixel_size_m  # m^2 per unit contour width, approx
    return np.log(specific_catchment_area / np.tan(slope_rad))


def compute_drainage_distance_km(flow_accum: np.ndarray, pixel_size_m: float,
                                  stream_percentile: float = 97.0) -> np.ndarray:
    """Treats the top (100 - stream_percentile)% of flow-accumulation cells
    as the stream/drainage network, then returns each cell's straight-line
    distance (km) to the nearest such cell — a landslide trigger factor via
    bank erosion/undercutting, distinct from road_distance."""
    threshold = np.percentile(flow_accum, stream_percentile)
    non_stream = flow_accum < threshold  # True everywhere except the stream network
    dist_px = distance_transform_edt(non_stream)
    return dist_px * pixel_size_m / 1000.0
