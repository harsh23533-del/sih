"""
Synthetic data generator — stands in for Week 2's real download_dem.py /
download_rainfall.py / download_osm_infra.py / fetch_nasa_coolr.py so the
full pipeline (features -> training -> risk engine -> dashboard) can be
built and validated end-to-end BEFORE real data is available.

Produces files in the exact formats the downstream scripts expect:
  data/terrain/sikkim_srtm30m.tif           (DEM GeoTIFF)
  data/rainfall/sikkim_rainfall_YYYY.nc     (one NetCDF per year, 2018-2024)
  data/infrastructure/sikkim_osm.geojson    (road LineStrings)
  data/historical_landslides/coolr_ner_labeled.csv  (positive events, label=1)

  -- literature-standard factors added on top of the original prototype --
  data/soil_geology/sikkim_soil.tif           (soil type, categorical)
  data/soil_geology/sikkim_geology.tif        (lithology, categorical)
  data/soil_geology/sikkim_faults.geojson     (fault traces, for fault_distance)
  data/soil_geology/sikkim_seismic_pga.tif    (peak ground acceleration proxy)
  data/soil_geology/sikkim_soil_moisture.tif  (baseline wetness/groundwater proxy)
  data/satellite/sikkim_ndvi.tif              (vegetation index, 0-1)
  data/satellite/sikkim_landuse.tif           (land use/cover, categorical)

  -- second round: hydrology, forecast/intensity rainfall, InSAR, exposure --
  data/terrain/sikkim_twi.tif                    (Topographic Wetness Index)
  data/terrain/sikkim_drainage_distance.tif      (distance to nearest stream, km)
  data/satellite/sikkim_soil_moisture_sat.tif    (satellite soil-moisture proxy)
  data/satellite/sikkim_insar_deformation.tif    (ground deformation, mm/yr)
  data/historical_landslides/sikkim_glacial_lakes.geojson (GLOF risk points)
  data/infrastructure/sikkim_population_density.tif       (exposure layer)
  rainfall NetCDFs now also carry a per-day "peak_intensity_frac" variable,
  used to derive rainfall_intensity_mm_hr, plus rainfall_forecast_24h/48h are
  computed at feature-build time by looking ahead in the same time series
  (with injected forecast error) — standing in for a real NWP rainfall
  forecast product.

Real-data swap-in map (see docs/Limitations_and_Assumptions.md):
  soil/geology       -> Bhuvan / GSI geology & soil maps
  seismic PGA        -> BIS seismic zonation (NER is mostly Zone V) / NDMA
  soil moisture       -> CGWB groundwater level data + antecedent rainfall
  NDVI / land use    -> Sentinel-2 (Copernicus) or Bhuvan LULC products
  fault_distance      -> GSI active fault database
  TWI / drainage      -> derived from a real DEM the same way (no swap needed)
  rainfall intensity/forecast -> IMD sub-daily gauges / IMD-NWP or ECMWF forecast API
  satellite soil moisture -> ISRO Bhuvan / NASA SMAP
  InSAR deformation   -> ISRO NISAR (upcoming) / ESA Sentinel-1 InSAR products
  population density  -> WorldPop / Census of India gridded population

This is clearly synthetic (random terrain + rainfall + a hand-placed set of
"landslide" points biased toward steep synthetic slopes so the model has a
learnable signal) — see docs/Limitations_and_Assumptions.md. Swap this
script out for the real Week 2 downloaders the moment real data is ready;
nothing downstream needs to change.

Usage:
    python generate_synthetic_data.py --out-dir ../../data
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter, distance_transform_edt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hydrology_features import compute_flow_accumulation, compute_twi, compute_drainage_distance_km

SIKKIM_BBOX = dict(south=27.00, north=28.13, west=88.00, east=88.93)
YEARS = list(range(2018, 2025))
N_POSITIVES = 120


def make_dem(out_path: str, seed: int = 0, size: int = 400):
    rng = np.random.default_rng(seed)
    b = SIKKIM_BBOX

    # Fractal-ish terrain: sum of a few random sine fields + noise, scaled to
    # a plausible Sikkim elevation range (~250m valleys to ~5500m peaks).
    y, x = np.mgrid[0:size, 0:size]
    elevation = np.zeros((size, size))
    for k in (2, 5, 11, 23):
        elevation += (1.0 / k) * np.sin(x / size * k * np.pi) * np.cos(y / size * k * np.pi)
    elevation += rng.normal(0, 0.05, size=(size, size))
    elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())
    elevation = 250 + elevation * 5250  # ~250m to ~5500m

    px_size_x = (b["east"] - b["west"]) / size
    px_size_y = (b["north"] - b["south"]) / size
    transform = from_origin(b["west"], b["north"], px_size_x, px_size_y)

    with rasterio.open(
        out_path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype=elevation.dtype, crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(elevation, 1)
    print(f"Synthetic DEM saved -> {out_path} ({size}x{size}, "
          f"{elevation.min():.0f}-{elevation.max():.0f}m)")
    return elevation, transform


def make_rainfall(out_dir: str, seed: int = 1, grid: int = 15):
    rng = np.random.default_rng(seed)
    b = SIKKIM_BBOX
    lats = np.linspace(b["south"], b["north"], grid)
    lons = np.linspace(b["west"], b["east"], grid)

    for year in YEARS:
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        # Monsoon (Jun-Sep) gets much higher mean daily rainfall, per NER climatology.
        month = dates.month.values
        base = np.where(np.isin(month, [6, 7, 8, 9]), 18.0, 3.0)
        rain = np.stack([
            rng.gamma(shape=1.2, scale=b_) if b_ > 0 else np.zeros_like(base)
            for b_ in base
        ]) if False else None

        # Build (time, lat, lon) rainfall with gamma noise per day, broadcast
        # over the grid with a little spatial variation.
        daily = rng.gamma(shape=1.2, scale=np.maximum(base, 0.1))  # (time,)
        spatial_noise = rng.normal(1.0, 0.15, size=(len(dates), grid, grid)).clip(0.3, 2.0)
        data = daily[:, None, None] * spatial_noise

        # Peak-hour intensity fraction: what share of the day's total rain
        # fell in its single heaviest hour. Monsoon convective bursts run
        # higher than steady winter drizzle — this is what lets us derive
        # rainfall_intensity_mm_hr downstream instead of only daily totals.
        monsoon_mask = np.isin(month, [6, 7, 8, 9])
        peak_frac_mean = np.where(monsoon_mask, 0.28, 0.16)
        peak_frac = rng.beta(a=4, b=8, size=(len(dates), grid, grid))
        peak_frac = (peak_frac * (peak_frac_mean[:, None, None] / 0.27)).clip(0.05, 0.6)

        import xarray as xr
        ds = xr.Dataset(
            {
                "rain": (["time", "lat", "lon"], data),
                "peak_intensity_frac": (["time", "lat", "lon"], peak_frac),
            },
            coords={"time": dates, "lat": lats, "lon": lons},
        )
        out_path = os.path.join(out_dir, f"sikkim_rainfall_{year}.nc")
        ds.to_netcdf(out_path)
    print(f"Synthetic rainfall NetCDFs saved -> {out_dir}/sikkim_rainfall_{{2018..2024}}.nc "
          f"(with rain + peak_intensity_frac variables)")


def make_roads(out_path: str, seed: int = 2, n_roads: int = 40):
    rng = np.random.default_rng(seed)
    b = SIKKIM_BBOX
    features = []
    for i in range(n_roads):
        lat0 = rng.uniform(b["south"], b["north"])
        lon0 = rng.uniform(b["west"], b["east"])
        length = rng.uniform(0.02, 0.12)
        angle = rng.uniform(0, 2 * np.pi)
        lat1 = np.clip(lat0 + length * np.sin(angle), b["south"], b["north"])
        lon1 = np.clip(lon0 + length * np.cos(angle), b["west"], b["east"])
        features.append({
            "type": "Feature",
            "properties": {"highway": "unclassified", "id": i},
            "geometry": {"type": "LineString", "coordinates": [[lon0, lat0], [lon1, lat1]]},
        })
    geojson = {"type": "FeatureCollection", "features": features}
    import json
    with open(out_path, "w") as f:
        json.dump(geojson, f)
    print(f"Synthetic OSM roads saved -> {out_path} ({n_roads} segments)")


def make_positives(out_path: str, elevation: np.ndarray, transform, seed: int = 3):
    """Places landslide points biased toward higher/steeper synthetic terrain
    and monsoon dates, so the model has a real learnable pattern."""
    rng = np.random.default_rng(seed)
    b = SIKKIM_BBOX
    size = elevation.shape[0]

    # Slope proxy via elevation gradient, to bias sampling toward steep areas
    gy, gx = np.gradient(elevation)
    slope_proxy = np.sqrt(gx ** 2 + gy ** 2)
    weights = slope_proxy.flatten()
    weights = weights / weights.sum()

    rows_cols = rng.choice(size * size, size=N_POSITIVES, p=weights, replace=True)
    rows = rows_cols // size
    cols = rows_cols % size

    records = []
    for r, c in zip(rows, cols):
        lon, lat = rasterio.transform.xy(transform, r, c)
        year = int(rng.choice(YEARS))
        month = int(rng.choice([6, 7, 7, 8, 8, 9, 5, 10]))  # monsoon-weighted
        day = int(rng.integers(1, 28))
        records.append({
            "latitude": lat, "longitude": lon,
            "date": f"{year}-{month:02d}-{day:02d}", "label": 1,
        })
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"Synthetic positive landslide events saved -> {out_path} ({len(records)} events)")


def _write_raster(out_path: str, array: np.ndarray, transform, dtype):
    with rasterio.open(
        out_path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=1, dtype=dtype, crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(array.astype(dtype), 1)


def make_faults(out_path: str, seed: int = 10, n_faults: int = 5):
    """A handful of long fault traces — same LineString pattern as
    make_roads(), just fewer and longer, to stand in for GSI's active
    fault database."""
    rng = np.random.default_rng(seed)
    b = SIKKIM_BBOX
    features = []
    for i in range(n_faults):
        lat0 = rng.uniform(b["south"], b["north"])
        lon0 = rng.uniform(b["west"], b["east"])
        length = rng.uniform(0.3, 0.9)
        angle = rng.uniform(0, 2 * np.pi)
        lat1 = np.clip(lat0 + length * np.sin(angle), b["south"], b["north"])
        lon1 = np.clip(lon0 + length * np.cos(angle), b["west"], b["east"])
        features.append({
            "type": "Feature",
            "properties": {"fault_type": "synthetic", "id": i},
            "geometry": {"type": "LineString", "coordinates": [[lon0, lat0], [lon1, lat1]]},
        })
    geojson = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w") as f:
        json.dump(geojson, f)
    print(f"Synthetic fault traces saved -> {out_path} ({n_faults} traces)")
    return features


def _fault_distance_raster_km(fault_features: list, size: int, transform):
    """Rasterizes fault traces onto the DEM grid and returns a per-pixel
    distance-to-nearest-fault grid in km, via a Euclidean distance
    transform (fast, avoids a geopandas dependency for this synthetic
    step)."""
    mask = np.ones((size, size), dtype=bool)
    for feat in fault_features:
        (lon0, lat0), (lon1, lat1) = feat["geometry"]["coordinates"]
        n_pts = 200
        for t in np.linspace(0, 1, n_pts):
            lon, lat = lon0 + t * (lon1 - lon0), lat0 + t * (lat1 - lat0)
            row, col = rasterio.transform.rowcol(transform, lon, lat)
            if 0 <= row < size and 0 <= col < size:
                mask[row, col] = False
    px_km = (SIKKIM_BBOX["east"] - SIKKIM_BBOX["west"]) / size * 111.0  # ~km per pixel
    dist_px = distance_transform_edt(mask)
    return dist_px * px_km


def make_soil_geology(out_dir: str, elevation: np.ndarray, transform, seed: int = 20):
    """Soil type + lithology (geology) as categorical rasters, spatially
    coherent with elevation/terrain rather than pure noise — plus fault
    traces, a seismic PGA proxy (boosted near faults, since NER sits in
    BIS Seismic Zone V), and a baseline soil-moisture/wetness index
    (valleys retain more moisture than ridges)."""
    rng = np.random.default_rng(seed)
    size = elevation.shape[0]

    # --- Soil type: 1=Rocky/Skeletal, 2=Sandy, 3=Loamy, 4=Clayey/Alluvial ---
    # Higher/steeper terrain -> thinner, rockier soil; valleys -> deeper alluvial soil.
    elev_norm = (elevation - elevation.min()) / (elevation.max() - elevation.min())
    noise = rng.normal(0, 0.08, size=(size, size))
    soil_score = elev_norm + noise
    soil = np.digitize(soil_score, bins=[0.25, 0.5, 0.75]) + 1  # -> 1..4
    soil = soil.astype(np.uint8)

    # --- Geology / lithology: spatially-clustered patches via a smoothed
    # random field, standing in for real rock-type boundaries.
    # 1=Gneiss 2=Schist 3=Phyllite 4=Quartzite 5=Alluvium
    raw = rng.normal(size=(size, size))
    smooth = gaussian_filter(raw, sigma=size / 12)
    geology = np.digitize(smooth, bins=np.quantile(smooth, [0.2, 0.4, 0.6, 0.8])) + 1
    geology = geology.astype(np.uint8)

    # --- Fault traces + distance-based seismic PGA proxy ---
    fault_features = make_faults(os.path.join(out_dir, "sikkim_faults.geojson"), seed=seed + 1)
    fault_dist_km = _fault_distance_raster_km(fault_features, size, transform)
    # NER sits mostly in BIS Seismic Zone V (PGA ~0.24-0.36g baseline);
    # add a boost close to fault traces plus a little spatial noise.
    seismic_pga = 0.24 + 0.15 * np.exp(-fault_dist_km / 8.0) + rng.normal(0, 0.01, size=(size, size))
    seismic_pga = seismic_pga.clip(0.15, 0.5)

    # --- Baseline soil moisture / groundwater-proxy wetness index (0-1) ---
    # Valleys (concave, lower than their surroundings) hold more moisture
    # than ridges — approximate via elevation minus a heavily-smoothed
    # version of itself (a cheap relief/concavity proxy).
    local_relief = elevation - gaussian_filter(elevation, sigma=size / 15)
    wetness = -local_relief
    wetness = (wetness - wetness.min()) / (wetness.max() - wetness.min())

    transforms_and_arrays = [
        ("sikkim_soil.tif", soil, "uint8"),
        ("sikkim_geology.tif", geology, "uint8"),
        ("sikkim_seismic_pga.tif", seismic_pga, "float32"),
        ("sikkim_soil_moisture.tif", wetness, "float32"),
    ]
    for fname, arr, dtype in transforms_and_arrays:
        _write_raster(os.path.join(out_dir, fname), arr, transform, dtype)
    print(f"Synthetic soil/geology/seismic/moisture rasters saved -> {out_dir}/")
    return geology


def make_satellite(out_dir: str, elevation: np.ndarray, transform, geology: np.ndarray,
                    seed: int = 30):
    """NDVI (vegetation) + land use/cover, correlated with elevation (alpine
    zone above ~3500m is sparsely vegetated) with a few random
    deforestation/agriculture/urban patches layered on top, standing in
    for a real Sentinel-2 / Bhuvan LULC extraction. Also writes the
    satellite soil-moisture and InSAR ground-deformation layers, which
    both need the same elevation/geology inputs.

    Returns the land_use array (needed downstream for population density)."""
    rng = np.random.default_rng(seed)
    size = elevation.shape[0]

    # Base NDVI: dense vegetation in the mid-elevation forest belt,
    # tapering off toward the alpine/snow zone above ~3500m.
    treeline = 3500.0
    base_ndvi = np.where(
        elevation < treeline,
        0.55 + 0.3 * (1 - np.abs(elevation - 1800) / 1800).clip(0, 1),
        0.35 * np.exp(-(elevation - treeline) / 800),
    )
    base_ndvi += rng.normal(0, 0.05, size=(size, size))

    # A handful of low-NDVI patches for deforestation/agriculture/urban clearings.
    landuse = np.ones((size, size), dtype=np.uint8)  # 1 = Forest (default)
    n_patches = 25
    for _ in range(n_patches):
        cy, cx = rng.integers(0, size), rng.integers(0, size)
        radius = rng.integers(4, 14)
        yy, xx = np.ogrid[:size, :size]
        patch = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        kind = rng.choice([2, 3, 4], p=[0.55, 0.15, 0.30])  # Agri / Urban / Barren
        landuse[patch] = kind
        ndvi_drop = {2: 0.25, 3: 0.45, 4: 0.35}[kind]
        base_ndvi[patch] -= ndvi_drop

    ndvi = base_ndvi.clip(0.0, 1.0)

    _write_raster(os.path.join(out_dir, "sikkim_ndvi.tif"), ndvi, transform, "float32")
    _write_raster(os.path.join(out_dir, "sikkim_landuse.tif"), landuse, transform, "uint8")
    print(f"Synthetic NDVI + land-use rasters saved -> {out_dir}/ "
          f"(land use codes: 1=Forest 2=Agriculture 3=Urban 4=Barren/Snow)")

    from scipy.ndimage import sobel
    px_size_x = abs(transform.a)
    px_size_y = abs(transform.e)
    dz_dx = sobel(elevation, axis=1) / (8 * px_size_x)
    dz_dy = sobel(elevation, axis=0) / (8 * px_size_y)
    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))

    deformation, sat_moisture = make_insar_and_satellite_moisture(elevation, geology, slope_deg)
    _write_raster(os.path.join(out_dir, "sikkim_insar_deformation.tif"), deformation,
                   transform, "float32")
    _write_raster(os.path.join(out_dir, "sikkim_soil_moisture_sat.tif"), sat_moisture,
                   transform, "float32")
    print(f"Synthetic InSAR deformation + satellite soil-moisture rasters saved -> {out_dir}/")

    return landuse


def make_hydrology(terrain_dir: str, dem_path: str, elevation: np.ndarray, transform):
    """TWI + drainage distance, purely derived from the DEM already
    generated — no extra external data source needed, real or synthetic."""
    from scipy.ndimage import sobel
    px_size_x = abs(transform.a)
    px_size_y = abs(transform.e)
    dz_dx = sobel(elevation, axis=1) / (8 * px_size_x)
    dz_dy = sobel(elevation, axis=0) / (8 * px_size_y)
    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))

    # Degrees-per-pixel -> rough meters-per-pixel at this latitude, for the
    # flow-accumulation / TWI formulas (approximate, fine for synthetic data).
    px_size_m = px_size_x * 111_000

    flow_accum = compute_flow_accumulation(elevation)
    twi = compute_twi(elevation, slope_deg, flow_accum, pixel_size_m=px_size_m)
    drainage_km = compute_drainage_distance_km(flow_accum, pixel_size_m=px_size_m)

    _write_raster(os.path.join(terrain_dir, "sikkim_twi.tif"), twi, transform, "float32")
    _write_raster(os.path.join(terrain_dir, "sikkim_drainage_distance.tif"), drainage_km,
                   transform, "float32")
    print(f"Synthetic TWI + drainage-distance rasters saved -> {terrain_dir}/ "
          f"(from D8 flow accumulation on the DEM)")


def make_insar_and_satellite_moisture(elevation: np.ndarray, geology: np.ndarray,
                                       slope_deg: np.ndarray, seed: int = 40):
    """Ground-deformation (InSAR-style) proxy and a *separate* satellite
    soil-moisture layer distinct from the ground-based baseline — coarser
    and noisier, the way a real remote-sensing product would be. Returns
    the two arrays; the caller writes them as rasters."""
    rng = np.random.default_rng(seed)
    size = elevation.shape[0]

    # Weaker/younger lithology (higher geology code, by our synthetic
    # convention) + steeper slope -> more creep/deformation.
    geology_factor = (geology.astype(float) - 1) / 4.0  # 0..1
    slope_factor = np.clip(slope_deg / 45.0, 0, 1)
    deformation = 2.0 + 10.0 * (0.5 * geology_factor + 0.5 * slope_factor)
    deformation += rng.normal(0, 1.0, size=(size, size))
    deformation = deformation.clip(0, None)

    # Satellite soil moisture: same rough spatial pattern as ground wetness
    # would have, but resampled coarser (blurred more) and with its own
    # sensor noise, distinct from the ground-station-style baseline.
    raw = rng.normal(size=(size, size))
    coarse = gaussian_filter(raw, sigma=size / 8)
    sat_moisture = (coarse - coarse.min()) / (coarse.max() - coarse.min())
    sat_moisture = (0.7 * sat_moisture + 0.3 * rng.uniform(0, 1, size=(size, size))).clip(0, 1)

    return deformation, sat_moisture


def make_glacial_lakes(out_path: str, elevation: np.ndarray, transform, seed: int = 50, n_lakes: int = 4):
    """A handful of high-elevation glacial lake points (>4200m), the way
    South Lhonak actually sits — for a GLOF (glacial lake outburst flood)
    proximity feature, a distinct hazard mechanism from rainfall-triggered
    slope failure."""
    rng = np.random.default_rng(seed)
    size = elevation.shape[0]
    high_rows, high_cols = np.where(elevation > 4200)
    if len(high_rows) == 0:
        high_rows, high_cols = np.where(elevation > np.percentile(elevation, 95))

    idx = rng.choice(len(high_rows), size=min(n_lakes, len(high_rows)), replace=False)
    features = []
    for i in idx:
        lon, lat = rasterio.transform.xy(transform, int(high_rows[i]), int(high_cols[i]))
        features.append({
            "type": "Feature",
            "properties": {"name": f"synthetic_glacial_lake_{i}"},
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
        })
    geojson = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w") as f:
        json.dump(geojson, f)
    print(f"Synthetic glacial lake points saved -> {out_path} ({len(features)} lakes)")


def make_population_density(out_path: str, land_use: np.ndarray, transform, seed: int = 60):
    """Population density proxy, tied to land use (urban patches dense,
    agriculture moderate, forest/barren sparse) — the "exposure" half of
    Risk = Hazard x Exposure, not just where the ground is unstable but
    where that matters for people."""
    rng = np.random.default_rng(seed)
    base = {1: 5, 2: 60, 3: 800, 4: 1}  # people per sq km, by land_use code
    density = np.vectorize(base.get)(land_use).astype(float)
    density *= rng.uniform(0.7, 1.3, size=land_use.shape)
    _write_raster(out_path, density, transform, "float32")
    print(f"Synthetic population-density raster saved -> {out_path}")


def main(args):
    terrain_dir = os.path.join(args.out_dir, "terrain")
    rainfall_dir = os.path.join(args.out_dir, "rainfall")
    infra_dir = os.path.join(args.out_dir, "infrastructure")
    hist_dir = os.path.join(args.out_dir, "historical_landslides")
    soil_geo_dir = os.path.join(args.out_dir, "soil_geology")
    satellite_dir = os.path.join(args.out_dir, "satellite")
    for d in (terrain_dir, rainfall_dir, infra_dir, hist_dir, soil_geo_dir, satellite_dir):
        os.makedirs(d, exist_ok=True)

    dem_path = os.path.join(terrain_dir, "sikkim_srtm30m.tif")
    elevation, transform = make_dem(dem_path)
    make_rainfall(rainfall_dir)
    make_roads(os.path.join(infra_dir, "sikkim_osm.geojson"))
    make_positives(os.path.join(hist_dir, "coolr_ner_labeled.csv"), elevation, transform)
    geology = make_soil_geology(soil_geo_dir, elevation, transform)
    make_hydrology(terrain_dir, dem_path, elevation, transform)
    land_use = make_satellite(satellite_dir, elevation, transform, geology)
    make_glacial_lakes(os.path.join(hist_dir, "sikkim_glacial_lakes.geojson"), elevation, transform)
    make_population_density(os.path.join(infra_dir, "sikkim_population_density.tif"),
                             land_use, transform)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="../../data")
    args = parser.parse_args()
    main(args)
