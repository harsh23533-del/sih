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

This is clearly synthetic (random terrain + rainfall + a hand-placed set of
"landslide" points biased toward steep synthetic slopes so the model has a
learnable signal) — see docs/Limitations_and_Assumptions.md. Swap this
script out for the real Week 2 downloaders the moment real data is ready;
nothing downstream needs to change.

Usage:
    python generate_synthetic_data.py --out-dir ../../data
"""
import argparse
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

SIKKIM_BBOX = dict(south=27.00, north=28.13, west=88.00, east=88.93)
YEARS = list(range(2018, 2025))
N_POSITIVES = 120


def make_dem(out_path: str, seed: int = 0, size: int = 300):
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

        import xarray as xr
        ds = xr.Dataset(
            {"rain": (["time", "lat", "lon"], data)},
            coords={"time": dates, "lat": lats, "lon": lons},
        )
        out_path = os.path.join(out_dir, f"sikkim_rainfall_{year}.nc")
        ds.to_netcdf(out_path)
    print(f"Synthetic rainfall NetCDFs saved -> {out_dir}/sikkim_rainfall_{{2018..2024}}.nc")


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


def main(args):
    terrain_dir = os.path.join(args.out_dir, "terrain")
    rainfall_dir = os.path.join(args.out_dir, "rainfall")
    infra_dir = os.path.join(args.out_dir, "infrastructure")
    hist_dir = os.path.join(args.out_dir, "historical_landslides")
    for d in (terrain_dir, rainfall_dir, infra_dir, hist_dir):
        os.makedirs(d, exist_ok=True)

    dem_path = os.path.join(terrain_dir, "sikkim_srtm30m.tif")
    elevation, transform = make_dem(dem_path)
    make_rainfall(rainfall_dir)
    make_roads(os.path.join(infra_dir, "sikkim_osm.geojson"))
    make_positives(os.path.join(hist_dir, "coolr_ner_labeled.csv"), elevation, transform)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="../../data")
    args = parser.parse_args()
    main(args)
