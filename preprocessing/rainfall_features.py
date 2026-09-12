"""
Compute multi-window cumulative rainfall features (1/3/7/15/30-day) at a given
lat/lon/date, from the Week 2 IMD NetCDF files (one per year, Sikkim-clipped).

Requirements:
    pip install xarray pandas numpy netCDF4

Usage:
    from rainfall_features import RainfallExtractor
    re = RainfallExtractor("../data/rainfall")   # folder with sikkim_rainfall_YYYY.nc files
    row = re.sample(27.33, 88.61, "2023-07-14")
"""
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

WINDOWS = [1, 3, 7, 15, 30]


class RainfallExtractor:
    def __init__(self, rainfall_dir: str):
        files = sorted(glob.glob(os.path.join(rainfall_dir, "sikkim_rainfall_*.nc")))
        if not files:
            raise FileNotFoundError(f"No sikkim_rainfall_*.nc files found in {rainfall_dir}")
        # Combine all years into one time-indexed dataset
        self.ds = xr.open_mfdataset(files, combine="by_coords")

    def sample(self, lat: float, lon: float, date: str) -> dict:
        """date: 'YYYY-MM-DD' — the event/observation date."""
        target_date = pd.Timestamp(date)
        point = self.ds.sel(lat=lat, lon=lon, method="nearest")

        result = {}
        for w in WINDOWS:
            start = target_date - pd.Timedelta(days=w - 1)
            window_slice = point.sel(time=slice(start, target_date))
            cumulative = float(window_slice["rain"].sum(skipna=True).values)
            result[f"rainfall_{w}d"] = cumulative

        return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rainfall-dir", required=True)
    parser.add_argument("--points", required=True, help="CSV with 'latitude','longitude','date' columns")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    re_ = RainfallExtractor(args.rainfall_dir)
    df = pd.read_csv(args.points)
    feats = df.apply(
        lambda r: re_.sample(r["latitude"], r["longitude"], r["date"]),
        axis=1, result_type="expand"
    )
    out = pd.concat([df, feats], axis=1)
    out.to_csv(args.out, index=False)
    print(f"Saved {len(out)} rows with rainfall window features -> {args.out}")
