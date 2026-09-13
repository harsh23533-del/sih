"""
Compute multi-window cumulative rainfall features (1/3/7/15/30-day) at a given
lat/lon/date, from the Week 2 IMD NetCDF files (one per year, Sikkim-clipped).
Also derives:
  - rainfall_intensity_mm_hr: peak-hour intensity on the event date, using
    the peak_intensity_frac variable the synthetic generator now writes
    (real IMD sub-daily/AWS gauge data would replace this directly).
  - rainfall_forecast_24h / rainfall_forecast_48h: look-ahead rainfall for
    the next 1-2 days, with injected forecast error — standing in for a
    real NWP/IMD forecast product, which would replace this look-ahead
    with an actual forecast API call.

Requirements:
    pip install xarray pandas numpy netCDF4

Usage:
    from rainfall_features import RainfallExtractor
    re = RainfallExtractor("../data/rainfall")   # folder with sikkim_rainfall_YYYY.nc files
    row = re.sample(27.33, 88.61, "2023-07-14")
"""
import glob
import hashlib
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
        self._has_intensity = "peak_intensity_frac" in self.ds.data_vars

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

        # Peak-hour rainfall intensity on the event date itself.
        if self._has_intensity:
            try:
                today = point.sel(time=target_date, method="nearest")
                rain_today = float(today["rain"].values)
                peak_frac = float(today["peak_intensity_frac"].values)
                result["rainfall_intensity_mm_hr"] = rain_today * peak_frac
            except (KeyError, ValueError):
                result["rainfall_intensity_mm_hr"] = np.nan
        else:
            result["rainfall_intensity_mm_hr"] = np.nan

        # Forecast look-ahead: real next-day(s) rainfall from the same
        # series, with injected forecast error (deterministic per
        # lat/lon/date so re-running gives the same "forecast").
        seed = int(hashlib.md5(f"{lat:.4f}_{lon:.4f}_{date}".encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        for h, days in [("24h", 1), ("48h", 2)]:
            end = target_date + pd.Timedelta(days=days)
            start = target_date + pd.Timedelta(days=1)
            try:
                future_slice = point.sel(time=slice(start, end))
                actual_future = float(future_slice["rain"].sum(skipna=True).values)
                forecast_error = rng.normal(1.0, 0.25)  # imperfect NWP forecast
                result[f"rainfall_forecast_{h}"] = max(0.0, actual_future * forecast_error)
            except (KeyError, ValueError):
                result[f"rainfall_forecast_{h}"] = np.nan

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
