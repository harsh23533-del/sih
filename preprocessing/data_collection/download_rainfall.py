"""
Download IMD gridded daily rainfall data and clip to Sikkim bounding box.

Requirements:
    pip install imdlib xarray

Usage:
    python download_rainfall.py --start 2015 --end 2025 --out ../data/rainfall
"""
import argparse
import os

import imdlib as imd


SIKKIM_BBOX = dict(min_lat=27.00, max_lat=28.13, min_lon=88.00, max_lon=88.93)


def download_and_clip(start_year: int, end_year: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    for year in range(start_year, end_year + 1):
        print(f"Downloading IMD rainfall grid for {year}...")
        # 'rain' variable, downloads to out_dir as .grd + .txt
        data = imd.get_data("rain", year, year, fn_format="yearwise", file_dir=out_dir)

        # Convert to xarray and clip to Sikkim bbox
        ds = data.get_xarray()
        ds_sikkim = ds.sel(
            lat=slice(SIKKIM_BBOX["min_lat"], SIKKIM_BBOX["max_lat"]),
            lon=slice(SIKKIM_BBOX["min_lon"], SIKKIM_BBOX["max_lon"]),
        )
        out_path = os.path.join(out_dir, f"sikkim_rainfall_{year}.nc")
        ds_sikkim.to_netcdf(out_path)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2015)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--out", type=str, default="../data/rainfall")
    args = parser.parse_args()

    download_and_clip(args.start, args.end, args.out)
