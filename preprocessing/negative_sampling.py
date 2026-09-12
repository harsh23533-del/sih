"""
Generate negative (non-landslide) background samples for training, matched
roughly on terrain and kept a safe distance from known positive events —
avoids the model learning a trivial "near a road/anywhere" decision boundary.

Strategy:
  1. Random-sample candidate points within the DEM's valid extent.
  2. Drop any candidate within `exclusion_radius_km` of a known historical
     landslide (positive event) — prevents label leakage / near-duplicate
     ambiguous points.
  3. Assign each negative a random date within the rainfall data's date range,
     avoiding the ~30 days *after* any known heavy-rain event date if you want
     to be extra conservative (kept simple here: fully random date).
  4. Output count = `neg_to_pos_ratio` x number of positive events (a common
     starting ratio is 1:1 to 1:3 — tune based on validation performance, not
     fixed a priori).

Requirements:
    pip install pandas numpy rasterio

Usage:
    python negative_sampling.py \\
        --dem ../data/terrain/sikkim_srtm30m.tif \\
        --positives ../data/historical_landslides/coolr_ner.csv \\
        --rainfall-start 2015-01-01 --rainfall-end 2025-12-31 \\
        --neg-to-pos-ratio 2 \\
        --exclusion-radius-km 2 \\
        --out ../data/processed/negative_samples.csv
"""
import argparse

import numpy as np
import pandas as pd
import rasterio


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def generate_negatives(dem_path, positives_df, n_needed, exclusion_km,
                        rainfall_start, rainfall_end, seed=42):
    rng = np.random.default_rng(seed)

    with rasterio.open(dem_path) as src:
        bounds = src.bounds  # left, bottom, right, top -> lon/lat bbox
        elevation = src.read(1).astype(float)
        elevation[elevation < -1000] = np.nan
        transform = src.transform

    pos_lat = positives_df["latitude"].values
    pos_lon = positives_df["longitude"].values

    negatives = []
    attempts = 0
    max_attempts = n_needed * 50  # safety cap

    while len(negatives) < n_needed and attempts < max_attempts:
        attempts += 1
        lat = rng.uniform(bounds.bottom, bounds.top)
        lon = rng.uniform(bounds.left, bounds.right)

        # check it's on valid land (not nodata) — sample nearest pixel
        row, col = rasterio.transform.rowcol(transform, lon, lat)
        try:
            elev = elevation[row, col]
        except IndexError:
            continue
        if np.isnan(elev):
            continue

        # exclusion buffer from all known positives
        d = haversine_km(lat, lon, pos_lat, pos_lon)
        if np.min(d) < exclusion_km:
            continue

        random_date = pd.Timestamp(rainfall_start) + pd.Timedelta(
            days=int(rng.integers(0, (pd.Timestamp(rainfall_end) - pd.Timestamp(rainfall_start)).days))
        )

        negatives.append({
            "latitude": lat, "longitude": lon,
            "date": random_date.strftime("%Y-%m-%d"),
            "label": 0,
        })

    if len(negatives) < n_needed:
        print(f"WARNING: only generated {len(negatives)}/{n_needed} negatives "
              f"after {max_attempts} attempts — consider lowering exclusion-radius-km.")

    return pd.DataFrame(negatives)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", required=True)
    parser.add_argument("--positives", required=True, help="CSV with latitude/longitude of known landslides")
    parser.add_argument("--rainfall-start", required=True)
    parser.add_argument("--rainfall-end", required=True)
    parser.add_argument("--neg-to-pos-ratio", type=float, default=2.0)
    parser.add_argument("--exclusion-radius-km", type=float, default=2.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pos_df = pd.read_csv(args.positives)
    n_needed = int(len(pos_df) * args.neg_to_pos_ratio)

    neg_df = generate_negatives(
        args.dem, pos_df, n_needed, args.exclusion_radius_km,
        args.rainfall_start, args.rainfall_end
    )
    neg_df.to_csv(args.out, index=False)
    print(f"Saved {len(neg_df)} negative samples -> {args.out}")
