"""
Master pipeline for Week 3: combines positive (historical landslide) events and
generated negative/background samples, runs terrain + rainfall + density/distance
feature extraction on all of them, cleans missing values, and writes the final
combined training table matching the schema from the project plan:

latitude, longitude, date, rainfall_1d, rainfall_3d, rainfall_7d, rainfall_15d,
rainfall_30d, elevation, slope, aspect, curvature, soil, geology, NDVI,
road_distance, drainage_distance, historical_landslide_density, label

Run this AFTER:
  1. negative_sampling.py        -> negative_samples.csv
  2. (positives already have latitude/longitude/date/label=1 from Week 2's
     coolr_ner.csv / GSI Bhukosh export — add a `label` column of 1s if missing)

Requirements:
    pip install pandas numpy

Usage:
    python build_feature_table.py \\
        --positives ../data/historical_landslides/coolr_ner_labeled.csv \\
        --negatives ../data/processed/negative_samples.csv \\
        --dem ../data/terrain/sikkim_srtm30m.tif \\
        --rainfall-dir ../data/rainfall \\
        --osm-roads ../data/infrastructure/sikkim_osm.geojson \\
        --out ../data/processed/final_feature_table.csv
"""
import argparse

import numpy as np
import pandas as pd

from terrain_features import TerrainExtractor
from rainfall_features import RainfallExtractor
from density_distance_features import add_landslide_density_and_distance, add_road_distance


def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # Drop rows with no coordinates or date — unusable
    df = df.dropna(subset=["latitude", "longitude", "date"])

    # For numeric feature columns, missing values are usually a DEM/rainfall
    # nodata edge case — impute with the column median rather than dropping,
    # since dropping would bias the (already rare) positive class
    numeric_cols = [
        "elevation", "slope", "aspect", "curvature", "ruggedness_tri",
        "rainfall_1d", "rainfall_3d", "rainfall_7d", "rainfall_15d", "rainfall_30d",
        "road_distance", "historical_landslide_density",
    ]
    for col in numeric_cols:
        if col in df.columns:
            median = df[col].median()
            df[col] = df[col].fillna(median)

    # Drop exact duplicate lat/lon/date rows
    df = df.drop_duplicates(subset=["latitude", "longitude", "date"])

    after = len(df)
    print(f"Cleaning: {before} -> {after} rows ({before - after} dropped)")
    return df


def main(args):
    pos_df = pd.read_csv(args.positives)
    neg_df = pd.read_csv(args.negatives)

    if "label" not in pos_df.columns:
        pos_df["label"] = 1

    combined = pd.concat([pos_df, neg_df], ignore_index=True)
    print(f"Combined {len(pos_df)} positives + {len(neg_df)} negatives = {len(combined)} rows")

    # 1. Terrain features
    te = TerrainExtractor(args.dem)
    terrain_feats = combined.apply(
        lambda r: te.sample(r["latitude"], r["longitude"]), axis=1, result_type="expand"
    )
    combined = pd.concat([combined, terrain_feats], axis=1)

    # 2. Rainfall window features
    rf = RainfallExtractor(args.rainfall_dir)
    rain_feats = combined.apply(
        lambda r: rf.sample(r["latitude"], r["longitude"], r["date"]), axis=1, result_type="expand"
    )
    combined = pd.concat([combined, rain_feats], axis=1)

    # 3. Historical landslide density + distance (use the FULL positive set as reference)
    combined = add_landslide_density_and_distance(
        combined, pos_df["latitude"].values, pos_df["longitude"].values, radius_km=5.0
    )

    # 4. Road distance
    combined = add_road_distance(combined, args.osm_roads)

    # 5. Clean
    combined = clean_missing(combined)

    # NOTE: soil, geology, NDVI columns are left as placeholders here — join those
    # in separately once you've extracted them from Bhuvan/GSI/Sentinel-2 rasters,
    # using the same TerrainExtractor-style point-sampling pattern.
    for col in ["soil", "geology", "NDVI"]:
        if col not in combined.columns:
            combined[col] = np.nan

    final_cols = [
        "latitude", "longitude", "date",
        "rainfall_1d", "rainfall_3d", "rainfall_7d", "rainfall_15d", "rainfall_30d",
        "elevation", "slope", "aspect", "curvature", "ruggedness_tri",
        "soil", "geology", "NDVI",
        "road_distance", "historical_landslide_density", "distance_to_nearest_landslide_m",
        "label",
    ]
    combined = combined[[c for c in final_cols if c in combined.columns]]
    combined.to_csv(args.out, index=False)
    print(f"Final feature table saved -> {args.out} ({len(combined)} rows, "
          f"{combined['label'].sum()} positive / {(combined['label'] == 0).sum()} negative)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--positives", required=True)
    parser.add_argument("--negatives", required=True)
    parser.add_argument("--dem", required=True)
    parser.add_argument("--rainfall-dir", required=True)
    parser.add_argument("--osm-roads", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args)
