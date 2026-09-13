"""
Master pipeline for Week 3: combines positive (historical landslide) events and
generated negative/background samples, runs terrain + rainfall + density/distance
feature extraction on all of them, cleans missing values, and writes the final
combined training table matching the schema from the project plan:

latitude, longitude, date, rainfall_1d, rainfall_3d, rainfall_7d, rainfall_15d,
rainfall_30d, rainfall_intensity_mm_hr, rainfall_forecast_24h, rainfall_forecast_48h,
elevation, slope, aspect, curvature, twi, drainage_distance_km,
soil, geology, NDVI, land_use, seismic_pga, fault_distance_km,
soil_moisture_index, soil_moisture_satellite, insar_deformation_mm_yr,
insolation_proxy, freeze_thaw_index, root_cohesion_proxy,
road_distance, historical_landslide_density, glacial_lake_distance_km,
population_density, exposure_index, label

Run this AFTER:
  1. negative_sampling.py        -> negative_samples.csv
  2. (positives already have latitude/longitude/date/label=1 from Week 2's
     coolr_ner.csv / GSI Bhukosh export — add a `label` column of 1s if missing)

Requirements:
    pip install pandas numpy rasterio

Usage:
    python build_feature_table.py \\
        --positives ../data/historical_landslides/coolr_ner_labeled.csv \\
        --negatives ../data/processed/negative_samples.csv \\
        --dem ../data/terrain/sikkim_srtm30m.tif \\
        --twi ../data/terrain/sikkim_twi.tif \\
        --drainage ../data/terrain/sikkim_drainage_distance.tif \\
        --rainfall-dir ../data/rainfall \\
        --osm-roads ../data/infrastructure/sikkim_osm.geojson \\
        --soil ../data/soil_geology/sikkim_soil.tif \\
        --geology ../data/soil_geology/sikkim_geology.tif \\
        --seismic ../data/soil_geology/sikkim_seismic_pga.tif \\
        --moisture ../data/soil_geology/sikkim_soil_moisture.tif \\
        --faults ../data/soil_geology/sikkim_faults.geojson \\
        --ndvi ../data/satellite/sikkim_ndvi.tif \\
        --landuse ../data/satellite/sikkim_landuse.tif \\
        --sat-moisture ../data/satellite/sikkim_soil_moisture_sat.tif \\
        --insar ../data/satellite/sikkim_insar_deformation.tif \\
        --population ../data/infrastructure/sikkim_population_density.tif \\
        --glacial-lakes ../data/historical_landslides/sikkim_glacial_lakes.geojson \\
        --out ../data/processed/final_feature_table.csv
"""
import argparse

import numpy as np
import pandas as pd

from terrain_features import TerrainExtractor
from rainfall_features import RainfallExtractor
from density_distance_features import add_landslide_density_and_distance, add_road_distance
from environmental_features import EnvironmentalExtractor, add_derived_environmental_features


def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # Drop rows with no coordinates or date — unusable
    df = df.dropna(subset=["latitude", "longitude", "date"])

    # For numeric feature columns, missing values are usually a DEM/rainfall
    # nodata edge case — impute with the column median rather than dropping,
    # since dropping would bias the (already rare) positive class
    numeric_cols = [
        "elevation", "slope", "aspect", "curvature", "ruggedness_tri",
        "twi", "drainage_distance_km",
        "rainfall_1d", "rainfall_3d", "rainfall_7d", "rainfall_15d", "rainfall_30d",
        "rainfall_intensity_mm_hr", "rainfall_forecast_24h", "rainfall_forecast_48h",
        "road_distance", "historical_landslide_density",
        "soil", "geology", "NDVI", "land_use", "seismic_pga",
        "fault_distance_km", "soil_moisture_baseline",
        "insolation_proxy", "freeze_thaw_index", "soil_moisture_index",
        "soil_moisture_satellite", "insar_deformation_mm_yr", "root_cohesion_proxy",
        "population_density", "exposure_index", "glacial_lake_distance_km",
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

    # 1. Terrain features (+ TWI / drainage distance, if provided)
    te = TerrainExtractor(args.dem, twi_path=args.twi, drainage_path=args.drainage)
    terrain_feats = combined.apply(
        lambda r: te.sample(r["latitude"], r["longitude"]), axis=1, result_type="expand"
    )
    combined = pd.concat([combined, terrain_feats], axis=1)

    # 2. Rainfall window + intensity + forecast features
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

    # 5. Soil / geology / NDVI / land use / seismic / moisture / fault distance
    #    (+ satellite moisture / InSAR / population / glacial lakes, if provided)
    env = EnvironmentalExtractor(
        soil_path=args.soil, geology_path=args.geology, ndvi_path=args.ndvi,
        landuse_path=args.landuse, seismic_path=args.seismic,
        moisture_path=args.moisture, faults_path=args.faults,
        sat_moisture_path=args.sat_moisture, insar_path=args.insar,
        population_path=args.population, glacial_lakes_path=args.glacial_lakes,
    )
    env_feats = combined.apply(
        lambda r: env.sample(r["latitude"], r["longitude"]), axis=1, result_type="expand"
    )
    combined = pd.concat([combined, env_feats], axis=1)

    # 6. Derived environmental features (insolation, freeze-thaw, dynamic soil
    #    moisture, root cohesion, exposure index)
    combined = add_derived_environmental_features(combined)

    # 7. Clean
    combined = clean_missing(combined)

    final_cols = [
        "latitude", "longitude", "date",
        "rainfall_1d", "rainfall_3d", "rainfall_7d", "rainfall_15d", "rainfall_30d",
        "rainfall_intensity_mm_hr", "rainfall_forecast_24h", "rainfall_forecast_48h",
        "elevation", "slope", "aspect", "curvature", "ruggedness_tri",
        "twi", "drainage_distance_km",
        "soil", "geology", "NDVI", "land_use",
        "seismic_pga", "fault_distance_km",
        "soil_moisture_baseline", "soil_moisture_index", "soil_moisture_satellite",
        "insar_deformation_mm_yr",
        "insolation_proxy", "freeze_thaw_index", "root_cohesion_proxy",
        "road_distance", "historical_landslide_density", "distance_to_nearest_landslide_m",
        "glacial_lake_distance_km", "population_density", "exposure_index",
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
    parser.add_argument("--twi", default=None)
    parser.add_argument("--drainage", default=None)
    parser.add_argument("--rainfall-dir", required=True)
    parser.add_argument("--osm-roads", required=True)
    parser.add_argument("--soil", required=True)
    parser.add_argument("--geology", required=True)
    parser.add_argument("--ndvi", required=True)
    parser.add_argument("--landuse", required=True)
    parser.add_argument("--seismic", required=True)
    parser.add_argument("--moisture", required=True)
    parser.add_argument("--faults", required=True)
    parser.add_argument("--sat-moisture", default=None)
    parser.add_argument("--insar", default=None)
    parser.add_argument("--population", default=None)
    parser.add_argument("--glacial-lakes", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args)
