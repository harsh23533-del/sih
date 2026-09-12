# Week 3 — Data Cleaning & Feature Engineering

**Goal:** turn Week 2's raw downloads into one clean, combined feature table ready for
model training (Week 4).

## Pipeline order

1. **`negative_sampling.py`** — generate background/negative samples
   - Randomly samples valid land points from the DEM extent
   - Excludes anything within `exclusion_radius_km` of a known historical landslide (avoids label leakage)
   - Assigns each a random date within your rainfall data's coverage
   - Ratio of negatives to positives is a *starting hypothesis* (default 2:1) — tune based on validation results, not fixed upfront

2. **`terrain_features.py`** — computes elevation, slope, aspect, profile curvature, and Terrain Ruggedness Index (TRI) from the DEM, sampled at each point (positive + negative)

3. **`rainfall_features.py`** — computes 1/3/7/15/30-day cumulative rainfall at each point's date, from the Week 2 IMD NetCDF files

4. **`density_distance_features.py`** — for each point:
   - `historical_landslide_density`: count of past landslides within 5km (tunable)
   - `distance_to_nearest_landslide_m`
   - `road_distance`: distance to nearest OSM road (reprojected to UTM 45N for meter accuracy)

5. **`build_feature_table.py`** — the master script. Combines everything above, cleans, and outputs the final table matching the plan's schema:
   ```
   latitude, longitude, date, rainfall_1d, rainfall_3d, rainfall_7d, rainfall_15d,
   rainfall_30d, elevation, slope, aspect, curvature, ruggedness_tri, soil, geology,
   NDVI, road_distance, historical_landslide_density, distance_to_nearest_landslide_m, label
   ```

## Cleaning decisions (and why)

| Decision | Reasoning |
|---|---|
| Drop rows missing lat/lon/date | Unusable — can't be feature-extracted at all |
| Impute missing numeric features with column median (not drop) | Landslide positives are already rare; dropping rows with a single missing feature would bias the dataset further against the minority class |
| Drop exact duplicate (lat, lon, date) rows | Removes accidental double-counting from merging multiple source catalogs (e.g. an event appearing in both COOLR and GSI Bhukosh) |
| Exclude negatives within `exclusion_radius_km` of any positive | Prevents ambiguous near-duplicate points that would confuse the model about the true decision boundary |

## Soil / Geology / NDVI — not yet automated
These three columns are left as placeholders in `build_feature_table.py`. Once you've pulled
Bhuvan soil/geology layers and computed NDVI from Sentinel-2 (or Google Earth Engine) as
mentioned in the Week 2 guide, sample them at each point the same way `terrain_features.py`
samples the DEM (point-in-raster lookup) and join on `latitude`/`longitude`.

## What to check before moving to Week 4
- Confirm the **positive/negative ratio** in the final table (`label` value counts) is what you intended
- Spot-check a handful of rows against the actual DEM/rainfall values in QGIS or by eyeballing coordinates — catching a coordinate reference system (CRS) mismatch here saves a lot of pain in Week 4
- Look at `rainfall_1d`/`rainfall_30d` distributions for positives vs. negatives — positives should visibly skew toward higher rainfall if the data pipeline is working correctly

## Output for this week
`data/processed/final_feature_table.csv` — the "Final combined feature table" deliverable from the original plan.
