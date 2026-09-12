"""
Compute:
  1. historical_landslide_density — count of past landslides within a radius (km)
  2. distance to nearest historical landslide (m)
  3. road_distance — distance to nearest OSM road (m)
  4. drainage_distance — distance to nearest drainage/river line (m, if you have one;
     OSM 'waterway' tag can substitute if a dedicated drainage layer isn't available)

Requirements:
    pip install pandas numpy scipy geopandas shapely pyproj

Usage:
    python density_distance_features.py \\
        --points ../data/historical_landslides/coolr_ner.csv \\
        --osm-roads ../data/infrastructure/sikkim_osm.geojson \\
        --out ../data/processed/points_with_density_distance.csv
"""
import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_landslide_density_and_distance(points_df: pd.DataFrame, landslide_lat, landslide_lon,
                                        radius_km: float = 5.0) -> pd.DataFrame:
    """For each row in points_df, compute density of nearby historical landslides
    and distance to the single nearest one. landslide_lat/lon are the FULL historical
    inventory (not the subset being scored) so events aren't excluded from their own density."""
    density = []
    nearest_dist = []

    for _, row in points_df.iterrows():
        d = haversine_km(row["latitude"], row["longitude"], landslide_lat, landslide_lon)
        density.append(int(np.sum(d <= radius_km)))
        nearest_dist.append(float(np.min(d)) * 1000)  # convert to meters

    points_df = points_df.copy()
    points_df["historical_landslide_density"] = density
    points_df["distance_to_nearest_landslide_m"] = nearest_dist
    return points_df


def add_road_distance(points_df: pd.DataFrame, osm_geojson_path: str) -> pd.DataFrame:
    """Distance (m) from each point to the nearest OSM road geometry.
    Reprojects to a local UTM zone for accurate metric distance."""
    roads = gpd.read_file(osm_geojson_path)
    roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])]

    points_gdf = gpd.GeoDataFrame(
        points_df.copy(),
        geometry=[Point(xy) for xy in zip(points_df["longitude"], points_df["latitude"])],
        crs="EPSG:4326",
    )

    # Reproject both to UTM 45N (covers Sikkim/NER) for meter-accurate distances
    utm_crs = "EPSG:32645"
    roads_utm = roads.to_crs(utm_crs)
    points_utm = points_gdf.to_crs(utm_crs)

    roads_union = roads_utm.geometry.union_all() if hasattr(roads_utm.geometry, "union_all") \
        else roads_utm.geometry.unary_union

    points_utm["road_distance"] = points_utm.geometry.apply(lambda p: p.distance(roads_union))

    points_df = points_df.copy()
    points_df["road_distance"] = points_utm["road_distance"].values
    return points_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True, help="CSV with latitude/longitude to score")
    parser.add_argument("--historical-landslides", required=True,
                         help="CSV of the FULL historical landslide inventory (lat/lon)")
    parser.add_argument("--osm-roads", required=True, help="GeoJSON of OSM roads (Week 2 output)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--radius-km", type=float, default=5.0)
    args = parser.parse_args()

    points_df = pd.read_csv(args.points)
    hist_df = pd.read_csv(args.historical_landslides)

    out_df = add_landslide_density_and_distance(
        points_df, hist_df["latitude"].values, hist_df["longitude"].values, args.radius_km
    )
    out_df = add_road_distance(out_df, args.osm_roads)

    out_df.to_csv(args.out, index=False)
    print(f"Saved {len(out_df)} rows with density/distance features -> {args.out}")
