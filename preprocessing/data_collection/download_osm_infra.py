"""
Download roads and settlements for the Sikkim bounding box from OpenStreetMap
via the Overpass API. No login required.

Requirements:
    pip install requests

Usage:
    python download_osm_infra.py --out ../data/infrastructure/sikkim_osm.geojson
"""
import argparse
import json

import requests

SIKKIM_BBOX = "27.00,88.00,28.13,88.93"  # south,west,north,east

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

QUERY = f"""
[out:json][timeout:120];
(
  way["highway"]({SIKKIM_BBOX});
  node["place"~"city|town|village"]({SIKKIM_BBOX});
);
out geom;
"""


def download(out_path: str):
    print("Querying Overpass API for Sikkim roads + settlements...")
    resp = requests.post(OVERPASS_URL, data={"data": QUERY})
    resp.raise_for_status()
    data = resp.json()

    with open(out_path, "w") as f:
        json.dump(data, f)

    print(f"Saved {len(data.get('elements', []))} features -> {out_path}")
    print("Tip: load into GeoPandas with gpd.read_file() after converting OSM JSON "
          "to GeoJSON (osmtogeojson or manual parsing), or use the `overpy` / "
          "`osmnx` library instead for a cleaner GeoDataFrame directly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="../data/infrastructure/sikkim_osm.json")
    args = parser.parse_args()

    download(args.out)
