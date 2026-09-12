"""
Download SRTM 30m DEM for the Sikkim bounding box via the OpenTopography REST API.

Get a free API key first:
    https://opentopography.org -> sign up -> My Account -> Request API Key

Requirements:
    pip install requests

Usage:
    python download_dem.py --api-key YOUR_KEY --out ../data/terrain/sikkim_srtm30m.tif
"""
import argparse

import requests

SIKKIM_BBOX = dict(south=27.00, north=28.13, west=88.00, east=88.93)

API_URL = "https://portal.opentopography.org/API/globaldem"


def download_dem(api_key: str, out_path: str, dem_type: str = "SRTMGL1"):
    params = {
        "demtype": dem_type,       # SRTMGL1 = SRTM 30m
        "south": SIKKIM_BBOX["south"],
        "north": SIKKIM_BBOX["north"],
        "west": SIKKIM_BBOX["west"],
        "east": SIKKIM_BBOX["east"],
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    print("Requesting DEM tile from OpenTopography...")
    resp = requests.get(API_URL, params=params, stream=True)
    resp.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Saved DEM -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--out", default="../data/terrain/sikkim_srtm30m.tif")
    parser.add_argument("--dem-type", default="SRTMGL1")
    args = parser.parse_args()

    download_dem(args.api_key, args.out, args.dem_type)
