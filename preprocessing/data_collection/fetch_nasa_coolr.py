"""
Download NASA COOLR (Cooperative Open Online Landslide Repository) global landslide
catalog and filter to Sikkim / NER bounding box. No login required.

Requirements:
    pip install pandas requests

Usage:
    python fetch_nasa_coolr.py --out ../data/historical_landslides/coolr_ner.csv
"""
import argparse

import pandas as pd
import requests

# NASA COOLR public CSV export (Global Landslide Catalog).
# Check https://gpm.nasa.gov/landslides/index.html for the current direct-download link
# if this URL has moved.
COOLR_CSV_URL = "https://data.nasa.gov/resource/dd9e-wu2v.csv?$limit=50000"

# Broad NER bounding box (covers all 8 states; narrow further to Sikkim if you only
# want Phase-1 data)
NER_BBOX = dict(min_lat=21.5, max_lat=29.5, min_lon=88.0, max_lon=97.5)


def fetch_and_filter(out_path: str):
    print("Downloading NASA COOLR catalog...")
    df = pd.read_csv(COOLR_CSV_URL)

    # Column names in COOLR exports are typically 'latitude'/'longitude' —
    # adjust here if the schema differs when you actually pull it.
    lat_col = "latitude" if "latitude" in df.columns else "event_lat"
    lon_col = "longitude" if "longitude" in df.columns else "event_lon"

    ner = df[
        (df[lat_col] >= NER_BBOX["min_lat"]) & (df[lat_col] <= NER_BBOX["max_lat"]) &
        (df[lon_col] >= NER_BBOX["min_lon"]) & (df[lon_col] <= NER_BBOX["max_lon"])
    ]

    ner.to_csv(out_path, index=False)
    print(f"Saved {len(ner)} NER landslide events -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="../data/historical_landslides/coolr_ner.csv")
    args = parser.parse_args()

    fetch_and_filter(args.out)
