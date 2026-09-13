"""
Live NDVI via NASA's ORNL DAAC MODIS/VIIRS Land Products REST API
(free, no key required, genuinely global coverage -- unlike geology
data sources, this is satellite-based so it isn't limited to
well-mapped regions).

Uses the MOD13Q1 product (MODIS/Terra Vegetation Indices, 16-day
composite, 250m resolution). Two calls:
    1. /dates -- find the most recent date this pixel has data for
       (there's a processing lag of roughly 1-2 months)
    2. /subset -- fetch just that date's NDVI band for the single
       center pixel (kmAboveBelow=0, kmLeftRight=0)

NDVI in the raw response is scaled by 10000 (e.g. 4517 -> 0.4517) and
uses -3000 as a fill value for cloud-obscured/no-data pixels, which we
treat as "no usable value" the same way a failed network call is.

API docs: https://modis.ornl.gov/data/modis_webservice.html
"""
import requests

BASE_URL = "https://modis.ornl.gov/rst/api/v1"
PRODUCT = "MOD13Q1"
BAND = "250m_16_days_NDVI"
FILL_VALUE = -3000
SCALE_FACTOR = 0.0001


def fetch_live_ndvi(lat: float, lon: float) -> dict | None:
    """Returns {"NDVI": <0-1 value>}, or None if no valid recent
    observation is available (cloud cover, ocean, API hiccup, etc.)."""
    try:
        dates_resp = requests.get(
            f"{BASE_URL}/{PRODUCT}/dates",
            params={"latitude": lat, "longitude": lon},
            timeout=8,
        )
        dates_resp.raise_for_status()
        dates = dates_resp.json().get("dates", [])
        if not dates:
            return None
        latest_modis_date = dates[-1]["modis_date"]

        subset_resp = requests.get(
            f"{BASE_URL}/{PRODUCT}/subset",
            params={
                "latitude": lat,
                "longitude": lon,
                "band": BAND,
                "startDate": latest_modis_date,
                "endDate": latest_modis_date,
                "kmAboveBelow": 0,
                "kmLeftRight": 0,
            },
            headers={"Accept": "application/json"},
            timeout=8,
        )
        subset_resp.raise_for_status()
        subset = subset_resp.json().get("subset", [])
        if not subset or not subset[0].get("data"):
            return None

        raw = subset[0]["data"][0]
        if raw is None or raw <= FILL_VALUE:
            return None

        ndvi = raw * SCALE_FACTOR
        return {"NDVI": round(min(max(ndvi, 0.0), 1.0), 4)}
    except Exception:
        return None
