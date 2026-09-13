"""
Live soil type via ISRIC SoilGrids' free REST API (no key required).

SoilGrids gives global-coverage topsoil (0-5cm) texture fractions and
coarse-fragment volume for any lat/lon. We map those onto this project's
existing 4-class scheme (see CATEGORY_LABELS in dashboard/app.py: 1=Rocky,
2=Sandy, 3=Loamy, 4=Clayey) with a simple, documented rule rather than a
full USDA texture-triangle classification, since the training data's
classes are already this coarse:

    high coarse-fragment volume (>15% by volume) -> Rocky
    else sand fraction dominant (>=50%)          -> Sandy
    else clay fraction dominant (>=40%)           -> Clayey
    else                                           -> Loamy

API doc: https://www.isric.org/explore/soilgrids/faq-soilgrids -- values
returned are means in g/kg (texture) and cm3/dm3 (coarse fragments) at
mapped depth intervals; we use the shallowest (0-5cm) interval.
"""
import requests

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPERTIES = ["clay", "sand", "cfvo"]  # clay %, sand %, coarse fragments (per-mille of volume)


def fetch_live_soil_type(lat: float, lon: float) -> dict | None:
    """Returns {"soil": <1-4 category code>} matching this project's
    existing CATEGORY_LABELS scheme, or None if the API call fails or
    returns no usable data for this point (e.g. open water)."""
    try:
        resp = requests.get(
            SOILGRIDS_URL,
            params={
                "lon": lon,
                "lat": lat,
                "property": PROPERTIES,
                "depth": "0-5cm",
                "value": "mean",
            },
            timeout=8,
        )
        resp.raise_for_status()
        layers = resp.json().get("properties", {}).get("layers", [])

        values = {}
        for layer in layers:
            name = layer.get("name")
            depths = layer.get("depths", [])
            if not depths:
                continue
            mean = depths[0].get("values", {}).get("mean")
            if mean is None:
                continue
            # SoilGrids returns values scaled by a factor (d_factor) to
            # keep them as integers -- divide back out to get real units.
            factor = layer.get("unit_measure", {}).get("d_factor", 1) or 1
            values[name] = mean / factor

        if "clay" not in values or "sand" not in values:
            return None

        clay_pct = values["clay"]
        sand_pct = values["sand"]
        coarse_pct = values.get("cfvo", 0.0)

        if coarse_pct > 15:
            code = 1  # Rocky
        elif sand_pct >= 50:
            code = 2  # Sandy
        elif clay_pct >= 40:
            code = 4  # Clayey
        else:
            code = 3  # Loamy

        return {"soil": code}
    except Exception:
        return None
