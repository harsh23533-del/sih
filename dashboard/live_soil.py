"""
Live soil moisture via Open-Meteo's free weather API (no key required).

Open-Meteo exposes modelled soil moisture at several depth bands. The
training data's three soil-moisture columns don't map onto these bands
exactly, so this is a best-effort bridge:
    soil_moisture_satellite = surface layer (0-1cm) -- closest analogue to
                               what a satellite radiometer actually observes
    soil_moisture_baseline  = deep layer (27-81cm) -- changes slowly, so it
                               stands in for a location's underlying wetness
    soil_moisture_index     = average of the surface + two middle layers --
                               a general "how wet is it right now" figure
"""
import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
SOIL_VARS = [
    "soil_moisture_0_to_1cm",
    "soil_moisture_3_to_9cm",
    "soil_moisture_9_to_27cm",
    "soil_moisture_27_to_81cm",
]


def fetch_live_soil_moisture(lat: float, lon: float) -> dict | None:
    """Returns soil_moisture_satellite/baseline/index for this point, or
    None if the API call fails or returns no usable data."""
    try:
        resp = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join(SOIL_VARS),
                "past_days": 1,
                "forecast_days": 1,
                "timezone": "auto",
            },
            timeout=6,
        )
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})

        def most_recent(var):
            for v in reversed(hourly.get(var, [])):
                if v is not None:
                    return v
            return None

        surface = most_recent("soil_moisture_0_to_1cm")
        shallow = most_recent("soil_moisture_3_to_9cm")
        mid = most_recent("soil_moisture_9_to_27cm")
        deep = most_recent("soil_moisture_27_to_81cm")
        if surface is None:
            return None

        out = {"soil_moisture_satellite": round(surface, 3)}
        if deep is not None:
            out["soil_moisture_baseline"] = round(deep, 3)
        blend = [v for v in [surface, shallow, mid] if v is not None]
        out["soil_moisture_index"] = round(sum(blend) / len(blend), 3)
        return out
    except Exception:
        return None
