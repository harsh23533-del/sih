"""
Live rainfall lookup via Open-Meteo's free weather API (no key required).

The training data's rainfall_* columns are synthetic. This module fills the
same columns with real, present-day values for a given lat/lon, so the risk
models can be evaluated against actual current weather instead of only the
dataset's synthetic numbers.

Column definitions (best-effort mapping onto Open-Meteo's daily/current data):
    rainfall_Nd              = sum of the last N *full* days' precipitation (mm)
    rainfall_intensity_mm_hr = current hour's precipitation rate (mm/hr)
    rainfall_forecast_24h    = today's forecast total precipitation (mm)
    rainfall_forecast_48h    = today + tomorrow's forecast total precipitation (mm)

These are a reasonable bridge to real data, not a guaranteed like-for-like
match to whatever exact definitions generated the synthetic training set --
treat rainfall_30d etc. as "last 30 real days of rain at this point", which
is what actually matters for a live risk check.
"""
import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# We always request the same window: 30 days of history + today + 2 more
# forecast days. This makes "today" a fixed offset from the end of the
# returned array, so we don't need to do timezone-aware date matching.
PAST_DAYS = 30
FORECAST_DAYS = 3  # today + tomorrow + day after


def fetch_live_rainfall(lat: float, lon: float) -> dict | None:
    """Returns a dict with rainfall_1d/3d/7d/15d/30d, rainfall_intensity_mm_hr,
    and rainfall_forecast_24h/48h for this location, using real weather data.
    Returns None if the API call fails (network issue, bad response, rate
    limit, etc.) -- callers should fall back to the synthetic dataset values
    in that case, so a weather-API hiccup never breaks the dashboard."""
    try:
        resp = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "precipitation_sum",
                "current": "precipitation",
                "past_days": PAST_DAYS,
                "forecast_days": FORECAST_DAYS,
                "timezone": "auto",
            },
            timeout=6,
        )
        resp.raise_for_status()
        data = resp.json()

        daily_precip = data["daily"]["precipitation_sum"]
        # "today" sits FORECAST_DAYS entries from the end of the array.
        idx_today = len(daily_precip) - FORECAST_DAYS
        if idx_today < 30:
            return None  # API returned less history than expected

        def past_n_days_sum(n: int) -> float:
            window = daily_precip[idx_today - n:idx_today]
            return round(sum(v for v in window if v is not None), 1)

        forecast_today = daily_precip[idx_today] or 0.0
        forecast_tomorrow = (
            daily_precip[idx_today + 1] if idx_today + 1 < len(daily_precip) else 0.0
        )
        current_precip = (data.get("current") or {}).get("precipitation") or 0.0

        return {
            "rainfall_1d": past_n_days_sum(1),
            "rainfall_3d": past_n_days_sum(3),
            "rainfall_7d": past_n_days_sum(7),
            "rainfall_15d": past_n_days_sum(15),
            "rainfall_30d": past_n_days_sum(30),
            "rainfall_intensity_mm_hr": round(current_precip, 1),
            "rainfall_forecast_24h": round(forecast_today, 1),
            "rainfall_forecast_48h": round(forecast_today + forecast_tomorrow, 1),
        }
    except Exception:
        return None
