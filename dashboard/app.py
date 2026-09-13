"""
Week 7 — Dashboard & Alerts (plain-language edition).

Auto-detects the visitor's browser location, snaps it to the nearest
scored sample point in the feature table, and leads with one big,
color-coded, plain-language status card (Safe / Stay Alert / Warning /
Leave the area) — no jargon, no numbers, nothing to type in. All the
technical detail (SHAP factors, rainfall bars, raw scores) sits behind a
single "See full analysis" expander, closed by default. A friendly map
shows the same colors for everyone else on the map.

Requirements:
    pip install streamlit pandas numpy plotly shap folium xgboost
                scikit-learn streamlit-js-eval streamlit-folium

Usage:
    streamlit run app.py -- --features ../data/processed/final_feature_table.csv \\
        --model-a ../models/best_model.pkl --model-b ../models/dynamic_hazard_model.pkl
"""
import argparse
import math
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_js_eval import get_geolocation
from streamlit_folium import st_folium

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "explainability"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "alerts"))

from risk_engine import load_models, score_single_location, score_table  # noqa: E402
from shap_analysis import explain_location  # noqa: E402
from alert_engine import process_alert  # noqa: E402
from gis_map import build_friendly_map  # noqa: E402
from live_weather import fetch_live_rainfall  # noqa: E402
from live_soil import fetch_live_soil_moisture  # noqa: E402
from live_terrain import fetch_live_terrain  # noqa: E402
from live_landslide_history import fetch_live_landslide_history  # noqa: E402


# ---------------------------------------------------------------------------
# Plain-language layer — maps the model's technical output onto colors,
# emoji, and everyday phrases a non-technical visitor can act on instantly.
# ---------------------------------------------------------------------------
LEVEL_UI = {
    "Very Low": dict(color="#2e7d32", emoji="✅", label="SAFE ZONE",
                      message="No landslide risk detected right now. Nothing to do."),
    "Low": dict(color="#66bb6a", emoji="🙂", label="ALL CLEAR",
                message="Conditions look normal. No action needed."),
    "Moderate": dict(color="#fbc02d", emoji="⚠️", label="STAY ALERT",
                      message="Some risk factors are building up. Keep an eye on the weather."),
    "High": dict(color="#f57c00", emoji="🚨", label="WARNING",
                 message="Risk is elevated. Be ready to move to safety if it keeps raining."),
    "Critical": dict(color="#c62828", emoji="🔴", label="LEAVE THE AREA NOW",
                      message="High landslide risk. Move to safety immediately and warn others nearby."),
}

# Friendly names + short explanations for every feature the models use, so
# the "why" panel never shows a raw column name to a non-technical visitor.
FRIENDLY_FACTORS = {
    "rainfall_1d": ("rain in the last day", "🌧️"),
    "rainfall_3d": ("rain over the last 3 days", "🌧️"),
    "rainfall_7d": ("rain over the last week", "🌧️"),
    "rainfall_15d": ("rain over the last 2 weeks", "🌧️"),
    "rainfall_30d": ("rain over the last month", "🌧️"),
    "slope": ("how steep the land is", "⛰️"),
    "aspect": ("which direction the slope faces", "🧭"),
    "curvature": ("how uneven the ground is", "🪨"),
    "ruggedness_tri": ("how rugged the terrain is", "🪨"),
    "elevation": ("how high up the area is", "🏔️"),
    "road_distance": ("distance from the nearest road", "🛣️"),
    "historical_landslide_density": ("past landslides nearby", "📜"),
    "distance_to_nearest_landslide_m": ("distance to the closest past landslide", "📜"),
    "susceptibility_score": ("overall ground stability", "🧱"),
    "soil": ("the type of soil here", "🟤"),
    "geology": ("the type of rock underneath", "🪨"),
    "NDVI": ("how much vegetation/tree cover there is", "🌳"),
    "land_use": ("how the land is being used (forest/farm/built-up)", "🏞️"),
    "seismic_pga": ("how earthquake-prone the area is", "🌐"),
    "fault_distance_km": ("distance to the nearest earthquake fault line", "🌐"),
    "soil_moisture_baseline": ("how naturally damp the ground stays", "💧"),
    "soil_moisture_index": ("how saturated the ground is right now", "💧"),
    "insolation_proxy": ("how much direct sun this slope gets", "☀️"),
    "freeze_thaw_index": ("how much the ground freezes and thaws", "❄️"),
    "rainfall_intensity_mm_hr": ("how hard it's raining right now", "⛈️"),
    "rainfall_forecast_24h": ("rain expected in the next day", "🔮"),
    "rainfall_forecast_48h": ("rain expected in the next 2 days", "🔮"),
    "twi": ("how much water this spot collects", "💦"),
    "drainage_distance_km": ("distance to the nearest stream", "🏞️"),
    "soil_moisture_satellite": ("satellite-measured ground wetness", "🛰️"),
    "insar_deformation_mm_yr": ("how much the ground is slowly shifting", "📡"),
    "root_cohesion_proxy": ("how well tree roots hold the soil together", "🌲"),
    "glacial_lake_distance_km": ("distance to the nearest glacial lake", "🏔️"),
    "population_density": ("how many people live nearby", "🏘️"),
    "exposure_index": ("how many people/places could be affected", "🏘️"),
}


# Human labels for the synthetic categorical codes (soil/geology/land_use),
# so the terrain table never shows a bare integer to a non-technical visitor.
CATEGORY_LABELS = {
    "soil": {1: "Rocky", 2: "Sandy", 3: "Loamy", 4: "Clayey"},
    "geology": {1: "Gneiss", 2: "Schist", 3: "Phyllite", 4: "Quartzite", 5: "Alluvium"},
    "land_use": {1: "Forest", 2: "Agriculture", 3: "Urban", 4: "Barren/Snow"},
}


def friendly_name(col: str) -> tuple:
    return FRIENDLY_FACTORS.get(col, (col.replace("_", " "), "•"))


@st.cache_data(show_spinner=False, ttl=3600)
def get_location_name(lat: float, lon: float) -> str:
    """Reverse-geocode via OpenStreetMap's free Nominatim API — no key
    needed. Cached per rounded coordinate (both to be a good citizen of a
    free public service, and since re-querying the same spot is pointless).
    Falls back to plain coordinates if the service is unreachable or slow,
    so a network hiccup never breaks the page."""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format": "json", "lat": lat, "lon": lon, "zoom": 14, "addressdetails": 1},
            headers={"User-Agent": "landslide-early-warning-dashboard/1.0"},
            timeout=4,
        )
        resp.raise_for_status()
        data = resp.json()
        addr = data.get("address", {})
        parts = [
            addr.get("village") or addr.get("town") or addr.get("city") or addr.get("suburb"),
            addr.get("county") or addr.get("state_district"),
            addr.get("state"),
        ]
        name = ", ".join(p for p in parts if p)
        return name or data.get("display_name", f"{lat:.3f}, {lon:.3f}")
    except Exception:
        return f"{lat:.3f}, {lon:.3f}"


@st.cache_data(show_spinner=False, ttl=3600)
def geocode_place_name(query: str):
    """Forward-geocode a typed place name via OpenStreetMap's free Nominatim
    search API — the mirror of get_location_name(). Returns (lat, lon,
    display_name) or None if nothing matched / the service is unreachable."""
    if not query or not query.strip():
        return None
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"format": "json", "q": query, "limit": 1, "addressdetails": 1},
            headers={"User-Agent": "landslide-early-warning-dashboard/1.0"},
            timeout=4,
        )
        resp.raise_for_status()
        results = resp.json()
        if not results:
            return None
        top = results[0]
        return float(top["lat"]), float(top["lon"]), top.get("display_name", query)
    except Exception:
        return None


def parse_cli_args():
    """Streamlit's CLI sometimes forwards the "--" separator to the script and
    sometimes strips it before handing off sys.argv, depending on version.
    Handle both."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    return parser.parse_args(argv)


@st.cache_resource
def get_models(model_a_path: str, model_b_path: str):
    return load_models(model_a_path, model_b_path)


@st.cache_data
def get_features(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False, ttl=600)
def get_live_rainfall(lat: float, lon: float):
    """Cached for 10 minutes (rain doesn't change second-to-second) and
    keyed on a coordinate rounded to ~1km, so nearby clicks/searches share
    one Open-Meteo call instead of each firing a fresh request."""
    return fetch_live_rainfall(lat, lon)


@st.cache_data(show_spinner=False, ttl=600)
def get_live_soil_moisture(lat: float, lon: float):
    return fetch_live_soil_moisture(lat, lon)


@st.cache_data(show_spinner=False, ttl=86400)
def get_live_terrain(lat: float, lon: float):
    """Terrain barely changes, so this is cached for a full day (also
    keeps us polite to the free Overpass/elevation APIs)."""
    return fetch_live_terrain(lat, lon)


@st.cache_data(show_spinner=False)
def get_live_landslide_history(lat: float, lon: float):
    """No network call (catalog is local), so no ttl needed -- it can only
    change if the repo's catalog files themselves change."""
    return fetch_live_landslide_history(lat, lon)


@st.cache_data(show_spinner=False)
def score_sample_points(map_df: pd.DataFrame, _model_a, _model_b) -> pd.DataFrame:
    """Batch-scores the map's sample points once and caches the result, so
    dragging the rain slider, ticking a checkbox, or tapping the map doesn't
    silently re-run 150 individual model predictions on every rerun. The
    leading underscore on _model_a/_model_b tells st.cache_data to key the
    cache on map_df's contents only, not on the (unhashable) model objects."""
    return score_table(map_df.copy(), _model_a, _model_b)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_row(df: pd.DataFrame, lat: float, lon: float):
    """Snap a raw GPS point to the nearest row we actually have terrain/rainfall
    features for — this demo can only score locations already in the feature
    table; a production version would extract features live from the DEM and
    rainfall rasters for the visitor's exact coordinates."""
    dists = df.apply(lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]), axis=1)
    idx = dists.idxmin()
    return df.loc[idx], float(dists.loc[idx])


def status_card(level: str, extra_note: str = ""):
    ui = LEVEL_UI.get(level, LEVEL_UI["Moderate"])
    st.markdown(
        f"""
        <div style="background:{ui['color']}; padding:28px 20px; border-radius:16px;
                    text-align:center; color:white; margin-bottom:14px;">
            <div style="font-size:52px; line-height:1;">{ui['emoji']}</div>
            <div style="font-size:28px; font-weight:800; letter-spacing:1px; margin-top:6px;">
                {ui['label']}
            </div>
            <div style="font-size:16px; margin-top:8px; opacity:0.95;">{ui['message']}</div>
            {f'<div style="font-size:13px; margin-top:10px; opacity:0.85;">{extra_note}</div>' if extra_note else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def ai_summary_sentence(factors: list) -> str:
    """A short, templated natural-language explanation stitched together
    from the model's own SHAP output — not a separate LLM call, but the
    same idea: turn the model's internal reasoning into a plain sentence."""
    if not factors:
        return "Not enough signal to explain this score in detail yet."
    raisers = [f for f in factors if f["contribution_pct"] > 0][:2]
    if raisers:
        names = [friendly_name(f["factor"])[0] for f in raisers]
        joined = " and ".join(names)
        return f"This is mainly because of {joined}."
    lowerers = [f for f in factors if f["contribution_pct"] < 0][:2]
    names = [friendly_name(f["factor"])[0] for f in lowerers]
    joined = " and ".join(names) if names else "current conditions"
    return f"Conditions look manageable right now, mainly thanks to {joined}."


def gauge_chart(score: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100", "font": {"size": 30}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "showticklabels": False},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "#eeeeee",
            "steps": [
                {"range": [0, 20], "color": "#2e7d3233"},
                {"range": [20, 40], "color": "#66bb6a33"},
                {"range": [40, 60], "color": "#fbc02d33"},
                {"range": [60, 80], "color": "#f57c0033"},
                {"range": [80, 100], "color": "#c6282833"},
            ],
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=10, b=10))
    return fig


def rainfall_chart(row: dict) -> go.Figure:
    windows = ["rainfall_1d", "rainfall_3d", "rainfall_7d", "rainfall_15d", "rainfall_30d"]
    labels = ["Last day", "Last 3 days", "Last week", "Last 2 weeks", "Last month"]
    values = [row.get(w, 0) for w in windows]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color="#1976d2",
        text=[f"{v:.0f} mm" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title="How much rain has fallen recently",
        height=280, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Millimetres of rain",
    )
    return fig


def main():
    args = parse_cli_args()
    st.set_page_config(page_title="Landslide Early Warning", layout="centered",
                        page_icon="⛰️")

    st.markdown(
        "<h2 style='margin-bottom:0;'>⛰️ Landslide Early Warning</h2>"
        "<p style='color:gray; margin-top:2px;'>Automatic, location-based risk check "
        "for North East India (Sikkim pilot).</p>",
        unsafe_allow_html=True,
    )

    model_a, model_b = get_models(args.model_a, args.model_b)
    df = get_features(args.features)

    if "clicked_latlon" not in st.session_state:
        st.session_state.clicked_latlon = None

    # --- Auto-detect the visitor's location, no typing required ------------
    # Only ask the browser once per session: get_geolocation() triggers a
    # real GPS/network location fix, and re-running it on every widget
    # interaction (slider drag, checkbox, map tap) is what was making every
    # click feel slow. Cache the result and only re-ask if it hasn't
    # resolved yet, or the user explicitly asks to refresh it.
    if "geo_loc" not in st.session_state:
        st.session_state.geo_loc = None
    if st.session_state.geo_loc is None:
        st.session_state.geo_loc = get_geolocation()
    loc = st.session_state.geo_loc

    with st.sidebar:
        st.header("🌧️ What-if: heavier or lighter rain?")
        st.caption("Slide to simulate a bigger or smaller storm on top of today's data.")
        rain_multiplier = st.slider("Rain intensity", 0.0, 3.0, 1.0, 0.1,
                                     format="%.1fx", label_visibility="collapsed")
        use_live_rain = st.checkbox(
            "Use live data (weather, soil moisture, terrain — not the synthetic dataset)",
            value=True,
            help="Fetches real rainfall/soil-moisture (Open-Meteo), real "
                 "elevation/slope/aspect/road & river distance (Open-Meteo Elevation "
                 "+ OpenStreetMap), and real past-landslide density/distance (this "
                 "project's own historical catalog) for the exact point you're "
                 "checking. Whatever a source can't provide (geology, land use, "
                 "seismic hazard, InSAR, etc.) still comes from the nearest "
                 "synthetic sample point."
        )
        st.divider()

        st.header("📍 Set a location")
        st.caption("Didn't get a location prompt, or want to check somewhere else? "
                   "Set it manually below, or tap anywhere on the map itself.")
        if st.button("🔄 Refresh my live location"):
            st.session_state.geo_loc = None
            st.rerun()
        if st.session_state.clicked_latlon:
            lat_c, lon_c = st.session_state.clicked_latlon
            st.info(f"📌 Showing the point you tapped on the map "
                    f"({lat_c:.4f}, {lon_c:.4f}).")
            if st.button("Clear map pin"):
                st.session_state.clicked_latlon = None
                st.rerun()
        manual = st.checkbox("Enter a location manually")
        manual_mode = None
        place_query = None
        manual_lat, manual_lon = None, None
        manual_idx = None
        if manual:
            manual_mode = st.radio(
                "How?", ["Type a place name", "Enter coordinates", "Pick a sample point"],
                label_visibility="collapsed",
            )
            if manual_mode == "Type a place name":
                place_query = st.text_input(
                    "Place name", placeholder="e.g. Gangtok, Sikkim",
                    label_visibility="collapsed",
                )
            elif manual_mode == "Enter coordinates":
                manual_lat = st.number_input("Latitude", value=27.33, format="%.4f")
                manual_lon = st.number_input("Longitude", value=88.61, format="%.4f")
            else:
                manual_idx = st.selectbox(
                    "Sample location", options=df.index.tolist(),
                    format_func=lambda i: f"#{i} ({df.loc[i, 'latitude']:.3f}, {df.loc[i, 'longitude']:.3f})",
                )

    user_latlon = None
    distance_note = ""

    if st.session_state.clicked_latlon:
        # Highest priority: the visitor just tapped an exact spot on the
        # map to ask "what about here?" — that beats any earlier manual
        # entry or auto-detected browser location.
        lat, lon = st.session_state.clicked_latlon
        user_latlon = (lat, lon)
        nearest, dist_km = nearest_row(df, lat, lon)
        base_row = nearest.to_dict()
        distance_note = (
            f"Showing risk for the exact point you tapped on the map "
            f"(closest monitored point is {dist_km:.1f} km away)."
        )
    elif manual_mode == "Type a place name" and place_query:
        found = geocode_place_name(place_query)
        if found:
            lat, lon, matched_name = found
            user_latlon = (lat, lon)
            nearest, dist_km = nearest_row(df, lat, lon)
            base_row = nearest.to_dict()
            distance_note = (
                f"Matched to \"{matched_name}\" — showing the closest monitored point "
                f"({dist_km:.1f} km away)."
            )
        else:
            st.warning("📍 Couldn't find that place. Try a different spelling, or "
                       "add the state/district (e.g. \"Gangtok, Sikkim\").")
            st.stop()
    elif manual_mode == "Enter coordinates" and manual_lat is not None:
        user_latlon = (manual_lat, manual_lon)
        nearest, dist_km = nearest_row(df, manual_lat, manual_lon)
        base_row = nearest.to_dict()
        distance_note = f"Showing the closest monitored point to those coordinates ({dist_km:.1f} km away)."
    elif manual_idx is not None:
        base_row = df.loc[manual_idx].to_dict()
        user_latlon = (base_row["latitude"], base_row["longitude"])
    elif manual and manual_mode == "Type a place name":
        st.info("📍 Type a place name in the sidebar to check its risk.")
        st.stop()
    elif loc and "coords" in loc:
        lat, lon = loc["coords"]["latitude"], loc["coords"]["longitude"]
        user_latlon = (lat, lon)
        nearest, dist_km = nearest_row(df, lat, lon)
        base_row = nearest.to_dict()
        distance_note = (
            f"Showing the closest monitored point to you ({dist_km:.1f} km away)."
        )
    elif loc and "error" in loc:
        st.warning("📍 Couldn't get your location automatically "
                   "(your browser may have blocked it). Pick one manually in the sidebar.")
        st.stop()
    else:
        st.info("📍 Detecting your location — allow location access if your browser asks...")
        st.stop()

    # --- Swap in live data for this exact point, if enabled ----------------
    # Whatever a live source can't provide (geology, land use, seismic
    # hazard, InSAR, etc.) still comes from the nearest synthetic sample
    # point (base_row) -- only the pieces we have a real source for
    # (rainfall, soil moisture, terrain, past-landslide history) get
    # overwritten.
    if use_live_rain and user_latlon:
        live_notes = []
        live_rain = get_live_rainfall(round(user_latlon[0], 2), round(user_latlon[1], 2))
        if live_rain:
            base_row.update(live_rain)
            live_notes.append("rainfall")
        live_soil = get_live_soil_moisture(round(user_latlon[0], 2), round(user_latlon[1], 2))
        if live_soil:
            base_row.update(live_soil)
            live_notes.append("soil moisture")
        live_terrain = get_live_terrain(round(user_latlon[0], 3), round(user_latlon[1], 3))
        if live_terrain:
            base_row.update(live_terrain)
            live_notes.append("terrain")
        live_landslide_history = get_live_landslide_history(
            round(user_latlon[0], 4), round(user_latlon[1], 4)
        )
        if live_landslide_history:
            base_row.update(live_landslide_history)
            live_notes.append("past-landslide history")
        if live_notes:
            distance_note = (distance_note + " " if distance_note else "") + \
                f"🌍 Using live {', '.join(live_notes)} for this exact point."
        else:
            distance_note = (distance_note + " " if distance_note else "") + \
                "⚠️ Live data unavailable right now — showing dataset values instead."

    # --- Score this location -------------------------------------------------
    scenario_row = dict(base_row)
    for col in ["rainfall_1d", "rainfall_3d", "rainfall_7d", "rainfall_15d", "rainfall_30d"]:
        if col in scenario_row and pd.notna(scenario_row[col]):
            scenario_row[col] = scenario_row[col] * rain_multiplier

    result = score_single_location(scenario_row, model_a, model_b)
    level = result["risk_level"]
    ui = LEVEL_UI.get(level, LEVEL_UI["Moderate"])

    # --- Who/where: a real place name, not just raw coordinates ------------
    # Round to ~110m before the reverse-geocode lookup so nearby map taps
    # reuse the cache instead of firing a fresh (up to 4s) network call for
    # every last-decimal-place difference in the click coordinates.
    location_name = get_location_name(round(user_latlon[0], 3), round(user_latlon[1], 3))
    st.markdown(
        f"<p style='text-align:center; color:gray; margin-bottom:4px;'>📍 {location_name}</p>",
        unsafe_allow_html=True,
    )

    # --- Front screen: just the color + plain notation, nothing technical ---
    status_card(level, extra_note=distance_note)

    factors = explain_location(scenario_row, model_a, model_b, top_n=5)
    st.markdown(f"**{ai_summary_sentence(factors)}**")

    alert = process_alert(
        location=f"({base_row['latitude']:.3f}, {base_row['longitude']:.3f})",
        risk_score=result["final_risk_score"],
        risk_level=level,
        system_action=result["system_action"],
    )
    if alert:
        st.markdown(f"📣 *{alert.message}*")

    st.markdown("#### 🗺️ Your area on the map")
    st.caption("Green = safe, yellow = stay alert, orange = warning, red = leave the area. "
               "The blue crosshair pin is your exact point — tap anywhere else on the map "
               "to check the risk there instead.")
    sample_size = min(len(df), 150)
    map_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df.copy()
    scored_sample = score_sample_points(map_df, model_a, model_b)
    fmap = build_friendly_map(scored_sample, user_location=user_latlon, user_risk_level=level,
                               user_location_name=location_name)
    map_state = st_folium(fmap, height=420, width=None, returned_objects=["last_clicked"],
                           key="risk_map")
    clicked = map_state.get("last_clicked") if map_state else None
    if clicked:
        new_latlon = (clicked["lat"], clicked["lng"])
        if new_latlon != st.session_state.clicked_latlon:
            st.session_state.clicked_latlon = new_latlon
            st.rerun()

    # --- Everything technical is shown directly — nothing hidden -----------
    st.markdown("---")
    st.markdown("### 🔍 Full technical analysis")
    st.caption(
        "Software-only, AI-driven decision-support tool — not a guarantee of "
        "landslide occurrence or non-occurrence. Validate real decisions with "
        "disaster-management authorities."
    )
    g1, g2 = st.columns([1, 1])
    with g1:
        st.plotly_chart(gauge_chart(result["final_risk_score"], ui["color"]),
                         width="stretch")
        st.caption(f"Risk score: {result['final_risk_score']} / 100 · "
                   f"24h hazard probability: {result['dynamic_hazard_score']:.2f}")
    with g2:
        st.plotly_chart(rainfall_chart(scenario_row), width="stretch")

    st.markdown("**Why this score — top contributing factors**")
    if factors:
        rows = []
        for f in factors:
            name, emoji = friendly_name(f["factor"])
            direction = "⬆️ raises risk" if f["contribution_pct"] > 0 else "⬇️ lowers risk"
            rows.append({"Factor": f"{emoji} {name}", "Effect": direction,
                         "Weight": f"{abs(f['contribution_pct']):.0f}%"})
        st.table(pd.DataFrame(rows))
    else:
        st.info("No clear standout factor for this location.")

    st.markdown("**Every parameter at this location — nothing hidden**")
    # Every feature column in the table (not a curated subset) — identifiers
    # like latitude/longitude/date/label are shown separately, not repeated here.
    excluded = {"latitude", "longitude", "date", "label"}
    env_cols = [c for c in base_row.keys() if c not in excluded]
    rows = []
    for c in env_cols:
        val = base_row[c]
        if c in CATEGORY_LABELS:
            val = CATEGORY_LABELS[c].get(int(val), val)
        elif isinstance(val, float):
            val = round(val, 3)
        name, emoji = friendly_name(c)
        # Cast to string: this column mixes numbers and category labels
        # (e.g. "Loamy"), and a single object column with mixed types makes
        # Streamlit's Arrow conversion silently retry/fall back on every
        # render — casting up front avoids that entirely.
        rows.append({"Parameter": f"{emoji} {name.capitalize()}", "Value": str(val)})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
