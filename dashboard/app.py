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
                scikit-learn streamlit-js-eval

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
import streamlit as st
from streamlit_js_eval import get_geolocation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "explainability"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "alerts"))

from risk_engine import load_models, score_single_location  # noqa: E402
from shap_analysis import explain_location  # noqa: E402
from alert_engine import process_alert  # noqa: E402
from gis_map import build_friendly_map  # noqa: E402


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
    "elevation": ("how high up the area is", "🏔️"),
    "road_distance": ("distance from the nearest road", "🛣️"),
    "historical_landslide_density": ("past landslides nearby", "📜"),
    "susceptibility_score": ("overall ground stability", "🧱"),
}


def friendly_name(col: str) -> tuple:
    return FRIENDLY_FACTORS.get(col, (col.replace("_", " "), "•"))


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

    # --- Auto-detect the visitor's location, no typing required ------------
    loc = get_geolocation()

    with st.sidebar:
        st.header("🌧️ What-if: heavier or lighter rain?")
        st.caption("Slide to simulate a bigger or smaller storm on top of today's data.")
        rain_multiplier = st.slider("Rain intensity", 0.0, 3.0, 1.0, 0.1,
                                     format="%.1fx", label_visibility="collapsed")
        st.divider()
        st.caption("Didn't get a location prompt, or said no by mistake? "
                   "Reload the page and allow location access when your browser asks.")
        manual = st.checkbox("Pick a location manually instead")
        manual_idx = None
        if manual:
            manual_idx = st.selectbox(
                "Sample location", options=df.index.tolist(),
                format_func=lambda i: f"#{i} ({df.loc[i, 'latitude']:.3f}, {df.loc[i, 'longitude']:.3f})",
            )

    user_latlon = None
    distance_note = ""

    if manual_idx is not None:
        base_row = df.loc[manual_idx].to_dict()
        user_latlon = (base_row["latitude"], base_row["longitude"])
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

    # --- Score this location -------------------------------------------------
    scenario_row = dict(base_row)
    for col in ["rainfall_1d", "rainfall_3d", "rainfall_7d", "rainfall_15d", "rainfall_30d"]:
        if col in scenario_row and pd.notna(scenario_row[col]):
            scenario_row[col] = scenario_row[col] * rain_multiplier

    result = score_single_location(scenario_row, model_a, model_b)
    level = result["risk_level"]
    ui = LEVEL_UI.get(level, LEVEL_UI["Moderate"])

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
    st.caption("Green = safe, yellow = stay alert, orange = warning, red = leave the area.")
    sample_size = min(len(df), 150)
    map_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df.copy()
    scored_sample = pd.DataFrame([
        {**row, **score_single_location(row, model_a, model_b)}
        for row in map_df.to_dict("records")
    ])
    fmap = build_friendly_map(scored_sample, user_location=user_latlon, user_risk_level=level)
    st.html(f'<div style="height:420px;">{fmap._repr_html_()}</div>', unsafe_allow_javascript=True)

    # --- Everything technical stays hidden until someone actually asks -----
    with st.expander("🔍 Tap to see the full technical analysis"):
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

        st.markdown("**Terrain & environment at this location**")
        env_cols = [c for c in ["elevation", "slope", "aspect", "curvature", "road_distance",
                                 "historical_landslide_density"] if c in base_row]
        friendly_env = {friendly_name(c)[0].capitalize(): base_row[c] for c in env_cols}
        st.table(pd.DataFrame([friendly_env]))


if __name__ == "__main__":
    main()
