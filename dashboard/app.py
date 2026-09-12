"""
Week 7 — Dashboard & Alerts.

The end-to-end Streamlit app: pick a location -> see its current
score/probability -> see the SHAP "why" -> drag the rainfall slider to
simulate a storm and watch the risk escalate live -> see the generated
warning message. This is the Final Demonstration Script from the plan,
built as one interactive app.

Requirements:
    pip install streamlit pandas numpy plotly shap folium streamlit-folium xgboost scikit-learn

Usage:
    streamlit run app.py -- --features ../data/processed/final_feature_table.csv \\
        --model-a ../models/best_model.pkl --model-b ../models/dynamic_hazard_model.pkl
"""
import argparse
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "explainability"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "alerts"))

from risk_engine import load_models, score_single_location, classify_risk  # noqa: E402
from shap_analysis import explain_location  # noqa: E402
from alert_engine import process_alert  # noqa: E402
from gis_map import build_risk_map  # noqa: E402


def parse_cli_args():
    """Streamlit passes its own args first; anything after `--` is ours."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []
    return parser.parse_args(argv)


@st.cache_resource
def get_models(model_a_path: str, model_b_path: str):
    return load_models(model_a_path, model_b_path)


@st.cache_data
def get_features(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def location_label(row: pd.Series, idx: int) -> str:
    date = row.get("date", "")
    return f"#{idx} — ({row['latitude']:.3f}, {row['longitude']:.3f}) {date}"


def rainfall_trend_chart(row: dict) -> go.Figure:
    windows = ["rainfall_1d", "rainfall_3d", "rainfall_7d", "rainfall_15d", "rainfall_30d"]
    labels = ["1-day", "3-day", "7-day", "15-day", "30-day"]
    values = [row.get(w, 0) for w in windows]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#1f77b4"))
    fig.update_layout(title="Cumulative rainfall by window (mm)", height=300,
                       margin=dict(l=10, r=10, t=40, b=10))
    return fig


def main():
    args = parse_cli_args()
    st.set_page_config(page_title="NER Landslide Early Warning", layout="wide")
    st.title("AI-Based Landslide Early Warning — NER India")
    st.caption("Software-only, AI-driven decision-support tool. Not a guarantee of "
               "landslide occurrence or non-occurrence — validate real decisions with "
               "disaster-management authorities.")

    model_a, model_b = get_models(args.model_a, args.model_b)
    df = get_features(args.features)

    st.sidebar.header("Select a location")
    idx = st.sidebar.selectbox(
        "Location", options=df.index.tolist(),
        format_func=lambda i: location_label(df.loc[i], i),
    )
    base_row = df.loc[idx].to_dict()

    st.sidebar.header("Rainfall scenario")
    st.sidebar.caption("Simulate a heavier or lighter rain event on top of the recorded data")
    rain_multiplier = st.sidebar.slider("Rainfall multiplier", 0.0, 3.0, 1.0, 0.1)

    scenario_row = dict(base_row)
    for col in ["rainfall_1d", "rainfall_3d", "rainfall_7d", "rainfall_15d", "rainfall_30d"]:
        if col in scenario_row and pd.notna(scenario_row[col]):
            scenario_row[col] = scenario_row[col] * rain_multiplier

    result = score_single_location(scenario_row, model_a, model_b)

    col1, col2, col3 = st.columns(3)
    col1.metric("Final risk score", f"{result['final_risk_score']} / 100")
    col2.metric("Risk level", result["risk_level"])
    col3.metric("24h dynamic hazard prob.", f"{result['dynamic_hazard_score']:.2f}")

    st.subheader("Terrain / environmental values at this location")
    env_cols = [c for c in ["elevation", "slope", "aspect", "curvature", "road_distance",
                             "historical_landslide_density"] if c in base_row]
    st.table(pd.DataFrame([{c: base_row[c] for c in env_cols}]))

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Rainfall (current scenario)")
        st.plotly_chart(rainfall_trend_chart(scenario_row), use_container_width=True)

    with right:
        st.subheader("Why this score — top contributing factors (SHAP)")
        factors = explain_location(scenario_row, model_a, model_b, top_n=5)
        if factors:
            factors_df = pd.DataFrame(factors).rename(
                columns={"factor": "Contributing Factor", "contribution_pct": "Approx. Contribution (%)"}
            )
            st.table(factors_df)
        else:
            st.info("No SHAP breakdown available for this row (all-zero contribution).")

    st.subheader("Early warning")
    alert = process_alert(
        location=f"({base_row['latitude']:.3f}, {base_row['longitude']:.3f})",
        risk_score=result["final_risk_score"],
        risk_level=result["risk_level"],
        system_action=result["system_action"],
    )
    if alert:
        st.error(alert.message)
    else:
        st.success(f"No warning triggered — current level is '{result['risk_level']}' "
                   f"({result['system_action']}).")

    st.subheader("Regional risk map")
    st.caption("Static prototype view (Week 6) — showing risk levels at each historical/sample point.")
    sample_size = min(len(df), 200)  # keep the map light for the prototype
    map_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df.copy()
    scored_sample = pd.DataFrame([
        {**row, **score_single_location(row, model_a, model_b)}
        for row in map_df.to_dict("records")
    ])
    fmap = build_risk_map(scored_sample)
    st.components.v1.html(fmap._repr_html_(), height=500)


if __name__ == "__main__":
    main()
