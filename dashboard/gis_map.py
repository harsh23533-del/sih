"""
Week 6 — GIS Map.

Builds an interactive Folium map of the NER study area, colour-coding every
scored location green -> red by its risk level (Week 5's risk_engine output),
with a click popup showing the score breakdown and terrain/rainfall context —
per the plan's GIS Dashboard section (state/district/point selection,
colour-coded risk, per-location detail on click).

This is the standalone map prototype for Week 6. Week 7 wires this same
scoring/colour logic into the full Streamlit dashboard (location search,
live rainfall-scenario slider, SHAP explanation panel, alerts).

Requirements:
    pip install pandas folium

Usage:
    python gis_map.py \\
        --scores ../models/risk_scores.csv \\
        --out ner_risk_map.html
    (run models/risk_engine.py first to produce risk_scores.csv, or pass
    --features/--model-a/--model-b instead of --scores to score on the fly)
"""
import argparse
import os
import sys

import folium
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

# Colour per risk level, matching Table 3's green -> red alert scale
RISK_COLORS = {
    "Very Low": "#2e7d32",
    "Low": "#8bc34a",
    "Moderate": "#f9c74f",
    "High": "#f3722c",
    "Critical": "#d00000",
}

# Rough NER India bounding box, used as the map's default view
NER_CENTER = (26.5, 92.5)
NER_DEFAULT_ZOOM = 6


def _popup_html(row: pd.Series) -> str:
    fields = [
        ("Risk score", f"{row.get('final_risk_score', 'n/a')} / 100"),
        ("Risk level", row.get("risk_level", "n/a")),
        ("System action", row.get("system_action", "n/a")),
        ("Susceptibility", row.get("susceptibility_score", "n/a")),
        ("24h dynamic hazard", row.get("dynamic_hazard_score", "n/a")),
        ("Date", row.get("date", "n/a")),
    ]
    for rain_col in ("rainfall_1d", "rainfall_3d", "rainfall_7d"):
        if rain_col in row:
            fields.append((rain_col, row[rain_col]))

    rows_html = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in fields)
    return f"<table>{rows_html}</table>"


def build_risk_map(scored_df: pd.DataFrame, center=NER_CENTER, zoom=NER_DEFAULT_ZOOM) -> folium.Map:
    """Builds a Folium map with one colour-coded, clickable marker per row
    of a Week 5 risk_engine-scored table (needs latitude, longitude,
    risk_level at minimum)."""
    fmap = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")

    legend_items = "".join(
        f'<i style="background:{color}"></i> {level}<br>' for level, color in RISK_COLORS.items()
    )
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 6px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-size: 13px;">
    <b>Risk level</b><br>{legend_items}
    <style>i {{width: 12px; height: 12px; display: inline-block; margin-right: 4px;
              border-radius: 50%;}}</style>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    for _, row in scored_df.iterrows():
        if pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
            continue
        color = RISK_COLORS.get(row.get("risk_level"), "#777777")
        folium.CircleMarker(
            location=(row["latitude"], row["longitude"]),
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(_popup_html(row), max_width=280),
            tooltip=f"{row.get('risk_level', 'n/a')} risk",
        ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    return fmap


def main(args):
    if args.scores:
        scored = pd.read_csv(args.scores)
    else:
        # Score on the fly using Week 5's risk engine
        from risk_engine import load_models, score_table  # local import, path added above

        df = pd.read_csv(args.features)
        model_a, model_b = load_models(args.model_a, args.model_b)
        scored = score_table(df, model_a, model_b)

    fmap = build_risk_map(scored)
    fmap.save(args.out)
    print(f"Saved interactive map ({len(scored)} points) -> {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", help="Pre-scored CSV from risk_engine.py (has risk_level etc.)")
    parser.add_argument("--features", help="Raw feature table (used with --model-a/--model-b instead of --scores)")
    parser.add_argument("--model-a", help="Path to Week 4 best_model.pkl")
    parser.add_argument("--model-b", help="Path to Week 5 dynamic_hazard_model.pkl")
    parser.add_argument("--out", default="ner_risk_map.html")
    args = parser.parse_args()
    if not args.scores and not (args.features and args.model_a and args.model_b):
        parser.error("Provide either --scores, or --features + --model-a + --model-b")
    main(args)
