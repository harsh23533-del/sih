# Week 6 — GIS Map

**Goal:** an interactive NER map, colour-coded by risk level, where clicking any point
shows its full score breakdown — the "GIS Dashboard" section's core requirement, built
here as a standalone prototype before Week 7 wraps it into the full Streamlit app.

## Files
- **`gis_map.py`** — builds a Folium map from a Week 5 `risk_engine`-scored table:
  - One circle marker per location, coloured green (Very Low) -> red (Critical) per
    Table 3's alert scale (`RISK_COLORS`)
  - Click popup shows: final risk score, risk level, system action, susceptibility
    score, 24h dynamic hazard score, date, and available rainfall windows
  - Built-in legend (bottom-left) mapping colour -> risk level
  - Can either load a pre-scored CSV (`--scores`, from `models/risk_engine.py`) or score
    a raw feature table on the fly (`--features` + `--model-a` + `--model-b`)
  - Uses plain OpenStreetMap tiles (CartoDB's free tiles now require an API key, so this
    avoids an extra signup step for the prototype)

## Running it
```bash
cd dashboard
pip install pandas folium

# Option A: score first, then map
python ../models/risk_engine.py \
    --features ../data/processed/final_feature_table.csv \
    --model-a ../models/best_model.pkl \
    --model-b ../models/dynamic_hazard_model.pkl \
    --out ../models/risk_scores.csv
python gis_map.py --scores ../models/risk_scores.csv --out ner_risk_map.html

# Option B: score and map in one step
python gis_map.py \
    --features ../data/processed/final_feature_table.csv \
    --model-a ../models/best_model.pkl \
    --model-b ../models/dynamic_hazard_model.pkl \
    --out ner_risk_map.html
```
Open `ner_risk_map.html` in a browser — pan/zoom/click work offline once generated.

## Notes / what's deferred to Week 7
- **State/district dropdown selection** (from the GIS Dashboard spec) needs an actual
  UI framework to be interactive rather than a static HTML file — that lands in Week 7's
  Streamlit dashboard, which reuses this same colour/popup logic.
- **Rainfall/risk trend chart per location** and the **rainfall-scenario slider** are also
  Week 7 dashboard features, not part of this static map prototype.
- District/state boundary polygons (GeoPandas overlay) can be added later if a NER
  admin-boundary shapefile is sourced — current version plots points only.

## Output for this week
`gis_map.py` + a generated `ner_risk_map.html` — matches the plan's "Interactive GIS map
prototype" deliverable.
