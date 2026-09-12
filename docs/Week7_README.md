# Week 7 — Dashboard & Alerts

**Goal:** wire Weeks 4-6 into one interactive Streamlit app — pick a location, see its
risk score, see *why* (SHAP), simulate a heavier storm, and see the resulting alert —
matching the plan's Final Demonstration Script end to end.

## Files
- **`explainability/shap_analysis.py`** — `explain_location(feature_dict, model_a,
  model_b, top_n=5)` runs `shap.TreeExplainer` over both models and returns the top-5
  contributing factors with a signed approximate % contribution (Table 4 style: heavy
  rainfall, slope, susceptibility, etc.). `susceptibility_score` (Model A's own output,
  also fed into Model B) is combined rather than double-counted across the two models.
- **`alerts/alert_engine.py`** — the plan's threshold-check + warning-generation stage:
  - `check_threshold()` — fires for High/Critical by default (`ALERT_LEVELS`, tune per
    district against the plan's false-alarm-tolerance metric)
  - `generate_message()` — human-readable warning text, always framed as a
    decision-support signal, not a guarantee (per the plan's Scientific & Ethical
    Positioning section)
  - `log_delivery()` — appends every alert attempt to a CSV delivery log. Prototype
    channel is `"dashboard"`; swap in real email/SMS/API calls for production while
    keeping the log, per the plan's accountability note
- **`dashboard/app.py`** — the Streamlit app itself:
  - Sidebar: location picker (from the feature table) + a rainfall-multiplier slider
    that scales all rainfall windows to simulate a heavier/lighter storm live
  - Metrics: final risk score, risk level, 24h dynamic hazard probability
  - Terrain/environmental values table for the selected location
  - Rainfall bar chart (Plotly) across the 1/3/7/15/30-day windows
  - SHAP top-5 factors table, recomputed for the current rainfall scenario
  - Alert panel — shows the generated warning message if High/Critical, otherwise a
    "no warning triggered" confirmation
  - Embedded regional risk map (reuses Week 6's `gis_map.build_risk_map`, sampled to
    200 points to keep the prototype responsive)

## Running it
```bash
cd dashboard
pip install streamlit pandas numpy plotly shap folium xgboost scikit-learn

streamlit run app.py -- \
    --features ../data/processed/final_feature_table.csv \
    --model-a ../models/best_model.pkl \
    --model-b ../models/dynamic_hazard_model.pkl
```
(the `--` before the app's own flags is required — Streamlit otherwise tries to parse
them itself)

## Notes / what to revisit
- SHAP's return shape differs slightly across `shap`/`scikit-learn`/`xgboost` versions
  for binary classifiers (list of two arrays vs. a single 3D array) —
  `_shap_contributions()` normalizes both cases to the positive-class values.
- The embedded map samples up to 200 points for responsiveness; for the full dataset,
  precompute scores with `models/risk_engine.py` and cache them instead of scoring live
  on every slider move.
- Dashboard response time target from the plan (< 3 seconds per location query) should
  be checked once real (larger) data is loaded — `st.cache_resource`/`st.cache_data` are
  already used for the model/feature-table loads to help with this.

## Output for this week
`dashboard/app.py` (Streamlit app) + `explainability/shap_analysis.py` +
`alerts/alert_engine.py` — matches the plan's "Functional end-to-end dashboard (app.py)"
deliverable.
