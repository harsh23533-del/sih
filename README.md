# AI-Based IoT-Free Geospatial Early Warning & Landslide Risk Monitoring System — NER India

Software-only, AI-driven early warning and risk monitoring system for landslides across
the North Eastern Region (NER) of India, using open rainfall/terrain/soil/satellite data
and machine learning (no physical sensors required).

## Project status (8-week build plan)
- [x] Week 1 — Research & Understanding (`docs/Week1_Literature_Context_Brief.md`)
- [x] Week 2 — Data Collection (`docs/Week2_Data_Collection_Guide.md`, `preprocessing/data_collection/`)
- [x] Week 3 — Cleaning & Feature Engineering (`docs/Week3_README.md`, `preprocessing/`)
- [x] Week 4 — Model Training (`docs/Week4_README.md`, `models/spatial_cv.py`, `models/train_models.py`)
- [x] Week 5 — Risk Engine (`docs/Week5_README.md`, `models/train_dynamic_hazard.py`, `models/risk_engine.py`)
- [x] Week 6 — GIS Map (`docs/Week6_README.md`, `dashboard/gis_map.py`)
- [x] Week 7 — Dashboard & Alerts (`docs/Week7_README.md`, `dashboard/app.py`, `explainability/shap_analysis.py`, `alerts/alert_engine.py`)
- [x] Week 8 — Testing, Documentation & Demo (`docs/Week8_README.md`, `tests/test_scenarios.py`, `docs/Limitations_and_Assumptions.md`)

All 8 weeks are now scaffolded **and the full pipeline runs end-to-end on synthetic
data** — one command reproduces everything:

```
bash run_synthetic_pipeline.sh
```

This generates synthetic terrain/rainfall/roads/landslide data
(`preprocessing/data_collection/generate_synthetic_data.py`), builds the feature table,
trains both models, scores risk, and builds the GIS map. Then launch the dashboard:

```
cd dashboard && streamlit run app.py -- \
  --features ../data/processed/final_feature_table.csv \
  --model-a ../models/best_model.pkl --model-b ../models/dynamic_hazard_model.pkl
```

See `docs/Limitations_and_Assumptions.md` before treating any output as
production-ready — every model here is trained-and-tested against **synthetic**
placeholder data, not real NER landslide records. When real data is ready, swap
`generate_synthetic_data.py`'s output for the real Week 2 downloaders
(`preprocessing/data_collection/download_dem.py`, `download_rainfall.py`,
`download_osm_infra.py`, `fetch_nasa_coolr.py`) and re-run the same script —
nothing else in the pipeline needs to change.

### Deployment
- `requirements.txt` — full pipeline (data gen + training + dashboard)
- `requirements-dashboard.txt` — lighter deploy-only deps for Streamlit
  Community Cloud (the dashboard only needs to *load* the committed
  `.pkl` models and `final_feature_table.csv`, not rasterio/geopandas/xarray)
- To deploy: push this repo to GitHub, then on
  [share.streamlit.io](https://share.streamlit.io) point a new app at
  `dashboard/app.py` with `requirements-dashboard.txt`, and set app arguments
  `-- --features data/processed/final_feature_table.csv --model-a models/best_model.pkl --model-b models/dynamic_hazard_model.pkl`

## Folder structure
```
sih/
├── data/              # rainfall, terrain, soil/geology, infrastructure, historical landslides, satellite
├── preprocessing/      # feature engineering (terrain, rainfall, density/distance, negative sampling)
│   └── data_collection/  # source download scripts (IMD, OpenTopography, NASA COOLR, OSM)
├── models/             # susceptibility_model.py, warning_model.py (Week 4)
├── explainability/      # shap_analysis.py (Week 4/7)
├── dashboard/           # app.py — Streamlit dashboard (Week 7)
├── alerts/              # alert_engine.py (Week 7)
├── tests/               # unit tests for each module
├── notebooks/           # EDA and model experimentation
└── docs/                # planning docs and week-by-week guides
```

## Phase 1 target
Sikkim (best historical inventory + DEM coverage), expanding to full NER in later phases.
