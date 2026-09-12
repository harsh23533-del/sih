#!/usr/bin/env bash
# Runs the FULL pipeline end-to-end on synthetic data: data generation ->
# negative sampling -> feature table -> Model A -> Model B -> risk scoring
# -> GIS map. Swap generate_synthetic_data.py for the real Week 2
# downloaders once real data is available — nothing else in this script
# needs to change.
set -e

mkdir -p data/processed

echo "== 1/6: Generating synthetic terrain/rainfall/roads/landslide data =="
(cd preprocessing/data_collection && python3 generate_synthetic_data.py --out-dir ../../data)

echo "== 2/6: Negative sampling =="
(cd preprocessing && python3 negative_sampling.py \
  --dem ../data/terrain/sikkim_srtm30m.tif \
  --positives ../data/historical_landslides/coolr_ner_labeled.csv \
  --rainfall-start 2018-01-01 --rainfall-end 2024-12-31 \
  --neg-to-pos-ratio 2 --exclusion-radius-km 2 \
  --out ../data/processed/negative_samples.csv)

echo "== 3/6: Building final feature table =="
(cd preprocessing && python3 build_feature_table.py \
  --positives ../data/historical_landslides/coolr_ner_labeled.csv \
  --negatives ../data/processed/negative_samples.csv \
  --dem ../data/terrain/sikkim_srtm30m.tif \
  --rainfall-dir ../data/rainfall \
  --osm-roads ../data/infrastructure/sikkim_osm.geojson \
  --out ../data/processed/final_feature_table.csv)

echo "== 4/6: Training Model A (susceptibility) =="
(cd models && python3 train_models.py \
  --features ../data/processed/final_feature_table.csv \
  --out-report week4_model_comparison_report.csv \
  --out-model best_model.pkl)

echo "== 5/6: Training Model B (dynamic hazard) =="
(cd models && python3 train_dynamic_hazard.py \
  --features ../data/processed/final_feature_table.csv \
  --model-a best_model.pkl \
  --out-report week5_dynamic_hazard_report.csv \
  --out-model dynamic_hazard_model.pkl)

echo "== 6/6: Risk scoring + GIS map =="
(cd models && python3 risk_engine.py \
  --features ../data/processed/final_feature_table.csv \
  --model-a best_model.pkl --model-b dynamic_hazard_model.pkl \
  --out risk_scores.csv)
(cd dashboard && python3 gis_map.py --scores ../models/risk_scores.csv --out ner_risk_map.html)

echo ""
echo "Done. To launch the dashboard:"
echo "  cd dashboard && streamlit run app.py -- \\"
echo "    --features ../data/processed/final_feature_table.csv \\"
echo "    --model-a ../models/best_model.pkl --model-b ../models/dynamic_hazard_model.pkl"
