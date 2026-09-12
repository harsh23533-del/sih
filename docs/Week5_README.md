# Week 5 — Risk Engine

**Goal:** layer a dynamic, rainfall-reactive model (Model B) on top of Week 4's static
susceptibility model (Model A), and combine both into a single 0-100 risk score with
alert-level classification.

## Files
- **`train_dynamic_hazard.py`** — trains Model B. Loads Model A (`best_model.pkl`),
  computes each row's `susceptibility_score`, and adds two derived dynamic features:
  - `rainfall_trend` — `rainfall_1d - (rainfall_3d / 3)`, a simple intensifying-rainfall
    signal (positive = last day heavier than the recent 3-day average)
  - `is_monsoon_season` — 1 if the record's month falls in Jun-Sep, else 0

  Model B is then trained (same spatial-CV / RF vs XGBoost comparison pattern as Week 4,
  reusing `spatial_cv.py`) on: `rainfall_1d/3d/7d/15d/30d`, `rainfall_trend`,
  `is_monsoon_season`, and `susceptibility_score`. Best model by PR-AUC is refit on full
  data and saved as `dynamic_hazard_model.pkl`.
- **`risk_engine.py`** — combines both models:
  ```
  Final Risk = 40% x Susceptibility (Model A) + 60% x Dynamic Hazard (Model B)
  ```
  scaled to 0-100, then classified per the plan's Table 3 thresholds:

  | Score | Level | System Action |
  |---|---|---|
  | 0-20 | Very Low | Normal monitoring |
  | 20-40 | Low | Continue monitoring |
  | 40-60 | Moderate | Watch status |
  | 60-80 | High | Warning / closer monitoring |
  | 80-100 | Critical | Immediate alert / authority review |

  Exposes `score_table()` for batch scoring a feature table, and
  `score_single_location(feature_dict, model_a, model_b)` for one-off lookups — this is
  what the Week 6/7 GIS dashboard will call when a user selects a point on the map.

## Running it
```bash
cd models
pip install pandas numpy scikit-learn xgboost

# 1. Train Model B (needs Week 4's best_model.pkl already saved)
python train_dynamic_hazard.py \
    --features ../data/processed/final_feature_table.csv \
    --model-a best_model.pkl \
    --out-report week5_dynamic_hazard_report.csv \
    --out-model dynamic_hazard_model.pkl

# 2. Run the combined risk engine
python risk_engine.py \
    --features ../data/processed/final_feature_table.csv \
    --model-a best_model.pkl \
    --model-b dynamic_hazard_model.pkl \
    --out risk_scores.csv
```

## Notes / what to revisit
- The 40/60 weighting is a starting hypothesis per the plan, not tuned — once real data
  is profiled, consider fitting a small logistic meta-model on
  `[susceptibility_score, dynamic_hazard_score] -> label` instead of a fixed blend.
- `rainfall_trend` and `is_monsoon_season` are simple placeholders for the plan's
  "rainfall intensity/trend, season" inputs — swap in an actual hourly-intensity feature
  if higher-resolution rainfall data becomes available (see Table 2's coarse-resolution
  risk/mitigation note).
- Backtest against 2-3 known historical extreme-rainfall events (Week 8 task) once this
  is wired up, to confirm the risk score escalates appropriately ahead of the event.

## Output for this week
`dynamic_hazard_model.pkl` (Model B) + `week5_dynamic_hazard_report.csv` (fold metrics) +
`risk_engine.py` producing a 0-100 `final_risk_score` with `risk_level`/`system_action`
per row — matches the plan's "Working `risk_engine.py` jo 0-100 score output kare"
deliverable.
