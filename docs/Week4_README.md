# Week 4 — Model Training & Comparison

**Goal:** train Random Forest and XGBoost on Week 3's final feature table, compare them
properly (spatial CV, not random split), and select a baseline model.

## Files
- **`spatial_cv.py`** — leave-one-block-out spatial cross-validation. Points are grouped
  into ~20km grid blocks (tunable via `--block-size-km`); each fold holds out whole blocks,
  not random points. This matters because landslide data is spatially autocorrelated — a
  random split lets nearby points leak between train/test and **overstates accuracy**, per
  the plan's Model Validation Protocol.
- **`train_models.py`** — trains both models across 5 spatial folds, reports:
  - Precision, Recall, F1
  - ROC-AUC and **PR-AUC** (PR-AUC is the primary metric — positive events are rare, so
    ROC-AUC alone can look deceptively good)
  - **False-negative rate**, tracked separately and reported per-fold, since a missed
    warning is far more costly than a false alarm
  - Picks the best model by mean PR-AUC across folds, refits it on the full dataset, and
    saves it as `best_model.pkl`

## Class imbalance handling
- Random Forest: `class_weight="balanced"`
- XGBoost: `scale_pos_weight` set to the actual negative:positive ratio in your data

These are starting points, not final settings — if PR-AUC is still weak, revisit the
negative-sampling ratio from Week 3 (`neg_to_pos_ratio`) before tuning hyperparameters
further.

## Running it
```bash
cd preprocessing/../models   # or wherever you place these scripts
pip install pandas numpy scikit-learn xgboost
python train_models.py \
    --features ../data/processed/final_feature_table.csv \
    --out-report week4_model_comparison_report.csv \
    --out-model best_model.pkl
```

## What to check before moving to Week 5
- Compare fold-to-fold variance in PR-AUC — high variance across spatial folds usually
  means your feature set or negative sampling needs more work before the risk engine is
  built on top of it
- Look at which features drive the most splits (feature importances) — sanity-check these
  against domain expectations (rainfall and slope should typically dominate)
- Confirm the false-negative rate is being tracked and reported alongside every score, not
  just the aggregate metrics — this becomes the headline number disaster-management
  reviewers will care about most

## Output for this week
`week4_model_comparison_report.csv` (per-fold metrics for both models) + `best_model.pkl`
(the selected baseline model, refit on full data) — matches the plan's "Baseline model
comparison report + best model selected" deliverable.

## Note: Model A vs Model B
This week trains the **static susceptibility model** (Model A) using terrain + soil/geology
features. The **dynamic hazard model** (Model B) — which layers rainfall + susceptibility
score to predict 24-hour probability — gets built in Week 5 as part of the risk engine,
reusing this same `train_models.py` pattern with rainfall features weighted more heavily
and susceptibility score added as an input feature.
