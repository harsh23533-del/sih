"""
Week 5 — Train Model B (Dynamic Hazard).

Model A (best_model.pkl, from Week 4) gives a static susceptibility score —
how landslide-prone a location is in general. Model B layers rainfall
accumulation/trend and season on top of that susceptibility score to
estimate the probability of a landslide in a near-term window (e.g. the
next 24 hours) — the model that actually reacts to an incoming storm,
per the project plan's AI Model Design.

Requirements:
    pip install pandas numpy scikit-learn xgboost

Usage:
    python train_dynamic_hazard.py \\
        --features ../data/processed/final_feature_table.csv \\
        --model-a best_model.pkl \\
        --out-report week5_dynamic_hazard_report.csv \\
        --out-model dynamic_hazard_model.pkl
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from spatial_cv import spatial_kfold_indices
from train_models import evaluate_model, run_spatial_cv

RAINFALL_COLS = ["rainfall_1d", "rainfall_3d", "rainfall_7d", "rainfall_15d", "rainfall_30d"]


def add_dynamic_features(df: pd.DataFrame, model_a: dict) -> pd.DataFrame:
    """Adds susceptibility_score (from Model A), a simple rainfall-trend
    feature, and a monsoon-season flag — the inputs Model B trains on."""
    df = df.copy()

    a_features = model_a["feature_cols"]
    available_a = [c for c in a_features if c in df.columns]
    missing_a = set(a_features) - set(available_a)
    if missing_a:
        print(f"WARNING: Model A missing columns in this table (filled with column mean): {missing_a}")
        for col in missing_a:
            df[col] = df[available_a].mean(axis=1) if available_a else 0.0

    df["susceptibility_score"] = model_a["model"].predict_proba(df[a_features].values)[:, 1]

    # Rainfall trend: is the last day heavier than the recent 3-day average?
    # Positive = intensifying rainfall, which the plan flags as a key dynamic signal.
    if {"rainfall_1d", "rainfall_3d"}.issubset(df.columns):
        df["rainfall_trend"] = df["rainfall_1d"] - (df["rainfall_3d"] / 3.0)
    else:
        df["rainfall_trend"] = 0.0

    # Season: NER monsoon (rainfall-triggered landslides cluster Jun-Sep)
    if "date" in df.columns:
        month = pd.to_datetime(df["date"], errors="coerce").dt.month
        df["is_monsoon_season"] = month.isin([6, 7, 8, 9]).astype(int)
    else:
        df["is_monsoon_season"] = 0

    return df


def main(args):
    df = pd.read_csv(args.features)

    with open(args.model_a, "rb") as f:
        model_a = pickle.load(f)
    print(f"Loaded Model A ({model_a['name']}), {len(model_a['feature_cols'])} static features")

    df = add_dynamic_features(df, model_a)

    dynamic_features = [c for c in RAINFALL_COLS if c in df.columns] + [
        "rainfall_trend", "is_monsoon_season", "susceptibility_score",
    ]
    # soil_moisture_index is itself dynamic (baseline wetness + recent rainfall),
    # so it belongs with Model B's near-term hazard signal, not just Model A.
    if "soil_moisture_index" in df.columns:
        dynamic_features.append("soil_moisture_index")
    df = df.dropna(subset=dynamic_features + ["label"])
    print(f"Training Model B on {len(df)} rows, {df['label'].sum()} positive, "
          f"{len(dynamic_features)} dynamic features: {dynamic_features}")

    models = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "xgboost": lambda: XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=(df["label"] == 0).sum() / max(df["label"].sum(), 1),
            eval_metric="aucpr", random_state=42, n_jobs=-1
        ),
    }

    all_results = []
    for name, builder in models.items():
        print(f"\nRunning spatial CV for {name} (Model B)...")
        fold_df = run_spatial_cv(df, dynamic_features, builder,
                                  n_splits=args.n_splits, block_size_km=args.block_size_km)
        fold_df["model"] = name
        all_results.append(fold_df)
        print(fold_df.mean(numeric_only=True))

    report = pd.concat(all_results, ignore_index=True)
    summary = report.groupby("model").mean(numeric_only=True).drop(columns=["fold"])
    print("\n=== Mean metrics across folds (Model B) ===")
    print(summary)

    best_name = summary["pr_auc"].idxmax()
    print(f"\nBest dynamic hazard model by PR-AUC: {best_name}")

    best_builder = models[best_name]
    best_model = best_builder()
    best_model.fit(df[dynamic_features].values, df["label"].values)

    report.to_csv(args.out_report, index=False)
    print(f"Full fold-by-fold report saved -> {args.out_report}")

    with open(args.out_model, "wb") as f:
        pickle.dump({"model": best_model, "feature_cols": dynamic_features, "name": best_name}, f)
    print(f"Model B (refit on full data) saved -> {args.out_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model-a", required=True, help="Path to Week 4 best_model.pkl")
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--out-model", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--block-size-km", type=float, default=20)
    args = parser.parse_args()
    main(args)
