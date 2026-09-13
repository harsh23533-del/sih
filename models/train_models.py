"""
Week 4 — Train Random Forest and XGBoost on the Week 3 final feature table,
evaluate with spatial cross-validation, and compare on precision, recall, F1,
ROC-AUC, PR-AUC, and false-negative rate (tracked separately since a missed
warning is costlier than a false alarm, per the project plan).

Requirements:
    pip install pandas numpy scikit-learn xgboost

Usage:
    python train_models.py \\
        --features ../../data/processed/final_feature_table.csv \\
        --out-report ../../models/week4_model_comparison_report.csv \\
        --out-model ../../models/best_model.pkl
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)
from xgboost import XGBClassifier

from spatial_cv import spatial_kfold_indices

FEATURE_COLS = [
    "elevation", "slope", "aspect", "curvature", "ruggedness_tri",
    "twi", "drainage_distance_km",
    "rainfall_1d", "rainfall_3d", "rainfall_7d", "rainfall_15d", "rainfall_30d",
    "road_distance", "historical_landslide_density", "distance_to_nearest_landslide_m",
    # Literature-standard factors added on top of the original terrain/rainfall set —
    # soil/geology/land_use are already integer-coded categories from the synthetic
    # rasters, so tree models can split on them directly with no extra encoding.
    "soil", "geology", "NDVI", "land_use",
    "seismic_pga", "fault_distance_km",
    "soil_moisture_baseline", "soil_moisture_index", "soil_moisture_satellite",
    "insar_deformation_mm_yr",
    "insolation_proxy", "freeze_thaw_index", "root_cohesion_proxy",
    "glacial_lake_distance_km", "population_density", "exposure_index",
]


def false_negative_rate(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tp) if (fn + tp) > 0 else np.nan


def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan,
        "false_negative_rate": false_negative_rate(y_test, y_pred),
    }


def run_spatial_cv(df: pd.DataFrame, feature_cols: list, model_builder, n_splits=5,
                    block_size_km=20):
    fold_metrics = []
    X = df[feature_cols].values
    y = df["label"].values

    for fold_i, (train_idx, test_idx) in enumerate(
        spatial_kfold_indices(df, n_splits=n_splits, block_size_km=block_size_km)
    ):
        model = model_builder()
        metrics = evaluate_model(model, X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        metrics["fold"] = fold_i
        fold_metrics.append(metrics)

    return pd.DataFrame(fold_metrics)


def main(args):
    df = pd.read_csv(args.features)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available_features)
    if missing:
        print(f"WARNING: missing feature columns (skipping): {missing}")

    df = df.dropna(subset=available_features + ["label"])
    print(f"Training on {len(df)} rows, {df['label'].sum()} positive / "
          f"{(df['label'] == 0).sum()} negative, {len(available_features)} features")

    models = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "xgboost": lambda: XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=(df["label"] == 0).sum() / max(df["label"].sum(), 1),
            eval_metric="aucpr", random_state=42, n_jobs=-1
        ),
    }

    all_results = []
    for name, builder in models.items():
        print(f"\nRunning spatial CV for {name}...")
        fold_df = run_spatial_cv(df, available_features, builder,
                                  n_splits=args.n_splits, block_size_km=args.block_size_km)
        fold_df["model"] = name
        all_results.append(fold_df)
        print(fold_df.mean(numeric_only=True))

    report = pd.concat(all_results, ignore_index=True)
    summary = report.groupby("model").mean(numeric_only=True).drop(columns=["fold"])
    print("\n=== Mean metrics across folds ===")
    print(summary)

    report.to_csv(args.out_report, index=False)
    print(f"\nFull fold-by-fold report saved -> {args.out_report}")

    # Pick best model by PR-AUC (most meaningful given class imbalance, per plan)
    best_name = summary["pr_auc"].idxmax()
    print(f"\nBest model by PR-AUC: {best_name}")

    best_builder = models[best_name]
    best_model = best_builder()
    best_model.fit(df[available_features].values, df["label"].values)

    with open(args.out_model, "wb") as f:
        pickle.dump({"model": best_model, "feature_cols": available_features, "name": best_name}, f)
    print(f"Best model (refit on full data) saved -> {args.out_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--out-model", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--block-size-km", type=float, default=20)
    args = parser.parse_args()
    main(args)
