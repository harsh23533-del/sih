"""
Week 5 — Risk Engine.

Combines Model A (static susceptibility, Week 4) and Model B (dynamic
hazard, Week 5) into a single 0-100 risk score, and classifies that score
into one of five alert levels with a recommended system action, per the
project plan's Risk Engine and Risk Classification & Alert Levels sections.

    Final Risk = 40% x Susceptibility + 60% x Dynamic Hazard

These weights are a starting hypothesis, not a scientific constant --
tune them against the historical dataset and review with a domain expert
before treating them as final (see docs/Week5_README.md).

Requirements:
    pip install pandas numpy scikit-learn xgboost

Usage (batch scoring of a feature table):
    python risk_engine.py \\
        --features ../data/processed/final_feature_table.csv \\
        --model-a best_model.pkl \\
        --model-b dynamic_hazard_model.pkl \\
        --out risk_scores.csv
"""
import argparse
import pickle

import pandas as pd

from train_dynamic_hazard import add_dynamic_features

# Static weight applied to susceptibility, dynamic weight to hazard. Tune per Table 3 note.
WEIGHT_SUSCEPTIBILITY = 0.40
WEIGHT_DYNAMIC_HAZARD = 0.60

# Table 3 — Prototype risk thresholds (requires calibration against local
# historical events and formal sign-off from disaster-management authorities).
RISK_LEVELS = [
    (0, 20, "Very Low", "Normal monitoring"),
    (20, 40, "Low", "Continue monitoring"),
    (40, 60, "Moderate", "Watch status"),
    (60, 80, "High", "Warning / closer monitoring"),
    (80, 101, "Critical", "Immediate alert / authority review"),
]


def classify_risk(score: float) -> tuple:
    """Returns (level, system_action) for a 0-100 risk score."""
    for low, high, level, action in RISK_LEVELS:
        if low <= score < high:
            return level, action
    return "Critical", "Immediate alert / authority review"


def load_models(model_a_path: str, model_b_path: str) -> tuple:
    with open(model_a_path, "rb") as f:
        model_a = pickle.load(f)
    with open(model_b_path, "rb") as f:
        model_b = pickle.load(f)
    return model_a, model_b


def score_table(df: pd.DataFrame, model_a: dict, model_b: dict) -> pd.DataFrame:
    """Scores every row of a feature table, returning susceptibility,
    dynamic hazard, combined risk score, level, and action."""
    df = add_dynamic_features(df, model_a)  # also computes susceptibility_score

    b_features = model_b["feature_cols"]
    df["dynamic_hazard_score"] = model_b["model"].predict_proba(df[b_features].values)[:, 1]

    df["final_risk_score"] = (
        WEIGHT_SUSCEPTIBILITY * df["susceptibility_score"] * 100
        + WEIGHT_DYNAMIC_HAZARD * df["dynamic_hazard_score"] * 100
    ).clip(0, 100)

    levels_actions = df["final_risk_score"].apply(classify_risk)
    df["risk_level"] = levels_actions.apply(lambda t: t[0])
    df["system_action"] = levels_actions.apply(lambda t: t[1])

    return df


def score_single_location(feature_dict: dict, model_a: dict, model_b: dict) -> dict:
    """Scores one location from a dict of feature values (e.g. from the
    Week 6/7 GIS dashboard on-click lookup). Returns a plain dict, ready
    for the dashboard or an alert message."""
    df = pd.DataFrame([feature_dict])
    scored = score_table(df, model_a, model_b)
    row = scored.iloc[0]
    return {
        "susceptibility_score": round(float(row["susceptibility_score"]), 3),
        "dynamic_hazard_score": round(float(row["dynamic_hazard_score"]), 3),
        "final_risk_score": round(float(row["final_risk_score"]), 1),
        "risk_level": row["risk_level"],
        "system_action": row["system_action"],
    }


def main(args):
    df = pd.read_csv(args.features)
    model_a, model_b = load_models(args.model_a, args.model_b)

    scored = score_table(df, model_a, model_b)

    out_cols = [c for c in ["latitude", "longitude", "date"] if c in scored.columns]
    out_cols += ["susceptibility_score", "dynamic_hazard_score", "final_risk_score",
                 "risk_level", "system_action"]
    scored[out_cols].to_csv(args.out, index=False)

    print(f"Scored {len(scored)} rows -> {args.out}")
    print("\nRisk level distribution:")
    print(scored["risk_level"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model-a", required=True, help="Path to Week 4 best_model.pkl")
    parser.add_argument("--model-b", required=True, help="Path to Week 5 dynamic_hazard_model.pkl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args)
