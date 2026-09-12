"""
Week 7 — Explainable AI (XAI).

Wraps SHAP over Model A (susceptibility) and Model B (dynamic hazard) so
every risk score shown on the dashboard can be paired with a "why" —
the top contributing factors and their approximate % contribution — per
the plan's Explainable AI section (Table 4 style output).

Requirements:
    pip install shap pandas numpy scikit-learn xgboost

Usage:
    from shap_analysis import explain_location
    factors = explain_location(feature_row, model_a, model_b, top_n=5)
"""
import sys
import os

import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from train_dynamic_hazard import add_dynamic_features  # noqa: E402


def _shap_contributions(model, X_row: pd.DataFrame, feature_cols: list) -> pd.Series:
    """Returns each feature's SHAP value for the positive (landslide) class,
    for a single-row input."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row[feature_cols])

    # Binary classifiers: SHAP's return shape varies by version/model —
    # normalize to a 1D array of per-feature SHAP values for the positive
    # (landslide) class.
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        arr = np.array(shap_values)
        if arr.ndim == 3:  # (n_samples, n_features, n_classes)
            values = arr[0, :, 1]
        elif arr.ndim == 2:  # (n_samples, n_features) — already positive-class values
            values = arr[0]
        else:
            values = arr

    return pd.Series(values, index=feature_cols)


def explain_location(feature_dict: dict, model_a: dict, model_b: dict, top_n: int = 5) -> list:
    """Returns the top_n contributing factors (across both models) for one
    location, as a list of {"factor": ..., "contribution_pct": ...} dicts —
    matching the plan's Table 4 (Contributing Factor / Approx. Contribution).

    Contribution % is each feature's |SHAP value| as a share of the total
    |SHAP value| across both models, signed to show direction (+ raises
    risk, - lowers it).
    """
    df = pd.DataFrame([feature_dict])
    df = add_dynamic_features(df, model_a)

    a_shap = _shap_contributions(model_a["model"], df, model_a["feature_cols"])
    b_shap = _shap_contributions(model_b["model"], df, model_b["feature_cols"])

    # susceptibility_score appears as an input to Model B too — combine it
    # with Model A's own contribution to that same underlying signal rather
    # than double counting it as two separate rows.
    combined = a_shap.add(b_shap, fill_value=0)

    total_abs = combined.abs().sum()
    if total_abs == 0:
        return []

    pct = (combined / total_abs * 100).sort_values(key=abs, ascending=False)

    return [
        {"factor": name, "contribution_pct": round(float(value), 1)}
        for name, value in pct.head(top_n).items()
    ]


if __name__ == "__main__":
    print("Import explain_location(feature_dict, model_a, model_b) from the dashboard "
          "or a script — this module has no standalone CLI, it needs live model objects.")
