"""
Week 8 — Testing, Documentation & Demo.

Two things, per the plan's Week 8 tasks:

1. Scenario tests — run the same location through normal / moderate /
   heavy-rain / critical rainfall scenarios and check the risk engine
   escalates sensibly (score goes up, level moves toward Critical) as
   rainfall increases. This is a structural sanity check on the pipeline,
   not a substitute for validation against real data.

2. Backtest scaffold — `backtest_event()` compares the risk score in the
   days leading up to a known historical landslide against a quiet-period
   baseline at the same location. NO real historical event data ships in
   this repo (see docs/Limitations_and_Assumptions.md) — plug in real
   GSI/NRSC/NASA COOLR event records here once Week 2's data collection
   is filled in, per the plan's "backtest against 2-3 known historical
   extreme-rainfall events" task.

Requirements:
    pip install pandas numpy scikit-learn xgboost

Usage:
    python test_scenarios.py --model-a ../models/best_model.pkl --model-b ../models/dynamic_hazard_model.pkl
"""
import argparse
import pickle
import sys
import os

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from risk_engine import score_single_location  # noqa: E402

# Multiplies a location's recorded rainfall to approximate each named
# scenario. These multipliers are illustrative, not calibrated against a
# real rainfall-intensity distribution for the region.
SCENARIOS = {
    "normal": 0.3,
    "moderate": 1.0,
    "heavy": 2.0,
    "critical": 3.5,
}

RAINFALL_COLS = ["rainfall_1d", "rainfall_3d", "rainfall_7d", "rainfall_15d", "rainfall_30d"]


def build_scenario_row(base_row: dict, multiplier: float) -> dict:
    row = dict(base_row)
    for col in RAINFALL_COLS:
        if col in row and pd.notna(row[col]):
            row[col] = row[col] * multiplier
    return row


def run_scenarios(base_row: dict, model_a: dict, model_b: dict) -> pd.DataFrame:
    """Scores the same location under each named rainfall scenario."""
    results = []
    for name, multiplier in SCENARIOS.items():
        scenario_row = build_scenario_row(base_row, multiplier)
        result = score_single_location(scenario_row, model_a, model_b)
        result["scenario"] = name
        result["rainfall_multiplier"] = multiplier
        results.append(result)
    return pd.DataFrame(results)


def check_monotonic_escalation(scenario_df: pd.DataFrame) -> list:
    """Returns a list of warning strings if risk does NOT increase as the
    rainfall scenario intensifies (normal -> moderate -> heavy -> critical).
    An empty list means the escalation behaved as expected."""
    ordered = scenario_df.set_index("scenario").loc[["normal", "moderate", "heavy", "critical"]]
    scores = ordered["final_risk_score"].tolist()

    warnings = []
    for i in range(len(scores) - 1):
        if scores[i + 1] < scores[i]:
            stage_a, stage_b = ordered.index[i], ordered.index[i + 1]
            warnings.append(
                f"Risk score DROPPED from {stage_a} ({scores[i]:.1f}) to {stage_b} "
                f"({scores[i + 1]:.1f}) — expected non-decreasing risk with heavier rainfall."
            )
    return warnings


def backtest_event(event_row: dict, quiet_row: dict, model_a: dict, model_b: dict) -> dict:
    """Scaffold: compares the risk score right before a known historical
    landslide (`event_row`, its rainfall as recorded leading up to the
    event) against a quiet-period baseline at the same location
    (`quiet_row`). A meaningful early-warning signal should show the
    event score clearly higher than the quiet baseline, ahead of the
    plan's 6-12 hour lead-time target.

    Three real, sourced Sikkim events suitable for this are listed in
    data/historical_landslides/known_events_sikkim.csv (Gangtok 1997-06-07,
    West Sikkim 2023-06-19, Mangan 2025-06-01) — build event_row for each
    by pulling that location's actual recorded rainfall in the days before
    from your Week 2 rainfall data, once collected. The CSV also lists two
    reference-only events (1968 floods, 2023 GLOF) that aren't suitable for
    a like-for-like quantitative backtest — see the notes column."""
    event_result = score_single_location(event_row, model_a, model_b)
    quiet_result = score_single_location(quiet_row, model_a, model_b)
    return {
        "event_risk_score": event_result["final_risk_score"],
        "event_risk_level": event_result["risk_level"],
        "quiet_baseline_score": quiet_result["final_risk_score"],
        "quiet_baseline_level": quiet_result["risk_level"],
        "escalated_correctly": event_result["final_risk_score"] > quiet_result["final_risk_score"],
    }


def main(args):
    with open(args.model_a, "rb") as f:
        model_a = pickle.load(f)
    with open(args.model_b, "rb") as f:
        model_b = pickle.load(f)

    df = pd.read_csv(args.features)
    base_row = df.iloc[args.row_index].to_dict()
    print(f"Base location: ({base_row.get('latitude')}, {base_row.get('longitude')})")

    scenario_df = run_scenarios(base_row, model_a, model_b)
    print("\n=== Scenario results ===")
    print(scenario_df[["scenario", "rainfall_multiplier", "final_risk_score", "risk_level"]]
          .to_string(index=False))

    warnings = check_monotonic_escalation(scenario_df)
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\nRisk escalated as expected across all four scenarios (normal -> critical).")

    if args.out:
        scenario_df.to_csv(args.out, index=False)
        print(f"\nScenario report saved -> {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="../data/processed/final_feature_table.csv")
    parser.add_argument("--row-index", type=int, default=0, help="Which row of the feature table to test scenarios on")
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--out", help="Optional path to save the scenario report CSV")
    args = parser.parse_args()
    main(args)
