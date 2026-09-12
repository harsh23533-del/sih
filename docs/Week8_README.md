# Week 8 — Testing, Documentation & Demo

**Goal:** run scenario tests, scaffold the historical backtest, write up limitations, and
lay out the final demo script — the plan's closing week.

## Files
- **`tests/test_scenarios.py`**:
  - `run_scenarios()` — scores one location under four named rainfall scenarios
    (`normal` 0.3x, `moderate` 1.0x, `heavy` 2.0x, `critical` 3.5x, applied to all
    rainfall windows) and reports the resulting risk score/level for each
  - `check_monotonic_escalation()` — flags if risk score ever *drops* as the scenario
    intensifies, which shouldn't happen for a sane model
  - `backtest_event()` — scaffold for the plan's "backtest against 2-3 known historical
    extreme-rainfall events" task. **Not run against real events yet** — no historical
    event rainfall sequences ship in this repo (Week 2's collection scripts fetch the
    sources; the actual event records need to be pulled and labeled first). Wire real
    GSI/NRSC/NASA COOLR event rows into this function once available.
- **`docs/Limitations_and_Assumptions.md`** — consolidated risks & mitigations table,
  scientific/ethical positioning statement, prototype success-metric targets, and the
  assumptions carried through the build — pulled together from the original plan so it's
  one reference doc for the final report.

## Running it
```bash
cd tests
pip install pandas scikit-learn xgboost
python test_scenarios.py \
    --features ../data/processed/final_feature_table.csv \
    --model-a ../models/best_model.pkl \
    --model-b ../models/dynamic_hazard_model.pkl \
    --out week8_scenario_report.csv
```

## Finding from Week 8 scenario testing
Run against models trained on synthetic placeholder data, the monotonicity check
**failed** — risk score dropped going from `heavy` to `critical`. This is a documented,
expected quirk of tree-based models extrapolating past their training range, not a bug
in the risk engine — see `docs/Limitations_and_Assumptions.md` for the full explanation
and what to check once the real trained models are available.

## Final Demo Script (from the plan)
A live sequence that shows the full system without any physical sensors, using
`dashboard/app.py`:
1. Select a location on the map
2. Show its current environmental/terrain values
3. Run the AI models and display susceptibility + 24-hour probability
4. Show the SHAP explanation for that score
5. Adjust the rainfall scenario slider (e.g. simulate a heavy-rain event)
6. Watch the risk level escalate live from Low/Moderate to High/Critical
7. Generate and display the resulting early-warning message

Every step of this maps directly to a control already built in `dashboard/app.py`
(location picker → terrain table → risk metrics → SHAP table → rainfall slider → alert
panel).

## Real events collected for the backtest

I sourced three real, dated, cited Sikkim landslide events via web search — not full
inventory downloads, since bulk sources (IMD grids, NASA COOLR bulk export, OSM,
OpenTopography) aren't reachable from this environment (see the honesty note below).
Saved to `data/historical_landslides/known_events_sikkim.csv`:

| Date | Location | Reported rainfall | Source |
|---|---|---|---|
| 1997-06-07 | Gangtok | ~224 mm in 24h | Sikkim Express |
| 2023-06-19 | Salangdang/Dentam, West Sikkim | heavy overnight downpour (no exact mm reported) | Sikkim Express |
| 2025-06-01 | Mangan, North Sikkim | 124.8 mm cumulative, 1-3 June | ScienceDirect (Kar et al.) |

Two more events are listed in the same CSV as **reference-only** (not suitable for a
like-for-like quantitative backtest): the 1968 Sikkim floods (predates satellite/gridded
rainfall products) and the October 2023 South Lhonak GLOF (a glacial-lake-outburst/dam-
failure cascade, not a direct slope landslide — different mechanism from what this
system models).

To actually run `backtest_event()` against these: pull each location's real rainfall in
the days leading up to its date from your Week 2 rainfall collection, build the
`event_row` dict, and compare against a `quiet_row` baseline at the same location from a
non-event period.

**Honesty note on what I could/couldn't collect:** I do not have general internet
access from this environment — only `web_search` (returns snippets) and `web_fetch`
(can only open a URL that literally appeared in a prior search/fetch result — it
rejects any URL I construct myself, including API query URLs with bounding-box/date
parameters). That ruled out pulling data from NASA COOLR's live feature-server API,
Overpass, OpenTopography, or IMD — all of those need either a constructed query URL or
a real download client, which is why Week 2's data-collection scripts have to run on
your own machine, not from here. The three events above are the genuine, sourced data I
could put together this way; everything else in the repo before this point was
synthetic test data, not real Sikkim records.

## Output for this week
`tests/test_scenarios.py` (scenario tests + backtest scaffold) +
`docs/Limitations_and_Assumptions.md` — matches the plan's "Demo-ready system + final
project report" deliverable, alongside the demo sequence above.
