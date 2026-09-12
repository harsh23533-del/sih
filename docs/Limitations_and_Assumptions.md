# Limitations, Assumptions & Success Metrics

Consolidated from the original project plan's Risk & Limitations, Scientific & Ethical
Positioning, and Executive Summary sections — kept as one reference doc for the Week 8
final report and demo.

## Key risks & mitigations

| Risk / Limitation | Mitigation |
|---|---|
| Sparse or inconsistent historical landslide records | Combine multiple inventories (GSI, NRSC, NASA COOLR); document known gaps |
| Severe class imbalance (few positive events) | Careful negative sampling, PR-AUC as primary metric, class-weighting/SMOTE where needed |
| Spatial data leakage inflating reported accuracy | Spatial (region-based) cross-validation instead of purely random splits |
| Rainfall data resolution too coarse for local slope-scale prediction | Cross-check IMD/gridded rainfall against any available local station data |
| Model drift as land use / deforestation changes over time | Scheduled retraining cadence and periodic NDVI/land-cover refresh |
| Over-trust in an automated score by end users | Always present the score with its SHAP explanation and explicit uncertainty framing |
| Tree models can behave non-monotonically outside their training rainfall range | Confirmed in Week 8 scenario testing (`tests/test_scenarios.py`) — see note below |

## Scientific & ethical positioning

This system should be presented as an **AI-based risk prediction and early-warning
decision-support tool**, not as a guarantee of landslide occurrence or non-occurrence.
Model performance depends directly on data quality, spatial resolution, the accuracy of
historical event labels, class imbalance, and local calibration. Any real evacuation,
road-closure, or emergency decision must be validated and issued by competent
disaster-management authorities — the system's role is to surface an evidence-based,
explainable signal that supports, not replaces, expert judgment.

## Success metrics (prototype targets)

| Metric | Target for prototype |
|---|---|
| Susceptibility model ROC-AUC | ≥ 0.80 on held-out spatial folds |
| Dynamic hazard model PR-AUC | ≥ 0.65 (class imbalance expected) |
| 24-hour warning lead time | ≥ 6-12 hours ahead of the event, where rainfall data allows |
| False-alarm tolerance | Tuned per district in consultation with disaster-management authorities |
| Dashboard response time | < 3 seconds per location query |
| Explainability | Top 5 SHAP contributing features shown for every risk score |

These are the targets to check the trained models against once real data (not the
synthetic data used for integration testing) is profiled — none of the numbers above are
guaranteed by the code alone.

## Finding from Week 8 scenario testing

Running `tests/test_scenarios.py` against models trained on synthetic data showed the
risk score does **not** always increase monotonically from `heavy` to `critical`
rainfall multipliers. This is expected behaviour for tree-based models (Random
Forest/XGBoost) when an input is scaled well outside the range they were trained on —
they can't extrapolate a trend beyond their training distribution the way a linear model
would. Two implications for real deployment:

1. **Recalibrate the scenario multipliers** in `tests/test_scenarios.py` against the
   real rainfall distribution for the region once historical data is loaded, rather than
   the illustrative `0.3x / 1.0x / 2.0x / 3.5x` placeholders used here.
2. **Re-run the monotonicity check on the real trained models** before demoing — if it
   still fails, the training data may need denser sampling at the high-rainfall end
   (upsampling extreme events, or adding synthetic extreme-rainfall rows) so the model
   has seen enough of that range to extrapolate sensibly.

## Assumptions carried through the prototype

- Phase 1 scope is a single district or Sikkim (best historical inventory + DEM
  coverage), per the plan's phased rollout — not yet generalized to all 8 NER states.
- The 40% susceptibility / 60% dynamic hazard weighting in `models/risk_engine.py` is a
  starting hypothesis, not a validated split — revisit with a logistic meta-model once
  real data is available.
- `rainfall_trend` and `is_monsoon_season` (Week 5) are simple placeholder proxies for
  "rainfall intensity/trend, season" — refine if higher-resolution rainfall data becomes
  available.

## Environmental factors added beyond the original plan

The original prototype only used terrain (elevation/slope/aspect/curvature) and
rainfall, with soil/geology/NDVI left as empty placeholder columns. Per a standard
landslide-susceptibility literature review, the following factors were added —
**still synthetic for now**, generated with the same spatially-coherent approach as
the terrain/rainfall (see `generate_synthetic_data.py`), and ready to swap for the
real source listed once available:

| Feature | What it captures | Real-data source to swap in |
|---|---|---|
| `soil` | Soil type (rocky/sandy/loamy/clayey) | Bhuvan / NBSS&LUP soil maps |
| `geology` | Lithology (rock type) | GSI geology maps |
| `fault_distance_km` | Distance to nearest active fault | GSI active fault database |
| `seismic_pga` | Earthquake shaking intensity proxy | BIS seismic zonation (NER is mostly Zone V) / NDMA |
| `NDVI` | Vegetation cover | Sentinel-2 (Copernicus) / Bhuvan |
| `land_use` | Forest / agriculture / urban / barren | Bhuvan or Sentinel-2 LULC products |
| `soil_moisture_baseline`, `soil_moisture_index` | Ground wetness/saturation | CGWB groundwater data + antecedent rainfall |
| `insolation_proxy` | Solar exposure (derived from aspect+slope, no new raster) | — computed, not sourced |
| `freeze_thaw_index` | Freeze-thaw cycling (derived from elevation) | — computed; more relevant at higher NER elevations than in the Sikkim valley pilot |

Adding these gives the models 23 features instead of 12 — but since they're still
synthetic, they do **not** yet make the reported accuracy trustworthy (see the
model-comparison reports' near-perfect scores, which reflect an easy synthetic
signal, not real-world skill). The real gain from these factors comes only once
they're swapped for their real sources above — soil, geology, and NDVI/land-use are
the three the landslide-susceptibility literature weights most heavily.
