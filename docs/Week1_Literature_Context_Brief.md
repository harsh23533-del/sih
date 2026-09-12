# Literature & Context Brief
## AI-Based IoT-Free Geospatial Early Warning & Landslide Risk Monitoring System — NER India
**Week 1 Output | Research & Understanding**

---

## 1. Landslide Trigger Factors

Landslides in hilly/mountainous terrain are driven by a combination of **static (predisposing)** and **dynamic (triggering)** factors:

| Factor Type | Factor | Why it matters |
|---|---|---|
| Static | Slope angle | 25°–45° slopes are most vulnerable; very steep faces may already have shed loose material |
| Static | Soil/geology | Weathered rock, loose colluvium, weak lithology (shale, phyllite) reduce shear strength |
| Static | Land use/deforestation | Loss of root cohesion destabilizes slopes; *jhoom* (shifting) cultivation is a major NER-specific driver |
| Static | Drainage/road proximity | Road-cutting and poor drainage concentrate water and undercut slopes |
| Dynamic | Rainfall (intensity + antecedent) | Intense monsoon rainfall is the dominant trigger; cumulative rainfall over prior days pre-saturates soil, lowering the threshold for failure |
| Dynamic | Seismicity | NER lies in Seismic Zone V — earthquakes can trigger slides independent of rainfall |

**Key takeaway for this project:** rainfall (single-day + multi-day cumulative) combined with terrain susceptibility explains most NER landslide events — this validates the two-model design (static susceptibility + dynamic rainfall-triggered hazard) already planned.

---

## 2. NER Geography & High-Risk Districts

NER India = 8 states (Arunachal Pradesh, Assam, Manipur, Meghalaya, Mizoram, Nagaland, Sikkim, Tripura), ~262,179 sq km, spanning valley floors to >6,000m elevation in Arunachal Pradesh. The region sits in Seismic Zone V and receives some of the highest rainfall in India (Meghalaya is the wettest state in the country).

**State-wise landslide susceptibility ranking** (per recent national-scale AHP/FR/Yc hybrid modeling, % of state area classified Very-High-Susceptibility):

| Rank (among NER states) | State | Very-High-Susceptibility Area |
|---|---|---|
| 1 | Nagaland | ~55% |
| 2 | Mizoram | ~53% |
| 3 | Arunachal Pradesh | ~52% |
| 4 | Sikkim | ~45% |
| 5 | Manipur | ~31% |
| 6 | Meghalaya | ~22% |
| — | Tripura, Assam | Low (mostly valley terrain) |

**Notable high-incidence districts / hotspots identified in recent GSI and academic studies:**
- **Mizoram:** Aizawl, Serchhip, Lawngtlai, Lunglei, Champhai, Siaha — worst-hit in the May–June 2025 Cyclone Remal event; Aizawl alone has a documented decade-long inventory of rainfall-triggered slides along NH-54/NH-6 corridors
- **Meghalaya:** East Khasi Hills (Mawlai-Umjapung), East/West Jaintia Hills (Sonapur, Jowai) — Shillong Plateau, ~40% of studied area highly susceptible
- **Sikkim:** Already has an operational GSI regional early-warning prototype (see below) — good candidate for Phase 1 due to best data/DEM coverage

**Implication for phased scope:** the plan's Phase 1 choice of Sikkim (or a single high-incidence district) aligns well with where GSI already has the richest inventory and infrastructure — this will make Week 2 data collection much easier than starting in a data-sparse state.

---

## 3. Existing Early-Warning Systems — Short Study

- **GSI National Landslide Susceptibility Mapping (NLSM):** Completed high-resolution (90m–30m) susceptibility mapping across 4.3 lakh sq km / 19 states, including the NER's Tertiary Belt. Database of ~91,000 historical landslides (33,904 field-validated) — this is a usable historical inventory source for Week 2.
- **GSI Regional Landslide Early Warning System (LEWS):** Grew out of the 2017 LANDSLIP program (with UK partners), first piloted in Darjeeling (WB) and Nilgiris (TN), later scaled to 14+ districts including **Sikkim** and **Nagaland**. As of 2024, GSI began issuing *experimental* early-warning bulletins during monsoon season, now operational (prototype stage) in 21 districts across 8 states.
- **Design pattern used by GSI's system:** rainfall-threshold-based triggering layered on static susceptibility zones — conceptually the same static+dynamic split this project uses, but GSI's is threshold-rule-based rather than ML-driven. **This project's differentiator:** ML models (RF/XGBoost) + SHAP explainability + a continuous 0–100 risk score instead of discrete threshold rules, which can better capture non-linear interactions and gives auditability for authorities.
- **Gap this project fills:** GSI's system is not yet dashboard-driven with location-level interactive querying, SHAP-based reasoning, or scenario simulation (rainfall sliders) — these are exactly the "Advanced/Future Features" already scoped in the existing plan.

---

## 4. Project Scope & Success Metrics (finalized)

**Scope:** Software-only, IoT-free, ML-driven landslide risk decision-support system for NER India, phased Sikkim/single-district → state-level → full 8-state rollout, using only open geospatial/meteorological data (no new physical sensors).

**Success metrics for prototype** (carried over and confirmed from the existing plan — these are well-chosen and standard for this problem class):

| Metric | Target |
|---|---|
| Susceptibility model ROC-AUC | ≥ 0.80 (spatial holdout) |
| Dynamic hazard model PR-AUC | ≥ 0.65 (class imbalance expected) |
| Warning lead time | 6–12 hours ahead of event, where rainfall data allows |
| Dashboard response time | < 3 seconds per query |
| Explainability | Top-5 SHAP features shown per score |

**Note:** given real-world imbalance (landslides are rare events), false-negative rate should be tracked as a primary operational metric alongside PR-AUC, since a missed warning is far costlier than a false alarm — this is already reflected in the Week 4 plan.

---

### Sources consulted
- GSI National Landslide Susceptibility Mapping (NLSM) program reports (2024–2025)
- Scientific Reports (Nature) — national-scale hybrid landslide susceptibility/risk mapping, 2025
- GSI Bhusanket incidence reports — Mizoram 2025 monsoon event
- Peer-reviewed susceptibility studies: Meghalaya (Taylor & Francis, 2022), Darjeeling MCDA study (Applied Sciences, 2023), Mizoram/Aizawl inventory (arXiv, 2026)
