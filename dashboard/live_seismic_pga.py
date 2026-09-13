"""
Live seismic_pga -- real official classification, no network call needed.

The synthetic seismic_pga was never grounded in an actual zone map: it was
a formula (0.24 baseline + a boost the closer a point was to one of the 5
fake synthetic fault traces, see generate_synthetic_data.py). Real seismic
zonation in India is a discrete government classification, not a
continuous function of distance to the nearest mapped fault -- so
"live" here means "the current official BIS zone factor for this region,"
not a per-pixel geospatial lookup.

Timeline (important -- this superseded an earlier, now-incorrect version
of this module):
  - Nov/Dec 2025: BIS notified a revised earthquake-design code, IS 1893
    (Part 1):2025, which introduced a new highest-severity "Zone VI"
    (Z = 0.75) covering the entire Himalayan arc including Sikkim,
    superseding the old Zone IV (Z = 0.24) that Sikkim carried under
    IS 1893:2016.
  - 13 Mar 2026: BIS WITHDREW the 2025 code after the Ministry of Housing
    and Urban Affairs raised concerns over construction-cost escalation
    (est. 10-15% for buildings, up to 50% for infrastructure in Zones V/VI)
    and inadequate stakeholder consultation. IS 1893 (Part 1):2016 is back
    in force as the current applicable standard.
  - As of this module's writing (Sep 2026), no reinstatement of the 2025
    code has been reported -- IS 1893:2016 remains current.

So the CURRENT official classification for Sikkim is Zone IV, Z = 0.24
under IS 1893:2016 -- the same value the original synthetic baseline
used, not the Zone VI/0.75 an earlier version of this module claimed.
This is a case where "live" data ended up matching the synthetic
baseline, because the regulatory change that would have moved it got
rolled back before this pilot's build -- a useful reminder that a
"real, current, sourced" value doesn't always look different from
the synthetic placeholder it replaces, and the module needs to be
revisited if the code is ever reinstated or revised again.

Sources:
  - Bureau of Indian Standards, IS 1893 (Part 1):2016 (currently in
    force zonation: Zones II-V, no Zone VI).
  - "Rollback of Seismic Code of 2025", Drishti IAS, 13 Mar 2026
    (https://www.drishtiias.com/daily-updates/daily-news-analysis/rollback-of-seismic-code-of-2025),
    citing Economic Times: BIS scraps new seismic zonation plan after
    govt flags construction cost surge.
  - ClearIAS, "New Seismic Zonation Map of India" (Mar 2026), confirming
    IS 1893 (Part 1):2016 is the standard currently in force.
"""

# Current official BIS zone factor for Sikkim under IS 1893 (Part 1):2016
# -- the presently-in-force standard after the 2025 revision's rollback.
_CURRENT_ZONE_FACTOR = 0.24
_CURRENT_ZONE_LABEL = "Zone IV (IS 1893:2016 -- currently in force)"


def fetch_live_seismic_pga(lat: float, lon: float) -> dict | None:
    """Real current seismic_pga for this point. Since the whole Sikkim/NER
    study region is uniformly Zone IV under the currently-in-force
    IS 1893:2016, this is a constant lookup rather than a per-pixel one --
    always returns a value (never None) for points inside the pilot's
    coverage, matching the None-means-fall-back-to-synthetic contract of
    the other live_* modules for consistency, even though it can't
    practically fail.
    """
    return {
        "seismic_pga": _CURRENT_ZONE_FACTOR,
        "seismic_zone_label": _CURRENT_ZONE_LABEL,
    }
