"""
Live seismic_pga -- real official classification, no network call needed.

The synthetic seismic_pga was never grounded in an actual zone map: it was
a formula (0.24 baseline + a boost the closer a point was to one of the 5
fake synthetic fault traces, see generate_synthetic_data.py). Real seismic
zonation in India is a discrete government classification, not a
continuous function of distance to the nearest mapped fault -- so
"live" here means "the current official BIS zone factor for this region,"
not a per-pixel geospatial lookup.

As of IS 1893 (Part 1):2025 -- the Bureau of Indian Standards' revised
earthquake-design code, published December 2025 -- the entire Himalayan
arc (Jammu & Kashmir/Ladakh through Himachal Pradesh, Uttarakhand, Sikkim,
and the North-East) was reclassified into a newly created, highest-severity
"Zone VI" with zone factor Z = 0.75 (superseding the old Zone IV, Z = 0.24,
that Sikkim carried under IS 1893:2016). Since this Sikkim/NER pilot's
entire study region sits inside that reclassified arc, the zone factor is
uniform across the area -- there is no sub-region within Sikkim that is
still Zone IV or V under the new code.

This gives a real_zone_factor of 0.75 rather than the old 0.24 baseline --
a large, source-backed jump that better reflects official current hazard
guidance for the region.

Real geospatial per-pixel BIS zone polygons (data.gov.in / bharatlas,
GODL-India licensed) exist for the older IS 1893:2016 map, but that map
is now superseded for this region and the polygon file is 7+MB (too large
to fetch and parse from here) -- not worth wiring in when a single
constant already reflects the current, correct classification for the
whole study area.

Sources:
  - BIS IS 1893 (Part 1):2025, Criteria for Earthquake Resistant Design
    of Structures -- General Provisions and Buildings.
  - Multiple engineering-literature summaries of the Dec 2025 revision
    (Zone VI, Z = 0.75) confirming Sikkim/Himalayan-arc reclassification.
"""

# Current official BIS zone factor for the whole Sikkim/NER Himalayan-arc
# study region under IS 1893 (Part 1):2025 -- Zone VI.
_CURRENT_ZONE_FACTOR = 0.75
_CURRENT_ZONE_LABEL = "Zone VI (IS 1893:2025)"


def fetch_live_seismic_pga(lat: float, lon: float) -> dict | None:
    """Real current seismic_pga for this point. Since the whole Sikkim/NER
    study region is uniformly reclassified as Zone VI under IS 1893:2025,
    this is a constant lookup rather than a per-pixel one -- always
    returns a value (never None) for points inside the pilot's coverage,
    matching the None-means-fall-back-to-synthetic contract of the other
    live_* modules for consistency, even though it can't practically fail.
    """
    return {
        "seismic_pga": _CURRENT_ZONE_FACTOR,
        "seismic_zone_label": _CURRENT_ZONE_LABEL,
    }
