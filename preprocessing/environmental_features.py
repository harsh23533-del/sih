"""
Environmental factors beyond the original terrain+rainfall set — added per
a literature review of standard landslide-susceptibility factors that were
previously left as empty placeholder columns (soil/geology/NDVI) or missing
entirely (land use, seismicity, soil moisture, fault proximity, solar
exposure, freeze-thaw). See docs/Limitations_and_Assumptions.md for the
real-data source each of these should be swapped for later.

Two kinds of features here:
  1. Sampled straight from a raster/vector layer (soil, geology, NDVI,
     land_use, seismic_pga, soil_moisture_baseline, fault_distance) —
     via EnvironmentalExtractor, same sample-at-a-point pattern as
     terrain_features.TerrainExtractor.
  2. Derived analytically from terrain already extracted (insolation_proxy,
     freeze_thaw_index, soil_moisture_index) — no extra raster needed,
     computed directly from elevation/slope/aspect + rainfall_30d.

Requirements:
    pip install rasterio numpy scipy geopandas shapely

Usage:
    from environmental_features import EnvironmentalExtractor, add_derived_environmental_features
    env = EnvironmentalExtractor(
        soil_path=".../sikkim_soil.tif", geology_path=".../sikkim_geology.tif",
        ndvi_path=".../sikkim_ndvi.tif", landuse_path=".../sikkim_landuse.tif",
        seismic_path=".../sikkim_seismic_pga.tif",
        moisture_path=".../sikkim_soil_moisture.tif",
        faults_path=".../sikkim_faults.geojson",
    )
    row = env.sample(27.33, 88.61)
"""
import numpy as np
import pandas as pd
import rasterio


class EnvironmentalExtractor:
    def __init__(self, soil_path: str, geology_path: str, ndvi_path: str,
                 landuse_path: str, seismic_path: str, moisture_path: str,
                 faults_path: str = None,
                 sat_moisture_path: str = None, insar_path: str = None,
                 population_path: str = None, glacial_lakes_path: str = None):
        self._soil = rasterio.open(soil_path)
        self._geology = rasterio.open(geology_path)
        self._ndvi = rasterio.open(ndvi_path)
        self._landuse = rasterio.open(landuse_path)
        self._seismic = rasterio.open(seismic_path)
        self._moisture = rasterio.open(moisture_path)

        self._soil_arr = self._soil.read(1)
        self._geology_arr = self._geology.read(1)
        self._ndvi_arr = self._ndvi.read(1)
        self._landuse_arr = self._landuse.read(1)
        self._seismic_arr = self._seismic.read(1)
        self._moisture_arr = self._moisture.read(1)

        self._fault_lines = None
        if faults_path:
            import json
            with open(faults_path) as f:
                gj = json.load(f)
            self._fault_lines = [
                feat["geometry"]["coordinates"] for feat in gj["features"]
            ]

        # Second round of factors — all optional so older callers/tests
        # that don't pass these paths still work.
        self._sat_moisture_src = rasterio.open(sat_moisture_path) if sat_moisture_path else None
        self._sat_moisture_arr = self._sat_moisture_src.read(1) if self._sat_moisture_src else None
        self._insar_src = rasterio.open(insar_path) if insar_path else None
        self._insar_arr = self._insar_src.read(1) if self._insar_src else None
        self._population_src = rasterio.open(population_path) if population_path else None
        self._population_arr = self._population_src.read(1) if self._population_src else None

        self._glacial_lakes = None
        if glacial_lakes_path:
            import json
            with open(glacial_lakes_path) as f:
                gj = json.load(f)
            self._glacial_lakes = [
                feat["geometry"]["coordinates"] for feat in gj["features"]
            ]

    def _sample_at(self, src, arr, lat, lon):
        try:
            row, col = src.index(lon, lat)
            return arr[row, col]
        except IndexError:
            return np.nan

    def _fault_distance_km(self, lat, lon) -> float:
        """Straight-line distance (km, haversine) to the nearest fault
        trace endpoint-to-endpoint segment — a simple point-to-segment
        approximation, adequate for this synthetic layer."""
        return self._nearest_point_or_line_km(lat, lon, self._fault_lines, is_line=True)

    def _glacial_lake_distance_km(self, lat, lon) -> float:
        """Distance to the nearest synthetic glacial lake — GLOF (glacial
        lake outburst flood) risk is a distinct mechanism from
        rainfall-triggered slope failure, but still location-relevant."""
        return self._nearest_point_or_line_km(lat, lon, self._glacial_lakes, is_line=False)

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2):
        r = 6371.0
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
        return 2 * r * np.arcsin(np.sqrt(a))

    def _nearest_point_or_line_km(self, lat, lon, features, is_line: bool) -> float:
        if not features:
            return np.nan
        best = np.inf
        for coords in features:
            if is_line:
                (lon0, lat0), (lon1, lat1) = coords
                for n in range(11):
                    t = n / 10.0
                    plon = lon0 + t * (lon1 - lon0)
                    plat = lat0 + t * (lat1 - lat0)
                    best = min(best, self._haversine_km(lat, lon, plat, plon))
            else:
                plon, plat = coords
                best = min(best, self._haversine_km(lat, lon, plat, plon))
        return float(best)

    def sample(self, lat: float, lon: float) -> dict:
        result = {
            "soil": self._sample_at(self._soil, self._soil_arr, lat, lon),
            "geology": self._sample_at(self._geology, self._geology_arr, lat, lon),
            "NDVI": self._sample_at(self._ndvi, self._ndvi_arr, lat, lon),
            "land_use": self._sample_at(self._landuse, self._landuse_arr, lat, lon),
            "seismic_pga": self._sample_at(self._seismic, self._seismic_arr, lat, lon),
            "soil_moisture_baseline": self._sample_at(self._moisture, self._moisture_arr, lat, lon),
            "fault_distance_km": self._fault_distance_km(lat, lon),
        }
        if self._sat_moisture_arr is not None:
            result["soil_moisture_satellite"] = self._sample_at(
                self._sat_moisture_src, self._sat_moisture_arr, lat, lon)
        if self._insar_arr is not None:
            result["insar_deformation_mm_yr"] = self._sample_at(
                self._insar_src, self._insar_arr, lat, lon)
        if self._population_arr is not None:
            result["population_density"] = self._sample_at(
                self._population_src, self._population_arr, lat, lon)
        if self._glacial_lakes is not None:
            result["glacial_lake_distance_km"] = self._glacial_lake_distance_km(lat, lon)
        return result


def add_derived_environmental_features(df: pd.DataFrame) -> pd.DataFrame:
    """Analytical features computed from columns already in df — no extra
    raster needed:

    - insolation_proxy: south-facing + steep slopes get more solar
      exposure (northern hemisphere) -> drier/faster freeze-thaw cycling.
      0 (little sun) to 1 (maximum exposure).
    - freeze_thaw_index: rises above the seasonal frost line (~3500m in
      this region) — a lower-weight, secondary factor for a Sikkim
      pilot, more relevant at the higher NER elevations.
    - soil_moisture_index: combines the static wetness baseline with how
      much rain has actually fallen recently (rainfall_30d), since real
      soil saturation is dynamic, not just terrain shape.
    - root_cohesion_proxy: forest holds slopes together far better than
      farmland/bare ground — derived from land_use + NDVI rather than a
      separate raster, since it's really a property of the vegetation
      already captured by those two columns.
    - exposure_index: the "who's actually affected" half of
      Risk = Hazard x Exposure x Vulnerability — combines population
      density with proximity to roads (denser, more accessible areas
      matter more for impact, even at equal hazard).
    """
    df = df.copy()

    if {"aspect", "slope"}.issubset(df.columns):
        south_facing = np.cos(np.radians(df["aspect"] - 180))  # 1 = due south
        slope_factor = np.clip(df["slope"] / 45.0, 0, 1)
        df["insolation_proxy"] = ((south_facing + 1) / 2 * 0.5 + slope_factor * 0.5).clip(0, 1)

    if "elevation" in df.columns:
        df["freeze_thaw_index"] = 1 / (1 + np.exp(-(df["elevation"] - 3500) / 300))

    if {"soil_moisture_baseline", "rainfall_30d"}.issubset(df.columns):
        rain_norm = (df["rainfall_30d"] - df["rainfall_30d"].min()) / (
            df["rainfall_30d"].max() - df["rainfall_30d"].min() + 1e-9
        )
        df["soil_moisture_index"] = (
            0.6 * df["soil_moisture_baseline"].fillna(df["soil_moisture_baseline"].median())
            + 0.4 * rain_norm
        ).clip(0, 1)

    if {"land_use", "NDVI"}.issubset(df.columns):
        # Forest(1) roots bind soil strongly; agriculture(2)/urban(3) much
        # less; barren(4) essentially none — scaled by how dense the
        # vegetation actually is (NDVI) within that land-use class.
        base_cohesion = df["land_use"].map({1: 0.8, 2: 0.35, 3: 0.15, 4: 0.05}).fillna(0.3)
        df["root_cohesion_proxy"] = (base_cohesion * (0.5 + 0.5 * df["NDVI"].clip(0, 1))).clip(0, 1)

    if {"population_density", "road_distance"}.issubset(df.columns):
        pop_norm = (df["population_density"] - df["population_density"].min()) / (
            df["population_density"].max() - df["population_density"].min() + 1e-9
        )
        accessibility = 1 / (1 + df["road_distance"].fillna(df["road_distance"].median()) / 1000.0)
        df["exposure_index"] = (0.7 * pop_norm + 0.3 * accessibility).clip(0, 1)

    return df
