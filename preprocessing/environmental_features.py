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
                 faults_path: str = None):
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
        if not self._fault_lines:
            return np.nan

        def haversine_km(lat1, lon1, lat2, lon2):
            r = 6371.0
            p1, p2 = np.radians(lat1), np.radians(lat2)
            dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
            a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
            return 2 * r * np.arcsin(np.sqrt(a))

        best = np.inf
        for coords in self._fault_lines:
            for n in range(11):
                t = n / 10.0
                (lon0, lat0), (lon1, lat1) = coords
                plon = lon0 + t * (lon1 - lon0)
                plat = lat0 + t * (lat1 - lat0)
                d = haversine_km(lat, lon, plat, plon)
                best = min(best, d)
        return float(best)

    def sample(self, lat: float, lon: float) -> dict:
        return {
            "soil": self._sample_at(self._soil, self._soil_arr, lat, lon),
            "geology": self._sample_at(self._geology, self._geology_arr, lat, lon),
            "NDVI": self._sample_at(self._ndvi, self._ndvi_arr, lat, lon),
            "land_use": self._sample_at(self._landuse, self._landuse_arr, lat, lon),
            "seismic_pga": self._sample_at(self._seismic, self._seismic_arr, lat, lon),
            "soil_moisture_baseline": self._sample_at(self._moisture, self._moisture_arr, lat, lon),
            "fault_distance_km": self._fault_distance_km(lat, lon),
        }


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

    return df
