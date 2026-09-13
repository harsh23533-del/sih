"""
Extract terrain features (elevation, slope, aspect, curvature, ruggedness/TRI)
from a DEM GeoTIFF (output of Week 2's download_dem.py) and sample them at
arbitrary lat/lon points (historical landslide points + negative samples).

Requirements:
    pip install rasterio numpy scipy

Usage:
    from terrain_features import TerrainExtractor
    te = TerrainExtractor("../data/terrain/sikkim_srtm30m.tif")
    row = te.sample(27.33, 88.61)   # -> dict of terrain features
"""
import numpy as np
import rasterio
from scipy.ndimage import sobel, generic_filter


class TerrainExtractor:
    def __init__(self, dem_path: str, twi_path: str = None, drainage_path: str = None):
        self.src = rasterio.open(dem_path)
        self.elevation = self.src.read(1).astype(float)
        self.elevation[self.elevation < -1000] = np.nan  # mask nodata/voids

        # The DEM's transform is in degrees (geographic CRS), but slope/
        # curvature need real ground distance — convert degrees-per-pixel
        # to meters-per-pixel using the DEM's center latitude. This was
        # previously used directly as if it were already in meters, which
        # made pixel size ~1000x too small and saturated slope near 90 deg
        # (see docs/Limitations_and_Assumptions.md).
        px_size_x_deg = abs(self.src.transform.a)
        px_size_y_deg = abs(self.src.transform.e)
        height, width = self.elevation.shape
        center_lat = self.src.xy(height // 2, width // 2)[1]
        meters_per_deg_lat = 111_320.0
        meters_per_deg_lon = 111_320.0 * np.cos(np.radians(center_lat))
        self.px_size_x = px_size_x_deg * meters_per_deg_lon
        self.px_size_y = px_size_y_deg * meters_per_deg_lat

        self._compute_derivatives()

        # TWI + drainage distance are precomputed rasters (see
        # hydrology_features.py / generate_synthetic_data.py) sampled the
        # same way as everything else, rather than recomputed per-point.
        self._twi_src = rasterio.open(twi_path) if twi_path else None
        self._twi = self._twi_src.read(1) if self._twi_src else None
        self._drainage_src = rasterio.open(drainage_path) if drainage_path else None
        self._drainage = self._drainage_src.read(1) if self._drainage_src else None

    def _compute_derivatives(self):
        dz_dx = sobel(self.elevation, axis=1) / (8 * self.px_size_x)
        dz_dy = sobel(self.elevation, axis=0) / (8 * self.px_size_y)

        self.slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))
        self.aspect_deg = (np.degrees(np.arctan2(dz_dy, -dz_dx)) + 360) % 360

        # Profile curvature (2nd derivative) — simple Laplacian approximation
        self.curvature = (
            sobel(dz_dx, axis=1) / self.px_size_x
            + sobel(dz_dy, axis=0) / self.px_size_y
        )

        # Terrain Ruggedness Index (TRI): mean absolute elevation diff to 8 neighbors
        def _tri(window):
            center = window[len(window) // 2]
            return np.nanmean(np.abs(window - center))

        self.tri = generic_filter(
            self.elevation, _tri, size=3, mode="nearest"
        )

    def _rowcol(self, lat: float, lon: float):
        row, col = self.src.index(lon, lat)
        return row, col

    def sample(self, lat: float, lon: float) -> dict:
        row, col = self._rowcol(lat, lon)
        try:
            result = {
                "elevation": float(self.elevation[row, col]),
                "slope": float(self.slope_deg[row, col]),
                "aspect": float(self.aspect_deg[row, col]),
                "curvature": float(self.curvature[row, col]),
                "ruggedness_tri": float(self.tri[row, col]),
            }
        except IndexError:
            result = {
                "elevation": np.nan, "slope": np.nan, "aspect": np.nan,
                "curvature": np.nan, "ruggedness_tri": np.nan,
            }
        if self._twi is not None:
            try:
                result["twi"] = float(self._twi[row, col])
            except IndexError:
                result["twi"] = np.nan
        if self._drainage is not None:
            try:
                result["drainage_distance_km"] = float(self._drainage[row, col])
            except IndexError:
                result["drainage_distance_km"] = np.nan
        return result


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", required=True, help="Path to DEM GeoTIFF")
    parser.add_argument("--points", required=True, help="CSV with 'latitude','longitude' columns")
    parser.add_argument("--out", required=True, help="Output CSV with terrain features appended")
    args = parser.parse_args()

    te = TerrainExtractor(args.dem)
    df = pd.read_csv(args.points)
    feats = df.apply(lambda r: te.sample(r["latitude"], r["longitude"]), axis=1, result_type="expand")
    out = pd.concat([df, feats], axis=1)
    out.to_csv(args.out, index=False)
    print(f"Saved {len(out)} rows with terrain features -> {args.out}")
