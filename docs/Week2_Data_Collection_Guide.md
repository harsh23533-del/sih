# Week 2 — Data Collection Guide
## Phase 1 target: Sikkim (single-district/state pilot)

**Important note:** Actual dataset files (GB-scale GRD/NetCDF/DEM/shapefiles) require registered
accounts on each source's own portal and must be downloaded from your own machine — this cannot
be done from within this chat's sandboxed environment (it only has network access to
GitHub/PyPI/npm). Below is the exact source, access method, and a ready-to-run script for each
category. Run the scripts in `scripts/` locally after registering where needed.

---

## 1. Rainfall

| Source | Resolution | Access | Registration needed? |
|---|---|---|---|
| **IMD Gridded Rainfall** (primary) | 0.25° daily, 1901–2024 | https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html — direct NetCDF/binary download by year | No |
| **IMD + GPM merged (finer)** | 0.25° daily | https://rcc.imdpune.gov.in/download.php | No |
| **GPM-IMERG** (satellite, backup/cross-check) | 0.1° 30-min/daily | https://gpm.nasa.gov/data/directory (via NASA GES DISC) | Yes — free Earthdata login |
| **CHIRPS** (backup) | 0.05° daily | https://data.chc.ucsb.edu/products/CHIRPS-2.0/ | No |

**Recommended library:** `IMDLIB` (Python) — handles download + read of IMD `.grd`/NetCDF files directly by year and bounding box.

```bash
pip install imdlib
```
See `scripts/download_rainfall.py`.

---

## 2. Terrain (DEM)

| Source | Resolution | Access |
|---|---|---|
| **OpenTopography — SRTM GL1** (recommended) | 30m | REST API, needs free API key (https://opentopography.org → My Account → Request API key) |
| Direct S3 bulk (no key needed) | 30m | `aws s3 cp s3://raster/SRTM_GL1/ . --recursive --endpoint-url https://opentopography.s3.sdsc.edu --no-sign-request` |
| CartoDEM (ISRO, India-specific, higher accuracy) | 30m/10m | via Bhuvan (https://bhuvan.nrsc.gov.in) — needs Bhuvan account |

See `scripts/download_dem.py` (uses OpenTopography REST API — just needs your API key pasted in).

---

## 3. Soil, Geology, NDVI, Land Cover

| Source | What | Access |
|---|---|---|
| **ISRO Bhuvan** | Soil, geomorphology, LULC for India | https://bhuvan.nrsc.gov.in (registration required) |
| **GSI geology maps** | Lithology, structural geology | https://bhukosh.gsi.gov.in (GSI's open geoscience data portal) |
| **Sentinel-2 (NDVI derivation)** | 10m multispectral, compute NDVI yourself | https://scihub.copernicus.eu or via Google Earth Engine (easier — no big downloads) |
| **ESA WorldCover** | 10m global land cover | https://worldcover2021.esa.int — direct download, no login |

**Practical tip:** for NDVI specifically, Google Earth Engine (free, browser-based) is far less painful than downloading raw Sentinel-2 tiles — I can write you an Earth Engine script if you want to go that route instead.

---

## 4. Historical Landslides (ground truth labels — most important dataset)

| Source | Coverage | Access |
|---|---|---|
| **GSI Bhukosh Landslide Inventory** (primary) | India-wide, includes NER incidence reports | https://bhukosh.gsi.gov.in — free portal, has a dedicated landslide layer |
| **GSI Bhusanket incidence reports** | Event-specific PDF reports w/ coordinates (used in our Week 1 research — e.g. the Mizoram 2025 report) | https://bhusanket.gsi.gov.in |
| **NASA COOLR (Global Landslide Catalog)** | Global, crowd + media sourced | https://gpm.nasa.gov/landslides/index.html — downloadable CSV/shapefile, no login |
| Academic literature (as cross-check) | Published inventories with lat/long | Papers found in Week 1 (Meghalaya 1330-event inventory, Mizoram/Aizawl 19-event inventory) — request supplementary data from authors if not public |

See `scripts/fetch_nasa_coolr.py` — NASA COOLR is the easiest one to fully automate since it needs no login.

---

## 5. Infrastructure (roads, settlements)

| Source | Access |
|---|---|
| **OpenStreetMap via Overpass API** | Free, no login — see `scripts/download_osm_infra.py` |
| Alternative: Geofabrik regional extracts | https://download.geofabrik.de/asia/india.html — pre-packaged India `.osm.pbf`, filter to Sikkim bounding box |

---

## Sikkim bounding box (for all scripts below)
```
min_lat = 27.00, max_lat = 28.13
min_lon = 88.00, max_lon = 88.93
```

## Suggested order of execution (least → most friction)
1. NASA COOLR historical landslides (no login) → `fetch_nasa_coolr.py`
2. OSM infrastructure (no login) → `download_osm_infra.py`
3. IMD rainfall via IMDLIB (no login) → `download_rainfall.py`
4. ESA WorldCover land cover (no login, direct download)
5. OpenTopography DEM (needs free API key) → `download_dem.py`
6. Bhuvan soil/geology + GSI Bhukosh landslide layer (needs account — do these last since registration takes a day or two to approve sometimes)

---

## Output for this week
Once you've run the scripts locally, you should have:
```
data/
├── rainfall/          (IMD .grd/.nc files, 2015–2025 for Sikkim bbox)
├── terrain/           (SRTM 30m GeoTIFF, Sikkim clipped)
├── soil_geology/       (Bhuvan/GSI layers, WorldCover clip)
├── historical_landslides/ (NASA COOLR CSV + GSI Bhukosh export)
├── infrastructure/     (OSM roads/settlements GeoJSON)
└── satellite/          (Sentinel-2/NDVI, optional if using GEE)
```
This matches the "Raw dataset folder ready" deliverable in the original plan.
