# Fact sheet (Phase 1, updated with Phase 3–4 measurements)

Status tags: **Verified-paper** (read in an uploaded primary source), **Measured** (computed by us on the files, script named), **To check**.

## EuroSAT (Helber et al., IEEE JSTARS 2019; `eurosat_paper.pdf`)

| Item | Value | Status / source |
|---|---|---|
| Images | 27,000 labelled, georeferenced, 10 classes | Verified-paper (abstract); Measured: 27,000 `.tif` (`build_eurosat_cache.py`) |
| Class counts | AnnualCrop 3000, Forest 3000, HerbaceousVegetation 3000, Highway 2500, Industrial 2500, Pasture 2000, PermanentCrop 2500, Residential 3000, River 2500, SeaLake 3000 | Measured (`phase4_qc.py`) |
| Patch | 64×64 px, ~640 m side; measured pixel size 10.005 × 10.008 m | Verified-paper; Measured (rasterio transform) |
| Format | GeoTIFF, 13 bands, **uint16**, (C,H,W) = (13,64,64); no band descriptions in files | Measured |
| **File band order** | **B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11, B12, B8A** | Measured (`phase3_band_order.py`, D-008) |
| Value range | 0 – 28,003 DN (TOA, ×10⁴ reflectance scale); B10 mean 12 DN | Measured |
| Georeferencing | Yes. CRS = UTM zones EPSG:32627–32636 (10 zones); lat 27.97–65.24°, lon −21.0–33.5° | Measured |
| Sensor | Sentinel-2A, 13 bands | Verified-paper |
| Coverage | 34 European countries, Urban Atlas cities | Verified-paper |
| Atmospheric correction | None (colour casts remain) | Verified-paper (Sec. III-B, Fig. 7) |
| Cleaning | Snow/ice, mislabelled, broken patches removed manually | Verified-paper |
| Resolution | All bands resampled to 10 m | Verified-paper |
| Product level | Not stated; DN range and no-correction statement are consistent with L1C TOA | To check (Interpretation) |
| Invalid pixels | 7 patches contain zeros (none > 1 % of pixels); no NaN; no 65,535 | Measured |
| Duplicates | 0 exact duplicates (SHA-1 of array bytes) | Measured |
| Spatial overlap | 527 pairs with overlapping footprints (111 cross-class); 20,619 pairs within 1 km; 15,926 patches (59 %) in a neighbour group; largest group 145 patches | Measured (`phase4_qc.py`) |
| Published results | ResNet-50 RGB pretrained 98.57 %, GoogLeNet 98.18 % (80/20); CI 98.30 %, SWIR 97.05 %; from scratch ≈ 2 pts lower (Table II: ResNet-50 96.43 % at 80/20) | Verified-paper (Tables II–IV) |
| Single-band study | RGB bands best; B05, B12 strong; B10 weakest (read from Fig. 9) | Verified-paper (figure) |
| Official split | None (random 80/20 per class) | Verified-paper |

## BigEarthNet

| Item | Value | Status / source |
|---|---|---|
| Uploaded paper | BigEarthNet-MM: 590,326 S1/S2 pairs, 10 countries, 125 S2 tiles June 2017–May 2018, sen2cor L2A, B10 absent; 120×120 (10 m) / 60×60 (20 m) / 20×20 (60 m) px; 19-class nomenclature from CLC 2018; 70,987 snow/cloud/shadow patches listed; 1–12 labels per patch | Verified-paper (`BigEarth.pdf`, Sumbul et al. 2021) |
| Splits (BEN-MM era) | 269,695 train pairs in the paper (other sets cut off in extraction); Papoutsis et al. use 295,118 / 147,559 / 147,559 S2 patches | Verified-paper |
| reBEN / v2.0 | 549,488 pairs; 1200 m patches; L2A; CLC 2018 19 classes; geographic split | From master file (arXiv 2407.03653) — primary PDF **not** in folder → To check |
| Bands used in literature | Papoutsis et al. drop B01, B09, B10; BEN-MM experiments drop B01, B09 | Verified-paper |
| File format, split file names, licence for v2.0 | | To check when downloaded |

## Sentinel-2 bands (EuroSAT paper Table I; ESA naming D-005)

| Band | λ (nm) | Region | Native res. | In 12-band set |
|---|---|---|---|---|
| B01 | 443 | Coastal aerosol | 60 m | yes |
| B02 | 490 | Blue | 10 m | yes |
| B03 | 560 | Green | 10 m | yes |
| B04 | 665 | Red | 10 m | yes |
| B05 | 705 | Red edge 1 | 20 m | yes |
| B06 | 740 | Red edge 2 | 20 m | yes |
| B07 | 783 | Red edge 3 | 20 m | yes |
| B08 | 842 | NIR | 10 m | yes |
| B8A | 865 | Narrow NIR | 20 m | yes |
| B09 | 945 | Water vapour | 60 m | yes |
| B10 | 1375 | SWIR cirrus | 60 m | **dropped** (one 13-band reference run) |
| B11 | 1610 | SWIR 1 | 20 m | yes |
| B12 | 2190 | SWIR 2 | 20 m | yes |
