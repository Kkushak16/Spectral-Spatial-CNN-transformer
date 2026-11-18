# Data inspection and QC report (Phases 3–4)

Scripts: `scripts/build_eurosat_cache.py`, `scripts/phase3_band_order.py`, `scripts/phase4_qc.py`, `scripts/phase4b_spectral_eda.py`. Raw outputs: `results/phase4_qc.json`, `results/phase4b_spectral.json`, `results/phase3_class_mean_spectra_raw.csv`, figures in `results/figures/`.

## Phase 3: what the files contain

* 27,000 GeoTIFFs in 10 class folders, all `(13, 64, 64)` **uint16**; no band descriptions in the files.
* Georeferenced: 10 UTM CRSs (EPSG:32627–32636), pixel ≈ 10.005 m; coordinates converted to WGS84 per patch.
* **Band order verified empirically = B01…B08, B09, B10, B11, B12, B8A** (D-008):
  * pixel correlation with B08: file ch12 r = 0.979 vs ch8 r = 0.736 → ch12 is B8A;
  * file ch9 mean 12 DN on every class → B10 (cirrus, near zero on clear scenes);
  * file ch8 dark for all classes (945 nm water-vapour absorption) → B09;
  * physics checks: Forest NIR/red = 6.3; SeaLake NIR (329) < green (792); water dark beyond 900 nm.
* Consequence: any loader that assumes Sentinel-2 wavelength order mislabels four bands (B8A↔B09↔B10↔B11↔B12 shift). This explains the inconsistent per-band means in Yassine et al. Table 2 flagged in the master file.
* Saved: `configs/band_map_eurosat.yaml` (file order, canonical 12-band order and file indices).

## Phase 4: quality checks and decisions

| Check | Result | Decision |
|---|---|---|
| Class balance | 2,000–3,000 per class (Pasture smallest) | Stratified splits; report macro-F1 |
| Agricultural share | 20.4 % (Annual+Permanent), 27.8 % with Pasture | Precision/recall/F1/PR curves for the binary analysis |
| Unreadable / wrong shape | 0 | — |
| NaN / saturation | none; max 28,003; 171 patches ≥ 16,000 DN (bright roofs/clouds) | Keep |
| Zero pixels | 7 patches, none > 1 % of pixels | Valid pixel = finite & > 0 (D-010) |
| Constant bands | 3 patches with constant B09 (60 m band over 640 m) | Keep |
| Exact duplicates | 0 | — |
| Overlapping footprints | **527 pairs** (centres < 640 m), 111 cross-class | Keep together in one split |
| Neighbours < 1 km | **20,619 pairs; 15,926 patches (59 %) in 3,742 groups; largest 145** | Grouped split primary (D-011) |
| Near-duplicate cosine | dominated by homogeneous SeaLake (414k pairs) | Not used (D-016) |
| Geography | lat 28–65°, lon −21–34°, 950 half-degree cells | Geographic 1° block split possible |
| Spectral separability | Fisher ratio top: B8A 2.94, B06 2.76, B08 2.72, B07 2.62, B11 2.52; lowest B01 0.64 | Bands do differ → band selection has something to learn |
| Spectral-only floor (B0) | RF on per-band mean/std: **12 bands 88.2 ± 0.2 % test**, RGB 74.7 % (3 seeds) | Strong evidence for H1 before any CNN |

Figures: `phase3_class_spectra.png`, `phase4_composites.png` (RGB + NIR false colour), `phase4_geography.png`, `phase4_class_counts.png`, `phase4_signatures_train.png`, `phase4_indices_train.png`.
