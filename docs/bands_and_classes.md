# Bands and classes (Phase 2)

Every "band X helps class Y" statement below is a **hypothesis** until Phase 9 tests it. Measured numbers are from the **grouped split, train part only** (`scripts/phase4b_spectral_eda.py`, `results/phase4b_spectral.json`).

## Band table

| Band | λ nm | Native | What it measures | Possible relevance (hypothesis) | Fisher ratio, 10 classes (measured) | Fisher, crop trio (measured) | Single-band RF val acc (measured) |
|---|---|---|---|---|---|---|---|
| B01 | 443 | 60 m | Coastal aerosol; atmospheric scattering | Low for land classes; may encode haze (uncorrected data) | 0.64 | 0.07 | 40.3 % |
| B02 | 490 | 10 m | Blue | Water, built-up | 0.83 | 0.26 | 43.9 % |
| B03 | 560 | 10 m | Green | Vegetation peak, water turbidity | 0.83 | 0.40 | 44.0 % |
| B04 | 665 | 10 m | Red (chlorophyll absorption) | Vegetation vs bare soil; crop phenology | 1.04 | **0.63** | 44.3 % |
| B05 | 705 | 20 m | Red edge 1 | Chlorophyll content; canopy | 1.27 | 0.34 | 45.3 % |
| B06 | 740 | 20 m | Red edge 2 | Canopy structure | **2.76** | 0.13 | 39.9 % |
| B07 | 783 | 20 m | Red edge 3 | Canopy structure / LAI | 2.62 | 0.20 | 41.0 % |
| B08 | 842 | 10 m | NIR (leaf mesophyll) | Vegetation vs water; biomass | 2.72 | 0.23 | 42.1 % |
| B8A | 865 | 20 m | Narrow NIR | Same as B08 (r = 0.979 with B08) | **2.94** | 0.23 | 41.7 % |
| B09 | 945 | 60 m | Water vapour absorption | Mostly atmosphere | 1.01 | 0.19 | 36.4 % |
| B10 | 1375 | 60 m | Cirrus (dropped) | Near zero on clear scenes (mean 12 DN) | — | — | — |
| B11 | 1610 | 20 m | SWIR 1 (leaf water, soil) | Moisture; roofs/materials; NDMI | 2.52 | 0.33 | 45.8 % |
| B12 | 2190 | 20 m | SWIR 2 (soil minerals, dry matter) | Bare soil / tillage; built-up | 1.89 | **0.56** | **47.9 %** |

Resolution consequence: 20 m bands were upsampled ×2 and 60 m ×6 to 10 m; they add spectral, not spatial, information.

**Indices** (standard definitions, D-015): NDVI = (B8 − B4)/(B8 + B4) measures green-vegetation vigour (chlorophyll absorption in red vs mesophyll scattering in NIR). NDMI = (B8 − B11)/(B8 + B11) measures canopy/leaf water content (SWIR absorption by water). Measured class means (train):

| Class | NDVI mean | NDVI std | NDMI mean | NDMI std |
|---|---|---|---|---|
| AnnualCrop | 0.381 | 0.169 | 0.081 | 0.158 |
| Forest | 0.711 | 0.044 | 0.331 | 0.060 |
| HerbaceousVegetation | 0.400 | 0.071 | 0.024 | 0.064 |
| Highway | 0.442 | 0.177 | 0.121 | 0.139 |
| Industrial | 0.224 | 0.191 | 0.009 | 0.132 |
| Pasture | 0.653 | 0.105 | 0.254 | 0.105 |
| PermanentCrop | 0.369 | 0.112 | 0.017 | 0.089 |
| Residential | 0.339 | 0.149 | 0.046 | 0.111 |
| River | 0.330 | 0.310 | 0.234 | 0.171 |
| SeaLake | −0.263 | 0.061 | 0.647 | 0.061 |

Note: NDMI is high for water (both bands tiny) — the index is only physically meaningful over vegetation; the auxiliary head is a distillation target, not a physical claim.

## Class cards

| Class | What it is | Visual cue | Bands hypothesised | Known confusions | Testable prediction |
|---|---|---|---|---|---|
| AnnualCrop | Arable fields, seasonal crops | Rectangular fields, bare-to-green mosaics | B04, B11, B12 (bare soil vs crop), red edge | PermanentCrop, Pasture, Herbaceous [H Fig 8] | Removing B11+B12 lowers AnnualCrop recall more than removing B01+B09; high within-class NDVI std (0.169 measured) |
| PermanentCrop | Orchards, vineyards, olive groves | Row structures, regular texture | B04, B12; texture (spatial) | AnnualCrop, Herbaceous | Gains from 12 bands over RGB are smaller than for AnnualCrop (texture-driven) |
| Pasture | Permanent grassland (agricultural meaning context-dependent) | Uniform green fields | NIR, red edge (dense green) | Herbaceous, AnnualCrop | NDVI mean 0.65 vs Herbaceous 0.40 → NIR/red bands should separate them; deleting B08+B8A raises Pasture↔Herbaceous confusion |
| HerbaceousVegetation | Natural grass/shrub | Irregular, mixed texture | Red edge, SWIR | Pasture, crops | Lowest per-class recall among vegetation classes in RGB models |
| Forest | Tree cover | Dark, coarse texture | NIR, red edge, SWIR1 (high NDMI 0.33) | Herbaceous | Near-perfect with ≥ 3 bands incl. one NIR/red-edge band |
| Highway | Major roads with surroundings | Linear features | Spatial pattern; SWIR for asphalt | River [H], Industrial | Highway↔River confusion depends on spatial, not spectral, cues → persists at k = 12 |
| Industrial | Industrial/commercial buildings | Large bright roofs | SWIR (materials), blue | Residential ([Y]: 6.84 % RGB → 0.75 % 13 bands) | Industrial→Residential error drops from RGB to 12 bands with the same CNN |
| Residential | Urban fabric | Dense small buildings, gardens | Visible + NIR (mix of roof and vegetation) | Industrial | Mostly spatial; small spectral gain |
| River | Rivers with banks | Linear water body | NIR/SWIR (water dark), green | Highway, SeaLake | Deleting all NIR+SWIR bands raises River→Highway errors |
| SeaLake | Open water | Uniform dark | Any NIR/SWIR band (water absorbs) | River | ≥ 99 % recall with k = 3 if one NIR/SWIR band is kept |

## Agricultural label rules (O4)

* **EuroSAT `agri_binary`** = AnnualCrop + PermanentCrop (5,500 / 27,000 = 20.4 % positive, measured).
* **EuroSAT `agri_pasture_binary`** = + Pasture (7,500 = 27.8 %). Reported separately because Pasture can be grazing land (agricultural use) or semi-natural grassland.
* **BigEarthNet** (19-class nomenclature; mapping to be confirmed against reBEN class list when downloaded): agricultural = *Arable land*, *Permanent crops*, *Pastures*(variant), *Complex cultivation patterns*, *Land principally occupied by agriculture, with significant areas of natural vegetation*, *Agro-forestry areas*. Mixed-label rule candidates for the Phase 5 subset test: (R1) "any agricultural label" → positive; (R2) "agricultural labels only" positive, "no agricultural label" negative, mixed patches excluded from the binary analysis.
