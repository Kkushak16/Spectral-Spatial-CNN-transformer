# Research linkage (Phase 6) and experiment protocol (Phase 7)

## Hypotheses with the experiment that could reject each

| ID | Hypothesis | Evidence so far | Rejecting experiment | Rejected if |
|---|---|---|---|---|
| H1 | 12 bands beat RGB with the same model | B0 RF: 88.2 % vs 74.7 % (spectral only) | B3 (12-band CNN) vs B2 (same CNN, RGB), 5 seeds, grouped split; also 10 % data | 12-band mean ≤ RGB mean + 1 seed-std |
| H2 | Learned gate at k ≤ 6 beats random, RGB, RGB+NIR, PCA-loading and mRMR subsets of equal size | none | S5 vs S1–S4 at k = 3, 4, 6; 5 seeds; ≥ 20 random draws for S1 | Gate not above the best fixed subset by > 1 seed-std at any k ≤ 6 |
| H3 | NDVI/NDMI head improves low-k accuracy or agri-F1 beyond a control head | crop-trio Fisher ranks B04/B12 above B08 → physics head may *not* point to crop bands | S6 (gate + physics) vs S6c (gate + control head) vs S5 at k = 3, 4; full and 10 % data | physics ≤ control within seed-std |
| H4 | Consistency improves robustness (noise, gain shift, band drop) and narrows geo gap | none | S7 vs S5 on corrupted test sets and on geo split | no improvement > seed-std on robustness metrics |
| H5 | Band explanations are faithful and stable | none | Deletion test top-m vs random-m (≥ 20 draws) vs bottom-m; Jaccard of selected bands across ≥ 3 seeds | top-m deletion drop ≤ random-m drop; Jaccard < 0.5 |
| H6 | k = 4–6 within **1.0 point** of the 12-band accuracy (pre-registered margin) at lower cost | none | S8 at k = 4, 6 vs B8 (12 bands), grouped split | gap > 1.0 pt (mean over 5 seeds) |
| H7 | Transformer adds value over the CNN at equal parameters | none | B8 vs parameter-matched B3-wide | B8 ≤ B3-wide within seed-std |

### Pair-level predictions (to be checked with confusion matrices)

| Pair | Prediction | Test |
|---|---|---|
| Forest vs Herbaceous | NIR + red-edge separate canopy density | band-drop of B06/B07/B08/B8A |
| Annual vs Permanent Crop vs Pasture | **Red (B04) and SWIR2 (B12)** carry most crop-trio separability (measured Fisher), not NIR | per-class recall vs band set; gate selections at k = 3/4 should include B04 or B12 |
| River vs SeaLake | spatial shape dominates; spectra similar | confusion persists at k = 12 |
| Industrial vs Residential | SWIR reduces Industrial→Residential errors | RGB vs 12-band, matched model |
| Highway vs River | spatial + spectral; NIR/SWIR (water dark) helps | spatial vs spectral ablation |
| Effect of deleting B08 | small (B8A and B07 are r ≈ 0.98 redundant) | single-band deletion at test time and retrained |

### Decisions taken in Phase 6

* **Static selection** (D-006). Image-adaptive input SE (B7) is a baseline only.
* **Control auxiliary head**: predict mean/std of a normalised difference of a fixed *non-physical* band pair chosen at random once (seeded) from bands not used by NDVI/NDMI, e.g. (B02 − B09)/(B02 + B09); same head size and loss. Compare gate frequencies with physics vs control vs no head (master issue 3).
* **Gate method**: implement both (hard-concrete, straight-through top-k); choose on validation at k = 4 in Phase 8.

## Protocol (pre-registered)

* Splits: grouped (primary), random (literature comparison), geo (generalisation); train fractions 1.0, 0.5, 0.2, 0.1.
* Same optimiser/schedule/epochs for every model (D-019); native 64×64; augmentation: flips + 90° rotations on all bands.
* Seeds: 5 on EuroSAT (0–4), 3 on BigEarthNet; report mean ± std and bootstrap 95 % CI of test accuracy; paired comparisons on the same test images.
* Metrics: accuracy, macro-F1, per-class recall, confusion, agricultural P/R/F1; params, FLOPs (fvcore), CPU latency, bands read.
* k and λ chosen on validation only; test evaluated once per final configuration.
* Success criterion: improvement larger than the seed std on both full-data and 10 %-data at equal (±10 %) parameters. H6 margin = 1.0 point.

## Baseline ladder status

| ID | Input | Model | Status |
|---|---|---|---|
| B0 | per-band mean/std | random forest | **done** (88.2 ± 0.2 % test, 12 bands; 74.7 % RGB) |
| B1 | RGB | ResNet-50 ImageNet-pretrained, 224 px | deferred to GPU (D-018) |
| B2 | RGB | compact CNN (same as B3), scratch | running |
| B3 | 12 bands | compact CNN, scratch | running |
| B4 | 12 bands | B3 with first conv from pretrained RGB weights | to do |
| B5 | 12 bands + NDs | B3 | to do |
| B6 | 12 bands | B3 + SE on features | config ready |
| B7 | 12 bands | B3 + soft SE on input bands | config ready |
| B8 | 12 bands | CNN → Transformer (64 tokens) | running |
| S1–S8 | subsets / gate / physics / consistency | — | Phase 8 |
