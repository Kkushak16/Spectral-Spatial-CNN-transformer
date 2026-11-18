# Phase report — AgriSpec-XAI (PG-ASST / COPS)

Updated 2026-11-12. For each phase: **what was done · what was used · what is special · what was achieved · gate**.
Numbers are measured by the named script unless marked as from a paper.

---

## Phase 0 — Setup and ground rules ✅

* **Done:** project package `agrispec` (`src/`), YAML config system with inheritance (`configs/base.yaml` → `configs/runs/*.yaml`), run directories with config + manifest (split/norm/band-map hashes) + history + metrics + checkpoint, a global `results/results.csv`, decision log, pytest suite.
* **Used:** Python 3.14 venv, PyTorch 2.15 (CPU), rasterio, numpy/pandas/scikit-learn, fvcore (FLOPs), captum/shap (for Phase 9), timm/torchvision.
* **Special:** the sparse **BandGate** (one logit per band; straight-through top-k *and* hard-concrete/L0) plus `export_k_channel`, which rebuilds the network with a genuinely *k-input-channel* first convolution — unselected bands are never read at inference.
* **Achieved:** smoke test on random (B, 12, 64, 64): for every k ∈ {3,4,6,8,12} and both gate types the gate keeps exactly k bands, unselected channels are exactly zero, gradients reach the gate, and the k-channel copy reproduces the masked model (max |Δlogit| ≈ 1e-8).
* **Hardware finding:** no NVIDIA GPU (Intel UHD only) → CPU training (D-001).
* **Gate:** passed (`scripts/smoke_test.py`, `tests/test_gate.py`).

## Phase 1 — Dataset knowledge and literature ✅

* **Done:** all 10 uploaded PDFs text-extracted and read for facts; web search on differentiable band selection; `docs/fact_sheet.md`, `docs/literature_review.md` (novelty table).
* **Special findings:** `BigEarth.pdf` is **BigEarthNet-MM (2021, 590,326 pairs)**, not reBEN v2.0 (549,488) — reBEN facts still need the primary paper (D-017). Hard-concrete/L0 band gates already exist for *hyperspectral* data (EHBS/CHBS 2024, SGBR-HC 2026), so the gate alone is not novel; the physics-distillation + consistency + faithfulness + leakage-aware evaluation package on Sentinel-2 was not found → candidate contribution, not a "first" claim.
* **Gate:** passed (every number tagged with a source; novelty table drafted).

## Phase 2 — Bands and classes ✅

* **Done:** 12-band table with measured separability, 10 class cards each with a testable prediction, NDVI/NDMI definitions and class means, agricultural label rules (EuroSAT binary, +Pasture variant, BigEarthNet draft mapping) — `docs/bands_and_classes.md`.
* **Gate:** passed.

## Phase 3 — Obtain and inspect the data ✅

* **Done:** read all 27,000 GeoTIFFs once into a uint16 cache (≈1 min, 2.9 GB) with a metadata table (CRS, coordinates, hashes, QC stats).
* **Key discovery — band order:** the EuroSAT MS files store **B01…B08, B09, B10, B11, B12, B8A** (B8A *last*). Proven from data: pixel correlation with B08 is 0.979 for file channel 12 vs 0.736 for channel 8; channel 9 averages 12 DN (cirrus); channel 8 is dark for every class (945 nm water vapour). A loader assuming standard order would mislabel four bands — this also explains the inconsistent band table in Yassine et al.
* **Georeferencing:** yes — 10 UTM zones, per-patch lat/lon recovered → geographic splits possible.
* **Gate:** passed (band order verified, georeferencing known; BigEarthNet split files pending its download).

## Phase 4 — Visualisation and quality checks ✅

* **Done:** composites (RGB, NIR false colour), class counts, geography map, invalid/constant/duplicate checks, spatial-overlap analysis, class spectral signatures and NDVI/NDMI by class (train split only), Fisher ratios, spectral-only baseline B0 — `docs/qc_report.md`.
* **Key discovery — spatial leakage:** 527 patch pairs have overlapping footprints (111 across classes) and **59 % of patches have another patch within 1 km**. A random split puts 2,271 such neighbour groups across train/test.
* **Decisive cheap test:** bands *do* differ between classes. A random forest on per-band mean/std only (no spatial information) reaches **88.2 ± 0.2 % (12 bands) vs 74.7 % (RGB)** test accuracy. Crop classes are best separated by **B04 (red) and B12 (SWIR2)**, not NIR.
* **Gate:** passed (every issue has a logged decision).

## Phase 5 — Data preparation pipeline ✅

* **Done:** three 70/10/20 stratified splits — `random` (literature-comparable), **`grouped`** (neighbour groups kept whole; primary), `geo` (1° blocks; geographic hold-out) — plus 10/20/50 % low-data subsets; train-only per-band normalisation; per-patch NDVI/NDMI mean/std targets; `EuroSATSplit` loader with band-aligned flips/rotations and physically plausible consistency views (shared gain + correlated noise, no colour jitter).
* **Training data flow:** cache (uint16) → select canonical bands → batch → float → (augment geometry on all bands together) → (two radiometric views for consistency) → normalise with train mean/std → model.
* **Gate:** passed — 21 pytest tests: flips change all bands identically; no path in two splits; grouped/geo splits keep every neighbour group in one split; B8A read from the right channel; normalisation fitted on train only.

## Phase 6 — Link data to the research question ✅

* **Done:** hypothesis table H1–H7, each with the experiment and the result that would reject it; pair-level predictions; static selection decided; control auxiliary head designed; H6 margin pre-registered at 1.0 point — `docs/hypotheses_and_plan.md`.
* **Gate:** passed.

## Phase 7 — Baselines, protocol, experiment plan ⏳ (in progress)

* **Done so far:** protocol fixed; baseline models implemented (compact ResNet B2/B3, SE B6, input-band SE B7, CNN→Transformer B8 with 64 region tokens and gated fusion); trainer with early stopping on val, single test evaluation, bootstrap CI, agricultural metrics, cost report; filter-selection subsets (mRMR, PCA loadings, 20 random draws, RGB, RGB+NIR); B1 ResNet-50 reproduction script (GPU-ready).
* **Running (seed 0, grouped split, 30 epochs, ~1 min/epoch on CPU):** B3 12-band CNN, B2 same CNN on RGB, B8 CNN–Transformer. Results will be added below.
* **Still open for the gate:** B1 ResNet-50 RGB reproduction (≈70 min/epoch on this CPU → GPU or overnight), seeds 1–4, low-data runs.

### Results so far

| ID | Model | Bands | Split | Seed | Val acc | Test acc | Test macro-F1 | Params | MFLOPs |
|---|---|---|---|---|---|---|---|---|---|
| B0 | Random forest (mean/std) | 12 | grouped | 0–2 | 87.1 % | 88.2 ± 0.2 % | 87.8 % | — | — |
| B0 | Random forest (mean/std) | RGB | grouped | 0–2 | 75.6 % | 74.7 % | 73.7 % | — | — |
| B2 | Compact CNN, scratch | RGB | grouped | 0 | 96.59 % | **96.65 %** [96.17, 97.11] | 96.56 % | 1.23 M | 64.2 |
| B3 | Same compact CNN, scratch | 12 | grouped | 0 | 98.44 % | **98.24 %** [97.89, 98.59] | 98.17 % | 1.23 M | 66.8 |
| B8 | CNN→Transformer (64 tokens) | 12 | grouped | 0 | 95.96 % | 95.76 % | 95.67 % | 1.59 M | 76.5 |

| B8 v2 | CNN→Transformer, full 30 epochs | 12 | grouped | 0 | 97.74 % | **98.15 %** [97.78, 98.50] | 98.08 % | 1.59 M | 76.5 |

*First B8 row is invalid* — early-stopped at epoch 14 (D-020); superseded by B8 v2. B8 v2 ≈ B3 (98.15 vs 98.24 %, CIs overlap) with 29 % more parameters → no evidence yet that the Transformer adds value (H7); needs seeds and a parameter-matched CNN.

**Early reading of H1 (one seed only — not yet a claim):** with the *same* model and parameters, 12 bands beat RGB by +1.6 points (98.24 vs 96.65 %; 95 % bootstrap CIs do not overlap). Largest per-class gains: PermanentCrop recall 91.2 → 97.2 %, Pasture 94.3 → 97.0 %, Herbaceous 94.5 → 96.7 %, River 95.8 → 98.0 %. Agricultural F1 (from the 10-class model) 95.6 → 98.3 %. Seeds 1–4 needed before H1 is accepted.
Context: published from-scratch ResNet-50 on RGB ≈ 96.4 % (Helber Table II, random split) — our compact RGB CNN at 96.65 % on the *stricter* grouped split is in the same range.
