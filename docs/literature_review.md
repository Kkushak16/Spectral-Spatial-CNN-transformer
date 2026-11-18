# Literature review and novelty table (Phase 1)

Uploaded PDFs were text-extracted to `literature/text/` and read for the facts below; web results are marked (web) and were read from abstracts/snippets only — confirm in full text before citing.

## Uploaded papers

| File | Paper | What it does | What it gives this project | Limits for our question |
|---|---|---|---|---|
| eurosat_paper.pdf | Helber et al. 2019, EuroSAT (JSTARS) | Dataset + CNN benchmark | ResNet-50 RGB 98.57 % (pretrained, 80/20); from scratch ≈ 96.4 %; single-band study (RGB > others, B05/B12 strong, B10 weakest); CI 98.30, SWIR 97.05 | Random split, single runs, no CIs, no multispectral full-band result |
| REP1.pdf | Yassine et al. 2021 (ISPRS Archives) | DenseNet201 RGB vs 13 bands vs 13 bands + indices | 96.83 → 98.78 → 99.58 %; pair-level error table | Single run; index formulas partly wrong; band-label table consistent with the **B8A-last file order** we measured (their "water vapour ≈ 12" is actually B10, "cirrus ≈ 1,820" is B11) |
| REP2.pdf | Li et al. 2024 review (Sensors) | Spectral/spatial attention, CNN–Transformer hybrids for HSI | Menu of modules (SE/ECA/CBAM, fusion, token designs) | Hyperspectral, per-pixel labels; not comparable to 12-band patch classification |
| Photogrammetry & Remote sensing.pdf | Papoutsis et al. 2023 (ISPRS J.) | Benchmark of CNN/MLP/ViT on BigEarthNet; EfficientNet/WRN + ECA | Drops B01, B09, B10; EfficientNet-B0/WRN-ECA SOTA with ~10× fewer params; reports training time and inference rate; EuroSAT 80/10/10 transfer | No band selection; attention on feature channels |
| BigEarth.pdf | Sumbul et al. 2021, BigEarthNet-MM (GRSM) | S1+S2 multi-label archive, 19-class nomenclature | Dataset facts (see fact sheet); classes "Arable land", "Permanent crops", "Pastures", "Complex cultivation patterns", "Land principally occupied by agriculture…" for the agricultural mapping | This is not reBEN v2.0 |
| MultiModal Dataset.pdf | Wang et al. 2023, SSL4EO-S12 (GRSM) | Unlabelled S1/S2 corpus; MoCo/DINO/MAE/data2vec pretraining | EuroSAT up to 99.1 % fine-tuned; linear probing on EuroSAT/BigEarthNet 10 % labels — low-label comparison point | Large pretrained models; no band selection |
| SatMAE.pdf | Cong et al. 2022 (NeurIPS) | MAE pretraining with spectral **band groups** (RGB+NIR / red edge / SWIR) and group encodings | EuroSAT 98.98 % (ViT-L); grouping by resolution + wavelength helps; drops B1, B9, B10 | Huge models; groups are hand-designed, not learned |
| MSMatch.pdf | Gómez & Meoni 2021 (JSTARS) | Semi-supervised FixMatch-style on EuroSAT RGB and MS | 95.86 % (MS) with 5 labels per class — low-data reference | Uses all bands; no selection |
| GLUSE.pdf | Le et al. 2026 (JSTARS) | ResNet + gated-linear-unit SE for onboard EO | EuroSAT 94.63 % with 33× fewer params than MobileViT (99.09 %); reports power, FLOPs, latency | Feature-channel attention, not band selection |
| MobileNetV3.pdf | Howard et al. 2019 | Efficient CNN with SE + h-swish (NAS) | Candidate compact encoder | ImageNet-scale design |

## Web search (band selection with differentiable gates)

| Work | Method | Data | Relevance |
|---|---|---|---|
| Abid et al. 2019, Concrete Autoencoders (arXiv 1901.09346) (web) | Concrete selector layer, annealed temperature, fixed k | genes, MNIST | Origin of differentiable k-feature selection |
| Sun et al. 2021, CAE-UBS (TGRS) (web) | Gumbel-softmax concrete AE, unsupervised band selection + entropy post-step | HSI | Unsupervised; not task-driven |
| Xu, Ahishali, Gabbouj 2024, Dropout Concrete AE (arXiv 2401.16522) (web) | Concrete selector + dropout ranking, direct k | HSI (Salinas, PaviaU, Indian Pines, KSC) | Direct-k selection |
| Supervised embedded band selection, EHBS (L0) / CHBS (concrete) (arXiv 2401.11420) (web) | Structured L0 and concrete selector inside classifier | HSI | Closest to our gate — but HSI, pixel classification |
| SGBR-HC 2026 (arXiv 2606.06684) (web) | Band ranking → hard-concrete gates with L0, spatially disjoint evaluation | HSI (Pavia U, Houston 2013) | Hard-concrete gates + spatially disjoint evaluation already exist for HSI |

## Novelty table (draft)

| Component | Exists already? | Where | What is left for us |
|---|---|---|---|
| Learned static band gate (hard-concrete / concrete / L0) | **Yes** | CAE, EHBS/CHBS, SGBR-HC (HSI) | Apply to 12-band Sentinel-2 patch classification with *genuine* k-channel deployment and cost measurement |
| Band grouping / spectral tokens | Yes | SatMAE | Learned vs hand-designed groups |
| Feature-channel attention on S2 | Yes | Papoutsis (ECA), GLUSE | Contrast with input-band gating (only the latter maps to wavelengths) |
| Index-based auxiliary loss (NDVI/NDMI distillation from full bands) | Not found in this search | — | **Candidate contribution**; must include a control head (H3) |
| Consistency under physically plausible radiometric views + band selection | Not found in this search | — | **Candidate contribution** (H4) |
| Deletion-test faithfulness + cross-seed stability of selected bands on S2 | Partly (XAI papers generally) | — | Systematic test on EuroSAT + BigEarthNet |
| Spatially disjoint evaluation | Yes for HSI (SGBR-HC); not standard for EuroSAT | — | **Our measured EuroSAT leakage (59 % of patches in neighbour groups)** and grouped/geo splits |

**Conclusion (Interpretation):** learned sparse band gates are not new; the claimable contribution is the *joint* package (gate + physics distillation + consistency + faithfulness tests + leakage-aware evaluation + measured cost) on Sentinel-2 — and only if experiments support it. Do not claim "first". A systematic search (Scopus/Google Scholar: "band selection" AND Sentinel-2 AND (gate OR concrete OR L0)) is still to be done before the write-up.
