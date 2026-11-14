"""Phase 3: verify EuroSAT MS band order empirically (do not trust names).

Evidence used
  1. class-mean spectra per raw channel index (vegetation bright NIR / dark red; water dark beyond visible;
     cirrus ~0 on clear scenes),
  2. pixel-wise correlation of every channel with channel 7 (assumed B08, 842 nm): the true B8A (865 nm)
     must be the most correlated channel with B08, far above the 945 nm water-vapour band or SWIR.
Writes configs/band_map_eurosat.yaml and results/figures/phase3_class_spectra.png.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agrispec import PROJECT_ROOT

P = PROJECT_ROOT
X = np.load(P / "data/processed/eurosat_ms_uint16.npy", mmap_mode="r")
meta = pd.read_csv(P / "data/processed/eurosat_meta.csv")
rng = np.random.default_rng(0)

# 1. class-mean spectra (300 random patches per class)
means = {}
for c, g in meta.groupby("class"):
    idx = np.sort(rng.choice(g["idx"].values, 300, replace=False))
    means[c] = X[idx].astype(np.float64).mean(axis=(0, 2, 3))
spec = pd.DataFrame(means).T
spec.columns = [f"ch{i}" for i in range(13)]

# 2. pixel-wise correlation with channel 7 over a random sample of patches
idx = np.sort(rng.choice(len(meta), 2000, replace=False))
S = X[idx].astype(np.float64).transpose(1, 0, 2, 3).reshape(13, -1)
corr = np.corrcoef(S)

print("Class-mean spectra (raw DN) by raw channel index:")
print(spec.round(0).astype(int).to_string())
print("\nPixel correlation of each channel with ch7:", np.round(corr[7], 3).tolist())
print("Global channel means:", np.round(S.mean(1), 1).tolist())

hyp_A = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]  # S2 order
hyp_B = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]  # B8A last

# decision rule: B8A is the channel (other than 7) most correlated with B08; cirrus is the channel with the
# smallest global mean.
# B8A can only sit at index 8 (S2 order) or 12 (appended last); B07 (783 nm) is also highly correlated
# with B08, so compare the two candidate positions only.
cand = [8, 12]
b8a = max(cand, key=lambda i: corr[7, i])
cirrus = int(np.argmin(S.mean(1)))
print(f"\nMost B08-correlated channel: ch{b8a} (r={corr[7, b8a]:.3f}); lowest-mean channel: ch{cirrus}")
order = hyp_B if (b8a == 12 and cirrus == 9) else hyp_A if (b8a == 8 and cirrus == 10) else None
assert order is not None, "band order matches neither hypothesis - inspect manually"
print("Verified order:", order)

# checks on physics with the verified order
pos = {b: i for i, b in enumerate(order)}
for c in ["Forest", "SeaLake", "AnnualCrop"]:
    r, n, s1 = spec.loc[c, f"ch{pos['B04']}"], spec.loc[c, f"ch{pos['B08']}"], spec.loc[c, f"ch{pos['B11']}"]
    print(f"{c:12s} red={r:7.0f} nir={n:7.0f} swir1={s1:7.0f} NIR/red={n / r:5.2f}")
assert spec.loc["Forest", f"ch{pos['B08']}"] > 2 * spec.loc["Forest", f"ch{pos['B04']}"], "vegetation NIR test failed"
assert spec.loc["SeaLake", f"ch{pos['B08']}"] < spec.loc["SeaLake", f"ch{pos['B03']}"], "water NIR test failed"

keep12 = [b for b in ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]]
band_map = {
    "dataset": "EuroSAT MS (GeoTIFF, 13 x 64 x 64 uint16)",
    "file_order": order,
    "verified_by": "class-mean spectra + pixel correlation with B08 (scripts/phase3_band_order.py)",
    "b8a_corr_with_b08": float(corr[7, pos["B8A"]]),
    "canonical_12": keep12,
    "canonical_12_file_indices": [pos[b] for b in keep12],
    "dropped": ["B10"],
    "note": "canonical_12 is Sentinel-2 wavelength order without B10; shared with BigEarthNet L2A",
}
with open(P / "configs/band_map_eurosat.yaml", "w") as f:
    yaml.safe_dump(band_map, f, sort_keys=False)

# figure
wl = {"B01": 443, "B02": 490, "B03": 560, "B04": 665, "B05": 705, "B06": 740, "B07": 783, "B08": 842,
      "B8A": 865, "B09": 945, "B10": 1375, "B11": 1610, "B12": 2190}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for c in spec.index:
    axes[0].plot(range(13), spec.loc[c].values, marker="o", label=c)
axes[0].set_xticks(range(13), [f"ch{i}" for i in range(13)])
axes[0].set_title("Raw channel index (file order)")
for c in spec.index:
    xs = [wl[b] for b in order if b != "B10"]
    ys = [spec.loc[c, f"ch{pos[b]}"] for b in order if b != "B10"]
    srt = np.argsort(xs)
    axes[1].plot(np.array(xs)[srt], np.array(ys)[srt], marker="o", label=c)
axes[1].set_xlabel("wavelength (nm), verified order, B10 omitted")
axes[1].set_title("Class-mean spectra (TOA DN)")
axes[1].legend(fontsize=7)
for a in axes:
    a.set_ylabel("mean DN")
plt.tight_layout()
(P / "results/figures").mkdir(parents=True, exist_ok=True)
plt.savefig(P / "results/figures/phase3_class_spectra.png", dpi=120)
spec.to_csv(P / "results/phase3_class_mean_spectra_raw.csv")
print("saved band map and figure")
