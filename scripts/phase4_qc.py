"""Phase 4: visualisation and quality checks on EuroSAT MS.

Checks: class balance, agricultural share, invalid/constant patches, exact duplicates, near duplicates,
spatial overlap / adjacency between patches (leakage), geographic distribution, composites.
Writes results/phase4_qc.json, data/processed/eurosat_groups.csv, figures in results/figures/.
Spectral statistics by class are computed later on the training split only (Phase 5).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from agrispec import PROJECT_ROOT

P = PROJECT_ROOT
FIG = P / "results/figures"
X = np.load(P / "data/processed/eurosat_ms_uint16.npy", mmap_mode="r")
meta = pd.read_csv(P / "data/processed/eurosat_meta.csv")
N = len(meta)
rep: dict = {}

# ---- counts --------------------------------------------------------------------------------------
counts = meta["class"].value_counts().sort_index()
rep["class_counts"] = counts.to_dict()
agri = {"AnnualCrop", "PermanentCrop"}
rep["agri_binary_positive"] = int(meta["class"].isin(agri).sum())
rep["agri_binary_positive_frac"] = round(rep["agri_binary_positive"] / N, 4)
rep["agri_plus_pasture_positive"] = int(meta["class"].isin(agri | {"Pasture"}).sum())
rep["agri_plus_pasture_frac"] = round(rep["agri_plus_pasture_positive"] / N, 4)

# ---- validity --------------------------------------------------------------------------------------
rep["shapes"] = meta["shape"].value_counts().to_dict()
rep["dtypes"] = meta["dtype"].value_counts().to_dict()
rep["global_min"], rep["global_max"] = int(meta["min"].min()), int(meta["max"].max())
rep["patches_with_any_zero"] = int((meta["zero_frac"] > 0).sum())
rep["patches_zero_frac_gt_1pct"] = int((meta["zero_frac"] > 0.01).sum())
rep["patches_with_constant_band"] = int((meta["n_const_bands"] > 0).sum())
const_by_n = meta.loc[meta["n_const_bands"] > 0, "n_const_bands"].value_counts().to_dict()
rep["constant_band_count_histogram"] = {int(k): int(v) for k, v in const_by_n.items()}
# which bands are constant? (B01/B09/B10 are 60 m -> upsampled; constancy within 640 m is plausible)
const_band_hits = np.zeros(13, int)
for i in meta.index[meta["n_const_bands"] > 0]:
    a = X[i].reshape(13, -1)
    const_band_hits += (a.std(1) == 0)
rep["constant_band_hits_by_file_index"] = const_band_hits.tolist()
# saturation
rep["patches_with_max_ge_16000"] = int((meta["max"] >= 16000).sum())
rep["patches_max_eq_65535"] = int((meta["max"] == 65535).sum())

# ---- exact duplicates ------------------------------------------------------------------------------
dup = meta[meta.duplicated("sha1", keep=False)].sort_values("sha1")
rep["exact_duplicate_patches"] = int(len(dup))
rep["exact_duplicate_groups"] = int(dup["sha1"].nunique())
rep["exact_duplicate_cross_class_groups"] = int((dup.groupby("sha1")["class"].nunique() > 1).sum())

# ---- spatial overlap / adjacency ------------------------------------------------------------------
rep["crs_counts"] = meta["crs"].value_counts().to_dict()
pairs_overlap, pairs_adjacent = [], []
for crs, g in meta.groupby("crs"):
    pts = g[["cx", "cy"]].values
    tree = cKDTree(pts)
    ids = g["idx"].values
    # patch side ~640 m: centres < 640 m apart => footprints overlap; < 1000 m => touching/adjacent
    for a, b in tree.query_pairs(r=999.0):
        d = np.hypot(*(pts[a] - pts[b]))
        (pairs_overlap if d < 639.0 else pairs_adjacent).append((ids[a], ids[b], d))
rep["pairs_overlapping_footprint"] = len(pairs_overlap)
rep["pairs_adjacent_within_1km"] = len(pairs_adjacent)
ov = pd.DataFrame(pairs_overlap, columns=["i", "j", "dist_m"])
if len(ov):
    ov["ci"] = meta.loc[ov.i, "class"].values
    ov["cj"] = meta.loc[ov.j, "class"].values
    rep["overlap_pairs_cross_class"] = int((ov.ci != ov.cj).sum())
    rep["overlap_distance_quantiles_m"] = ov.dist_m.quantile([0, .25, .5, .75, 1]).round(1).to_dict()
# groups: connected components over overlapping-or-adjacent pairs + exact duplicates -> keep together in splits
edges = [(i, j) for i, j, _ in pairs_overlap + pairs_adjacent]
for _, g in dup.groupby("sha1"):
    ids = g["idx"].tolist()
    edges += [(ids[0], k) for k in ids[1:]]
if edges:
    e = np.array(edges)
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(N, N))
    ncomp, comp = connected_components(A, directed=False)
else:
    comp = np.arange(N)
sizes = pd.Series(comp).value_counts()
rep["leakage_groups"] = int(len(sizes))
rep["leakage_groups_size_gt1"] = int((sizes > 1).sum())
rep["patches_in_groups_gt1"] = int(sizes[sizes > 1].sum())
rep["largest_group"] = int(sizes.max())

# geographic blocks (0.5 degree cells) for the geographic split
meta["group"] = comp
meta["geo_cell"] = (np.floor(meta["lat"] / 0.5).astype(int).astype(str) + "_" +
                    np.floor(meta["lon"] / 0.5).astype(int).astype(str))
rep["geo_cells_0p5deg"] = int(meta["geo_cell"].nunique())
rep["lat_range"] = [round(meta.lat.min(), 3), round(meta.lat.max(), 3)]
rep["lon_range"] = [round(meta.lon.min(), 3), round(meta.lon.max(), 3)]
meta[["idx", "path", "class", "label", "lat", "lon", "crs", "sha1", "group", "geo_cell"]].to_csv(
    P / "data/processed/eurosat_groups.csv", index=False)

# ---- near duplicates (different bytes, nearly identical content) ----------------------------------
# 16x16 block means of B02,B03,B04,B08 (file idx 1,2,3,7), per-patch standardised, cosine similarity
feat = np.empty((N, 4 * 256), np.float32)
for s in range(0, N, 2000):
    a = X[s:s + 2000][:, [1, 2, 3, 7]].astype(np.float32).reshape(-1, 4, 16, 4, 16, 4).mean((3, 5))
    a = a.reshape(len(a), -1)
    a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-6)
    feat[s:s + 2000] = a / np.linalg.norm(a, axis=1, keepdims=True)
near = []
for s in range(0, N, 1000):
    sim = feat[s:s + 1000] @ feat.T
    for r in range(sim.shape[0]):
        sim[r, s + r] = -1
    ii, jj = np.nonzero(sim > 0.995)
    near += [(s + i, j, float(sim[i, j])) for i, j in zip(ii, jj) if s + i < j]
nd = pd.DataFrame(near, columns=["i", "j", "cos"])
nd = nd[meta.loc[nd.i, "sha1"].values != meta.loc[nd.j, "sha1"].values] if len(nd) else nd
rep["near_duplicate_pairs_cos_gt_0.995"] = int(len(nd))
if len(nd):
    nd["same_group"] = meta.loc[nd.i, "group"].values == meta.loc[nd.j, "group"].values
    nd["ci"], nd["cj"] = meta.loc[nd.i, "class"].values, meta.loc[nd.j, "class"].values
    rep["near_duplicate_pairs_not_already_grouped"] = int((~nd.same_group).sum())
    rep["near_duplicate_pairs_cross_class"] = int((nd.ci != nd.cj).sum())
    rep["near_duplicate_pairs_by_class"] = nd[nd.ci == nd.cj].ci.value_counts().to_dict()
    nd.to_csv(P / "results/phase4_near_duplicates.csv", index=False)

with open(P / "results/phase4_qc.json", "w") as f:
    json.dump(rep, f, indent=2, default=int)
print(json.dumps(rep, indent=2, default=int))

# ---- figures --------------------------------------------------------------------------------------
def stretch(x, lo=2, hi=98):
    a, b = np.percentile(x, (lo, hi))
    return np.clip((x - a) / (b - a + 1e-6), 0, 1)

classes = sorted(meta["class"].unique())
rng = np.random.default_rng(1)
fig, axes = plt.subplots(len(classes), 6, figsize=(10, 17))
for r, c in enumerate(classes):
    ids = rng.choice(meta.index[meta["class"] == c], 3, replace=False)
    for k, i in enumerate(ids):
        a = X[i].astype(np.float32)
        rgb = np.stack([stretch(a[3]), stretch(a[2]), stretch(a[1])], -1)
        cir = np.stack([stretch(a[7]), stretch(a[3]), stretch(a[2])], -1)
        axes[r, 2 * k].imshow(rgb); axes[r, 2 * k + 1].imshow(cir)
    axes[r, 0].set_ylabel(c, fontsize=8)
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
axes[0, 0].set_title("RGB", fontsize=8); axes[0, 1].set_title("NIR false colour", fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "phase4_composites.png", dpi=110); plt.close()

fig, ax = plt.subplots(figsize=(8, 7))
for c in classes:
    m = meta["class"] == c
    ax.scatter(meta.lon[m], meta.lat[m], s=1, label=c)
ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.legend(markerscale=8, fontsize=7); ax.set_title("EuroSAT patch centres")
plt.tight_layout(); plt.savefig(FIG / "phase4_geography.png", dpi=110); plt.close()

fig, ax = plt.subplots(figsize=(8, 4))
counts.plot.bar(ax=ax); ax.set_title("EuroSAT MS class counts"); plt.tight_layout()
plt.savefig(FIG / "phase4_class_counts.png", dpi=110); plt.close()
print("figures saved")
