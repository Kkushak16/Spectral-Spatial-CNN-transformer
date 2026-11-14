"""Phase 4 (spectral part) + baseline B0, computed on the grouped split.

Statistics/figures use TRAIN only. Random forest B0 is fit on train, model choices on val; test once.
  * class spectral signatures (mean +- std of patch-mean DN) - figure
  * NDVI / NDMI distributions by class - figure
  * per-band Fisher ratio (between-class / within-class variance of patch means), all 10 classes and crop trio
  * B0: random forest on per-band patch mean + std (12 bands -> 24 features); also RGB-only and per single band
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from agrispec import PROJECT_ROOT
from agrispec.data.eurosat import CANONICAL_12, EuroSATSplit

P = PROJECT_ROOT
ds = EuroSATSplit(split_csv="splits/eurosat_grouped_s0.csv", norm_yaml="configs/norm_eurosat_grouped_s0.yaml",
                  bands="all12")
parts = {p: ds.load(p) for p in ("train", "val", "test")}
classes = ds.classes


def feats(a):
    x = a.x.astype(np.float32)
    return np.concatenate([x.mean((2, 3)), x.std((2, 3))], 1)


F = {p: feats(a) for p, a in parts.items()}
Y = {p: a.y for p, a in parts.items()}
tr_mean = F["train"][:, :12]

# ---- signatures ---------------------------------------------------------------------------------
sig = pd.DataFrame(tr_mean, columns=CANONICAL_12)
sig["class"] = [classes[i] for i in Y["train"]]
g_mean, g_std = sig.groupby("class").mean(), sig.groupby("class").std()
wl = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190]
fig, ax = plt.subplots(figsize=(10, 5.5))
for c in classes:
    ax.errorbar(wl, g_mean.loc[c], yerr=g_std.loc[c], marker="o", capsize=2, label=c, alpha=.85)
ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("patch-mean DN (TOA, train split)")
ax.set_title("EuroSAT class spectral signatures (mean ± std across patches)"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(P / "results/figures/phase4_signatures_train.png", dpi=120); plt.close()

# ---- indices ----------------------------------------------------------------------------------
t = pd.read_csv(P / "data/processed/eurosat_index_targets.csv").set_index("idx")
tr_rows = ds.df.iloc[ds.indices("train")]
ti = t.loc[tr_rows["idx"]].copy(); ti["class"] = tr_rows["class"].values
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, col in zip(axes, ["ndvi_mean", "ndmi_mean"]):
    ax.boxplot([ti.loc[ti["class"] == c, col] for c in classes], showfliers=False)
    ax.set_xticks(range(1, 11), classes, rotation=40, ha="right", fontsize=8); ax.set_title(col + " (train)")
plt.tight_layout(); plt.savefig(P / "results/figures/phase4_indices_train.png", dpi=120); plt.close()
idx_table = ti.groupby("class")[["ndvi_mean", "ndvi_std", "ndmi_mean", "ndmi_std"]].mean().round(3)

# ---- Fisher ratio ----------------------------------------------------------------------------
def fisher(xm, y):
    overall = xm.mean(0)
    sb = sum((y == k).sum() * (xm[y == k].mean(0) - overall) ** 2 for k in np.unique(y))
    sw = sum(((xm[y == k] - xm[y == k].mean(0)) ** 2).sum(0) for k in np.unique(y))
    return sb / sw


fis_all = fisher(tr_mean, Y["train"])
crop = [classes.index(c) for c in ("AnnualCrop", "PermanentCrop", "Pasture")]
m = np.isin(Y["train"], crop)
fis_crop = fisher(tr_mean[m], Y["train"][m])

# ---- B0 random forest -----------------------------------------------------------------------
def rf_eval(cols, seed=0, final=False):
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    rf.fit(F["train"][:, cols], Y["train"])
    out = {}
    for p in (["val", "test"] if final else ["val"]):
        pr = rf.predict(F[p][:, cols])
        out[p] = {"acc": round(accuracy_score(Y[p], pr), 4), "macro_f1": round(f1_score(Y[p], pr, average="macro"), 4)}
    return out


all_cols = list(range(24))
rgb_cols = [CANONICAL_12.index(b) for b in ("B04", "B03", "B02")]
rgb_cols = rgb_cols + [c + 12 for c in rgb_cols]
b0 = {f"seed{s}": rf_eval(all_cols, s, final=True) for s in range(3)}
b0_rgb = {f"seed{s}": rf_eval(rgb_cols, s, final=True) for s in range(3)}
single = {b: rf_eval([i, i + 12])["val"]["acc"] for i, b in enumerate(CANONICAL_12)}

rep = {
    "fisher_all_classes": dict(zip(CANONICAL_12, np.round(fis_all, 3).tolist())),
    "fisher_crop_trio": dict(zip(CANONICAL_12, np.round(fis_crop, 3).tolist())),
    "index_means_by_class_train": idx_table.to_dict(orient="index"),
    "B0_rf_12band_meanstd": b0,
    "B0_rf_rgb_meanstd": b0_rgb,
    "single_band_rf_val_acc": single,
}
with open(P / "results/phase4b_spectral.json", "w") as f:
    json.dump(rep, f, indent=2)
print(json.dumps({k: v for k, v in rep.items() if k != "index_means_by_class_train"}, indent=1))
print(idx_table.to_string())
