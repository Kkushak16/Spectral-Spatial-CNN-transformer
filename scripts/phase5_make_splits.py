"""Phase 5: splits, normalisation statistics, physics index targets.

Splits (all 70/10/20, stratified by class, fixed split seed 0; model seeds vary separately - D-012):
  random   : plain stratified per-patch split (comparable with Helber et al. protocol)
  grouped  : spatial-leakage groups (overlapping/adjacent <1 km footprints) kept in one split  [PRIMARY]
  geo      : 1-degree geographic blocks kept in one split (geographic hold-out)
Low-data: nested stratified subsets of each train split at 10/20/50 % (columns lowdata_10 ...).

Normalisation: per-band mean/std (float64) over TRAIN pixels only, canonical 12 bands (+B10 for the 13-band run).
Index targets: per-patch NDVI/NDMI mean and std from raw DN on valid pixels; z-scoring constants from TRAIN only.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from agrispec import PROJECT_ROOT
from agrispec.data.indices import index_targets_batch
from agrispec.utils.config import file_hash

P = PROJECT_ROOT
SPLIT_SEED = 0
X = np.load(P / "data/processed/eurosat_ms_uint16.npy", mmap_mode="r")
g = pd.read_csv(P / "data/processed/eurosat_groups.csv")
band_map = yaml.safe_load(open(P / "configs/band_map_eurosat.yaml"))
y = g["label"].values
N = len(g)


def split_random() -> np.ndarray:
    s = np.empty(N, dtype=object)
    sss = StratifiedShuffleSplit(1, test_size=0.2, random_state=SPLIT_SEED)
    trval, te = next(sss.split(np.zeros(N), y))
    s[te] = "test"
    sss2 = StratifiedShuffleSplit(1, test_size=0.125, random_state=SPLIT_SEED)  # 0.125*0.8 = 0.1
    tr, va = next(sss2.split(np.zeros(len(trval)), y[trval]))
    s[trval[tr]], s[trval[va]] = "train", "val"
    return s


def split_grouped(groups: np.ndarray) -> np.ndarray:
    """10 stratified group folds: folds 0-1 test (20 %), fold 2 val (10 %), rest train."""
    s = np.empty(N, dtype=object)
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SPLIT_SEED)
    for f, (_, idx) in enumerate(sgkf.split(np.zeros(N), y, groups)):
        s[idx] = "test" if f < 2 else "val" if f == 2 else "train"
    return s


def lowdata(split: np.ndarray, frac: float) -> np.ndarray:
    tr = np.nonzero(split == "train")[0]
    keep = np.zeros(N, bool)
    sss = StratifiedShuffleSplit(1, train_size=frac, random_state=SPLIT_SEED)
    sel, _ = next(sss.split(np.zeros(len(tr)), y[tr]))
    keep[tr[sel]] = True
    return keep


def main():
    # geographic block key: 1 deg cell of the leakage group's first member (keeps groups whole)
    cell = (np.floor(g.lat).astype(int).astype(str) + "_" + np.floor(g.lon).astype(int).astype(str))
    first = g.groupby("group")["idx"].transform("min")
    geo_key = cell.values[first.values]

    splits = {"random": split_random(), "grouped": split_grouped(g["group"].values),
              "geo": split_grouped(pd.factorize(geo_key)[0])}
    (P / "splits").mkdir(exist_ok=True)
    (P / "results").mkdir(exist_ok=True)
    summary = {}
    for name, s in splits.items():
        df = g[["idx", "path", "class", "label", "group"]].copy()
        df["geo_block"] = geo_key
        df["split"] = s
        # nested low-data subsets (10 % subset of 20 % subset of 50 % is not required; independent stratified draws)
        for frac in (0.1, 0.2, 0.5):
            df[f"lowdata_{int(frac * 100)}"] = lowdata(s, frac)
        out = P / f"splits/eurosat_{name}_s{SPLIT_SEED}.csv"
        df.to_csv(out, index=False)
        # leakage diagnostics
        grp_splits = df.groupby("group")["split"].nunique()
        geo_splits = df.groupby("geo_block")["split"].nunique()
        summary[name] = {
            "file": out.name, "hash": file_hash(out),
            "counts": df["split"].value_counts().to_dict(),
            "class_frac_test": df[df.split == "test"]["class"].value_counts(normalize=True).round(3).to_dict(),
            "leakage_groups_spanning_splits": int((grp_splits > 1).sum()),
            "geo_blocks_spanning_splits": int((geo_splits > 1).sum()),
        }
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "class_frac_test"} for k, v in summary.items()},
                     indent=2))

    # ---- normalisation + index targets per split ------------------------------------------------
    order = band_map["file_order"]
    idx12 = band_map["canonical_12_file_indices"]
    idx13 = [order.index(b) for b in ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10",
                                      "B11", "B12"]]
    # per-patch index targets (deterministic function of each patch; no fitting)
    tg = []
    for s0 in range(0, N, 2000):
        tg.append(index_targets_batch(X[s0:s0 + 2000], order))
    tg = pd.DataFrame(np.concatenate(tg), columns=["ndvi_mean", "ndvi_std", "ndmi_mean", "ndmi_std",
                                                   "valid_frac"])
    tg.insert(0, "idx", np.arange(N))
    tg.to_csv(P / "data/processed/eurosat_index_targets.csv", index=False)

    for name, s in splits.items():
        tr = np.nonzero(s == "train")[0]
        s1 = np.zeros(13); s2 = np.zeros(13); n = 0
        for s0 in range(0, len(tr), 1000):
            a = X[np.sort(tr[s0:s0 + 1000])].astype(np.float64)
            s1 += a.sum(axis=(0, 2, 3)); s2 += (a ** 2).sum(axis=(0, 2, 3)); n += a.shape[0] * 64 * 64
        mean = s1 / n
        std = np.sqrt(s2 / n - mean ** 2)
        tgt = tg.iloc[tr]
        norm = {
            "fitted_on": f"splits/eurosat_{name}_s{SPLIT_SEED}.csv train ({len(tr)} patches)",
            "file_order": order,
            "mean_file_order": mean.round(4).tolist(),
            "std_file_order": std.round(4).tolist(),
            "canonical_12": band_map["canonical_12"],
            "canonical_12_file_indices": idx12,
            "canonical_13_file_indices": idx13,
            "index_target_mean": tgt[["ndvi_mean", "ndvi_std", "ndmi_mean", "ndmi_std"]].mean().round(6).tolist(),
            "index_target_std": tgt[["ndvi_mean", "ndvi_std", "ndmi_mean", "ndmi_std"]].std().round(6).tolist(),
        }
        with open(P / f"configs/norm_eurosat_{name}_s{SPLIT_SEED}.yaml", "w") as f:
            yaml.safe_dump(norm, f, sort_keys=False)
    with open(P / "results/phase5_splits.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("index target valid_frac min:", tg.valid_frac.min(), " patches with <100% valid:", int((tg.valid_frac < 1).sum()))


if __name__ == "__main__":
    main()
