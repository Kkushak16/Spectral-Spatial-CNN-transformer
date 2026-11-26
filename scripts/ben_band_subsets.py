"""Filter band-selection baselines for BigEarthNet (training split only, CPU).

Same method as scripts/phase7_band_rankings.py (EuroSAT), adapted to 19-label multi-label data:
  mRMR    relevance = mutual information of each band's patch mean and std with each label, averaged over
          the 19 labels; redundancy = mean |pixel correlation| with the bands already chosen.
  PCA     PCA on standardised pixels; band score = sum over the top 3 components of
          explained_variance_ratio * loading^2; top-k bands by score.
  random  seeded draws of k bands.
Writes configs/band_subsets_ben.yaml.

    python scripts/ben_band_subsets.py --data D:/ben_shards --support D:/ben_support
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
CANON = [("b60", 0), ("b10", 0), ("b10", 1), ("b10", 2), ("b20", 0), ("b20", 1), ("b20", 2), ("b10", 3),
         ("b20", 3), ("b60", 1), ("b20", 4), ("b20", 5)]
UP = {"b10": 1, "b20": 2, "b60": 6}  # nearest-neighbour factor to the 10 m grid (120 x 120)
KS = [3, 4, 6, 8]
OUT = Path(__file__).resolve().parents[1] / "configs" / "band_subsets_ben.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="D:/ben_shards")
    ap.add_argument("--support", default="D:/ben_support")
    ap.add_argument("--shards", type=int, default=12, help="training shards to sample, evenly spaced")
    ap.add_argument("--pixels", type=int, default=2000, help="patches used for pixel correlation and PCA")
    a = ap.parse_args()

    index = pd.read_csv(Path(a.data) / "index.csv")
    meta = pd.read_parquet(Path(a.support) / "metadata.parquet", columns=["patch_id", "labels"])
    classes = sorted({c for labels in meta["labels"] for c in labels})
    labels = dict(zip(meta["patch_id"], meta["labels"]))
    train = sorted(index.loc[index["split"] == "train", "shard"].unique())
    names = [train[i] for i in np.linspace(0, len(train) - 1, a.shards).astype(int)]

    feats, ys, pix = [], [], []
    rng = np.random.default_rng(0)
    for name in names:
        with np.load(Path(a.data) / name) as z:
            g = {k: z[k] for k in ("b10", "b20", "b60")}
            ids = z["patch_id"].astype(str)
        x = [g[k][:, j].astype(np.float32) for k, j in CANON]  # 12 arrays at native resolution
        feats.append(np.stack([b.mean((1, 2)) for b in x] + [b.std((1, 2)) for b in x], 1))
        y = np.zeros((len(ids), len(classes)), np.uint8)
        for i, p in enumerate(ids):
            y[i, [classes.index(c) for c in labels[p]]] = 1
        ys.append(y)
        take = rng.choice(len(ids), max(1, a.pixels // len(names)), replace=False)
        full = [np.repeat(np.repeat(g[k][take, j], UP[k], 1), UP[k], 2) for k, j in CANON]
        pix.append(np.stack(full, 0).reshape(12, -1).astype(np.float64))
        print(f"{name}: {len(ids)} patches", flush=True)
    feat, y, pix = np.concatenate(feats), np.concatenate(ys), np.concatenate(pix, 1)
    print(f"{len(feat):,} patches, {pix.shape[1]:,} pixels")

    mi = np.mean([mutual_info_classif(feat, y[:, c], random_state=0) for c in range(len(classes))], 0)
    rel = mi[:12] + mi[12:]
    corr = np.abs(np.corrcoef(pix))
    chosen = [int(np.argmax(rel))]
    while len(chosen) < 12:
        rest = [i for i in range(12) if i not in chosen]
        score = [rel[i] - corr[i, chosen].mean() * rel.max() for i in rest]  # scaled difference form
        chosen.append(rest[int(np.argmax(score))])
    mrmr_order = [BANDS[i] for i in chosen]

    zs = (pix - pix.mean(1, keepdims=True)) / pix.std(1, keepdims=True)
    pca = PCA().fit(zs.T)
    score3 = (pca.explained_variance_ratio_[:3, None] * pca.components_[:3] ** 2).sum(0)
    pca_order = [BANDS[i] for i in np.argsort(-score3)]

    ordered = lambda bs: sorted(bs, key=BANDS.index)
    out = {
        "source": f"BigEarthNet v2 train split, {len(names)} shards, {len(feat)} patches",
        "rel_mi": dict(zip(BANDS, np.round(rel, 4).tolist())),
        "pca_explained_variance_ratio": np.round(pca.explained_variance_ratio_, 4).tolist(),
        "mrmr_order": mrmr_order,
        "pca_loading_order": pca_order,
        "natural": {3: ["B02", "B03", "B04"], 4: ["B02", "B03", "B04", "B08"]},
        "mrmr": {k: ordered(mrmr_order[:k]) for k in KS},
        "pca": {k: ordered(pca_order[:k]) for k in KS},
        "random": {k: [ordered(np.random.default_rng(100 + d).choice(BANDS, k, replace=False).tolist()) for d in range(5)]
                   for k in KS},
    }
    OUT.write_text(yaml.safe_dump(out, sort_keys=False, default_flow_style=None))
    print("MI relevance:", out["rel_mi"])
    print("mRMR order:", mrmr_order)
    print("PCA order:", pca_order, "EVR:", out["pca_explained_variance_ratio"][:4])
    for k in KS:
        print(f"k={k}  mRMR {out['mrmr'][k]}  PCA {out['pca'][k]}  random[0] {out['random'][k][0]}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
