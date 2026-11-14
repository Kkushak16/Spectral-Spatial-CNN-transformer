"""Phase 7: filter band-selection baselines (train split only).

S3 mRMR: greedy max-relevance (mutual information with class) min-redundancy (mean |corr| with chosen bands),
          on per-patch band means + per-pixel correlation for redundancy.
S4 PCA loading ranking: PCA on standardised pixels (12 bands); band score = sum over components of
          explained_variance_ratio * loading^2; top-k bands by score (raw PCA components are not bands).
S1 random subsets: 20 seeded draws per k.
S2 natural subsets: RGB (k=3), RGB+NIR (k=4).
Writes configs/band_subsets_eurosat.yaml
"""

from __future__ import annotations

import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

from agrispec import PROJECT_ROOT
from agrispec.data.eurosat import CANONICAL_12, EuroSATSplit

P = PROJECT_ROOT
ds = EuroSATSplit(bands="all12")
tr = ds.load("train")
xm = tr.x.astype(np.float32).mean((2, 3))
xs = tr.x.astype(np.float32).std((2, 3))
feat = np.concatenate([xm, xs], 1)

# relevance: MI of (mean, std) per band with class (sum of the two)
mi = mutual_info_classif(feat, tr.y, random_state=0)
rel = mi[:12] + mi[12:]
rng = np.random.default_rng(0)
pix = tr.x[rng.choice(len(tr), 1500, replace=False)].astype(np.float64).transpose(1, 0, 2, 3).reshape(12, -1)
C = np.abs(np.corrcoef(pix))

chosen = [int(np.argmax(rel))]
while len(chosen) < 12:
    rest = [i for i in range(12) if i not in chosen]
    score = [rel[i] - C[i, chosen].mean() * rel.max() for i in rest]  # scaled difference form
    chosen.append(rest[int(np.argmax(score))])
mrmr_order = [CANONICAL_12[i] for i in chosen]

z = (pix - pix.mean(1, keepdims=True)) / pix.std(1, keepdims=True)
pca = PCA().fit(z.T)
score = (pca.explained_variance_ratio_[:, None] * pca.components_ ** 2).sum(0)
# variance-weighted loadings favour bands in the largest correlated block; use top-3 components only
score3 = (pca.explained_variance_ratio_[:3, None] * pca.components_[:3] ** 2).sum(0)
pca_order = [CANONICAL_12[i] for i in np.argsort(-score3)]

ks = [3, 4, 6, 8]
subsets = {
    "rel_mi": dict(zip(CANONICAL_12, np.round(rel, 4).tolist())),
    "pca_explained_variance_ratio": np.round(pca.explained_variance_ratio_, 4).tolist(),
    "mrmr_order": mrmr_order,
    "pca_loading_order": pca_order,
    "S2_natural": {3: ["B02", "B03", "B04"], 4: ["B02", "B03", "B04", "B08"]},
    "S3_mrmr": {k: sorted(mrmr_order[:k], key=CANONICAL_12.index) for k in ks},
    "S4_pca": {k: sorted(pca_order[:k], key=CANONICAL_12.index) for k in ks},
    "S1_random": {k: [sorted(np.random.default_rng(100 + d).choice(CANONICAL_12, k, replace=False).tolist(),
                             key=CANONICAL_12.index) for d in range(20)] for k in ks},
}
with open(P / "configs/band_subsets_eurosat.yaml", "w") as f:
    yaml.safe_dump(subsets, f, sort_keys=False, default_flow_style=None)
print("MI relevance:", subsets["rel_mi"])
print("PCA EVR:", subsets["pca_explained_variance_ratio"][:5])
print("mRMR order:", mrmr_order)
print("PCA-loading order:", pca_order)
for k in ks:
    print(k, "mRMR", subsets["S3_mrmr"][k], "| PCA", subsets["S4_pca"][k])
