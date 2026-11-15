"""Build the physics auxiliary-head targets without re-running QC or re-splitting.

Writes data/processed/eurosat_index_targets.csv: per-patch NDVI/NDMI mean and std on valid pixels,
computed from raw DN (standard definitions, see src/agrispec/data/indices.py). Needed only when
training with loss.lambda_index > 0. The z-scoring constants already live in configs/norm_eurosat_*_s0.yaml
(fitted on each split's training part). Requires data/processed/eurosat_ms_uint16.npy
(scripts/build_eurosat_cache.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from agrispec import PROJECT_ROOT
from agrispec.data.indices import index_targets_batch

P = PROJECT_ROOT


def main():
    X = np.load(P / "data/processed/eurosat_ms_uint16.npy", mmap_mode="r")
    order = yaml.safe_load(open(P / "configs/band_map_eurosat.yaml"))["file_order"]
    rows = [index_targets_batch(X[s:s + 2000], order) for s in range(0, len(X), 2000)]
    tg = pd.DataFrame(np.concatenate(rows), columns=["ndvi_mean", "ndvi_std", "ndmi_mean", "ndmi_std", "valid_frac"])
    tg.insert(0, "idx", np.arange(len(tg)))
    out = P / "data/processed/eurosat_index_targets.csv"
    tg.to_csv(out, index=False)
    print(f"wrote {out} ({len(tg)} patches; min valid fraction {tg.valid_frac.min():.4f})")


if __name__ == "__main__":
    main()
