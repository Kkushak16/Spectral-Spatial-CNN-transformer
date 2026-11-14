"""Normalised-difference indices from raw Sentinel-2 DN (standard definitions, brief Sec. M3).

NDVI = (B8 - B4) / (B8 + B4 + eps)      NDMI = (B8 - B11) / (B8 + B11 + eps)
Valid pixel (EuroSAT, no cloud mask - decision D-010): all used bands finite and > 0.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + EPS)


def index_targets_batch(x: np.ndarray, order: list[str]) -> np.ndarray:
    """x: (B, C, H, W) raw DN in ``order``. Returns (B, 5): ndvi mean/std, ndmi mean/std, valid fraction."""
    x = np.asarray(x, dtype=np.float64)
    b4, b8, b11 = (x[:, order.index(b)] for b in ("B04", "B08", "B11"))
    valid = np.isfinite(b4) & np.isfinite(b8) & np.isfinite(b11) & (b4 > 0) & (b8 > 0) & (b11 > 0)
    ndvi, ndmi = nd(b8, b4), nd(b8, b11)
    out = np.zeros((x.shape[0], 5))
    for i in range(x.shape[0]):
        v = valid[i]
        if v.sum() == 0:
            out[i] = np.nan
            continue
        out[i] = [ndvi[i][v].mean(), ndvi[i][v].std(), ndmi[i][v].mean(), ndmi[i][v].std(), v.mean()]
    return out
