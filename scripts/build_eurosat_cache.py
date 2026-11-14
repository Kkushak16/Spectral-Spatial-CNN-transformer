"""Phase 3: read every EuroSAT MS GeoTIFF once.

Outputs
  data/processed/eurosat_ms_uint16.npy   (N, 13, 64, 64) uint16, raw file band order
  data/processed/eurosat_meta.csv        one row per patch: path, class, label, CRS, centre
                                         coordinates (UTM and WGS84), hashes, QC statistics
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform

from agrispec import PROJECT_ROOT

ROOT = PROJECT_ROOT / "EuroSAT_MS" / "EuroSAT_MS"
OUT = PROJECT_ROOT / "data" / "processed"


def read_one(path: str):
    with rasterio.open(path) as src:
        arr = src.read()
        crs = src.crs.to_string() if src.crs else ""
        b = src.bounds
        res = src.res
    cx, cy = (b.left + b.right) / 2, (b.top + b.bottom) / 2
    lon, lat = (np.nan, np.nan)
    if crs:
        xs, ys = warp_transform(crs, "EPSG:4326", [cx], [cy])
        lon, lat = xs[0], ys[0]
    a = arr.astype(np.float64)
    meta = {
        "crs": crs, "cx": cx, "cy": cy, "lon": lon, "lat": lat, "res_x": res[0], "res_y": res[1],
        "shape": "x".join(map(str, arr.shape)), "dtype": str(arr.dtype),
        "sha1": hashlib.sha1(arr.tobytes()).hexdigest(),
        "min": int(arr.min()), "max": int(arr.max()),
        "zero_frac": float((arr == 0).mean()),
        "n_const_bands": int((arr.reshape(arr.shape[0], -1).std(1) == 0).sum()),
        "band_means": a.reshape(a.shape[0], -1).mean(1).round(2).tolist(),
    }
    return arr, meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    classes = sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    rows = []
    for ci, c in enumerate(classes):
        files = sorted((ROOT / c).glob("*.tif"), key=lambda p: int(p.stem.split("_")[-1]))
        rows += [{"path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"), "class": c, "label": ci} for p in files]
    n = len(rows)
    cache = np.lib.format.open_memmap(OUT / "eurosat_ms_uint16.npy", mode="w+", dtype=np.uint16, shape=(n, 13, 64, 64))
    paths = [str(PROJECT_ROOT / r["path"]) for r in rows]
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, (arr, meta) in enumerate(ex.map(read_one, paths, chunksize=64)):
            assert arr.shape == (13, 64, 64), (paths[i], arr.shape)
            cache[i] = arr
            rows[i].update(meta)
            if i % 3000 == 0:
                print(f"{i}/{n}", flush=True)
    cache.flush()
    df = pd.DataFrame(rows)
    df.insert(0, "idx", range(n))
    df.to_csv(OUT / "eurosat_meta.csv", index=False)
    print("done", n, "patches;", df["class"].value_counts().to_dict())


if __name__ == "__main__":
    main()
