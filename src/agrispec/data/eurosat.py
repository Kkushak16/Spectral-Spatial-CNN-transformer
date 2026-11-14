"""EuroSAT MS loader: in-RAM uint16 cache, canonical band selection, train-only normalisation,
band-aligned augmentation, physics index targets and consistency views.

Designed for CPU training: the selected bands of the needed split are held in RAM as uint16 and
converted / normalised / augmented per batch (vectorised), avoiding per-sample Python overhead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from agrispec import PROJECT_ROOT

CANONICAL_12 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
BAND_SETS = {
    "all12": CANONICAL_12,
    "all13": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"],
    "rgb": ["B04", "B03", "B02"],
    "rgbn": ["B04", "B03", "B02", "B08"],
}
AGRI = {"AnnualCrop", "PermanentCrop"}


def resolve_bands(bands) -> list[str]:
    return BAND_SETS[bands] if isinstance(bands, str) else list(bands)


# ---------------------------------------------------------------------------------------------------
# augmentation (whole tensor -> all bands transformed identically)
# ---------------------------------------------------------------------------------------------------
def geometric_augment(x: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    """Random h/v flips and 90-degree rotations, drawn per sample, applied to all bands of a sample."""
    b = x.shape[0]
    r = torch.rand(b, 3, generator=gen)
    hf, vf, rot = r[:, 0] < 0.5, r[:, 1] < 0.5, r[:, 2] < 0.5
    x = torch.where(hf.view(-1, 1, 1, 1), x.flip(-1), x)
    x = torch.where(vf.view(-1, 1, 1, 1), x.flip(-2), x)
    x = torch.where(rot.view(-1, 1, 1, 1), x.transpose(-1, -2), x)  # transpose + flips = all 8 dihedral ops
    return x


def radiometric_view(x_raw: torch.Tensor, gain_sigma: float = 0.03, band_jitter: float = 0.01,
                     noise_sigma: float = 0.01, gen: torch.Generator | None = None) -> torch.Tensor:
    """Physically plausible perturbation of raw DN (decision D-014):
    * per-patch gain shared by all bands (illumination / uncorrected atmosphere), with a small per-band jitter
      (calibration), keeping inter-band ratios almost unchanged;
    * spatially varying noise field shared across bands (correlated), scaled by each band's patch mean.
    No per-band colour jitter."""
    b, c = x_raw.shape[:2]
    gain = 1 + gain_sigma * torch.randn(b, 1, 1, 1, generator=gen)
    gain = gain * (1 + band_jitter * torch.randn(b, c, 1, 1, generator=gen))
    field_ = torch.randn(b, 1, *x_raw.shape[2:], generator=gen)
    scale = x_raw.mean(dim=(2, 3), keepdim=True)
    return (x_raw * gain + noise_sigma * scale * field_).clamp_min(0)


# ---------------------------------------------------------------------------------------------------
@dataclass
class EuroSATSplit:
    """Holds one split of EuroSAT in RAM (uint16, selected bands in requested order)."""

    split_csv: str = "splits/eurosat_grouped_s0.csv"
    norm_yaml: str = "configs/norm_eurosat_grouped_s0.yaml"
    cache: str = "data/processed/eurosat_ms_uint16.npy"
    bands: object = "all12"
    label_set: str = "multiclass"
    with_index_targets: bool = False
    full_bands_for_targets: bool = True
    _cache_arr: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        root = PROJECT_ROOT
        self.df = pd.read_csv(root / self.split_csv)
        self.norm = yaml.safe_load(open(root / self.norm_yaml))
        self.order = self.norm["file_order"]
        self.band_names = resolve_bands(self.bands)
        self.file_idx = [self.order.index(b) for b in self.band_names]
        self.mean = torch.tensor([self.norm["mean_file_order"][i] for i in self.file_idx]).view(1, -1, 1, 1).float()
        self.std = torch.tensor([self.norm["std_file_order"][i] for i in self.file_idx]).view(1, -1, 1, 1).float()
        self.classes = sorted(self.df["class"].unique())
        if self.with_index_targets:
            t = pd.read_csv(root / "data/processed/eurosat_index_targets.csv").set_index("idx")
            cols = ["ndvi_mean", "ndvi_std", "ndmi_mean", "ndmi_std"]
            mu = np.array(self.norm["index_target_mean"]); sd = np.array(self.norm["index_target_std"])
            self.targets = ((t.loc[self.df["idx"], cols].values - mu) / sd).astype(np.float32)

    # labels ---------------------------------------------------------------------------------------
    def labels(self, rows: pd.DataFrame) -> np.ndarray:
        if self.label_set == "multiclass":
            return rows["label"].values.astype(np.int64)
        if self.label_set == "agri_binary":
            return rows["class"].isin(AGRI).values.astype(np.int64)
        if self.label_set == "agri_pasture_binary":
            return rows["class"].isin(AGRI | {"Pasture"}).values.astype(np.int64)
        raise ValueError(self.label_set)

    def indices(self, part: str, lowdata: int | None = None) -> np.ndarray:
        m = self.df["split"] == part
        if lowdata and part == "train":
            m &= self.df[f"lowdata_{lowdata}"]
        return np.nonzero(m.values)[0]

    def load(self, part: str, lowdata: int | None = None) -> "ArraySet":
        rows_i = self.indices(part, lowdata)
        rows = self.df.iloc[rows_i]
        if self._cache_arr is None:
            self._cache_arr = np.load(PROJECT_ROOT / self.cache, mmap_mode="r")
        cache_idx = rows["idx"].values
        order = np.argsort(cache_idx)  # sorted reads from the memmap are much faster
        x = np.empty((len(rows), len(self.file_idx), 64, 64), np.uint16)
        sel = self._cache_arr[cache_idx[order]]
        x[order] = sel[:, self.file_idx]
        t = self.targets[rows_i] if self.with_index_targets else None
        return ArraySet(x, self.labels(rows), rows["path"].values, t, self.mean, self.std)


@dataclass
class ArraySet:
    x: np.ndarray          # (N, C, 64, 64) uint16 raw DN
    y: np.ndarray
    paths: np.ndarray
    targets: np.ndarray | None
    mean: torch.Tensor
    std: torch.Tensor

    def __len__(self):
        return len(self.y)

    def normalise(self, x_raw: torch.Tensor) -> torch.Tensor:
        return (x_raw - self.mean) / self.std

    def batches(self, batch_size: int, shuffle: bool = False, augment: bool = False, views: int = 1,
                seed: int = 0, view_kwargs: dict | None = None):
        """Yield dicts: x (normalised, B,C,H,W), y, [t], and for views=2 also x2 (second radiometric view
        sharing the same geometry)."""
        n = len(self)
        gen = torch.Generator().manual_seed(seed)
        order = torch.randperm(n, generator=gen).numpy() if shuffle else np.arange(n)
        for s in range(0, n, batch_size):
            ib = np.sort(order[s:s + batch_size])
            xr = torch.from_numpy(self.x[ib].astype(np.float32))
            if augment:
                xr = geometric_augment(xr, gen)
            out = {"y": torch.from_numpy(self.y[ib]), "idx": torch.from_numpy(ib)}
            if views == 2:
                out["x"] = self.normalise(radiometric_view(xr, gen=gen, **(view_kwargs or {})))
                out["x2"] = self.normalise(radiometric_view(xr, gen=gen, **(view_kwargs or {})))
            else:
                out["x"] = self.normalise(xr)
            if self.targets is not None:
                out["t"] = torch.from_numpy(self.targets[ib])
            yield out
