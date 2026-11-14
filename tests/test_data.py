"""Phase 5 gate: band-aligned augmentation, no leakage between splits, correct band mapping."""

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from agrispec import PROJECT_ROOT
from agrispec.data.eurosat import EuroSATSplit, geometric_augment, radiometric_view
from agrispec.data.indices import index_targets_batch

P = PROJECT_ROOT
SPLITS = ["random", "grouped", "geo"]


def test_flip_changes_all_bands_identically():
    torch.manual_seed(0)
    x = torch.randn(64, 1, 16, 16).repeat(1, 12, 1, 1)  # every band identical
    x = x * torch.arange(1, 13).view(1, 12, 1, 1)         # bands differ only by a constant factor
    xa = geometric_augment(x.clone(), torch.Generator().manual_seed(1))
    ratio = xa / torch.arange(1, 13).view(1, 12, 1, 1)
    assert torch.allclose(ratio, ratio[:, :1].expand_as(ratio)), "bands were transformed differently"
    assert not torch.equal(xa, x), "augmentation did nothing"


def test_radiometric_view_keeps_band_ratios_close():
    x = torch.rand(8, 12, 64, 64) * 3000 + 500
    v = radiometric_view(x, gen=torch.Generator().manual_seed(0))
    r0 = x[:, 7] / x[:, 3]
    r1 = v[:, 7] / v[:, 3]
    assert ((r1 / r0 - 1).abs().median() < 0.05)


@pytest.mark.parametrize("name", SPLITS)
def test_no_overlap_between_splits(name):
    df = pd.read_csv(P / f"splits/eurosat_{name}_s0.csv")
    parts = {s: set(df.loc[df.split == s, "path"]) for s in ("train", "val", "test")}
    assert not parts["train"] & parts["test"]
    assert not parts["train"] & parts["val"]
    assert not parts["val"] & parts["test"]
    assert sum(len(v) for v in parts.values()) == 27000
    for frac in (10, 20, 50):
        assert df.loc[df[f"lowdata_{frac}"], "split"].eq("train").all()


@pytest.mark.parametrize("name", ["grouped", "geo"])
def test_leakage_groups_stay_in_one_split(name):
    df = pd.read_csv(P / f"splits/eurosat_{name}_s0.csv")
    assert (df.groupby("group")["split"].nunique() == 1).all()


def test_band_mapping_and_physics():
    ds = EuroSATSplit(bands=["B04", "B08", "B8A", "B10"])
    a = ds.load("val")
    raw = np.load(P / "data/processed/eurosat_ms_uint16.npy", mmap_mode="r")
    i = int(ds.df.iloc[ds.indices("val")[0]]["idx"])
    order = yaml.safe_load(open(P / "configs/band_map_eurosat.yaml"))["file_order"]
    assert np.array_equal(a.x[0, 1], raw[i, order.index("B08")])
    assert np.array_equal(a.x[0, 2], raw[i, 12])  # B8A is the last channel in the files
    assert a.x[:, 3].mean() < 50                   # cirrus near zero


def test_normalisation_fitted_on_train_only():
    norm = yaml.safe_load(open(P / "configs/norm_eurosat_grouped_s0.yaml"))
    assert "train" in norm["fitted_on"]
    ds = EuroSATSplit()
    tr = ds.load("train")
    sample = np.random.default_rng(0).choice(len(tr), 3000, replace=False)
    x = tr.normalise(torch.from_numpy(tr.x[np.sort(sample)].astype(np.float32)))
    assert x.mean(dim=(0, 2, 3)).abs().max() < 0.2   # every band roughly centred
    assert x.mean().abs() < 0.2 and (x.std() - 1).abs() < 0.3


def test_index_targets_vegetation_vs_water():
    order = yaml.safe_load(open(P / "configs/band_map_eurosat.yaml"))["file_order"]
    raw = np.load(P / "data/processed/eurosat_ms_uint16.npy", mmap_mode="r")
    df = pd.read_csv(P / "data/processed/eurosat_meta.csv")
    forest = index_targets_batch(raw[df.index[df["class"] == "Forest"][:50]], order)
    water = index_targets_batch(raw[df.index[df["class"] == "SeaLake"][:50]], order)
    assert forest[:, 0].mean() > 0.5 and water[:, 0].mean() < 0.1
