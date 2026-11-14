"""Train one configuration.

  python scripts/train.py configs/runs/b3_12band.yaml --set seed=1 train.epochs=5 dataset.train_fraction=0.1
"""

from __future__ import annotations

import argparse

import yaml

from agrispec.train.trainer import train
from agrispec.utils.config import load_config


def parse_sets(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = yaml.safe_load(v)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*")
    a = ap.parse_args()
    train(load_config(a.config, parse_sets(a.set)))
