"""Run configuration: one YAML per run, seeds, run directories."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from agrispec import PROJECT_ROOT


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Load a YAML config. A config may name a parent via `inherit: <file>`."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = deep_update(load_config(path.parent / parent), cfg)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def file_hash(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_run_dir(cfg: dict, root: str | Path = "results/runs") -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{cfg.get('name', 'run')}_s{cfg.get('seed', 0)}_{config_hash(cfg)}"
    run_dir = PROJECT_ROOT / root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return run_dir


def save_json(obj: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2, default=str)
