"""Repeated test passes of saved BigEarthNet models (CPU, no retraining).

Each model scores the test split several times:
  d0..d7   the 8 flips and 90-degree rotations of every patch (labels do not change). The model was
           trained with these, so each pass is an equally valid test; we report the mean and std of the
           8 scores, and the score of their averaged prediction (test-time augmentation, "TTA").
  gain_0.9, gain_1.1, noise_0.03
           radiometric corruptions for the robustness test of hypothesis H4: every band of a patch scaled
           by 0.9 or 1.1, or Gaussian noise with std 3 % of each band's patch mean added to each pixel.

Nothing is chosen on the test split: every pass is reported, never the best one.

    python scripts/ben_tta.py                       # all models in MODELS, full test split
    python scripts/ben_tta.py --shards 6 --runs ben_b8_rgb/seed_0     # quick check on 6 test shards
Writes <run>/tta_test.json for each run and results/runs/ben_tta.csv.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("ben_job", ROOT / "kaggle_jobs" / "ben_s7_cons_k4" / "train_ben.py")
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)  # data loading, band order and metrics exactly as in training
from agrispec.models.factory import build_model  # noqa: E402

MODELS = {
    "RGB": [f"ben_b8_rgb/seed_{s}" for s in range(3)],
    "Gate k=3": ["ben_s5_gate_k3/seed_0"],
    "RGB+NIR": [f"ben_b8_rgbnir/seed_{s}" for s in range(3)],
    "Gate k=4": ["ben_s5_gate_k4/seed_0"],
    "Gate k=6": ["ben_s5_gate_k6/seed_0", "ben_s5_gate_k6_rerun/seed_1", "ben_s5_gate_k6_rerun/seed_2"],
    "Fixed 6": [f"ben_b8_fixed6/seed_{s}" for s in range(3)],
    "12 bands": ["ben_b8_cnntr_12band_s0", "ben_b8_cnntr_12band_s1_s2/seed_1", "ben_b8_cnntr_12band_s1_s2/seed_2"],
}
DIHEDRAL = [f"d{i}" for i in range(8)]
CORRUPT = ["gain_0.9", "gain_1.1", "noise_0.03"]


def variant(raw: torch.Tensor, name: str, seed: int) -> torch.Tensor:
    if name.startswith("d"):
        i = int(name[1:])
        x = raw.rot90(i % 4, (-2, -1))
        return x.flip(-1) if i >= 4 else x
    if name.startswith("gain_"):
        return raw * float(name[5:])
    g = torch.Generator().manual_seed(seed)  # same noise for every model
    sigma = float(name[6:])
    return (raw + sigma * raw.mean((2, 3), keepdim=True) * torch.randn(raw.shape, generator=g)).clamp_min(0)


def load_model(run: str):
    cfg = json.loads((RUNS / run / "config.json").read_text())
    mcfg, bands = cfg["model"], cfg["bands"]
    gated = "gate" in mcfg
    model = build_model(mcfg, 12 if gated else len(bands), len(cfg["classes"]))
    if gated:
        model.gate.frozen_mask = torch.zeros(12)  # buffer size must match before loading
    model.load_state_dict(torch.load(RUNS / run / "best.pt", map_location="cpu", weights_only=False)["model"])
    norm = json.loads((RUNS / run / "norm.json").read_text())
    keep = None if gated else torch.tensor([job.BANDS.index(b) for b in bands])
    mean = torch.tensor(norm["mean"], dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor(norm["std"], dtype=torch.float32).view(1, -1, 1, 1)
    return model.eval(), keep, mean, std, cfg["classes"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[r for rs in MODELS.values() for r in rs])
    ap.add_argument("--shards", type=int, default=0, help="test shards to use, evenly spaced; 0 = all")
    ap.add_argument("--data", default="D:/ben_shards")
    ap.add_argument("--support", default="D:/ben_support")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--force", action="store_true", help="recompute runs that already have tta_test.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    index = pd.read_csv(job.find_one(a.data, "index.csv"))
    meta = pd.read_parquet(job.find_one(a.support, "metadata.parquet"), columns=["patch_id", "labels", "split"])
    frame = index.merge(meta, on="patch_id", how="left", validate="one_to_one", suffixes=("", "_meta"))
    classes = sorted({c for labels in meta["labels"] for c in labels})
    te = job.Split(Path(a.data), frame[frame["split"] == "test"], classes, a.shards)
    full = a.shards == 0
    print(f"test patches: {te.n:,} ({len(te.shards)} shards)", flush=True)
    names = DIHEDRAL + CORRUPT

    for run in a.runs:
        out = RUNS / run / ("tta_test.json" if full else f"tta_test_{a.shards}shards.json")
        if out.exists() and not a.force:
            print(f"{run}: already done", flush=True)
            continue
        model, keep, mean, std, cls = load_model(run)
        assert cls == classes
        P = {v: np.zeros((te.n, len(classes)), np.float32) for v in names}
        Y = np.zeros((te.n, len(classes)), np.uint8)
        t0 = time.time()
        with torch.inference_mode():
            for b in job.loader(job.ShardStream(te, a.batch, False), 0):
                raw = job.compose_raw(b, "cpu")
                i = b["idx"].numpy()
                Y[i] = b["y"].numpy().astype(np.uint8)
                for v in names:
                    x = (variant(raw, v, int(i[0])) - mean) / std
                    x = x if keep is None else x[:, keep]
                    P[v][i] = torch.sigmoid(model(x)).numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = {v: job.metrics(Y, P[v], classes) for v in names}
            tta = job.metrics(Y, np.mean([P[v] for v in DIHEDRAL], 0), classes)
        d = [res[v]["macro_mAP"] for v in DIHEDRAL]
        summary = {"run": run, "patches": te.n, "minutes": round((time.time() - t0) / 60, 1),
                   "d0_macro_mAP": round(d[0], 4), "dihedral_mean": round(float(np.mean(d)), 4),
                   "dihedral_std": round(float(np.std(d, ddof=1)), 4), "tta_macro_mAP": round(tta["macro_mAP"], 4),
                   "tta_micro_mAP": round(tta["micro_mAP"], 4), "tta_macro_f1": round(tta["macro_f1"], 4),
                   **{f"{v}_macro_mAP": round(res[v]["macro_mAP"], 4) for v in CORRUPT}}
        if full:  # pass d0 must reproduce the predictions saved during training (stored as float16)
            saved = np.load(RUNS / run / "preds_test.npz")
            gap = float(np.abs(saved["p"].astype(np.float32) - P["d0"]).max())
            assert np.array_equal(saved["y"], Y) and gap < 5e-3, (run, gap)
            summary["d0_vs_saved_max_gap"] = round(gap, 5)
        out.write_text(json.dumps({**summary, "per_variant": {v: {k: r[k] for k in ("macro_mAP", "micro_mAP", "macro_f1", "micro_f1")}
                                                              for v, r in res.items()}, "tta": tta}, indent=2))
        print(f"{run}: " + "  ".join(f"{k}={v}" for k, v in summary.items() if k != "run"), flush=True)

    rows = []
    for model_name, runs in MODELS.items():
        for run in runs:
            f = RUNS / run / ("tta_test.json" if full else f"tta_test_{a.shards}shards.json")
            if f.exists():
                s = json.loads(f.read_text())
                rows.append({"model": model_name, **{k: v for k, v in s.items() if k not in ("per_variant", "tta")}})
    if rows and full:
        with open(RUNS / "ben_tta.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
        print("wrote", RUNS / "ben_tta.csv")


if __name__ == "__main__":
    main()
