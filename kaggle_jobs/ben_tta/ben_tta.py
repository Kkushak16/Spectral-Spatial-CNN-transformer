"""Repeated test passes of saved BigEarthNet models (no retraining). Runs locally or as a Kaggle GPU script.

Each model scores the test split several times:
  d0..d7   the 8 flips and 90-degree rotations of every patch (labels do not change). The models were
           trained with these, so each pass is an equally valid test; we report the mean and std of the
           8 scores, and the score of their averaged prediction (test-time augmentation, "TTA").
  gain_0.9, gain_1.1, noise_0.03
           radiometric corruptions for the robustness test of hypothesis H4: every band of a patch scaled
           by 0.9 or 1.1, or Gaussian noise with std 3 % of each band's patch mean added to each pixel
           (the same noise for every model).

Nothing is chosen on the test split: every pass is reported, never the best one. The test data are read
once; every model scores each batch in turn.

Local:   python scripts/ben_tta.py [--shards 6] [--runs ben_b8_rgb/seed_0 ...]
Kaggle:  kaggle_jobs/ben_tta (weights dataset: <run with / replaced by __>__best.pt, __config.json,
         __norm.json, __preds_test.npz, plus train_ben_job.py)
Writes ben_tta.csv and ben_tta_full.json (local: results/runs/, Kaggle: /kaggle/working/).
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

KAGGLE = os.path.isdir("/kaggle/input")
ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() and not KAGGLE else None
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
NAMES = DIHEDRAL + CORRUPT


def variant(raw: torch.Tensor, name: str, seed: int) -> torch.Tensor:
    if name.startswith("d"):
        i = int(name[1:])
        x = raw.rot90(i % 4, (-2, -1))
        return x.flip(-1) if i >= 4 else x
    if name.startswith("gain_"):
        return raw * float(name[5:])
    g = torch.Generator().manual_seed(seed)  # same noise for every model
    noise = torch.randn(raw.shape, generator=g).to(raw.device)
    return (raw + float(name[6:]) * raw.mean((2, 3), keepdim=True) * noise).clamp_min(0)


class Files:
    """Run files either in results/runs/<run>/ or flat in one folder as <run with / -> __>__<file>."""

    def __init__(self, flat: str | None):
        self.flat = Path(flat) if flat else None

    def __call__(self, run: str, name: str) -> Path:
        if self.flat:
            return self.flat / f"{run.replace('/', '__')}__{name}"
        return ROOT / "results" / "runs" / run / name


def load_model(build_model, files: Files, run: str, bands_all: list[str], device):
    cfg = json.loads(files(run, "config.json").read_text())
    mcfg, bands = cfg["model"], cfg["bands"]
    gated = "gate" in mcfg
    model = build_model(mcfg, 12 if gated else len(bands), len(cfg["classes"]))
    if gated:
        model.gate.frozen_mask = torch.zeros(12)  # buffer size must match before loading
    model.load_state_dict(torch.load(files(run, "best.pt"), map_location="cpu", weights_only=False)["model"])
    norm = json.loads(files(run, "norm.json").read_text())
    keep = None if gated else torch.tensor([bands_all.index(b) for b in bands], device=device)
    mean = torch.tensor(norm["mean"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std = torch.tensor(norm["std"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    return {"model": model.to(device).eval(), "keep": keep, "mean": mean, "std": std, "classes": cfg["classes"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[r for rs in MODELS.values() for r in rs])
    ap.add_argument("--shards", type=int, default=0, help="test shards to use, evenly spaced; 0 = all")
    ap.add_argument("--data", default="/kaggle/input" if KAGGLE else "D:/ben_shards")
    ap.add_argument("--support", default="/kaggle/input" if KAGGLE else "D:/ben_support")
    ap.add_argument("--weights", default="/kaggle/input" if KAGGLE else None, help="flat weights folder (Kaggle)")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--threads", type=int, default=14)
    a, _ = ap.parse_known_args()
    torch.set_num_threads(a.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    if KAGGLE:
        job_file = Path(glob.glob("/kaggle/input/**/train_ben_job.py", recursive=True)[0])
        weights = str(job_file.parent)
        out_dir = Path("/kaggle/working")
    else:
        job_file = ROOT / "kaggle_jobs" / "ben_s7_cons_k4" / "train_ben.py"
        weights = a.weights
        out_dir = ROOT / "results" / "runs"
    spec = importlib.util.spec_from_file_location("ben_job", job_file)
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)  # data loading, band order and metrics exactly as in training
    meta_path = job.find_one(a.support, "metadata.parquet")
    build_model = job.import_build_model(meta_path.parent)
    files = Files(weights)

    index = pd.read_csv(job.find_one(a.data, "index.csv"))
    meta = pd.read_parquet(meta_path, columns=["patch_id", "labels", "split"])
    frame = index.merge(meta, on="patch_id", how="left", validate="one_to_one", suffixes=("", "_meta"))
    classes = sorted({c for labels in meta["labels"] for c in labels})
    te = job.Split(job.find_one(a.data, "index.csv").parent, frame[frame["split"] == "test"], classes, a.shards)
    print(f"device={device} test patches: {te.n:,} ({len(te.shards)} shards), models: {len(a.runs)}", flush=True)

    models = {r: load_model(build_model, files, r, job.BANDS, device) for r in a.runs}
    assert all(m["classes"] == classes for m in models.values())
    P = {r: {v: np.zeros((te.n, len(classes)), np.float16) for v in NAMES} for r in a.runs}
    Y = np.zeros((te.n, len(classes)), np.uint8)
    t0, done = time.time(), 0
    with torch.inference_mode():
        for b in job.loader(job.ShardStream(te, a.batch, False), 0):
            raw = job.compose_raw(b, device)
            i = b["idx"].numpy()
            Y[i] = b["y"].numpy().astype(np.uint8)
            for v in NAMES:
                xv = variant(raw, v, int(i[0]))
                for r, m in models.items():
                    x = (xv - m["mean"]) / m["std"]
                    with torch.autocast("cuda", enabled=amp):
                        logits = m["model"](x if m["keep"] is None else x[:, m["keep"]])
                    P[r][v][i] = torch.sigmoid(logits.float()).cpu().numpy()
            done += len(i)
            if done // a.batch % 20 == 0:
                print(f"  {done:,}/{te.n:,} patches, {(time.time() - t0) / 60:.1f} min", flush=True)

    rows, full_out = [], {}
    name_of = {r: n for n, rs in MODELS.items() for r in rs}
    for r in a.runs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = {v: job.metrics(Y, P[r][v].astype(np.float32), classes) for v in NAMES}
            tta = job.metrics(Y, np.mean([P[r][v].astype(np.float32) for v in DIHEDRAL], 0), classes)
        d = [res[v]["macro_mAP"] for v in DIHEDRAL]
        row = {"model": name_of.get(r, r), "run": r, "patches": te.n,
               "d0_macro_mAP": round(d[0], 4), "dihedral_mean": round(float(np.mean(d)), 4),
               "dihedral_std": round(float(np.std(d, ddof=1)), 4), "tta_macro_mAP": round(tta["macro_mAP"], 4),
               "tta_micro_mAP": round(tta["micro_mAP"], 4), "tta_macro_f1": round(tta["macro_f1"], 4),
               "tta_micro_f1": round(tta["micro_f1"], 4),
               **{f"{v}_macro_mAP": round(res[v]["macro_mAP"], 4) for v in CORRUPT}}
        saved = files(r, "preds_test.npz")
        if a.shards == 0 and saved.exists():  # pass d0 must reproduce the predictions saved in training
            s = np.load(saved)
            gap = float(np.abs(s["p"].astype(np.float32) - P[r]["d0"].astype(np.float32)).max())
            row["d0_vs_saved_max_gap"] = round(gap, 5)
            if not np.array_equal(s["y"], Y) or gap > 5e-3:
                print(f"WARNING {r}: d0 differs from saved predictions (max gap {gap})", flush=True)
        rows.append(row)
        full_out[r] = {"per_variant": {v: {k: x[k] for k in ("macro_mAP", "micro_mAP", "macro_f1", "micro_f1")}
                                       for v, x in res.items()}, "tta": tta}
        print("  ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    tag = "" if a.shards == 0 else f"_{a.shards}shards"
    with open(out_dir / f"ben_tta{tag}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    (out_dir / f"ben_tta{tag}_full.json").write_text(json.dumps(full_out, indent=2))
    print(f"wrote {out_dir / f'ben_tta{tag}.csv'}  total {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
