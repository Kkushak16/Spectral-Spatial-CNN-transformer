"""Free accuracy gains on BigEarthNet from saved predictions (CPU only, no retraining).

1. Seed ensemble: average the predicted probabilities of the three seeds.
2. Per-class thresholds: for each class, pick the threshold that maximises F1 on the
   validation split, then apply it unchanged to the test split (the default is 0.5).

mAP does not depend on thresholds, so step 2 changes only F1. Test labels are never
used to choose anything.

    python scripts/ben_postprocess.py            # writes results/runs/ben_postprocess.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

RUNS = Path(__file__).resolve().parents[1] / "results" / "runs"
GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
MODELS = {
    "RGB (3 bands)": ["ben_b8_rgb/seed_0", "ben_b8_rgb/seed_1", "ben_b8_rgb/seed_2"],
    "12 bands": ["ben_b8_cnntr_12band_s0", "ben_b8_cnntr_12band_s1_s2/seed_1", "ben_b8_cnntr_12band_s1_s2/seed_2"],
    "Learned 6 bands (2 seeds cut short)": ["ben_s5_gate_k6/seed_0", "ben_s5_gate_k6/seed_1", "ben_s5_gate_k6/seed_2"],
}


def load(run: str, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(RUNS / run / f"preds_{split}.npz")
    return d["patch_id"], d["y"].astype(np.uint8), d["p"].astype(np.float32)


def scores(y: np.ndarray, p: np.ndarray, thr: np.ndarray | float = 0.5) -> dict:
    present = y.any(0)
    ap = [average_precision_score(y[:, c], p[:, c]) for c in np.flatnonzero(present)]
    hard = (p >= thr).astype(np.uint8)
    return {"macro_mAP": float(np.mean(ap)), "micro_mAP": float(average_precision_score(y.ravel(), p.ravel())),
            "macro_f1": float(f1_score(y[:, present], hard[:, present], average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y, hard, average="micro", zero_division=0))}


def best_thresholds(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    thr = np.full(y.shape[1], 0.5, np.float32)
    for c in range(y.shape[1]):
        if not y[:, c].any():
            continue
        pos = y[:, c] == 1
        f1 = []
        for t in GRID:
            hit = p[:, c] >= t
            tp = np.sum(hit & pos)
            f1.append(2 * tp / max(hit.sum() + pos.sum(), 1))
        thr[c] = GRID[int(np.argmax(f1))]
    return thr


def main() -> None:
    rows = []
    for model, runs in MODELS.items():
        val, test = [load(r, "val") for r in runs], [load(r, "test") for r in runs]
        for split in (val, test):  # all seeds must score the same patches in the same order
            assert all(np.array_equal(split[0][0], s[0]) and np.array_equal(split[0][1], s[1]) for s in split)
        yv, yt = val[0][1], test[0][1]
        variants = [(f"seed {i}", v[2], t[2]) for i, (v, t) in enumerate(zip(val, test))]
        variants.append(("3-seed ensemble", np.mean([v[2] for v in val], 0), np.mean([t[2] for t in test], 0)))
        for name, pv, pt in variants:
            thr = best_thresholds(yv, pv)
            base, tuned = scores(yt, pt), scores(yt, pt, thr)
            rows.append({"model": model, "variant": name, **{f"test_{k}": round(v, 4) for k, v in base.items()},
                         "test_macro_f1_tuned": round(tuned["macro_f1"], 4), "test_micro_f1_tuned": round(tuned["micro_f1"], 4)})
            print(f"{model:36s} {name:16s} " + "  ".join(f"{k} {v:.4f}" for k, v in rows[-1].items() if k.startswith("test_")))
    out = RUNS / "ben_postprocess.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
