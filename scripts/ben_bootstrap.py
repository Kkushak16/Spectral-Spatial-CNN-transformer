"""Repeated test evaluations for BigEarthNet without retraining (CPU only, from saved predictions).

1. Scene bootstrap: the test split has 119,825 patches from 100 Sentinel-2 scenes (product + tile).
   Neighbouring patches are correlated, so we resample whole scenes with replacement, 1,000 times,
   and recompute macro mAP each time. This gives a mean and a 95 % interval for every model.
2. Paired comparisons: both models are scored on the same resampled scenes, so the interval of the
   difference shows whether a gap is larger than test-sample noise.
3. Per-country scores: macro mAP separately for each country with at least 2,000 test patches.

The bootstrap measures uncertainty from the choice of test images only. Seed-to-seed variation,
from retraining, is reported separately as the std over seeds.

    python scripts/ben_bootstrap.py      # writes results/runs/ben_bootstrap.csv, ben_bootstrap_pairs.csv,
                                         # ben_by_country.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

RUNS = Path(__file__).resolve().parents[1] / "results" / "runs"
META = Path("D:/ben_support/metadata.parquet")
B, CHUNK, SEED = 1000, 50, 0
MIN_COUNTRY, MIN_POS = 2000, 10
MODELS = {
    "RGB": [f"ben_b8_rgb/seed_{s}" for s in range(3)],
    "Gate k=3": ["ben_s5_gate_k3/seed_0"],
    "RGB+NIR": [f"ben_b8_rgbnir/seed_{s}" for s in range(3)],
    "Gate k=4": ["ben_s5_gate_k4/seed_0"],
    "Gate k=6": ["ben_s5_gate_k6/seed_0", "ben_s5_gate_k6_rerun/seed_1", "ben_s5_gate_k6_rerun/seed_2"],
    "Fixed 6": [f"ben_b8_fixed6/seed_{s}" for s in range(3)],
    "12 bands": ["ben_b8_cnntr_12band_s0", "ben_b8_cnntr_12band_s1_s2/seed_1", "ben_b8_cnntr_12band_s1_s2/seed_2"],
}
PAIRS = [("Gate k=3", "RGB"), ("Gate k=4", "RGB+NIR"), ("Gate k=4", "12 bands"), ("Gate k=6", "Fixed 6"),
         ("Fixed 6", "RGB+NIR"), ("12 bands", "RGB")]


def weighted_ap(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Average precision of one class under many weightings at once (w: R x N), with sklearn's
    handling of tied scores: precision is taken at the end of each run of equal scores."""
    order = np.argsort(-s, kind="stable")
    y, s, w = y[order], s[order], w[:, order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # last index of each tied run
    tp = np.cumsum(w * y, 1)[:, ends]
    fp = np.cumsum(w * (1 - y), 1)[:, ends]
    gain = np.diff(tp, axis=1, prepend=0.0)
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
    total = tp[:, -1]
    return np.divide((gain * prec).sum(1), total, out=np.full(len(total), np.nan), where=total > 0)


def macro_map(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> np.ndarray:
    aps = np.stack([weighted_ap(y[:, c], p[:, c], w) for c in range(y.shape[1]) if y[:, c].any()], 1)
    return np.nanmean(aps, 1)


def main() -> None:
    first = np.load(RUNS / MODELS["RGB"][0] / "preds_test.npz")
    ids, y = first["patch_id"].astype(str), first["y"].astype(np.float64)
    meta = pd.read_parquet(META, columns=["patch_id", "country"]).set_index("patch_id").loc[ids]
    scene = pd.Series(ids).str.rsplit("_", n=2).str[0]
    codes, names = pd.factorize(scene)
    print(f"{len(ids):,} test patches from {len(names)} scenes")

    preds = {}
    for model, runs in MODELS.items():
        ps = []
        for r in runs:
            d = np.load(RUNS / r / "preds_test.npz")
            assert np.array_equal(d["patch_id"].astype(str), ids) and np.array_equal(d["y"], first["y"]), r
            ps.append(d["p"].astype(np.float64))
        preds[model] = ps

    # check the fast AP against scikit-learn on the unweighted test set
    p0 = preds["RGB"][0]
    ref = np.mean([average_precision_score(y[:, c], p0[:, c]) for c in range(y.shape[1]) if y[:, c].any()])
    mine = macro_map(y, p0, np.ones((1, len(y))))[0]
    assert abs(ref - mine) < 1e-9, (ref, mine)
    print(f"fast AP matches scikit-learn: {mine:.6f}")

    rng = np.random.default_rng(SEED)
    draws = np.stack([np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names)) for _ in range(B)])
    boot = {m: np.zeros((len(ps), B)) for m, ps in preds.items()}  # model -> seeds x resamples
    for lo in range(0, B, CHUNK):
        w = draws[lo:lo + CHUNK][:, codes].astype(np.float64)
        for m, ps in preds.items():
            for i, p in enumerate(ps):
                boot[m][i, lo:lo + CHUNK] = macro_map(y, p, w)
        print(f"  resamples {min(lo + CHUNK, B)}/{B}", flush=True)

    rows = []
    for m, runs in MODELS.items():
        full = [macro_map(y, p, np.ones((1, len(y))))[0] for p in preds[m]]
        for i, r in enumerate(runs):
            b = boot[m][i]
            rows.append({"model": m, "run": r, "test_macro_mAP": round(full[i], 4), "boot_mean": round(b.mean(), 4),
                         "boot_std": round(b.std(ddof=1), 4), "ci_low": round(np.percentile(b, 2.5), 4),
                         "ci_high": round(np.percentile(b, 97.5), 4)})
        b = boot[m].mean(0)  # mean over seeds, resample by resample
        rows.append({"model": m, "run": f"mean of {len(runs)} seed(s)", "test_macro_mAP": round(np.mean(full), 4),
                     "boot_mean": round(b.mean(), 4), "boot_std": round(b.std(ddof=1), 4),
                     "ci_low": round(np.percentile(b, 2.5), 4), "ci_high": round(np.percentile(b, 97.5), 4)})
        print(f"{m:10s} {np.mean(full):.4f}  95% interval {rows[-1]['ci_low']:.4f}-{rows[-1]['ci_high']:.4f}")
    write(RUNS / "ben_bootstrap.csv", rows)

    pair_rows = []
    for a, b_ in PAIRS:
        d = boot[a].mean(0) - boot[b_].mean(0)
        pair_rows.append({"model_a": a, "model_b": b_, "seeds_a": len(MODELS[a]), "seeds_b": len(MODELS[b_]),
                          "diff_mean": round(d.mean(), 4), "ci_low": round(np.percentile(d, 2.5), 4),
                          "ci_high": round(np.percentile(d, 97.5), 4), "share_a_better": round((d > 0).mean(), 3)})
        print(f"{a} - {b_}: {d.mean():+.4f}  95% interval {pair_rows[-1]['ci_low']:+.4f} to "
              f"{pair_rows[-1]['ci_high']:+.4f}  A better in {pair_rows[-1]['share_a_better']:.1%}")
    write(RUNS / "ben_bootstrap_pairs.csv", pair_rows)

    country = meta["country"].to_numpy()
    keep = [c for c, n in pd.Series(country).value_counts().items() if n >= MIN_COUNTRY]
    c_rows = []
    for m, ps in preds.items():
        row = {"model": m}
        for c in keep:
            sel = country == c
            ys = y[sel]
            cls = ys.sum(0) >= MIN_POS
            row[c] = round(np.mean([np.mean([average_precision_score(ys[:, j], p[sel][:, j]) for j in np.flatnonzero(cls)])
                                    for p in ps]), 4)
        vals = [row[c] for c in keep]
        row["mean_over_countries"], row["std_over_countries"] = round(np.mean(vals), 4), round(np.std(vals, ddof=1), 4)
        c_rows.append(row)
        print(f"{m:10s} by country: " + "  ".join(f"{c[:4]} {row[c]:.3f}" for c in keep))
    write(RUNS / "ben_by_country.csv", c_rows)


def write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
