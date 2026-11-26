"""Per-seed summary CSVs for BigEarthNet runs (overall metrics and per-class AP).

    python scripts/ben_summary.py

Each row holds one metric for every seed plus the mean and the sample std (ddof = 1).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "results" / "runs"
OVERALL = ["macro_mAP", "micro_mAP", "macro_f1", "micro_f1"]
SUMMARIES = {
    # seed 0 of the first k = 6 run was complete; seeds 1 and 2 are the full-length reruns
    "ben_s5_gate_k6_summary_3seeds.csv": ["ben_s5_gate_k6/seed_0", "ben_s5_gate_k6_rerun/seed_1", "ben_s5_gate_k6_rerun/seed_2"],
    "ben_b8_fixed6_summary_3seeds.csv": [f"ben_b8_fixed6/seed_{s}" for s in range(3)],
    "ben_b8_rgbnir_summary_3seeds.csv": [f"ben_b8_rgbnir/seed_{s}" for s in range(3)],
}


def summarise(out: str, runs: list[str]) -> None:
    ms = [json.loads((RUNS / r / "metrics.json").read_text()) for r in runs]
    rows = []
    for split in ("val", "test"):
        items = [("overall", k, [m[split][k] for m in ms]) for k in OVERALL]
        classes = sorted(ms[0][split]["per_class_AP"])
        items += [("class_AP", c, [m[split]["per_class_AP"][c] for m in ms]) for c in classes]
        for kind, name, vals in items:
            rows.append([split, kind, name, *vals, float(np.mean(vals)), float(np.std(vals, ddof=1))])
    with open(RUNS / out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "kind", "name", *[f"seed{i}" for i in range(len(ms))], "mean", "std"])
        w.writerows(rows)
    test = {r[2]: r for r in rows if r[0] == "test" and r[1] == "overall"}
    print(out, "  ".join(f"{k} {test[k][-2]:.4f}±{test[k][-1]:.4f}" for k in OVERALL))


if __name__ == "__main__":
    for out, runs in SUMMARIES.items():
        summarise(out, runs)
