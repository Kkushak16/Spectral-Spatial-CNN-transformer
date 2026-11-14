"""Generic trainer: AdamW + cosine, early stopping on validation, test once with the best checkpoint.

Loss (brief): L = L_task + lambda_sparse * L_sparse + lambda_index * L_index + lambda_consistency * L_cons
Terms with weight 0 are skipped. Every run writes: config.yaml, history.csv, metrics.json, best.pt and one
row in results/results.csv.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from agrispec import PROJECT_ROOT
from agrispec.data.eurosat import AGRI, EuroSATSplit
from agrispec.models.factory import build_model
from agrispec.utils.config import config_hash, file_hash, make_run_dir, save_json, seed_everything
from agrispec.utils.profiling import cost_report


class IndexHead(nn.Module):
    """Physics auxiliary head: predicts z-scored [NDVI mean, NDVI std, NDMI mean, NDMI std]."""

    def __init__(self, d: int, out: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(inplace=True), nn.Linear(64, out))

    def forward(self, z):
        return self.net(z)


def _gate(model):
    return getattr(model, "gate", None)


def evaluate(model, data, bs=256, classes=None) -> dict:
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for b in data.batches(bs):
            lo = model(b["x"])
            probs.append(lo.softmax(1)); preds.append(lo.argmax(1))
    p = torch.cat(preds).numpy(); y = data.y
    out = {"acc": float(accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro")), "n": int(len(y))}
    if classes is not None and len(classes) == 10:
        out["per_class_recall"] = dict(zip(classes, np.round(confusion_matrix(y, p, normalize="true").diagonal(), 4)))
        out["confusion"] = confusion_matrix(y, p).tolist()
        agri_ids = [classes.index(c) for c in AGRI]
        yb, pb = np.isin(y, agri_ids), np.isin(p, agri_ids)
        pr, rc, f1, _ = precision_recall_fscore_support(yb, pb, average="binary", zero_division=0)
        out["agri_from_multiclass"] = {"precision": float(pr), "recall": float(rc), "f1": float(f1)}
    out["_pred"] = p
    return out


def bootstrap_ci(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n, len(y)))
    accs = (y[idx] == p[idx]).mean(1)
    return [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]


def train(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    torch.set_num_threads(cfg.get("num_threads", 10))
    d, m, lc, tc = cfg["dataset"], cfg["model"], cfg["loss"], cfg["train"]
    use_index = lc.get("lambda_index", 0) > 0
    use_cons = lc.get("lambda_consistency", 0) > 0
    ds = EuroSATSplit(split_csv=d["split"], norm_yaml=d["norm"], bands=d["bands"], label_set=d["label_set"],
                      with_index_targets=use_index)
    lowdata = None if d.get("train_fraction", 1.0) >= 1 else int(round(d["train_fraction"] * 100))
    tr, va = ds.load("train", lowdata), ds.load("val")
    in_ch = len(ds.band_names)
    ncls = 10 if d["label_set"] == "multiclass" else 2
    model = build_model(m, in_ch, ncls)
    aux = IndexHead(getattr(model, "encoder", model).out_dim if hasattr(getattr(model, "encoder", model), "out_dim")
                    else 128) if use_index else None
    params = list(model.parameters()) + (list(aux.parameters()) if aux else [])
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    steps = tc["epochs"] * int(np.ceil(len(tr) / tc["batch_size"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=steps, pct_start=0.1) \
        if tc.get("scheduler") == "onecycle" else torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    run_dir = make_run_dir(cfg)
    manifest = {"split_hash": file_hash(PROJECT_ROOT / d["split"]), "norm_hash": file_hash(PROJECT_ROOT / d["norm"]),
                "band_map_hash": file_hash(PROJECT_ROOT / "configs/band_map_eurosat.yaml"), "bands": ds.band_names,
                "n_train": len(tr), "n_val": len(va), "config_hash": config_hash(cfg)}
    save_json(manifest, run_dir / "manifest.json")
    print(f"[{cfg['name']}] run_dir={run_dir.name} train={len(tr)} val={len(va)} bands={ds.band_names}", flush=True)

    best, best_ep, hist, bad = -1.0, -1, [], 0
    t_start = time.time()
    for ep in range(tc["epochs"]):
        model.train(); aux and aux.train()
        t0, tot, n = time.time(), {"task": 0.0, "sparse": 0.0, "index": 0.0, "cons": 0.0}, 0
        for b in tr.batches(tc["batch_size"], shuffle=True, augment=bool(tc.get("augment")),
                            views=2 if use_cons else 1, seed=cfg["seed"] * 1000 + ep):
            x, y = b["x"], b["y"]
            if use_index or use_cons:
                z = model.embed(x)
                logits = model.head(z)
            else:
                logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=tc.get("label_smoothing", 0.0))
            tot["task"] += loss.item() * len(y)
            g = _gate(model)
            if g is not None and lc.get("lambda_sparse", 0) > 0:
                ls = g.sparsity_loss(); loss = loss + lc["lambda_sparse"] * ls; tot["sparse"] += ls.item() * len(y)
            if use_index:
                li = F.smooth_l1_loss(aux(z), b["t"]); loss = loss + lc["lambda_index"] * li
                tot["index"] += li.item() * len(y)
            if use_cons:
                lo2 = model(b["x2"])
                lcs = F.kl_div(F.log_softmax(lo2, 1), F.softmax(logits, 1).detach(), reduction="batchmean") + \
                    F.kl_div(F.log_softmax(logits, 1), F.softmax(lo2, 1).detach(), reduction="batchmean")
                loss = loss + lc["lambda_consistency"] * 0.5 * lcs + 0.5 * F.cross_entropy(lo2, y)
                tot["cons"] += lcs.item() * len(y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            n += len(y)
        ev = evaluate(model, va)
        row = {"epoch": ep, "time_s": round(time.time() - t0, 1), **{k: v / n for k, v in tot.items()},
               "val_acc": ev["acc"], "val_f1": ev["macro_f1"], "lr": sched.get_last_lr()[0]}
        g = _gate(model)
        if g is not None:
            row["bands"] = ",".join(ds.band_names[i] for i in g.selected_indices().tolist())
        hist.append(row)
        print(f"  ep{ep:02d} {row['time_s']:5.1f}s task={row['task']:.4f} val_acc={ev['acc']:.4f} "
              f"f1={ev['macro_f1']:.4f}" + (f" bands={row.get('bands')}" if g is not None else ""), flush=True)
        if ev["acc"] > best:
            best, best_ep, bad = ev["acc"], ep, 0
            torch.save({"model": model.state_dict(), "aux": aux.state_dict() if aux else None}, run_dir / "best.pt")
        else:
            bad += 1
            if bad >= tc.get("early_stop_patience", 1000):
                print(f"  early stop at epoch {ep}", flush=True)
                break
    pd.DataFrame(hist).to_csv(run_dir / "history.csv", index=False)
    train_time = time.time() - t_start

    # ---- final evaluation: best checkpoint, val + test (test touched once per run) -----------------
    model.load_state_dict(torch.load(run_dir / "best.pt")["model"])
    res = {"name": cfg["name"], "seed": cfg["seed"], "best_epoch": best_ep, "train_time_s": round(train_time, 1)}
    final_parts = ["val"] + (["test"] if cfg.get("eval_test", True) else [])
    for part in final_parts:
        data = va if part == "val" else ds.load("test")
        ev = evaluate(model, data, classes=ds.classes if ncls == 10 else None)
        p = ev.pop("_pred")
        ev["acc_ci95"] = bootstrap_ci(data.y, p)
        res[part] = ev
    g = _gate(model)
    if g is not None:
        res["selected_bands"] = [ds.band_names[i] for i in g.selected_indices().tolist()]
        res["gate_probabilities"] = dict(zip(ds.band_names, np.round(g.probabilities().numpy(), 4).tolist()))
    cost_model = model
    if g is not None:
        from agrispec.models.gate import export_k_channel
        cost_model = export_k_channel(g, model.encoder)
    res["cost"] = cost_report(cost_model, len(res.get("selected_bands", ds.band_names)))
    res["cost"]["params_train_incl_aux"] = sum(p.numel() for p in params)
    save_json(res, run_dir / "metrics.json")

    row = {"run": run_dir.name, "name": cfg["name"], "seed": cfg["seed"], "split": Path(d["split"]).stem,
           "bands": "+".join(res.get("selected_bands", ds.band_names)), "n_bands": len(res.get("selected_bands", ds.band_names)),
           "label_set": d["label_set"], "train_fraction": d.get("train_fraction", 1.0),
           "val_acc": res["val"]["acc"], "val_f1": res["val"]["macro_f1"],
           "test_acc": res.get("test", {}).get("acc"), "test_f1": res.get("test", {}).get("macro_f1"),
           "params": res["cost"]["params"], "mflops": (res["cost"]["flops"] or 0) / 1e6,
           "latency_ms": round(res["cost"]["latency_ms_median"], 3), "best_epoch": best_ep,
           "train_time_s": res["train_time_s"]}
    csv = PROJECT_ROOT / "results/results.csv"
    pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
    print(json.dumps({k: v for k, v in row.items()}, default=str), flush=True)
    return res
