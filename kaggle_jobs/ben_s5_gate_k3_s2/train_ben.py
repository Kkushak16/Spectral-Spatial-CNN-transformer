"""Train the CNN-Transformer on BigEarthNet v2 / reBEN (multi-label, 19 classes).

K = 12 is baseline B8 (all bands). K < 12 is S5: a learned top-k band gate. The gate learns during the
first 70 % of the epochs, the bands are then frozen and the network fine-tunes on them, and the final
model is exported with a genuine k-channel first layer, so unselected bands are never read.

Runs as a Kaggle script (MODE below selects smoke or full) and locally for testing:
    python kaggle_jobs/ben_b8/train_ben.py --mode smoke --workers 0 --epochs 1 \
        --data D:/ben_shards --support D:/ben_support --out <dir>

Inputs (found by file name, so Kaggle's mount path does not matter)
    index.csv + shard_*.npz          from scripts/pack_bigearthnet.py
    metadata.parquet + model *.py    support dataset (labels, src/agrispec/models)

Outputs: config.json, norm.json, history.csv, best.pt, metrics.json, preds_val.npz, preds_test.npz
Model selection uses validation macro mAP; the test split is evaluated once, at the end.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

MODE = "full"  # "smoke" = small end-to-end check, "full" = whole dataset
SEEDS = [2]  # trained one after the other, each saved to <out>/seed_<n>
K = 3  # bands kept by the gate; 12 = no gate (baseline B8)
GATE = "topk_st"  # straight-through top-k
FREEZE_AT = 0.7  # fraction of epochs after which the selected bands are frozen
INPUT_BANDS = None  # fixed input bands, e.g. ["B04", "B03", "B02"]; None = all 12

PRESETS = {
    "smoke": {"epochs": 2, "train_shards": 4, "eval_shards": 2, "stat_shards": 2},
    "full": {"epochs": 20, "train_shards": 0, "eval_shards": 0, "stat_shards": 12},
}
# Canonical band order of the project (D-009): B01..B08, B8A, B09, B11, B12, as (group, position in group)
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
CANON = [("b60", 0), ("b10", 0), ("b10", 1), ("b10", 2), ("b20", 0), ("b20", 1), ("b20", 2), ("b10", 3),
         ("b20", 3), ("b60", 1), ("b20", 4), ("b20", 5)]
IMG = 120


def find_one(root: str, name: str) -> Path:
    hits = glob.glob(os.path.join(glob.escape(root), "**", name), recursive=True)
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one {name} under {root}, found {hits}")
    return Path(hits[0])


def import_build_model(support: Path):
    """Use the project source when present, otherwise the copy in the support dataset."""
    here = Path(__file__).resolve().parents if "__file__" in globals() else []
    src = next((d / "src" for d in here if (d / "src" / "agrispec" / "models" / "factory.py").exists()), None)
    if src is not None:
        sys.path.insert(0, str(src))
    else:
        pkg = Path("/tmp/agrispec_pkg/agrispec")
        (pkg / "models").mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").touch()
        (pkg / "models" / "__init__.py").touch()
        for f in ("cnn.py", "gate.py", "transformer.py", "factory.py"):
            shutil.copy(support / f, pkg / "models" / f)
        sys.path.insert(0, str(pkg.parent))
    from agrispec.models.factory import build_model
    return build_model


class Split:
    """Per-shard patch ids and multi-hot labels for one split, in shard row order."""

    def __init__(self, root: Path, frame: pd.DataFrame, classes: list[str], max_shards: int):
        self.root = root
        pos = {c: i for i, c in enumerate(classes)}
        names = sorted(frame["shard"].unique())
        if max_shards:
            names = names[:: max(1, len(names) // max_shards)][:max_shards]
        self.shards, offset = {}, 0
        for name, g in frame[frame["shard"].isin(names)].groupby("shard", sort=True):
            g = g.sort_values("row")
            if not np.array_equal(g["row"].to_numpy(), np.arange(len(g))):
                raise ValueError(f"{name}: rows are not contiguous")
            y = np.zeros((len(g), len(classes)), np.uint8)
            for i, labels in enumerate(g["labels"]):
                y[i, [pos[c] for c in labels]] = 1
            self.shards[name] = (g["patch_id"].to_numpy(dtype=str), y, offset)
            offset += len(g)
        self.n = offset

    def ids(self) -> np.ndarray:
        return np.concatenate([v[0] for v in self.shards.values()])

    def load(self, name: str):
        ids, y, offset = self.shards[name]
        with np.load(self.root / name) as z:
            if not np.array_equal(z["patch_id"].astype(str), ids):
                raise ValueError(f"{name}: patch ids do not match index.csv")
            return z["b10"], z["b20"], z["b60"], y, np.arange(offset, offset + len(ids))


class ShardStream(IterableDataset):
    """Yields ready-made batches. Training mixes `mix` random shards at a time, because each
    shard covers only one or two Sentinel-2 tiles and would otherwise give very uniform batches."""

    def __init__(self, split: Split, batch_size: int, train: bool, mix: int = 4, seed: int = 0):
        self.split, self.bs, self.train, self.mix, self.seed = split, batch_size, train, mix, seed
        self.epoch = 0

    def _batch(self, parts, i):
        return {"b10": torch.from_numpy(parts[0][i].astype(np.float32)),
                "b20": torch.from_numpy(parts[1][i].astype(np.float32)),
                "b60": torch.from_numpy(parts[2][i].astype(np.float32)),
                "y": torch.from_numpy(parts[3][i].astype(np.float32)),
                "idx": torch.from_numpy(parts[4][i])}

    def __iter__(self):
        info = get_worker_info()
        wid, n = (info.id, info.num_workers) if info else (0, 1)
        order = list(self.split.shards)
        rng = np.random.default_rng([self.seed, self.epoch, wid])
        if self.train:
            np.random.default_rng([self.seed, self.epoch]).shuffle(order)
        mine = order[wid::n]
        step = self.mix if self.train else 1
        carry = None
        for k in range(0, len(mine), step):
            loaded = [self.split.load(s) for s in mine[k : k + step]]
            if carry is not None:
                loaded.append(carry)
            parts = [np.concatenate(c) for c in zip(*loaded)]
            perm = rng.permutation(len(parts[0])) if self.train else np.arange(len(parts[0]))
            end = len(perm) - len(perm) % self.bs if self.train else len(perm)
            for s in range(0, end, self.bs):
                yield self._batch(parts, perm[s : s + self.bs])
            rest = perm[end:]
            carry = tuple(p[rest] for p in parts) if len(rest) else None


def loader(ds: ShardStream, workers: int) -> DataLoader:
    return DataLoader(ds, batch_size=None, num_workers=workers, pin_memory=torch.cuda.is_available(),
                      prefetch_factor=4 if workers else None)


def band_stats(split: Split, n_shards: int):
    names = list(split.shards)
    names = [names[i] for i in np.linspace(0, len(names) - 1, min(n_shards, len(names))).astype(int)]
    count, s1, s2 = np.zeros(12), np.zeros(12), np.zeros(12)
    for name in names:
        b10, b20, b60, *_ = split.load(name)
        groups = {"b10": b10, "b20": b20, "b60": b60}
        for c, (g, j) in enumerate(CANON):
            v = groups[g][:, j].astype(np.float64)
            count[c] += v.size
            s1[c] += v.sum()
            s2[c] += (v * v).sum()
    mean = s1 / count
    return mean, np.sqrt(np.maximum(s2 / count - mean**2, 1e-6))


def compose(b: dict, device, mean, std) -> torch.Tensor:
    """Upsample 20 m and 60 m bands to 120x120 and stack in canonical order, normalised."""
    g = {"b10": b["b10"].to(device, non_blocking=True)}
    for k in ("b20", "b60"):
        g[k] = F.interpolate(b[k].to(device, non_blocking=True), size=(IMG, IMG), mode="bilinear", align_corners=False)
    x = torch.stack([g[k][:, j] for k, j in CANON], 1)
    return (x - mean) / std


def augment(x: torch.Tensor) -> torch.Tensor:
    n = len(x)
    m = torch.rand(n, device=x.device) < 0.5
    x[m] = x[m].flip(-1)
    m = torch.rand(n, device=x.device) < 0.5
    x[m] = x[m].flip(-2)
    m = torch.rand(n, device=x.device) < 0.5
    x[m] = x[m].rot90(1, (-2, -1))
    return x


def metrics(y: np.ndarray, p: np.ndarray, classes: list[str]) -> dict:
    ap = np.full(len(classes), np.nan)
    for c in range(len(classes)):
        if y[:, c].any():
            ap[c] = average_precision_score(y[:, c], p[:, c])
    hard = (p >= 0.5).astype(np.uint8)
    present = y.any(0)
    return {"macro_mAP": float(np.nanmean(ap)), "micro_mAP": float(average_precision_score(y.ravel(), p.ravel())),
            "macro_f1": float(f1_score(y[:, present], hard[:, present], average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y, hard, average="micro", zero_division=0)),
            "classes_scored": int(present.sum()), "n": int(len(y)),
            "per_class_AP": {c: (None if np.isnan(v) else round(float(v), 4)) for c, v in zip(classes, ap)}}


@torch.no_grad()
def predict(model, dl, n, n_cls, device, mean, std, amp):
    model.eval()
    P, Y = np.zeros((n, n_cls), np.float32), np.zeros((n, n_cls), np.uint8)
    seen = 0
    for b in dl:
        with torch.autocast("cuda", enabled=amp):
            logits = getattr(model, "from_full", model)(compose(b, device, mean, std))
        i = b["idx"].numpy()
        P[i] = torch.sigmoid(logits.float()).cpu().numpy()
        Y[i] = b["y"].numpy().astype(np.uint8)
        seen += len(i)
    if seen != n:
        raise RuntimeError(f"evaluation saw {seen} patches, expected {n}")
    return Y, P


def main() -> None:
    kaggle = os.path.isdir("/kaggle/input")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=MODE, choices=list(PRESETS))
    ap.add_argument("--data", default="/kaggle/input")
    ap.add_argument("--support", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working" if kaggle else "results/ben_local")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--bands", nargs="+", default=INPUT_BANDS or BANDS, choices=BANDS)
    ap.add_argument("--max-hours", type=float, default=4.5, help="per seed: stop training early to leave time for testing")
    args, _ = ap.parse_known_args()
    base = Path(args.out)
    for seed in args.seeds:
        run(argparse.Namespace(**{**vars(args), "seed": seed, "out": str(base / f"seed_{seed}")}))
        torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> None:
    preset = PRESETS[args.mode]
    epochs = args.epochs or preset["epochs"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    gpu = torch.cuda.get_device_name(0) if amp else "none"
    print(f"mode={args.mode} device={device} gpu={gpu} torch={torch.__version__} cpus={os.cpu_count()}", flush=True)
    if args.mode == "full" and not amp:
        raise SystemExit("full mode needs a GPU, none is available")

    index_path = find_one(args.data, "index.csv")
    meta_path = find_one(args.support, "metadata.parquet")
    build_model = import_build_model(meta_path.parent)
    index = pd.read_csv(index_path)
    meta = pd.read_parquet(meta_path, columns=["patch_id", "labels", "split"])
    frame = index.merge(meta, on="patch_id", how="left", validate="one_to_one", suffixes=("", "_meta"))
    if frame["labels"].isna().any() or (frame["split"] != frame["split_meta"]).any():
        raise SystemExit("index.csv and metadata.parquet disagree on patches or splits")
    classes = sorted({c for labels in meta["labels"] for c in labels})
    if len(classes) != 19:
        raise SystemExit(f"expected 19 classes, found {len(classes)}")

    root = index_path.parent
    def part(name, cap):
        return Split(root, frame[frame["split"] == name], classes, cap)
    tr, va, te = part("train", preset["train_shards"]), part("validation", preset["eval_shards"]), \
        part("test", preset["eval_shards"])
    print(f"patches train={tr.n:,} val={va.n:,} test={te.n:,} shards={len(tr.shards)}/{len(va.shards)}/{len(te.shards)}",
          flush=True)

    t0 = time.time()
    mean, std = band_stats(tr, preset["stat_shards"])
    print(f"band stats in {time.time() - t0:.0f}s  " + "  ".join(f"{b}:{m:.0f}/{s:.0f}" for b, m, s in zip(BANDS, mean, std)),
          flush=True)
    (out / "norm.json").write_text(json.dumps({"bands": BANDS, "mean": mean.tolist(), "std": std.tolist()}, indent=2))
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=device).view(1, -1, 1, 1)

    mcfg = {"name": "cnn_transformer", "embed_dim": 128, "heads": 4, "depth": 2, "token_stage": 2, "img_size": IMG}
    gated = args.k < 12
    freeze_ep = max(1, int(FREEZE_AT * epochs))
    if gated:
        if epochs < 2:
            raise SystemExit("the band gate needs at least 2 epochs")
        mcfg.update(gate=GATE, k=args.k)
    if gated and len(args.bands) != 12:
        raise SystemExit("the band gate selects from all 12 bands")
    model = build_model(mcfg, len(args.bands), len(classes)).to(device)
    if len(args.bands) != 12:  # the model only ever sees the chosen bands
        keep = torch.tensor([BANDS.index(b) for b in args.bands], device=device)
        model.register_forward_pre_hook(lambda _m, inputs: (inputs[0][:, keep],))
    n_params = sum(p.numel() for p in model.parameters())
    cfg = {**vars(args), "epochs": epochs, "model": mcfg, "classes": classes, "bands": args.bands, "params": n_params,
           "n_train": tr.n, "n_val": va.n, "n_test": te.n, "gpu": gpu}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"model params={n_params:,}", flush=True)

    steps = max(1, tr.n // args.batch_size)
    gate_params = list(model.gate.parameters()) if gated else []
    gate_ids = {id(p) for p in gate_params}
    groups = [{"params": [p for p in model.parameters() if id(p) not in gate_ids]}]
    if gated:
        groups.append({"params": gate_params, "weight_decay": 0.0})  # decay must not pull the band scores together
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=epochs * steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    tr_ds = ShardStream(tr, args.batch_size, True, seed=args.seed)
    va_dl = loader(ShardStream(va, args.batch_size * 2, False), args.workers)
    te_dl = loader(ShardStream(te, args.batch_size * 2, False), args.workers)

    fields = ["epoch", "loss", "train_s", "train_patches_per_s", "val_s", "val_macro_mAP", "val_micro_mAP",
              "val_macro_f1", "val_micro_f1", "lr", "bands"]
    with open(out / "history.csv", "w", newline="") as fh:
        csv.writer(fh).writerow(fields)
    best, best_ep = -1.0, -1
    def current_bands():
        return [BANDS[i] for i in model.gate.selected_indices().tolist()] if gated else args.bands

    for ep in range(epochs):
        if gated and ep == freeze_ep:
            model.gate.freeze()
            print(f"bands frozen before epoch {ep}: {current_bands()}", flush=True)
        model.train()
        tr_ds.epoch = ep
        t0, total, seen, done = time.time(), 0.0, 0, 0
        for b in loader(tr_ds, args.workers):
            y = b["y"].to(device, non_blocking=True)
            x = augment(compose(b, device, mean_t, std_t))
            with torch.autocast("cuda", enabled=amp):
                logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            done += 1
            total += loss.item() * len(y)
            seen += len(y)
            if done % 200 == 0:
                print(f"  ep{ep} step {done}/{steps} loss={total / seen:.4f} "
                      f"{seen / (time.time() - t0):,.0f} patches/s", flush=True)
        train_s = time.time() - t0
        for _ in range(done, steps):  # keep the schedule aligned if workers dropped a few partial batches
            sched.step()
        t1 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = metrics(*predict(model, va_dl, va.n, len(classes), device, mean_t, std_t, amp), classes)
        val_s = time.time() - t1
        row = [ep, round(total / seen, 5), round(train_s, 1), round(seen / train_s, 1), round(val_s, 1),
               round(ev["macro_mAP"], 5), round(ev["micro_mAP"], 5), round(ev["macro_f1"], 5),
               round(ev["micro_f1"], 5), sched.get_last_lr()[0], "+".join(current_bands())]
        with open(out / "history.csv", "a", newline="") as fh:
            csv.writer(fh).writerow(row)
        print(f"ep{ep:02d} loss={row[1]:.4f} train={train_s / 60:.1f}min ({row[3]:,.0f}/s) val={val_s / 60:.1f}min "
              f"macro_mAP={ev['macro_mAP']:.4f} micro_mAP={ev['micro_mAP']:.4f} micro_f1={ev['micro_f1']:.4f}"
              + (f" bands={row[-1]}" if gated else ""), flush=True)
        if (not gated or ep >= freeze_ep) and ev["macro_mAP"] > best:  # only models with final, frozen bands
            best, best_ep = ev["macro_mAP"], ep
            torch.save({"model": model.state_dict(), "epoch": ep, "config": cfg}, out / "best.pt")
        elapsed = time.time() - t_start
        if elapsed + 1.2 * (train_s + val_s) + 2 * val_s > args.max_hours * 3600 and ep + 1 < epochs \
                and (not gated or ep >= freeze_ep):
            print(f"stopping after epoch {ep}: time budget of {args.max_hours} h reached", flush=True)
            break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    final_model = model
    extra = {}
    if gated:
        from agrispec.models.gate import export_k_channel
        model.eval()
        final_model = export_k_channel(model.gate, model.encoder).to(device).eval()
        with torch.no_grad():
            x = compose(next(iter(loader(ShardStream(va, 32, False), 0))), device, mean_t, std_t)
            gap = (model(x) - final_model.from_full(x)).abs().max().item()
        if gap > 1e-3:
            raise RuntimeError(f"k-channel model differs from the gated model (max logit gap {gap})")
        torch.save({"model": final_model.state_dict(), "bands": current_bands(), "config": cfg}, out / "best_k_channel.pt")
        extra = {"selected_bands": current_bands(), "k": args.k, "k_channel_logit_gap": gap,
                 "params_k_channel": sum(p.numel() for p in final_model.parameters()),
                 "gate_probabilities": dict(zip(BANDS, np.round(model.gate.probabilities().cpu().numpy(), 4).tolist()))}
        print(f"selected bands: {current_bands()}  k-channel logit gap={gap:.2e}", flush=True)
    res = {"name": f"S5_gate_k{args.k}_ben" if gated else f"B8_cnntr_{'rgb' if len(args.bands) == 3 else '12band'}_ben", **extra, "mode": args.mode, "seed": args.seed, "best_epoch": best_ep,
           "epochs_planned": epochs, "epochs_run": ep + 1, "params": n_params, "gpu": gpu}
    for name, dl, split in (("val", va_dl, va), ("test", te_dl, te)):
        y, p = predict(final_model, dl, split.n, len(classes), device, mean_t, std_t, amp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res[name] = metrics(y, p, classes)
        np.savez_compressed(out / f"preds_{name}.npz", patch_id=split.ids(), y=y, p=p.astype(np.float16))
    res["total_hours"] = round((time.time() - t_start) / 3600, 3)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print("FINAL " + json.dumps({k: res[k] for k in ("best_epoch", "epochs_run", "total_hours")}), flush=True)
    for name in ("val", "test"):
        r = res[name]
        print(f"FINAL {name}: macro_mAP={r['macro_mAP']:.4f} micro_mAP={r['micro_mAP']:.4f} "
              f"macro_f1={r['macro_f1']:.4f} micro_f1={r['micro_f1']:.4f} n={r['n']:,}", flush=True)


if __name__ == "__main__":
    main()
