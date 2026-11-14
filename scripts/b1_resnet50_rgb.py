"""Baseline B1: reproduce Helber et al. 2019 (ResNet-50, ImageNet-pretrained, RGB, 80/20 class-wise split,
reported 98.57 %). Phase 7 gate: land within ~1 point.

Uses the RGB JPEG release (EuroSAT_RGB/), resized 64 -> 224, ImageNet normalisation, flips/rot90.
Protocol from the paper: first train the last layer (lr 0.01), then fine-tune the whole network (lr 1e-3 -> 1e-4).
Runs on CUDA if available (recommended: ~1-2 min/epoch on a T4); on this CPU ~70 min/epoch.

  python scripts/b1_resnet50_rgb.py --head-epochs 1 --ft-epochs 5 --seed 0
  python scripts/b1_resnet50_rgb.py --split grouped   # same model on our leakage-safe split (uses MS->RGB order map)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_rgb(split: str, seed: int):
    files = sorted((ROOT / "EuroSAT_RGB").glob("*/*.jpg"))
    classes = sorted({f.parent.name for f in files})
    y = np.array([classes.index(f.parent.name) for f in files])
    X = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files]).transpose(0, 3, 1, 2)  # uint8
    if split == "paper80_20":
        tr, te = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=seed).split(X, y))
        return X, y, tr, te, classes
    # our grouped split: map JPG names to MS paths (same class + same number)
    df = pd.read_csv(ROOT / f"splits/eurosat_{split}_s0.csv")
    key = {Path(p).stem: s for p, s in zip(df["path"], df["split"])}
    s = np.array([key[f.stem] for f in files])
    return X, y, np.nonzero(s != "test")[0], np.nonzero(s == "test")[0], classes


def batches(X, y, idx, bs, train, device, gen):
    order = idx[torch.randperm(len(idx), generator=gen).numpy()] if train else idx
    for s in range(0, len(order), bs):
        b = order[s:s + bs]
        x = torch.from_numpy(X[b]).float() / 255.0
        if train:
            if torch.rand(1, generator=gen) < 0.5: x = x.flip(-1)
            if torch.rand(1, generator=gen) < 0.5: x = x.flip(-2)
            if torch.rand(1, generator=gen) < 0.5: x = x.transpose(-1, -2)
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        yield ((x - MEAN) / STD).to(device), torch.from_numpy(y[b]).to(device)


def evaluate(model, X, y, idx, device):
    model.eval(); correct = 0
    with torch.no_grad():
        for x, t in batches(X, y, idx, 128, False, device, None):
            correct += (model(x).argmax(1) == t).sum().item()
    return correct / len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="paper80_20")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head-epochs", type=int, default=1)
    ap.add_argument("--ft-epochs", type=int, default=5)
    a = ap.parse_args()
    from torchvision.models import ResNet50_Weights, resnet50
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, tr, te, classes = load_rgb(a.split, a.seed)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)
    gen = torch.Generator().manual_seed(a.seed)
    log = []
    phases = [("head", a.head_epochs, 1e-2, model.fc.parameters()), ("finetune", a.ft_epochs, 1e-3, model.parameters())]
    for name, epochs, lr, params in phases:
        if epochs == 0:
            continue
        for p in model.parameters():
            p.requires_grad = name == "finetune"
        for p in model.fc.parameters():
            p.requires_grad = True
        opt = torch.optim.SGD([p for p in params if p.requires_grad], lr=lr, momentum=0.9, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr / 10)
        for ep in range(epochs):
            model.train(); t0 = time.time()
            for x, t in batches(X, y, tr, 64, True, device, gen):
                loss = F.cross_entropy(model(x), t)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            acc = evaluate(model, X, y, te, device)
            log.append({"phase": name, "epoch": ep, "test_acc": acc, "time_s": time.time() - t0})
            print(log[-1], flush=True)
    out = ROOT / "results" / f"B1_resnet50_rgb_{a.split}_s{a.seed}.json"
    json.dump({"split": a.split, "seed": a.seed, "final_test_acc": log[-1]["test_acc"], "log": log,
               "published": 0.9857, "note": "paper80_20 has no validation set (as in Helber et al.)"}, open(out, "w"), indent=2)
    print("final", log[-1]["test_acc"])


if __name__ == "__main__":
    main()
