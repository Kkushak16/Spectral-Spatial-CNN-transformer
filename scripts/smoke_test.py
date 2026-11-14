"""Phase 0 gate: smoke test on random (B, 12, 64, 64) tensors.

Checks
  1. the gate masks bands (unselected channels are exactly zero at eval),
  2. a k-channel copy with unselected channels physically removed runs,
  3. the k-channel copy reproduces the masked model's predictions,
  4. one training step updates the gate logits (gradient flows through the gate),
  5. cost report (params / FLOPs / latency) is produced.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from agrispec.models import BandGate, GatedClassifier, SmallResNet, export_k_channel
from agrispec.utils.config import load_config, seed_everything
from agrispec.utils.profiling import cost_report


def check(k: int, mode: str) -> dict:
    gate = BandGate(12, k, mode=mode)
    enc = SmallResNet(in_ch=12, num_classes=10)
    model = GatedClassifier(gate, enc)
    x = torch.randn(8, 12, 64, 64)
    y = torch.randint(0, 10, (8,))

    # one training step: gradient must reach the gate logits
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    before = gate.logits.detach().clone()
    loss = F.cross_entropy(model(x), y) + 0.1 * gate.sparsity_loss()
    loss.backward()
    assert gate.logits.grad is not None and gate.logits.grad.abs().sum() > 0, "no gradient on gate"
    opt.step()
    assert not torch.equal(before, gate.logits.detach()), "gate logits did not move"

    model.eval()
    idx = gate.selected_indices()
    assert len(idx) == k, f"gate kept {len(idx)} bands, expected {k}"
    masked = gate(x)
    unsel = [i for i in range(12) if i not in set(idx.tolist())]
    assert masked[:, unsel].abs().sum() == 0, "unselected bands not zeroed"
    assert torch.equal(masked[:, idx], x[:, idx]), "selected bands altered"

    kmodel = export_k_channel(gate, enc)
    x_k = x[:, idx].contiguous()
    assert x_k.shape[1] == k
    with torch.no_grad():
        ref = model(x)
        out = kmodel(x_k)
    max_err = (ref - out).abs().max().item()
    assert torch.allclose(ref, out, atol=1e-4), f"k-channel mismatch {max_err}"
    assert torch.equal(ref.argmax(1), out.argmax(1))
    cost = cost_report(kmodel, k)
    return {"k": k, "mode": mode, "bands": idx.tolist(), "max_abs_diff": max_err, **cost}


def main() -> int:
    cfg = load_config("configs/base.yaml")
    seed_everything(cfg["seed"])
    torch.set_num_threads(cfg.get("num_threads", 8))
    rows = []
    for mode in ("topk_st", "hard_concrete"):
        for k in cfg["k_grid"]:
            rows.append(check(k, mode))
    print(f"{'mode':14s} {'k':>2s} {'bands':38s} {'max|diff|':>10s} {'params':>8s} {'MFLOPs':>8s} {'ms/img':>7s}")
    for r in rows:
        print(f"{r['mode']:14s} {r['k']:2d} {str(r['bands']):38s} {r['max_abs_diff']:10.2e} "
              f"{r['params']:8d} {r['flops'] / 1e6:8.1f} {r['latency_ms_median']:7.2f}")
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
