import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from smoke_test import check  # noqa: E402

from agrispec.models import BandGate


@pytest.mark.parametrize("mode", ["topk_st", "hard_concrete"])
@pytest.mark.parametrize("k", [3, 4, 6, 8, 12])
def test_gate_and_k_channel_export(mode, k):
    torch.manual_seed(0)
    r = check(k, mode)
    assert len(r["bands"]) == k
    assert r["max_abs_diff"] < 1e-4


def test_frozen_gate_keeps_selection():
    g = BandGate(12, 4, mode="topk_st")
    g.freeze()
    sel = g.selected_indices().tolist()
    with torch.no_grad():
        g.logits.add_(torch.randn(12) * 10)
    assert g.selected_indices().tolist() == sel
    g.train()
    m = g.mask()
    assert m.sum().item() == 4
