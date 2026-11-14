"""Build models from a config dict."""

from __future__ import annotations

import torch.nn as nn

from agrispec.models.cnn import GatedClassifier, SmallResNet
from agrispec.models.gate import BandGate
from agrispec.models.transformer import CNNTransformer


def build_model(mcfg: dict, in_ch: int, num_classes: int) -> nn.Module:
    name = mcfg["name"]
    common = dict(widths=tuple(mcfg.get("widths", (32, 64, 128, 256))), blocks=tuple(mcfg.get("blocks", (1, 1, 1, 1))),
                  dropout=mcfg.get("dropout", 0.2))
    if name == "small_resnet":                       # B3 / B2-style
        enc = SmallResNet(in_ch, num_classes, **common)
    elif name == "small_resnet_se":                  # B6: SE on feature channels
        enc = SmallResNet(in_ch, num_classes, se=True, **common)
    elif name == "small_resnet_inse":                # B7: soft SE over input bands
        enc = SmallResNet(in_ch, num_classes, in_se=True, **common)
    elif name == "cnn_transformer":                  # B8 / PG-ASST encoder
        enc = CNNTransformer(in_ch, num_classes, embed_dim=mcfg.get("embed_dim", 128), heads=mcfg.get("heads", 4),
                             depth=mcfg.get("depth", 2), token_stage=mcfg.get("token_stage", 2),
                             fusion=mcfg.get("fusion", "gated"), **common)
    else:
        raise ValueError(name)
    gate_mode = mcfg.get("gate", "none")
    if gate_mode != "none":
        gate = BandGate(in_ch, mcfg["k"], mode=gate_mode, temperature=mcfg.get("gate_temperature", 1.0))
        return GatedClassifier(gate, enc)
    return enc
