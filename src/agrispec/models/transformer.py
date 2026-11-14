"""CNN -> Transformer hybrid (baseline B8; encoder of PG-ASST, modules B-D of the brief).

The CNN stage at stride 8 (8x8 = 64 region tokens for a 64x64 patch, decision D-007) is projected to
``embed_dim`` tokens with learned 2-D positional embeddings, passed through ``depth`` pre-norm Transformer
layers, mean-pooled, and fused with the pooled final CNN features by a learned sigmoid gate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from agrispec.models.cnn import SmallResNet


class CNNTransformer(nn.Module):
    def __init__(self, in_ch: int = 12, num_classes: int = 10, widths=(32, 64, 128, 256), blocks=(1, 1, 1, 1),
                 token_stage: int = 2, embed_dim: int = 128, heads: int = 4, depth: int = 2, mlp_ratio: float = 2.0,
                 dropout: float = 0.2, attn_dropout: float = 0.0, fusion: str = "gated", in_se: bool = False):
        super().__init__()
        self.cnn = SmallResNet(in_ch, num_classes, widths, blocks, dropout=dropout, in_se=in_se)
        self.token_stage = token_stage
        tok_ch = widths[token_stage]
        grid = 64 // (2 * 2 ** token_stage)        # stem stride 2, stage i>0 stride 2
        self.grid = grid
        self.proj = nn.Conv2d(tok_ch, embed_dim, 1)
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(embed_dim, heads, int(embed_dim * mlp_ratio), attn_dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False) if depth > 0 else None
        self.norm = nn.LayerNorm(embed_dim)
        self.cnn_proj = nn.Linear(self.cnn.out_dim, embed_dim)
        self.fusion = fusion
        if fusion == "gated":
            self.fuse_gate = nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.Sigmoid())
            out = embed_dim
        elif fusion == "concat":
            out = 2 * embed_dim
        else:
            raise ValueError(fusion)
        self.out_dim = out
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out, num_classes))
        self.last_attn_tokens: torch.Tensor | None = None

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        maps = self.cnn.feature_maps(x)
        tok = self.proj(maps[self.token_stage]).flatten(2).transpose(1, 2) + self.pos   # (B, T, D)
        if self.transformer is not None:
            tok = self.transformer(tok)
        t = self.norm(tok).mean(1)
        c = self.cnn_proj(self.cnn.pool(maps[-1]))
        if self.fusion == "gated":
            g = self.fuse_gate(torch.cat([c, t], 1))
            return g * c + (1 - g) * t
        return torch.cat([c, t], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))
