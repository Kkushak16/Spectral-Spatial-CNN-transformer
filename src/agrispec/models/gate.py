"""Static sparse band gate (module M1 / A of the brief).

One trainable logit per input band, shared across all images (static selection,
see decision D-006). Two relaxations are provided:

* ``topk_st``  - straight-through top-k: the forward pass uses an exact k-hot mask,
  the backward pass uses the gradient of a temperature-scaled sigmoid.
* ``hard_concrete`` - Louizos et al. 2018 stretched hard-concrete gates with an
  expected-L0 penalty; at evaluation the k bands with the largest log-alpha are kept.

At evaluation both gates return an exact k-hot {0,1} mask, so a model whose first
layer is linear in the input (a conv) can be rewritten as a genuine k-channel model
by slicing that layer's weights (see ``export_k_channel``).
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


class BandGate(nn.Module):
    def __init__(self, n_bands: int = 12, k: int = 12, mode: str = "topk_st", temperature: float = 1.0,
                 beta: float = 2 / 3, gamma: float = -0.1, zeta: float = 1.1, init_logit: float = 0.0,
                 init_noise: float = 0.01):
        super().__init__()
        if not 1 <= k <= n_bands:
            raise ValueError(f"k={k} must be in [1, {n_bands}]")
        if mode not in {"topk_st", "hard_concrete", "none"}:
            raise ValueError(mode)
        self.n_bands, self.k, self.mode = n_bands, k, mode
        self.temperature = temperature
        self.beta, self.gamma, self.zeta = beta, gamma, zeta
        # small noise breaks ties so top-k is well defined from step 0
        self.logits = nn.Parameter(init_logit + init_noise * torch.randn(n_bands))
        self.register_buffer("frozen_mask", torch.empty(0))

    # ---- selection ---------------------------------------------------------------
    def selected_indices(self) -> torch.Tensor:
        """Sorted indices of the k kept bands (deterministic, used at evaluation)."""
        if self.frozen_mask.numel():
            return torch.nonzero(self.frozen_mask > 0).flatten()
        return torch.topk(self.logits.detach(), self.k).indices.sort().values

    def hard_mask(self) -> torch.Tensor:
        m = torch.zeros(self.n_bands, device=self.logits.device)
        m[self.selected_indices()] = 1.0
        return m

    def freeze(self) -> None:
        """Fix the current selection (stage 4: fine-tune a genuine k-channel model)."""
        self.frozen_mask = self.hard_mask().detach().clone()
        self.logits.requires_grad_(False)

    # ---- training-time masks -----------------------------------------------------
    def _hard_concrete_sample(self) -> torch.Tensor:
        u = torch.rand_like(self.logits).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.logits) / self.beta)
        return (s * (self.zeta - self.gamma) + self.gamma).clamp(0, 1)

    def train_mask(self) -> torch.Tensor:
        if self.mode == "hard_concrete":
            return self._hard_concrete_sample()
        # straight-through top-k
        soft = torch.sigmoid(self.logits / self.temperature)
        hard = self.hard_mask()
        return hard + soft - soft.detach()

    def mask(self) -> torch.Tensor:
        if self.mode == "none" or self.k == self.n_bands and not self.frozen_mask.numel():
            return torch.ones(self.n_bands, device=self.logits.device)
        if self.training and not self.frozen_mask.numel():
            return self.train_mask()
        return self.hard_mask()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask().view(1, -1, 1, 1)

    # ---- regularisers --------------------------------------------------------------
    def expected_l0(self) -> torch.Tensor:
        """Expected number of open gates (hard-concrete) or sum of sigmoid probabilities (top-k)."""
        if self.mode == "hard_concrete":
            return torch.sigmoid(self.logits - self.beta * math.log(-self.gamma / self.zeta)).sum()
        return torch.sigmoid(self.logits / self.temperature).sum()

    def sparsity_loss(self) -> torch.Tensor:
        """Penalise deviation of the expected open-gate count from k (normalised)."""
        return ((self.expected_l0() - self.k) / self.n_bands) ** 2

    def probabilities(self) -> torch.Tensor:
        return torch.sigmoid(self.logits.detach())


def _first_conv(module: nn.Module) -> tuple[nn.Module, str, nn.Conv2d]:
    """Return (parent, attribute name, conv) for the first Conv2d that consumes the input."""
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            parent = module
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            return parent, parts[-1], m
    raise ValueError("no Conv2d found")


class KChannelModel(nn.Module):
    """Deployment model: reads only the k selected bands. ``band_indices`` refer to the
    12-band input order; callers must pass x[:, band_indices] (or use ``from_full``)."""

    def __init__(self, encoder: nn.Module, band_indices: torch.Tensor):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("band_indices", band_indices.clone())

    def forward(self, x_k: torch.Tensor):
        return self.encoder(x_k)

    def from_full(self, x_full: torch.Tensor):
        return self.encoder(x_full[:, self.band_indices])


def export_k_channel(gate: BandGate, encoder: nn.Module) -> KChannelModel:
    """Build a genuine k-input-channel copy of ``encoder``.

    Valid because at evaluation the gate multiplies unselected bands by exactly 0 and
    selected ones by exactly 1, and the first layer is a convolution (linear in input),
    so dropping the zeroed input channels and their weight slices is an identity.
    Padding mode must be 'zeros' (zero-padded zero channels contribute nothing).
    """
    idx = gate.selected_indices().cpu()
    enc = copy.deepcopy(encoder)
    parent, attr, conv = _first_conv(enc)
    if conv.groups != 1:
        raise ValueError("first conv must have groups=1")
    if conv.padding_mode != "zeros":
        raise ValueError("first conv must use zero padding")
    new = nn.Conv2d(len(idx), conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    conv.dilation, 1, conv.bias is not None, padding_mode=conv.padding_mode)
    with torch.no_grad():
        new.weight.copy_(conv.weight[:, idx])
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    setattr(parent, attr, new)
    return KChannelModel(enc.eval(), idx)
