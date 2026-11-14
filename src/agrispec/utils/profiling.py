"""Cost reporting: parameters, FLOPs, CPU latency, peak memory."""

from __future__ import annotations

import time
import tracemalloc

import torch
import torch.nn as nn


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def count_flops(model: nn.Module, x: torch.Tensor) -> int | None:
    """Multiply-accumulate count (fvcore convention: 1 FMA = 1 'flop'). None if fvcore unavailable."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None
    model.eval()
    with torch.no_grad():
        fca = FlopCountAnalysis(model, x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return int(fca.total())


@torch.no_grad()
def measure_latency(model: nn.Module, x: torch.Tensor, warmup: int = 5, reps: int = 30) -> dict:
    """Median per-batch wall-clock latency on the current device (ms)."""
    model.eval()
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"latency_ms_median": times[len(times) // 2], "latency_ms_p90": times[int(0.9 * len(times))],
            "batch_size": x.shape[0]}


@torch.no_grad()
def peak_cpu_memory_mb(model: nn.Module, x: torch.Tensor) -> float:
    """Peak Python-tracked allocation during one forward pass (approximate on CPU)."""
    model.eval()
    tracemalloc.start()
    model(x)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 2**20


def cost_report(model: nn.Module, in_ch: int, size: int = 64, batch: int = 1) -> dict:
    x = torch.randn(batch, in_ch, size, size)
    rep = {"params": count_params(model), "flops": count_flops(model, x), "input_bands": in_ch}
    rep.update(measure_latency(model, x))
    return rep
