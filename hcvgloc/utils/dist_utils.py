"""
dist_utils.py
--------------
DDP utility functions for HCVGLoc distributed training.

Provides:
    - Process group setup / teardown
    - Rank-aware print / log guards
    - All-reduce aggregation for metrics
    - CUDA event-based timing for benchmarking
    - Gradient norm computation
"""

import os
import time
import socket
from contextlib import contextmanager
from typing import Dict, Any

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------

def setup_distributed(backend: str = "nccl") -> tuple:
    """
    Initialise DDP process group from torchrun environment variables.

    torchrun sets: RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT

    Returns:
        (rank, local_rank, world_size)
    """
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)

    if world_size > 1:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        dist.barrier()

    return rank, local_rank, world_size


def teardown_distributed():
    """Cleanly destroy the process group."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """True only on rank 0 (or when not using DDP)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def barrier():
    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Metric aggregation across ranks
# ---------------------------------------------------------------------------

def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a scalar tensor across all DDP ranks."""
    if not dist.is_initialized():
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t


def reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_reduce_dict(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Average a dict of scalar tensors across all ranks.

    Usage:
        metrics = {'loss': loss_tensor, 'nce': nce_tensor}
        reduced = all_reduce_dict(metrics)
        # reduced = {'loss': 0.432, 'nce': 0.187}
    """
    if not dist.is_initialized():
        return {k: v.item() for k, v in metrics.items()}

    keys = sorted(metrics.keys())
    tensor = torch.stack([metrics[k].float() for k in keys])
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return {k: tensor[i].item() for i, k in enumerate(keys)}


# ---------------------------------------------------------------------------
# CUDA timing utilities
# ---------------------------------------------------------------------------

class CUDATimer:
    """
    Accurate GPU timing using CUDA events.

    Usage:
        timer = CUDATimer()
        with timer.measure("forward"):
            output = model(x)
        print(timer.elapsed_ms("forward"))
    """

    def __init__(self):
        self._times: Dict[str, float] = {}

    @contextmanager
    def measure(self, label: str = "default"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        self._times[label] = start.elapsed_time(end)   # milliseconds

    def elapsed_ms(self, label: str = "default") -> float:
        return self._times.get(label, 0.0)

    def reset(self):
        self._times.clear()

    def summary(self) -> Dict[str, float]:
        return dict(self._times)


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------

def compute_grad_norm(model: torch.nn.Module, norm_type: float = 2.0) -> float:
    """Compute the global gradient norm across all model parameters."""
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return 0.0
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]),
        norm_type,
    )
    return total_norm.item()


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def print_rank0(*args, **kwargs):
    """Print only from rank 0."""
    if is_main_process():
        print(*args, **kwargs)


def format_eta(seconds: float) -> str:
    """Format seconds into human-readable ETA string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"
