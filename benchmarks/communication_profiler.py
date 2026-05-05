"""
communication_profiler.py
--------------------------
Profiles AllReduce (gradient synchronization) latency vs. compute time
per training step in DDP training. Measures compute-to-communication
ratio across batch sizes and reports communication overhead %.

This is the core distributed systems experiment: it answers
"How much of our training time is spent waiting for gradient sync?"

Usage:
    torchrun --nproc_per_node=4 --master_port=29501 \\
        benchmarks/communication_profiler.py \\
        --batch_sizes 16 32 64 128 \\
        --steps 30

Output:
    results/tables/comm_profile_W{world_size}.json
    results/plots/comm_overhead.png  (rank 0 only)
"""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Proxy model (mimics AVGL++ Stage 1 forward cost)
# ---------------------------------------------------------------------------

class ProxyStage1(nn.Module):
    """
    Lightweight proxy model that approximates the compute load of
    CoarseLocalizer without importing the full model stack.
    Useful for isolated communication benchmarking.
    """

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),   nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(512, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.conv_stack(x).flatten(1)), p=2, dim=-1)


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

@contextmanager
def cuda_timer():
    """Yields elapsed milliseconds via CUDA events (accurate GPU timing)."""
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    yield
    e.record()
    torch.cuda.synchronize()
    cuda_timer._last_ms = s.elapsed_time(e)


# ---------------------------------------------------------------------------
# Core profiling loop
# ---------------------------------------------------------------------------

def profile_comm_vs_compute(args) -> list:
    """Run profiling across all requested batch sizes. Returns result list."""
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model = ProxyStage1(embed_dim=512).to(device)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=3e-4)

    results = []

    for bs in args.batch_sizes:
        dummy = torch.randn(bs, 3, 224, 224, device=device)

        compute_times, comm_times, total_times = [], [], []

        # Warmup
        for _ in range(5):
            optimizer.zero_grad()
            out = ddp_model(dummy)
            loss = out.norm()
            loss.backward()
            optimizer.step()

        # Profiling steps
        for step in range(args.steps):
            optimizer.zero_grad()

            # ── Compute (forward + backward, no sync yet) ──────────────────
            # Temporarily disable DDP gradient synchronization so we can
            # time compute and communication separately.
            with cuda_timer():
                with ddp_model.no_sync():
                    out = ddp_model(dummy)
                    loss = out.norm()
                    loss.backward()
            compute_ms = cuda_timer._last_ms

            # ── Communication (AllReduce across GPUs) ──────────────────────
            with cuda_timer():
                for param in ddp_model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            comm_ms = cuda_timer._last_ms

            optimizer.step()

            compute_times.append(compute_ms)
            comm_times.append(comm_ms)
            total_times.append(compute_ms + comm_ms)

        # Aggregate (skip first 3 steps as outliers)
        skip = 3
        avg_compute = sum(compute_times[skip:]) / len(compute_times[skip:])
        avg_comm    = sum(comm_times[skip:])    / len(comm_times[skip:])
        avg_total   = avg_compute + avg_comm
        c2c_ratio   = avg_compute / max(avg_comm, 1e-6)
        comm_pct    = 100.0 * avg_comm / max(avg_total, 1e-6)
        throughput  = (bs * world_size * 1000.0) / avg_total  # images/sec

        if rank == 0:
            print(
                f"[Batch={bs:3d}] world={world_size} "
                f"Compute={avg_compute:.2f}ms  "
                f"AllReduce={avg_comm:.2f}ms  "
                f"C2C={c2c_ratio:.2f}x  "
                f"CommOverhead={comm_pct:.1f}%  "
                f"Throughput={throughput:.0f}img/s"
            )

        results.append({
            "batch_size":       bs,
            "world_size":       world_size,
            "avg_compute_ms":   round(avg_compute, 3),
            "avg_allreduce_ms": round(avg_comm, 3),
            "c2c_ratio":        round(c2c_ratio, 3),
            "comm_overhead_pct": round(comm_pct, 2),
            "throughput_img_s": round(throughput, 1),
        })

    dist.destroy_process_group()
    return results


# ---------------------------------------------------------------------------
# Plotting (rank 0 only)
# ---------------------------------------------------------------------------

def plot_results(results: list, output_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("[plot] matplotlib/pandas not available — skipping plots.")
        return

    df = pd.DataFrame(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Stacked bar: compute vs comm time
    ax = axes[0]
    x = range(len(df))
    ax.bar(x, df["avg_compute_ms"], label="Compute (ms)", color="steelblue")
    ax.bar(x, df["avg_allreduce_ms"], bottom=df["avg_compute_ms"],
           label="AllReduce (ms)", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([f"B={b}" for b in df["batch_size"]])
    ax.set_ylabel("Time (ms)")
    ax.set_title("Compute vs AllReduce Time per Step")
    ax.legend()

    # 2. C2C ratio
    ax2 = axes[1]
    ax2.plot(df["batch_size"], df["c2c_ratio"], "go-", linewidth=2)
    ax2.axhline(1.0, color="red", linestyle="--", label="C2C = 1 (equal)")
    ax2.set_xlabel("Batch Size (per GPU)")
    ax2.set_ylabel("Compute / Communication ratio")
    ax2.set_title("Compute-to-Communication Ratio")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Communication overhead %
    ax3 = axes[2]
    ax3.bar([f"B={b}" for b in df["batch_size"]], df["comm_overhead_pct"],
            color="mediumpurple")
    ax3.set_ylabel("Communication Overhead (%)")
    ax3.set_title("AllReduce Overhead as % of Total Step Time")
    ax3.axhline(10, color="green", linestyle="--", label="10% threshold")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Communication Profiling — world_size={results[0]['world_size']}")
    plt.tight_layout()
    out_path = output_dir / f"comm_overhead_W{results[0]['world_size']}.png"
    plt.savefig(out_path, dpi=150)
    print(f"[Saved] {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="HCVGLoc DDP Communication Profiler")
    p.add_argument("--batch_sizes", nargs="+", type=int, default=[16, 32, 64, 128])
    p.add_argument("--steps", type=int, default=30,
                   help="Number of profiling steps per batch size (warmup excluded)")
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--wandb", action="store_true",
                   help="Log run to wandb (rank 0 only).")
    p.add_argument("--wandb_project", type=str, default="hcvgloc-distributed")
    p.add_argument("--wandb_entity", type=str, default="w2c-lab")
    p.add_argument("--wandb_group", type=str, default="comm_profiler")
    return p.parse_args()


def _log_to_wandb(args, results, json_path: Path, plot_path: Path):
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; skipping wandb logging.")
        return

    world_size = results[0]["world_size"]
    run_name = f"comm_profile_W{world_size}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=run_name,
        config={
            "world_size": world_size,
            "batch_sizes": args.batch_sizes,
            "steps": args.steps,
        },
        reinit=True,
    )

    # Per-batch-size metrics as a wandb Table, plus one step per batch size.
    table = wandb.Table(columns=list(results[0].keys()))
    for row in results:
        table.add_data(*row.values())
        wandb.log({f"comm/{k}": v for k, v in row.items()
                   if k not in ("batch_size", "world_size")},
                  step=row["batch_size"])

    wandb.log({"comm_profile_table": table})
    if plot_path.exists():
        wandb.log({"comm_overhead_plot": wandb.Image(str(plot_path))})
    wandb.save(str(json_path))
    wandb.finish()
    print(f"[wandb] comm profiler run synced ({run_name})")


if __name__ == "__main__":
    args = parse_args()
    results = profile_comm_vs_compute(args)

    # Rank 0: save results and plot
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        out = Path(args.output_dir)
        world_size = results[0]["world_size"]

        table_dir = out / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        json_path = table_dir / f"comm_profile_W{world_size}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Saved] {json_path}")

        plot_results(results, out / "plots")
        plot_path = out / "plots" / f"comm_overhead_W{world_size}.png"

        if args.wandb:
            _log_to_wandb(args, results, json_path, plot_path)

        print("\n[Communication profiling complete]")
